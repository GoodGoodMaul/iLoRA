import json
import math
import re
import shutil
import string
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root so we can import shared utils from root-level package.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.config import PeftConfig
from peft.tuners.lora import LoraLayer, Linear
from peft.tuners.lora.bnb import Linear8bitLt
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformers import PreTrainedModel

from args import Args
from iLoRA_model import ILoRAMatrix

from wrapperbase import AverageMeter, WrapperBase

args = Args()
args.checkpoint_path = "checkpoints/best_model_iLoRA"


def _apply_lora_override(
    input_tensor: torch.Tensor, override_weight: torch.Tensor
) -> torch.Tensor:
    """
    Apply a LoRA override weight that may be shared across the batch or
    specified per-sample.
    """
    if override_weight.dim() == 2:
        return F.linear(input_tensor, override_weight)

    if override_weight.dim() == 3:
        weight_t = override_weight.transpose(-1, -2)
        # The model sometimes presents inputs as (seq_len, batch, dim) instead
        # of (batch, seq_len, dim); align the override accordingly.
        if input_tensor.dim() == 2:
            b_in = input_tensor.shape[0]
            b_w = weight_t.shape[0]
            if b_w == 1:
                weight_t = weight_t.expand(b_in, -1, -1)
            elif b_w != b_in:
                weight_t = weight_t.mean(dim=0, keepdim=True).expand(b_in, -1, -1)
            return torch.matmul(input_tensor.unsqueeze(1), weight_t).squeeze(1)
        if input_tensor.dim() == 3:
            b_in, t_in = input_tensor.shape[0], input_tensor.shape[1]
            b_w = weight_t.shape[0]
            if b_w == b_in:
                return torch.matmul(input_tensor, weight_t)
            if b_w == t_in:
                return torch.matmul(input_tensor.transpose(0, 1), weight_t).transpose(
                    0, 1
                )
            if b_w == 1:
                weight_t = weight_t.expand(b_in, -1, -1)
            else:
                weight_t = weight_t.mean(dim=0, keepdim=True).expand(b_in, -1, -1)
            return torch.matmul(input_tensor, weight_t)
        raise ValueError(
            f"Unsupported tensor ranks for batched LoRA override: "
            f"input={input_tensor.dim()}, weight={override_weight.dim()}"
        )

    raise ValueError(
        f"LoRA override tensor must have rank 2 or 3, but received rank {override_weight.dim()}"
    )


def ilora_linear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    previous_dtype = x.dtype

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            override_weight = None
            if hasattr(self, "lora_A_override"):
                override_weight = self.lora_A_override.get(active_adapter)

            compute_dtype = (
                override_weight.dtype
                if isinstance(override_weight, torch.Tensor)
                else lora_A.weight.dtype
            )
            x_compute = x if x.dtype == compute_dtype else x.to(compute_dtype)
            dropped = dropout(x_compute)
            if isinstance(override_weight, torch.Tensor):
                down = _apply_lora_override(dropped, override_weight)
            else:
                down = lora_A(dropped)
            output = lora_B(down) * scaling
            if output.dtype != result.dtype:
                output = output.to(result.dtype)
            result = result + output

    return result.to(previous_dtype)


def ilora_8bitlinear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            expected_dtype = result.dtype
            override_weight = None
            if hasattr(self, "lora_A_override"):
                override_weight = self.lora_A_override.get(active_adapter)
            compute_dtype = (
                override_weight.dtype
                if isinstance(override_weight, torch.Tensor)
                else lora_A.weight.dtype
            )
            x_compute = x if x.dtype == compute_dtype else x.to(compute_dtype)
            dropped = dropout(x_compute)
            if isinstance(override_weight, torch.Tensor):
                down = _apply_lora_override(dropped, override_weight)
            else:
                down = lora_A(dropped)
            output = lora_B(down)
            if requires_conversion and output.dtype != expected_dtype:
                output = output.to(expected_dtype)
            output = output * scaling
            result += output

    return result


class ILoRAWrapper(WrapperBase):
    """Deterministic LoRA model augmented with iLoRA overrides."""

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        super().__init__(model, peft_config, args, accelerator, adapter_name)

        self._lora_layers_in_order: list[LoraLayer] = []
        self._ilora_params_registered: bool = False
        self.ilora_matrix: Optional[ILoRAMatrix] = None
        self._last_lora_layer: Optional[LoraLayer] = None
        self._latest_ilora_outputs: Dict[str, Any] = {}
        self.class_token_ids: Dict[str, int] = {}
        self._embeddings_unfrozen: bool = False
        self.use_ilora: bool = bool(getattr(self.args, "use_ilora", True))

        self._modify_lora_layers(self.base_model)
        if not self._lora_layers_in_order:
            raise RuntimeError("iLoRA integration requires at least one LoRA layer.")
        self._last_lora_layer = self._lora_layers_in_order[-1]

        input_embedding_dim = getattr(self.args, "ilora_input_dim", None)
        if input_embedding_dim is None:
            input_embedding_dim = self._last_lora_layer.in_features
        llm_embedding_dim = self._last_lora_layer.in_features
        if self.use_ilora:
            self.ilora_matrix = ILoRAMatrix(
                self.args,
                input_embedding_dim=input_embedding_dim,
                llm_embedding_dim=llm_embedding_dim,
            )
            self.add_module("ilora_matrix", self.ilora_matrix)
            self._register_ilora_parameters_with_optimizer()

        if args.load_lora_path is not None:
            self.load_adapter(args.load_lora_path, adapter_name)

        self.best_val_auroc = -1.0
        self.best_save_dir = Path(self.args.checkpoint_path)
        self._ilora_state_filename = "ilora_matrix.pt"
        if self.accelerator.is_local_main_process:
            self.best_save_dir.mkdir(parents=True, exist_ok=True)

    def enable_class_token_embedding_training(self):
        """Unfreeze input/output embeddings so class tokens can learn."""
        if self._embeddings_unfrozen or self.opt is None:
            return
        existing = set()
        for group in self.opt.param_groups:
            for p in group.get("params", []):
                existing.add(id(p))

        params_to_add = []
        emb = getattr(self.base_model, "get_input_embeddings", lambda: None)()
        if emb is not None and hasattr(emb, "weight"):
            emb.weight.requires_grad_(True)
            if id(emb.weight) not in existing:
                params_to_add.append(emb.weight)
        lm_head = getattr(self.base_model, "get_output_embeddings", lambda: None)()
        if lm_head is not None and hasattr(lm_head, "weight"):
            lm_head.weight.requires_grad_(True)
            if id(lm_head.weight) not in existing:
                if lm_head is not emb:
                    params_to_add.append(lm_head.weight)
        if params_to_add:
            self.opt.add_param_group(
                {"params": params_to_add, "weight_decay": self.args.opt_wd}
            )
            self._embeddings_unfrozen = True

    def _maybe_save_best(self, auroc_value: float):
        if auroc_value is None or not math.isfinite(auroc_value) or auroc_value <= getattr(self, "best_val_auroc", -1.0):
            return

        if self.accelerator.is_main_process:
            self.best_val_auroc = auroc_value
            self.best_save_dir.mkdir(parents=True, exist_ok=True)

            output_dir = Path(self.best_save_dir)

            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # unwrap the PEFT model itself so we only save adapter weights
            unwrapped = self.accelerator.unwrap_model(self)
            unwrapped.save_pretrained(output_dir, save_function=self.accelerator.save)
            if hasattr(self, "tokenizer"):
                self.tokenizer.save_pretrained(output_dir)
            if self.ilora_matrix is not None:
                ilora_state = self.accelerator.get_state_dict(self.ilora_matrix)
                torch.save(ilora_state, output_dir / self._ilora_state_filename)

            print(f"✅ Best model saved to {output_dir} (AUROC={auroc_value:.4f})")
        self.accelerator.wait_for_everyone()

    # def load_adapter(self, model_id, adapter_name, *args, **kwargs):
    #     result = super().load_adapter(model_id, adapter_name, *args, **kwargs)

    #     self._ilora_loaded = False
    #     if self.ilora_matrix is not None:
    #         ilora_checkpoint = Path(model_id) / self._ilora_state_filename
    #         if ilora_checkpoint.exists():
    #             try:
    #                 ilora_state = torch.load(ilora_checkpoint, map_location="cpu")
    #                 self.ilora_matrix.load_state_dict(ilora_state)
    #             except (OSError, RuntimeError) as exc:
    #                 if self.accelerator.is_main_process:
    #                     print(
    #                         f"Warning: failed to load iLoRA weights from {ilora_checkpoint}: {exc}"
    #                     )
    #         elif self.accelerator.is_main_process:
    #             print(
    #                 f"Warning: iLoRA checkpoint {ilora_checkpoint} not found; "
    #                 "keeping current iLoRA weights."
    #             )
    #     print("loaded ilora weights")
    #     self.accelerator.wait_for_everyone()
    #     return result


    def load_adapter(self, model_id, adapter_name, *args, **kwargs):

        result = super().load_adapter(model_id, adapter_name, *args, **kwargs)


        self._ilora_loaded = False


        if self.ilora_matrix is not None:
            ilora_checkpoint = Path(model_id) / self._ilora_state_filename
            if ilora_checkpoint.exists():
                try:
                    ilora_state = torch.load(ilora_checkpoint, map_location="cpu")

                    self.ilora_matrix.load_state_dict(ilora_state, strict=True)

                    base_param = next(self.base_model.parameters(), None)
                    if base_param is not None:
                        self.ilora_matrix.to(base_param.device)
                    self._ilora_loaded = True
                    if self.accelerator.is_main_process:
                        print(f"✅ Loaded iLoRA weights from {ilora_checkpoint}")
                except (OSError, RuntimeError) as exc:
                    if self.accelerator.is_main_process:
                        print(f"Warning: failed to load iLoRA weights from {ilora_checkpoint}: {exc}")
            else:
                if self.accelerator.is_main_process:
                    print(f"Warning: iLoRA checkpoint {ilora_checkpoint} not found; keeping current iLoRA weights.")


        self.accelerator.wait_for_everyone()
        return result

    def _modify_lora_layers(self, module):
        """
        Recursively go through the model and modify LoraLayer instances.
        """
        for _, child in module.named_children():
            if isinstance(child, LoraLayer):
                if child not in self._lora_layers_in_order:
                    self._lora_layers_in_order.append(child)
                if isinstance(child, Linear):
                    self._wrap_lora_layer(child)
                    setattr(
                        child,
                        "forward",
                        ilora_linear_forward.__get__(child, child.__class__),
                    )
                elif isinstance(child, Linear8bitLt):
                    self._wrap_lora_layer(child)
                    setattr(
                        child,
                        "forward",
                        ilora_8bitlinear_forward.__get__(child, child.__class__),
                    )
            else:
                self._modify_lora_layers(child)

    def _wrap_lora_layer(self, lora_layer: LoraLayer):
        overrides = getattr(lora_layer, "lora_A_override", None)
        if overrides is None:
            overrides = {}
        for adapter_name in lora_layer._active_adapter:
            overrides.setdefault(adapter_name, None)
        lora_layer.lora_A_override = overrides

    def _register_ilora_parameters_with_optimizer(self):
        if (
            self.ilora_matrix is None
            or self._ilora_params_registered
            or not hasattr(self, "opt")
            or self.opt is None
        ):
            return

        ilora_params = list(self.ilora_matrix.parameters())
        if not ilora_params:
            return

        base_param = next(self.base_model.parameters(), None)
        if base_param is not None:
            self.ilora_matrix.to(base_param.device)

        self.opt.add_param_group(
            {
                "params": ilora_params,
                "weight_decay": self.args.opt_wd,
            }
        )
        self._ilora_params_registered = True

    @staticmethod
    def _to_device_tensor(
        value: Any, device: torch.device, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        if value is None:
            raise RuntimeError("iLoRA requires tensor inputs, but received None.")
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device)
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            return tensor
        return torch.as_tensor(
            value, device=device, dtype=dtype if dtype is not None else torch.float32
        )

    def _extract_ilora_inputs_from_batch(
        self, batch: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if batch is None:
            raise RuntimeError("iLoRA requires batch data containing textf/qmask/umask.")

        if isinstance(batch, Mapping):
            batch_dict = dict(batch)
        else:
            try:
                batch_dict = dict(batch)
            except TypeError as exc:
                raise RuntimeError(
                    "iLoRA batch must be convertible to a dict containing textf/qmask/umask."
                ) from exc

        ilora_inputs = batch_dict.get("ilora_inputs")
        if ilora_inputs is not None:
            if not isinstance(ilora_inputs, dict):
                raise TypeError("batch['ilora_inputs'] must be a dict with iLoRA tensors.")
            return ilora_inputs

        direct = {key: batch_dict.get(key) for key in ("textf", "qmask", "umask")}
        if all(value is not None for value in direct.values()):
            return direct

        meta = batch_dict.get("meta")
        if isinstance(meta, dict):
            meta_inputs = {key: meta.get(key) for key in ("textf", "qmask", "umask")}
            if all(value is not None for value in meta_inputs.values()):
                return meta_inputs

        raise RuntimeError(
            "iLoRA inputs missing from batch; expected 'textf', 'qmask', and 'umask'."
        )

    def _coerce_ilora_inputs(
        self, ilora_inputs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        if ilora_inputs is not None:
            if not isinstance(ilora_inputs, dict):
                raise TypeError("ilora_inputs must be a dict with iLoRA tensors.")
            return ilora_inputs

        if "batch" in kwargs:
            return self._extract_ilora_inputs_from_batch(kwargs["batch"])

        direct = {key: kwargs.get(key) for key in ("textf", "qmask", "umask")}
        if all(value is not None for value in direct.values()):
            return direct

        raise RuntimeError(
            "iLoRA inputs must be provided via 'ilora_inputs' or include textf/qmask/umask."
        )

    # def _update_last_lora_A_from_ilora(self, ilora_inputs: Dict[str, Any]) -> None:
    #     if self._last_lora_layer is None or self.ilora_matrix is None:
    #         raise RuntimeError("iLoRA override requested before initialization.")

    #     textf = ilora_inputs.get("textf")
    #     qmask = ilora_inputs.get("qmask")
    #     umask = ilora_inputs.get("umask")
    #     missing = [
    #         name
    #         for name, value in (("textf", textf), ("qmask", qmask), ("umask", umask))
    #         if value is None
    #     ]
    #     if missing:
    #         raise RuntimeError(f"iLoRA inputs missing keys: {', '.join(missing)}")

    #     adapter_names = list(self._last_lora_layer.lora_A.keys())
    #     if not adapter_names:
    #         raise RuntimeError("No active LoRA adapters found for iLoRA override.")

    #     reference_weight = self._last_lora_layer.lora_A[adapter_names[0]].weight
    #     device = reference_weight.device
    #     dtype = reference_weight.dtype

    #     textf_tensor = self._to_device_tensor(textf, device=device, dtype=torch.float32)
    #     if textf_tensor.dim() == 2:
    #         textf_tensor = textf_tensor.unsqueeze(1)
    #     qmask_tensor = self._to_device_tensor(
    #         qmask, device=device, dtype=torch.float32
    #     )
    #     umask_tensor = self._to_device_tensor(
    #         umask, device=device, dtype=torch.float32
    #     )

    #     lora_A_batch, kl_g, kl_b, relation_val = self.ilora_matrix(
    #         textf_tensor,
    #         qmask_tensor,
    #         umask_tensor,
    #     )

    #     override_weight = lora_A_batch.to(device=device, dtype=dtype).detach()
    #     if override_weight.dim() == 2:
    #         if override_weight.shape != reference_weight.shape:
    #             raise RuntimeError(
    #                 "iLoRA override shape mismatch: "
    #                 f"expected {reference_weight.shape}, got {override_weight.shape}"
    #             )
    #     elif override_weight.dim() == 3:
    #         if override_weight.shape[-2:] != reference_weight.shape:
    #             raise RuntimeError(
    #                 "iLoRA override shape mismatch: "
    #                 f"expected {reference_weight.shape}, got {override_weight.shape[-2:]}"
    #             )
    #         if override_weight.shape[0] != qmask_tensor.shape[1]:
    #             raise RuntimeError(
    #                 "iLoRA override batch size mismatch: "
    #                 f"expected {qmask_tensor.shape[1]}, got {override_weight.shape[0]}"
    #             )
    #         # override_weight = override_weight.mean(dim=0)
    #     else:
    #         raise RuntimeError(
    #             f"Unsupported iLoRA override tensor rank {override_weight.dim()}"
    #         )

    #     for adapter_name in self._last_lora_layer.lora_A_override.keys():
    #         self._last_lora_layer.lora_A_override[adapter_name] = override_weight

    #     self._latest_ilora_outputs = {
    #         "kl_g": kl_g.detach(),
    #         "kl_b": kl_b.detach(),
    #         "relation_val": relation_val.detach() if isinstance(relation_val, torch.Tensor) else relation_val,
    #         "override_shape": tuple(override_weight.shape),
    #     }
    #     return kl_g, kl_b

    # def _compute_ilora_regularizer(
    #     self, device: torch.device, dtype: torch.dtype
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Combine iLoRA-specific KL terms into a single penalty term.
    #     """
    #     zero = torch.zeros((), device=device, dtype=dtype)
    #     outputs = self._latest_ilora_outputs or {}

    #     def _ensure_tensor(value: Any) -> torch.Tensor:
    #         if isinstance(value, torch.Tensor):
    #             return value.to(device=device, dtype=dtype)
    #         if value is None:
    #             return zero.clone()
    #         return torch.as_tensor(value, device=device, dtype=dtype)

    #     kl_g = _ensure_tensor(outputs.get("kl_g"))
    #     kl_b = _ensure_tensor(outputs.get("kl_b"))

    #     penalty = zero.clone()
    #     if self.args.ilora_loss_weight_laplace:
    #         penalty = penalty + self.args.ilora_loss_weight_laplace * kl_g
    #     if self.args.ilora_loss_weight_binomial:
    #         penalty = penalty + self.args.ilora_loss_weight_binomial * kl_b

    #     return penalty, kl_g, kl_b

    def _update_last_lora_A_from_ilora(self, ilora_inputs: Dict[str, Any]):
        if not self.use_ilora:
            reference_weight = None
            if self._last_lora_layer is not None and self._last_lora_layer.lora_A:
                first_adapter = next(iter(self._last_lora_layer.lora_A.keys()))
                reference_weight = self._last_lora_layer.lora_A[first_adapter].weight
            base_param = next(self.base_model.parameters(), None)
            device = (
                reference_weight.device
                if reference_weight is not None
                else (base_param.device if base_param is not None else torch.device("cpu"))
            )
            dtype = (
                reference_weight.dtype
                if reference_weight is not None
                else (base_param.dtype if base_param is not None else torch.float32)
            )
            zero_kl = torch.zeros((), device=device, dtype=dtype)
            self._clear_last_lora_override()
            self._latest_ilora_outputs = {
                "kl_g": zero_kl,
                "kl_b": zero_kl,
                "relation_val": None,
                "override_shape": tuple(),
            }
            return zero_kl, zero_kl

        if self._last_lora_layer is None or self.ilora_matrix is None:
            raise RuntimeError("iLoRA override requested before initialization.")

        textf = ilora_inputs.get("textf")
        qmask = ilora_inputs.get("qmask")
        umask = ilora_inputs.get("umask")
        missing = [name for name, value in (("textf", textf), ("qmask", qmask), ("umask", umask)) if value is None]
        if missing:
            raise RuntimeError(f"iLoRA inputs missing keys: {', '.join(missing)}")

        adapter_names = list(self._last_lora_layer.lora_A.keys())
        if not adapter_names:
            raise RuntimeError("No active LoRA adapters found for iLoRA override.")

        reference_weight = self._last_lora_layer.lora_A[adapter_names[0]].weight
        device = reference_weight.device
        dtype = reference_weight.dtype

        textf_tensor = self._to_device_tensor(textf, device=device, dtype=torch.float32)
        if textf_tensor.dim() == 2:
            textf_tensor = textf_tensor.unsqueeze(1)
        qmask_tensor = self._to_device_tensor(qmask, device=device, dtype=torch.float32)
        umask_tensor = self._to_device_tensor(umask, device=device, dtype=torch.float32)

        if (
            torch.isnan(textf_tensor).any()
            or torch.isnan(qmask_tensor).any()
            or torch.isnan(umask_tensor).any()
        ):
            if self.accelerator.is_local_main_process:
                self.accelerator.print("[iLoRA DEBUG] NaN detected in iLoRA inputs; skipping override and returning zeros.")
            zero_kl = torch.zeros((), device=device, dtype=dtype)
            zero_override = torch.zeros_like(reference_weight)
            for adapter_name in self._last_lora_layer.lora_A_override.keys():
                self._last_lora_layer.lora_A_override[adapter_name] = zero_override
            self._latest_ilora_outputs = {
                "kl_g": zero_kl,
                "kl_b": zero_kl,
                "relation_val": None,
                "override_shape": tuple(zero_override.shape),
            }
            return zero_kl, zero_kl

        lora_A_batch, kl_g, kl_b, relation_val = self.ilora_matrix(
            textf_tensor, qmask_tensor, umask_tensor
        )
        override_weight = lora_A_batch.to(device=device, dtype=dtype).detach()
        has_nan = torch.isnan(override_weight).any() or torch.isinf(override_weight).any()
        if has_nan and self.accelerator.is_local_main_process:
            self.accelerator.print("[iLoRA DEBUG] override_weight contains NaN/Inf before injection.")
        expect_2d = reference_weight.shape

        if override_weight.dim() == 2:
            if override_weight.shape != expect_2d:
                raise RuntimeError(
                    f"iLoRA override shape mismatch: expected {expect_2d}, got {override_weight.shape}"
                )
        elif override_weight.dim() == 3:
            if override_weight.shape[-2:] != expect_2d:
                raise RuntimeError(
                    f"iLoRA override shape mismatch: expected {expect_2d}, got {override_weight.shape[-2:]}"
                )
            B_ilora = override_weight.shape[0]
            B_cur = int(qmask_tensor.shape[1])
            if B_ilora not in (1, B_cur):
                raise RuntimeError(
                    f"iLoRA override batch size mismatch: expected 1 or {B_cur}, got {B_ilora}"
                )
            if B_ilora == 1 and B_cur > 1:
                override_weight = override_weight.expand(B_cur, -1, -1)
        else:
            raise RuntimeError(f"Unsupported iLoRA override tensor rank {override_weight.dim()}")

        for adapter_name in self._last_lora_layer.lora_A_override.keys():
            self._last_lora_layer.lora_A_override[adapter_name] = override_weight

        self._latest_ilora_outputs = {
            "kl_g": kl_g.detach(),
            "kl_b": kl_b.detach(),
            "relation_val": relation_val.detach() if isinstance(relation_val, torch.Tensor) else relation_val,
            "override_shape": tuple(override_weight.shape),
        }
        return kl_g, kl_b


    def _clear_last_lora_override(self):
        if self._last_lora_layer is not None and hasattr(self._last_lora_layer, "lora_A_override"):
            for k in self._last_lora_layer.lora_A_override.keys():
                self._last_lora_layer.lora_A_override[k] = None


    def _extract_answer(self, text: str) -> str:
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*?\})', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data.get("answer", "").strip()
        except (json.JSONDecodeError, AttributeError):
            pass
        answer_match = re.search(r'"answer"\s*:\s*"(.*?)"', text, re.DOTALL)
        return answer_match.group(1).strip() if answer_match else ""

    def _normalize_answer(self, s: str) -> str:
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            return text.translate(
                str.maketrans("", "", string.punctuation.replace("?", ""))
            )

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _compute_em_from_texts(self, extracted_preds, extracted_labels) -> float:
        em_scores = []
        for pred, gt in zip(extracted_preds, extracted_labels):
            if gt == "?":
                em_scores.append(1.0 if pred == "?" else 0.0)
                continue
            if pred == "?":
                em_scores.append(0.0)
                continue

            pred_norm = self._normalize_answer(pred)
            gt_norm = self._normalize_answer(gt)
            em_scores.append(1.0 if pred_norm == gt_norm else 0.0)

        return float(np.mean(em_scores)) if em_scores else 0.0

    def _compute_f1_from_texts(self, extracted_preds, extracted_labels) -> float:
        f1_scores = []
        for pred, gt in zip(extracted_preds, extracted_labels):
            if gt == "?":
                f1_scores.append(1.0 if pred == "?" else 0.0)
                continue
            if pred == "?":
                f1_scores.append(0.0)
                continue

            pred_toks = self._normalize_answer(pred).split()
            gt_toks = self._normalize_answer(gt).split()

            if not gt_toks and not pred_toks:
                f1_scores.append(1.0)
                continue
            if not gt_toks or not pred_toks:
                f1_scores.append(0.0)
                continue

            pred_c, gt_c = Counter(pred_toks), Counter(gt_toks)
            common = sum((pred_c & gt_c).values())
            if common == 0:
                f1_scores.append(0.0)
                continue

            precision = common / sum(pred_c.values())
            recall = common / sum(gt_c.values())
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

        return float(np.mean(f1_scores)) if f1_scores else 0.0

    def forward_logits(
        self,
        input_ids,
        attention_mask,
        **kwargs,
    ) -> torch.Tensor:

        return self.base_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

    def compute_autoregressive_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(-100)
        valid_tokens = int(valid_mask.sum().item())
        if valid_tokens == 0:
            return None, 0
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss, valid_tokens

    def _log_nan_debug(self, step_idx, logits, nll, kl_g, kl_b, batch):
        """Emit debug info when NaN/Inf appears to localize the source."""
        if not self.accelerator.is_local_main_process:
            return

        def stats(name, tensor):
            if tensor is None:
                return f"{name}: None"
            t = tensor.detach()
            t = t if isinstance(t, torch.Tensor) else torch.tensor(t)
            finite = t[torch.isfinite(t)]
            fin_min = float(finite.min().item()) if finite.numel() > 0 else float("nan")
            fin_max = float(finite.max().item()) if finite.numel() > 0 else float("nan")
            return (
                f"{name}: shape={tuple(t.shape)} "
                f"nan={torch.isnan(t).any().item()} "
                f"inf={torch.isinf(t).any().item()} "
                f"min={fin_min} "
                f"max={fin_max}"
            )

        bad_params = []
        for name, param in self.base_model.named_parameters():
            if not param.requires_grad:
                continue
            data = param.data
            if torch.isnan(data).any() or torch.isinf(data).any():
                bad_params.append(name)
                if len(bad_params) >= 5:
                    break

        try:
            ilora_shape = batch["ilora_inputs"]["textf"].shape
        except Exception:
            ilora_shape = "n/a"

        self.accelerator.print(
            "[NaN DEBUG] "
            f"step={step_idx} | "
            f"{stats('logits', logits)} | "
            f"{stats('nll', nll)} | "
            f"{stats('kl_g', kl_g)} | "
            f"{stats('kl_b', kl_b)} | "
            f"input_ids={tuple(batch['input_ids'].shape)} "
            f"labels={tuple(batch['labels'].shape)} "
            f"ilora_textf={ilora_shape}"
        )
        if bad_params:
            self.accelerator.print(f"[NaN DEBUG] Parameters with NaN/Inf (first 5): {bad_params}")

    def fit(self, train_loader, eval_loader):
        nll_losses = AverageMeter()
        total_losses = AverageMeter()
        ilora_penalties = AverageMeter()
        ilora_kl_g_meter = AverageMeter()
        ilora_kl_b_meter = AverageMeter()
        samples_seen = 0

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}",
            leave=False,
        ) as pbar:
            for i, batch in enumerate(train_loader):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                if self.args.use_ilora:
                    ilora_inputs = self._extract_ilora_inputs_from_batch(batch)
                    kl_g, kl_b = self._update_last_lora_A_from_ilora(ilora_inputs)
                else:
                    self._clear_last_lora_override()
                    kl_g, kl_b = None, None
                logits = self.forward_logits(
                    input_ids,
                    attention_mask,
                )
                nll, n_valid_tokens = self.compute_autoregressive_loss(logits, labels)
                if self.accelerator.is_local_main_process and i < 5:
                    valid_per_sample = (labels != -100).sum(dim=1)
                    first_tokens = []
                    first_tokens_decoded = []
                    for j in range(min(2, labels.size(0))):
                        idxs = (labels[j] != -100).nonzero(as_tuple=True)[0]
                        if len(idxs) > 0:
                            tok_id = int(labels[j, idxs[0]].item())
                            first_tokens.append(tok_id)
                            if hasattr(self, "tokenizer"):
                                first_tokens_decoded.append(self.tokenizer.decode([tok_id]))
                        else:
                            first_tokens.append(None)
                            first_tokens_decoded.append(None)
                    self.accelerator.print(
                        f"[NLL DEBUG] step={i} n_valid_tokens={n_valid_tokens} "
                        f"valid_per_sample={valid_per_sample.tolist()} "
                        f"first_label_tokens={first_tokens} "
                        f"first_label_decoded={first_tokens_decoded}"
                    )
                if n_valid_tokens == 0:
                    if self.accelerator.is_local_main_process:
                        self.accelerator.print(f"[WARN] step={i} skip batch (no valid LM tokens after masking). input_ids={tuple(input_ids.shape)}")
                    continue

                zero_kl = torch.zeros((), device=nll.device, dtype=nll.dtype)
                kl_g = kl_g if isinstance(kl_g, torch.Tensor) else zero_kl
                kl_b = kl_b if isinstance(kl_b, torch.Tensor) else zero_kl
                if (
                    torch.isnan(logits).any()
                    or not torch.isfinite(nll)
                    or (kl_g is not None and torch.isnan(kl_g).any())
                    or (kl_b is not None and torch.isnan(kl_b).any())
                ):
                    self._log_nan_debug(
                        step_idx=i,
                        logits=logits,
                        nll=nll,
                        kl_g=kl_g,
                        kl_b=kl_b,
                        batch=batch,
                    )
                kl_g = kl_g.to(device=nll.device, dtype=nll.dtype)
                kl_b = kl_b.to(device=nll.device, dtype=nll.dtype)
                ilora_penalty = torch.zeros((), device=nll.device, dtype=nll.dtype) + self.args.ilora_loss_weight_laplace * kl_g + self.args.ilora_loss_weight_binomial * kl_b
                total_loss = nll + ilora_penalty

                # if self._last_lora_layer is not None and hasattr(self._last_lora_layer, "lora_A_override"):
                #     for k in self._last_lora_layer.lora_A_override.keys():
                #         self._last_lora_layer.lora_A_override[k] = None

                self.accelerator.backward(total_loss)

                if self._last_lora_layer is not None and hasattr(self._last_lora_layer, "lora_A_override"):
                    for k in self._last_lora_layer.lora_A_override.keys():
                        self._last_lora_layer.lora_A_override[k] = None
                if self.args.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.base_model.parameters(), self.args.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                # print("nll:", nll.detach().cpu().item())
                # print("ilora_penalty:", ilora_penalty.detach().cpu().item())
                # print("kl_g:", kl_g.detach().cpu().item())
                # print("kl_b:", kl_b.detach().cpu().item())
                # print("total_loss:", total_loss.detach().cpu().item())
                # print(total_loss)

                total_loss_val = float(total_loss.detach().cpu().item())
                nll_loss = float(nll.detach().cpu().item())
                ilora_penalty_val = float(ilora_penalty.detach().cpu().item())
                kl_g_val = float(kl_g.detach().cpu().item())
                kl_b_val = float(kl_b.detach().cpu().item())

                references = self.accelerator.gather(batch["labels"])
                if self.accelerator.num_processes > 1:
                    if i == len(train_loader) - 1:
                        references = references[
                            : len(train_loader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += references.shape[0]
                len_batch = references.shape[0]

                total_losses.update(total_loss_val, len_batch)
                nll_losses.update(nll_loss, len_batch)
                ilora_penalties.update(ilora_penalty_val, len_batch)
                ilora_kl_g_meter.update(kl_g_val, len_batch)
                ilora_kl_b_meter.update(kl_b_val, len_batch)

                if self.accelerator.is_local_main_process:
                    pbar.set_postfix(
                        nll=f"{nll_loss:.3f}",
                        ilora=f"{ilora_penalty_val:.3f}",
                    )
                    if self.wandb_logger is not None:
                        self.wandb_logger.log(
                            {
                                "train_loss": total_losses.avg,
                                "train_nll_loss": nll_losses.avg,
                                "ilora_penalty": ilora_penalties.avg,
                                "ilora_kl_laplace": ilora_kl_g_meter.avg,
                                "ilora_kl_binomial": ilora_kl_b_meter.avg,
                                "lr": self.opt.param_groups[0]["lr"],
                            }
                        )

                self.step += self.accelerator.num_processes
                pbar.update(1)
                if self.step >= self.args.eval_per_steps:
                    self.step -= self.args.eval_per_steps
                    auroc = self.evaluate_class_tokens(eval_loader)
                    self._maybe_save_best(auroc)

    def evaluate_losses(self, eval_loader):
        """Evaluate average language-model loss (with iLoRA penalty) on a loader."""
        was_training = self.training
        self.eval()
        device = self.base_model.device
        total_losses = AverageMeter()
        nll_losses = AverageMeter()
        ilora_penalties = AverageMeter()
        ilora_kl_g_meter = AverageMeter()
        ilora_kl_b_meter = AverageMeter()
        samples_seen = 0

        with torch.no_grad(), torch.inference_mode():
            for step_idx, batch in enumerate(
                tqdm(
                    eval_loader,
                    desc="Evaluating (loss)",
                    leave=False,
                )
            ):
                if self.args.use_ilora:
                    ilora_inputs = self._extract_ilora_inputs_from_batch(batch)
                    kl_g, kl_b = self._update_last_lora_A_from_ilora(ilora_inputs)
                else:
                    self._clear_last_lora_override()
                    kl_g, kl_b = None, None

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = self.forward_logits(
                    input_ids,
                    attention_mask,
                )
                nll, n_valid_tokens = self.compute_autoregressive_loss(logits, labels)
                if n_valid_tokens == 0:
                    if self.accelerator.is_local_main_process:
                        self.accelerator.print(
                            f"[WARN] eval step={step_idx} skip batch (no valid LM tokens after masking). "
                            f"input_ids={tuple(input_ids.shape)}"
                        )
                    continue

                zero_kl = torch.zeros((), device=nll.device, dtype=nll.dtype)
                kl_g = kl_g if isinstance(kl_g, torch.Tensor) else zero_kl
                kl_b = kl_b if isinstance(kl_b, torch.Tensor) else zero_kl

                kl_g = kl_g.to(device=nll.device, dtype=nll.dtype)
                kl_b = kl_b.to(device=nll.device, dtype=nll.dtype)
                ilora_penalty = (
                    torch.zeros((), device=nll.device, dtype=nll.dtype)
                    + self.args.ilora_loss_weight_laplace * kl_g
                    + self.args.ilora_loss_weight_binomial * kl_b
                )
                total_loss = nll + ilora_penalty

                references = self.accelerator.gather(labels)
                if self.accelerator.num_processes > 1:
                    if step_idx == len(eval_loader) - 1:
                        references = references[
                            : len(eval_loader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += references.shape[0]
                len_batch = references.shape[0]

                total_losses.update(float(total_loss.detach().cpu().item()), len_batch)
                nll_losses.update(float(nll.detach().cpu().item()), len_batch)
                ilora_penalties.update(float(ilora_penalty.detach().cpu().item()), len_batch)
                ilora_kl_g_meter.update(float(kl_g.detach().cpu().item()), len_batch)
                ilora_kl_b_meter.update(float(kl_b.detach().cpu().item()), len_batch)

        self.train(was_training)
        if total_losses.count == 0:
            metrics = {
                "total_loss": float("nan"),
                "nll_loss": float("nan"),
                "ilora_penalty": float("nan"),
                "ilora_kl_laplace": float("nan"),
                "ilora_kl_binomial": float("nan"),
            }
            if self.accelerator.is_local_main_process:
                self.accelerator.print("[WARN] Evaluation produced no valid tokens; returning NaN metrics.")
        else:
            metrics = {
                "total_loss": total_losses.avg,
                "nll_loss": nll_losses.avg,
                "ilora_penalty": ilora_penalties.avg,
                "ilora_kl_laplace": ilora_kl_g_meter.avg,
                "ilora_kl_binomial": ilora_kl_b_meter.avg,
            }
        if self.accelerator.is_local_main_process and self.wandb_logger is not None:
            self.wandb_logger.log({f"eval_{k}": v for k, v in metrics.items()})
        return metrics

    def evaluate_class_tokens(self, eval_loader):
        """Classification-style eval using class tokens, aligned with new_lora_uc_cd.py.

        Returns:
            float: AUROC (for backward compatibility). For full metric dict, call
                   `evaluate_class_tokens_with_metrics(..., return_metrics=True)`.
        """
        return self.evaluate_class_tokens_with_metrics(eval_loader, return_metrics=False)

    def evaluate_class_tokens_with_metrics(self, eval_loader, return_metrics: bool = True):
        """Classification-style eval that can optionally return the full metrics dict."""
        was_training = self.training
        self.eval()
        device = self.base_model.device

        y_true: list[int] = []
        y_pred: list[str] = []
        y_prob_uc: list[float] = []
        per_ds_true: dict[str, list[int]] = {}
        per_ds_pred: dict[str, list[str]] = {}
        per_ds_prob_uc: dict[str, list[float]] = {}

        with torch.no_grad(), torch.inference_mode():
            for step_idx, batch in enumerate(
                tqdm(
                    eval_loader,
                    desc="Evaluating (class tokens)",
                    leave=False,
                )
            ):
                if self.args.use_ilora:
                    ilora_inputs = self._extract_ilora_inputs_from_batch(batch)
                    self._update_last_lora_A_from_ilora(ilora_inputs)
                else:
                    self._clear_last_lora_override()
                    kl_g, kl_b = None, None

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                label_text = batch.get("label_text", [])
                sub_dataset = batch.get("sub_dataset", [])
                class_token_id_batch = batch.get("class_token_id", None)
                if class_token_id_batch is None:
                    raise RuntimeError("class_token_id missing from batch for classification eval.")

                B = input_ids.size(0)
                for i in range(B):
                    attn_i = attention_mask[i]
                    labels_i = labels[i]
                    ids_i = input_ids[i]

                    valid_len = int(attn_i.sum().item())
                    seq_len = int(attn_i.size(0))
                    padding_side = getattr(getattr(self, "tokenizer", None), "padding_side", "right")
                    if padding_side == "left":
                        start = seq_len - valid_len
                        end = seq_len
                    else:
                        start = 0
                        end = valid_len

                    window_labels = labels_i[start:end]
                    non_mask_pos = (window_labels != -100).nonzero(as_tuple=True)[0]
                    if (
                        self.accelerator.is_main_process
                        and step_idx == 0
                        and i == 0
                    ):
                        non_mask_count = int((window_labels != -100).sum().item())
                        print(
                            f"[EVAL DEBUG] padding_side={padding_side} valid_len={valid_len} "
                            f"start={start} end={end} non_mask_count={non_mask_count}"
                        )

                    if len(non_mask_pos) > 0:
                        first_ans_offset = int(non_mask_pos[0].item())
                        first_ans_idx = start + first_ans_offset
                    else:
                        first_ans_idx = end

                    if first_ans_idx <= start:
                        first_ans_idx = min(end, start + 1)

                    prompt_ids = ids_i[start:first_ans_idx].unsqueeze(0)
                    prompt_mask = torch.ones_like(prompt_ids, device=device)

                    outputs = self.base_model(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        use_cache=False,
                    )
                    logits = outputs.logits[:, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)

                    uc_id = int(self.class_token_ids.get("UC"))
                    cd_id = int(self.class_token_ids.get("CD"))
                    lp_uc = float(log_probs[0, uc_id].item())
                    lp_cd = float(log_probs[0, cd_id].item())

                    denom = torch.logsumexp(torch.tensor([lp_uc, lp_cd], device=log_probs.device), dim=0)
                    prob_uc = float(torch.exp(torch.tensor(lp_uc, device=log_probs.device) - denom).item())

                    pred_label = "UC" if lp_uc >= lp_cd else "CD"
                    true_label_text = str(label_text[i])

                    y_true.append(1 if true_label_text == "UC" else 0)
                    y_pred.append(pred_label)
                    y_prob_uc.append(prob_uc)
                    ds_name = (
                        str(sub_dataset[i])
                        if isinstance(sub_dataset, list) and i < len(sub_dataset)
                        else "unknown"
                    )
                    per_ds_true.setdefault(ds_name, []).append(y_true[-1])
                    per_ds_pred.setdefault(ds_name, []).append(pred_label)
                    per_ds_prob_uc.setdefault(ds_name, []).append(prob_uc)

        metrics = {
            "accuracy": accuracy_score(y_true, [1 if p == "UC" else 0 for p in y_pred]),
            "f1_uc": f1_score(y_true, [1 if p == "UC" else 0 for p in y_pred], pos_label=1, zero_division=0.0),
        }
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_prob_uc)
        except Exception:
            metrics["auroc"] = 0.0
        try:
            metrics["auprc"] = average_precision_score(y_true, y_prob_uc)
        except Exception:
            metrics["auprc"] = 0.0

        per_dataset_metrics: dict[str, dict[str, float]] = {}
        for ds_name, truths in per_ds_true.items():
            preds = per_ds_pred.get(ds_name, [])
            probs = per_ds_prob_uc.get(ds_name, [])
            ds_metrics = {
                "accuracy": accuracy_score(truths, [1 if p == "UC" else 0 for p in preds]) if truths else 0.0,
                "f1_uc": f1_score(truths, [1 if p == "UC" else 0 for p in preds], pos_label=1, zero_division=0.0)
                if truths
                else 0.0,
            }
            try:
                ds_metrics["auroc"] = roc_auc_score(truths, probs)
            except Exception:
                ds_metrics["auroc"] = 0.0
            try:
                ds_metrics["auprc"] = average_precision_score(truths, probs)
            except Exception:
                ds_metrics["auprc"] = 0.0
            per_dataset_metrics[ds_name] = ds_metrics
        metrics["per_dataset_metrics"] = per_dataset_metrics

        if self.accelerator.is_local_main_process:
            print(
                f"[VAL] Accuracy: {metrics['accuracy']:.4f} | F1 (UC): {metrics['f1_uc']:.4f} | AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f}"
            )
            if per_dataset_metrics:
                ds_msg = "; ".join(
                    f"{k}: acc={v.get('accuracy', 0.0):.4f}, f1_uc={v.get('f1_uc', 0.0):.4f}, "
                    f"auroc={v.get('auroc', 0.0):.4f}, auprc={v.get('auprc', 0.0):.4f}"
                    for k, v in per_dataset_metrics.items()
                )
                print(f"[VAL] Per-dataset metrics -> {ds_msg}")
            if self.wandb_logger is not None:
                self.wandb_logger.log({f"eval_{k}": v for k, v in metrics.items()})

        self.train(was_training)
        if return_metrics:
            return metrics
        return metrics.get("auroc", 0.0)

    def prepare_for_fit_evaluate(self, dataset, wandb_logger=None):
        """
        Prepare the model for training and evaluation.
        """
        self.wandb_logger = wandb_logger
        train_loader, val_loader = dataset.train_dataloader, dataset.val_dataloader

        if hasattr(dataset, "tokenizer"):
            self.tokenizer = dataset.tokenizer

        num_update_steps_per_epoch = len(train_loader)
        if self.args.max_train_steps == 0:
            self.args.max_train_steps = (
                self.args.n_epochs * num_update_steps_per_epoch
            )
        self.args.n_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )
        warmup_steps = self.args.warmup_steps if self.args.warmup_steps > 0 else int(
            self.args.max_train_steps * self.args.warmup_ratio
        )

        if self.args.early_stop_steps > 0:
            self.earlystop_n_epochs = (
                math.ceil(self.args.early_stop_steps / num_update_steps_per_epoch)
                if self.args.ood_ori_dataset is None
                else 0
            )
        else:
            self.earlystop_n_epochs = 0
            if self.accelerator.is_local_main_process:
                print("len(train_loader):", len(train_loader))
                print("num of epochs:", self.args.n_epochs)
                print(
                    f"[LR DEBUG] total_steps={self.args.max_train_steps} "
                    f"warmup_ratio={self.args.warmup_ratio} warmup_steps={warmup_steps}"
                )
        self.step = 0

        # Unfreeze embeddings before accelerator.prepare so newly added class tokens can learn
        self.enable_class_token_embedding_training()

        (
            self.base_model,
            self.opt,
            train_loader,
            val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.base_model,
            self.opt,
            train_loader,
            val_loader,
            self.scheduler,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self._register_ilora_parameters_with_optimizer()
