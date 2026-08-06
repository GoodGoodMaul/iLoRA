import json
import math
import re
import shutil
import string
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.config import PeftConfig
from peft.tuners.lora import LoraLayer, Linear
from peft.tuners.lora.bnb import Linear8bitLt
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedModel

from args import Args
from iLoRA_model import ILoRAMatrix
from run.evaluation import *
from wrapperbase import WrapperBase

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
        if input_tensor.dim() == 2:
            return torch.matmul(input_tensor.unsqueeze(1), weight_t).squeeze(1)
        if input_tensor.dim() == 3:
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
    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        super().__init__(model, peft_config, args, accelerator, adapter_name)

        self.wandb_logger = None

        self._lora_layers_in_order: list[LoraLayer] = []
        self._ilora_params_registered: bool = False
        self.ilora_matrix: Optional[ILoRAMatrix] = None
        self._last_lora_layer: Optional[LoraLayer] = None
        self._latest_ilora_outputs: Dict[str, Any] = {}

        self._modify_lora_layers(self.base_model)
        if not self._lora_layers_in_order:
            raise RuntimeError("iLoRA integration requires at least one LoRA layer.")
        self._last_lora_layer = self._lora_layers_in_order[-1]

        input_embedding_dim = getattr(self.args, "ilora_input_dim", None)
        if input_embedding_dim is None:
            input_embedding_dim = self._last_lora_layer.in_features
        llm_embedding_dim = self._last_lora_layer.in_features
        self.ilora_matrix = ILoRAMatrix(
            self.args,
            input_embedding_dim=input_embedding_dim,
            llm_embedding_dim=llm_embedding_dim,
        )
        self.add_module("ilora_matrix", self.ilora_matrix)
        self._register_ilora_parameters_with_optimizer()

        if args.load_lora_path is not None:
            self.load_adapter(args.load_lora_path, adapter_name)

        self.best_val_f1 = -1.0
        self.best_save_dir = Path(self.args.checkpoint_path)
        self._ilora_state_filename = "ilora_matrix.pt"
        if self.accelerator.is_local_main_process:
            self.best_save_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_save_best(self, f1_value: float):
        if f1_value is None or f1_value <= self.best_val_f1:
            return

        if self.accelerator.is_main_process:
            self.best_val_f1 = f1_value
            self.best_save_dir.mkdir(parents=True, exist_ok=True)

            output_dir = Path(self.best_save_dir)

            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            unwrapped = self.accelerator.unwrap_model(self)
            unwrapped.save_pretrained(output_dir, save_function=self.accelerator.save)
            if hasattr(self, "tokenizer"):
                self.tokenizer.save_pretrained(output_dir)
            if self.ilora_matrix is not None:
                ilora_state = self.accelerator.get_state_dict(self.ilora_matrix)
                torch.save(ilora_state, output_dir / self._ilora_state_filename)

            print(f"✅ Best model saved to {output_dir}(F1={f1_value:.4f})")
        self.accelerator.wait_for_everyone()



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



    def _update_last_lora_A_from_ilora(self, ilora_inputs: Dict[str, Any]):
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

        lora_A_batch, kl_g, kl_b, relation_val = self.ilora_matrix(
            textf_tensor, qmask_tensor, umask_tensor
        )
        override_weight = lora_A_batch.to(device=device, dtype=dtype).detach()

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
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss

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
                ilora_inputs = self._extract_ilora_inputs_from_batch(batch)
                kl_g, kl_b = self._update_last_lora_A_from_ilora(ilora_inputs)
                logits = self.forward_logits(
                    input_ids,
                    attention_mask,
                )
                nll = self.compute_autoregressive_loss(logits, labels)

                kl_g = kl_g.to(device=nll.device, dtype=nll.dtype)
                kl_b = kl_b.to(device=nll.device, dtype=nll.dtype)
                ilora_penalty = torch.zeros((), device=nll.device, dtype=nll.dtype) + self.args.ilora_loss_weight_laplace * kl_g + self.args.ilora_loss_weight_binomial * kl_b

                total_loss = nll + ilora_penalty



                self.accelerator.backward(total_loss)

                if self._last_lora_layer is not None and hasattr(self._last_lora_layer, "lora_A_override"):
                    for k in self._last_lora_layer.lora_A_override.keys():
                        self._last_lora_layer.lora_A_override[k] = None
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()


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
                    f1 = self.evaluate_autoregressive(eval_loader)
                    self._maybe_save_best(f1)

    def evaluate_autoregressive(self, eval_loader):
        was_training = self.training
        self.eval()

        generation_config = getattr(self, "generation_config", None)
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        total_f1_sum = 0.0
        total_em_sum = 0.0
        total_count = 0
        samples_seen = 0
        examples_to_print = []

        device = self.base_model.device

        with torch.no_grad(), torch.inference_mode():
            is_main = (
                self.accelerator.is_local_main_process
                if hasattr(self, "accelerator")
                else True
            )
            with tqdm(
                total=len(eval_loader),
                desc="Evaluating (Autoregressive)",
                dynamic_ncols=True,
                leave=True,
                position=0,
                disable=not is_main,
            ) as t:
                for step, batch in enumerate(eval_loader):
                    ilora_inputs = self._extract_ilora_inputs_from_batch(batch)
                    self._update_last_lora_A_from_ilora(ilora_inputs)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    B = input_ids.size(0)

                    prompt_ids_list = []

                    for i in range(B):
                        attn_i = attention_mask[i]
                        labels_i = labels[i]
                        ids_i = input_ids[i]

                        valid_len = int(attn_i.sum().item())
                        seq_len = int(attn_i.size(0))
                        start = seq_len - valid_len
                        end = seq_len

                        window_labels = labels_i[start:end]
                        non_mask_pos = (window_labels != -100).nonzero(as_tuple=True)[0]

                        if len(non_mask_pos) > 0:
                            first_ans_offset = int(non_mask_pos[0].item())
                            first_ans_idx = start + first_ans_offset
                        else:
                            first_ans_idx = end

                        if first_ans_idx <= start:
                            first_ans_idx = min(end, start + 1)

                        p_ids = ids_i[start:first_ans_idx]
                        prompt_ids_list.append(p_ids)

                    self.tokenizer.padding_side = "left"
                    fallback_candidates = [
                        getattr(self.tokenizer, "bos_token_id", None),
                        getattr(self.tokenizer, "eos_token_id", None),
                        getattr(self.tokenizer, "unk_token_id", None),
                        1,
                    ]
                    FALLBACK_ID = next(tid for tid in fallback_candidates if tid is not None)

                    prompt_list = []
                    for i in range(B):
                        p_ids = prompt_ids_list[i]
                        L = p_ids.size(0)
                        if L == 0:
                            prompt_list.append([int(FALLBACK_ID)])
                        else:
                            prompt_list.append(p_ids.tolist())

                    batch_inputs = self.tokenizer.pad(
                        {"input_ids": prompt_list},
                        padding=True,
                        return_tensors="pt",
                    )
                    padded_prompts = batch_inputs["input_ids"].to(device)
                    padded_masks = batch_inputs["attention_mask"].to(device)

                    row_sums = padded_masks.sum(dim=1)
                    bad_rows = (row_sums == 0).nonzero(as_tuple=True)[0]
                    if bad_rows.numel() > 0:
                        padded_masks[bad_rows, -1] = 1

                    try:
                        outputs = self.base_model.generate(
                            input_ids=padded_prompts,
                            attention_mask=padded_masks,
                            generation_config=generation_config,
                            return_dict_in_generate=False,
                            output_scores=False,
                        )
                    finally:
                        self._clear_last_lora_override()

                    batch_preds = []
                    batch_gts = []
                    for i in range(B):
                        cut = padded_prompts.size(1)
                        gen_ids_i = outputs[i, cut:]
                        pred_raw = self.tokenizer.decode(
                            gen_ids_i, skip_special_tokens=True
                        )

                        ref_ids = labels[i][labels[i] != -100]
                        gt_raw = self.tokenizer.decode(
                            ref_ids, skip_special_tokens=True
                        )

                        pred_ans = self._extract_answer(pred_raw)
                        gt_ans = self._extract_answer(gt_raw)

                        batch_preds.append(pred_ans)
                        batch_gts.append(gt_ans)

                        if (
                            step == 0
                            and self.accelerator.is_local_main_process
                            and len(examples_to_print) < 3
                        ):
                            prompt_ids = prompt_ids_list[i]
                            prompt_text = self.tokenizer.decode(
                                prompt_ids, skip_special_tokens=True
                            )
                            examples_to_print.append(
                                {
                                    "prompt": prompt_text,
                                    "ground_truth_raw": gt_raw,
                                    "ground_truth_extracted": gt_ans,
                                    "prediction_raw": pred_raw,
                                    "prediction_extracted": pred_ans,
                                }
                            )

                    per_sample_f1 = []
                    per_sample_em = []
                    for p, g in zip(batch_preds, batch_gts):
                        f1_i = self._compute_f1_from_texts([p], [g])
                        per_sample_f1.append(f1_i)
                        em_i = self._compute_em_from_texts([p], [g])
                        per_sample_em.append(em_i)
                    f1_tensor = torch.tensor(
                        per_sample_f1, device=device, dtype=torch.float32
                    )
                    em_tensor = torch.tensor(
                        per_sample_em, device=device, dtype=torch.float32
                    )

                    gathered_f1 = self.accelerator.gather(f1_tensor)
                    gathered_em = self.accelerator.gather(em_tensor)
                    if self.accelerator.num_processes > 1:
                        if step == len(eval_loader) - 1:
                            gathered_f1 = gathered_f1[
                                : len(eval_loader.dataset) - samples_seen
                            ]
                            gathered_em = gathered_em[
                                : len(eval_loader.dataset) - samples_seen
                            ]
                        else:
                            samples_seen += f1_tensor.shape[0]

                    total_f1_sum += gathered_f1.sum().item()
                    total_em_sum += gathered_em.sum().item()
                    total_count += gathered_f1.numel()

                    avg_f1_display = total_f1_sum / max(1, total_count)
                    avg_em_display = total_em_sum / max(1, total_count)
                    t.set_postfix(
                        {"val_f1": f"{avg_f1_display:.4f}", "val_em": f"{avg_em_display:.4f}"}
                    )
                    t.update(1)

        final_f1 = total_f1_sum / max(1, total_count)
        final_em = total_em_sum / max(1, total_count)
        print(
            f"\nValidation F1 Score (Autoregressive): {final_f1:.4f} | EM: {final_em:.4f}\n"
        )
        if self.accelerator.is_local_main_process:
            print("=" * 80)
            print(">>> Sample Autoregressive Generation Results <<<")
            print("=" * 80)
            for i, ex in enumerate(examples_to_print):
                print(f"\n--- Sample {i+1} ---")
                print(f"Prompt:\n{ex['prompt']}")
                print(f"\n---> Ground Truth (Raw):\n{ex['ground_truth_raw']}")
                print(f"---> Model Prediction (Raw):\n{ex['prediction_raw']}")
                print("\n" + "-" * 20)
                print(
                    f"---> Ground Truth (Extracted): '{ex['ground_truth_extracted']}'"
                )
                print(
                    f"---> Model Prediction (Extracted): '{ex['prediction_extracted']}'"
                )
                print("-" * 50)
            print("=" * 80 + "\n")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if self.accelerator.is_local_main_process and self.wandb_logger is not None:
            self.wandb_logger.log({"val_f1": final_f1, "val_em": final_em})

        self.train(was_training)
        return final_f1

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
        self.step = 0

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

        self.generation_config = GenerationConfig(
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self._register_ilora_parameters_with_optimizer()
