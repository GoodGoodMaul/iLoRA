import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable (so we can import utils, run, etc.)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from peft import LoraConfig, TaskType
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)

from args import Args
from ilora_wrapper import ILoRAWrapper

from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]

def maybe_set_seed(seed: Optional[int], deterministic: bool = True) -> Optional[int]:
    if seed is None:
        return None
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)
    set_seed(seed_int)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed_int


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train iLoRA on microbe feature pairs.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/yes_no_ilora_uc_cd.yaml",
        help="Path to iLoRA YAML config.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path, config_path: str | Path) -> Path:
    path_value = Path(path_value)
    if path_value.is_absolute():
        return path_value
    config_dir = Path(config_path).parent
    candidate = config_dir / path_value
    if candidate.exists():
        return candidate
    return path_value


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    config_path = Path(path) if path is not None else Path("config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_chat_prompt(messages: List[Dict[str, Any]], tokenizer, thinking_mode: bool) -> str:
    """Mirror yes_no_lora prompt construction: drop assistant turns and add generation prompt."""
    if not messages:
        raise ValueError("Record is missing messages.")

    usable = [m for m in messages if m.get("role") != "assistant"]
    if not usable:
        usable = [messages[0]]

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return apply_template(
                usable,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_mode,
            )
        except TypeError:
            return apply_template(
                usable,
                tokenize=False,
                add_generation_prompt=True,
            )
    return usable[-1]["content"]


def tokenize_example_mask_prompt(
    example: Dict[str, Any],
    tokenizer,
    max_seq_length: int,
    thinking_mode: bool,
) -> Dict[str, List[int]]:
    """Match yes_no_lora tokenization so LM and eval use the same prompts."""
    messages = example["messages"]
    if not messages:
        raise ValueError("Sample is missing 'messages' content.")

    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    if assistant_msg is None:
        raise ValueError("Sample is missing assistant message.")

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            prompt_ids = apply_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=thinking_mode,
            )
        except TypeError:
            prompt_ids = apply_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
    else:
        prompt_text = build_chat_prompt(prompt_messages, tokenizer, thinking_mode)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    answer_text = assistant_msg.get("content", "")
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    # Sanity: ensure mask actually exists; otherwise training/eval will read prompt tokens.
    if all(v != -100 for v in labels):
        print(
            "[TOKENIZE WARN] No -100 in labels; prompt masking failed. "
            f"len_prompt={len(prompt_ids)} len_answer={len(answer_ids)}"
        )

    if len(input_ids) > max_seq_length:
        overflow = len(input_ids) - max_seq_length
        keep_prompt = max(len(prompt_ids) - overflow, 0)
        input_ids = prompt_ids[-keep_prompt:] + answer_ids
        labels = [-100] * keep_prompt + answer_ids

    attention_mask = [1] * len(input_ids)
    pad_length = max_seq_length - len(input_ids)
    if pad_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
        labels = labels + [-100] * pad_length
        attention_mask = attention_mask + [0] * pad_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def format_chat_template(
    example: Dict[str, Any],
    tokenizer,
    max_length: int = 2048,
    thinking_mode: bool = False,
) -> Dict[str, List[int]]:
    messages = example["messages"]

    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            enable_thinking=thinking_mode,
        )
    except TypeError:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )

    input_ids = list(tokenized["input_ids"])
    attention_mask = list(tokenized.get("attention_mask", [1] * len(input_ids)))
    labels = [-100] * len(input_ids)

    prefix_messages: List[Dict[str, str]] = []
    prev_len = 0
    for message in messages:
        prefix_messages.append(message)
        try:
            prefix_tokens = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                enable_thinking=thinking_mode,
            )["input_ids"]
        except TypeError:
            prefix_tokens = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )["input_ids"]
        current_len = len(prefix_tokens)
        if message.get("role") == "assistant" and current_len > prev_len:
            labels[prev_len:current_len] = prefix_tokens[prev_len:current_len]
        prev_len = current_len

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        attention_mask = attention_mask[-max_length:]
        labels = labels[-max_length:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def initialize_model_and_tokenizer(config: Dict[str, Any], args: Args):
    model_name = config["model_name"]
    tokenizer_name = config.get("tokenizer_name") or model_name
    trust_remote_code = bool(config.get("trust_remote_code", False))
    # Force float32 for stability.

    # load_dtype = torch.float32
    load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=load_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(load_dtype)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model, tokenizer


def ensure_class_tokens(tokenizer, class_tokens: Dict[str, str]) -> Dict[str, int]:
    existing_special = set(tokenizer.all_special_tokens)
    existing_vocab = set(tokenizer.get_vocab().keys())
    to_add = [
        tok
        for tok in class_tokens.values()
        if tok not in existing_special and tok not in existing_vocab
    ]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        print(f"[INFO] Added class tokens: {to_add}")
    else:
        print(f"[INFO] Class tokens already exist: {class_tokens.values()}")
    token_ids: Dict[str, int] = {}
    for label, token in class_tokens.items():
        tok_id = tokenizer.convert_tokens_to_ids(token)
        if tok_id is None or tok_id < 0:
            ids = tokenizer(token, add_special_tokens=False).input_ids
            if not ids:
                raise ValueError(f"Failed to obtain token id for class token '{token}' (label={label}).")
            tok_id = ids[-1]
            print(f"[INFO] Derived class token id via tokenization for '{token}': {tok_id}")
        token_ids[label] = int(tok_id)
    return token_ids


def create_peft_config(args: Args) -> LoraConfig:
    target_modules = args.target_modules or [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias=args.lora_bias,
        modules_to_save=args.modules_to_save,
        inference_mode=False,
    )


class IBDSplitDataset(Dataset):
    def __init__(
        self,
        subset,
        tokenizer,
        args: Args,
        embedder: "IBDSentenceEmbedder",
    ):
        self.subset = subset
        self.tokenizer = tokenizer
        self.args = args
        self.embedder = embedder

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.subset[idx]
        chat_fields = tokenize_example_mask_prompt(
            example,
            self.tokenizer,
            self.args.max_seq_len,
            thinking_mode=getattr(self.args, "thinking_mode", False),
        )
        if not hasattr(self, "_debug_printed"):
            labels_arr = chat_fields["labels"]
            non_mask = sum(1 for v in labels_arr if v != -100)
            first_non_mask = next((v for v in labels_arr if v != -100), None)
            decoded = (
                self.tokenizer.convert_ids_to_tokens([first_non_mask])[0]
                if first_non_mask is not None
                else None
            )
            print(
                f"[DATA DEBUG] idx={idx} len_input={len(chat_fields['input_ids'])} "
                f"len_labels={len(labels_arr)} non_mask={non_mask} "
                f"first_non_mask_id={first_non_mask} decoded={decoded}"
            )
            self._debug_printed = True
        raw_pairs = example.get("top_significant_feature_pairs") or []
        if isinstance(raw_pairs, str):
            try:
                raw_pairs = json.loads(raw_pairs)
            except Exception:
                raw_pairs = [raw_pairs]
        pair_tokens: List[Dict[str, List[int]]] = []
        for pair in raw_pairs:
            pair_text = str(pair)
            encoded = self.tokenizer(
                pair_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.args.feature_max_length,
            )
            pair_tokens.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                }
            )

        if not pair_tokens:
            encoded = self.tokenizer(
                "(empty,0.0)",
                add_special_tokens=True,
                truncation=True,
                max_length=self.args.feature_max_length,
            )
            pair_tokens.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                }
            )

        sample = {
            "input_ids": chat_fields["input_ids"],
            "attention_mask": chat_fields["attention_mask"],
            "labels": chat_fields["labels"],
            "label_text": example.get("label"),
            "class_token_id": example.get("class_token_id", -1),
            "sub_dataset": example.get("sub_dataset") or example.get("source_dataset"),
        }

        if getattr(self.args, "use_ilora", True):
            textf = self.embedder.encode_tokens(pair_tokens)
            seq_len = textf.shape[0]
            qmask = torch.ones(seq_len, 1, dtype=torch.float32)
            umask = torch.ones(seq_len, dtype=torch.float32)
            sample["ilora_inputs"] = {
                "textf": textf,
                "qmask": qmask,
                "umask": umask,
            }
        return sample


class IBDSentenceEmbedder:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        pad_token_id: int,
        device: torch.device,
        max_length: int = 256,
        batch_size: int = 8,
    ):
        self.model = model
        self.pad_token_id = pad_token_id
        self.device = device
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.hidden_size = getattr(self.model.config, "hidden_size", None)
        try:
            self._orig_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self._orig_dtype = torch.float32

    def encode_tokens(self, token_dicts: List[Dict[str, List[int]]]) -> torch.Tensor:
        if not token_dicts:
            hidden_size = self.hidden_size
            if hidden_size is None:
                raise ValueError("Model config must define hidden_size for embedding generation.")
            return torch.empty((0, hidden_size), dtype=torch.float32)

        if not hasattr(self, "_nan_warned"):
            self._nan_warned = 0

        embeddings: List[torch.Tensor] = []
        original_training_mode = self.model.training
        converted = False
        try:
            # Only upcast if the base model is in float16; allow bf16/float32 to remain.
            model_dtype = next(self.model.parameters()).dtype
            if model_dtype == torch.float16:
                self.model = self.model.to(torch.float32)
                converted = True
            self.model.eval()
            with torch.no_grad():
                if not hasattr(self, "_param_check_done"):
                    bad_params = []
                    for name, param in self.model.named_parameters():
                        if not param.requires_grad:
                            continue
                        data = param.data
                        if torch.isnan(data).any() or torch.isinf(data).any():
                            bad_params.append(name)
                        if len(bad_params) >= 5:
                            break
                    if bad_params:
                        print(f"[EMBED DEBUG] Model parameters already contain NaN/Inf: {bad_params}")
                    else:
                        print("[EMBED DEBUG] Parameter check: no NaN/Inf detected in model weights.")
                    self._param_check_done = True
                for start in range(0, len(token_dicts), self.batch_size):
                    chunk = token_dicts[start : start + self.batch_size]
                    input_tensors: List[torch.Tensor] = []
                    mask_tensors: List[torch.Tensor] = []
                    for item in chunk:
                        input_ids = item.get("input_ids")
                        attention_mask = item.get("attention_mask")
                        if input_ids is None or attention_mask is None:
                            raise ValueError("Token dicts must include 'input_ids' and 'attention_mask'.")
                        input_tensors.append(
                            torch.tensor(
                                input_ids[: self.max_length],
                                dtype=torch.long,
                            )
                        )
                        mask_tensors.append(
                            torch.tensor(
                                attention_mask[: self.max_length],
                                dtype=torch.long,
                            )
                        )

                    batch_input_ids = pad_sequence(
                        input_tensors,
                        batch_first=True,
                        padding_value=self.pad_token_id,
                    )
                    batch_attention = pad_sequence(
                        mask_tensors,
                        batch_first=True,
                        padding_value=0,
                    )

                    batch_input_ids = batch_input_ids.to(self.device)
                    batch_attention = batch_attention.to(self.device)

                    if torch.isnan(batch_input_ids).any() or torch.isinf(batch_input_ids).any():
                        print(f"[EMBED DEBUG] NaN/Inf in batch_input_ids at start={start}")
                    attn_sum = batch_attention.sum(dim=1)
                    if (attn_sum == 0).any() and self._nan_warned < 5:
                        print(f"[EMBED DEBUG] Attention rows with zero sum at start={start}")

                    outputs = self.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden_states = outputs.hidden_states[-1]
                    if (
                        torch.isnan(hidden_states).any()
                        or torch.isinf(hidden_states).any()
                    ):
                        if self._nan_warned < 5:
                            hs_finite = hidden_states[torch.isfinite(hidden_states)]
                            hs_min = hs_finite.min().item() if hs_finite.numel() > 0 else float("nan")
                            hs_max = hs_finite.max().item() if hs_finite.numel() > 0 else float("nan")
                            print(
                                "[EMBED DEBUG] NaN/Inf in hidden_states before pooling; "
                                f"batch_start={start} shape={tuple(hidden_states.shape)} "
                                f"finite_min={hs_min:.4e} finite_max={hs_max:.4e}"
                            )
                            self._nan_warned += 1
                    attention = batch_attention.unsqueeze(-1).to(hidden_states.dtype)
                    pooled = (hidden_states * attention).sum(dim=1)
                    lengths = attention.sum(dim=1).clamp(min=1.0)
                    pooled = pooled / lengths
                    if (
                        torch.isnan(pooled).any()
                        or torch.isinf(pooled).any()
                    ):
                        if self._nan_warned < 5:
                            finite = pooled[torch.isfinite(pooled)]
                            fin_min = finite.min().item() if finite.numel() > 0 else float("nan")
                            fin_max = finite.max().item() if finite.numel() > 0 else float("nan")
                            emb_w = self.model.get_input_embeddings().weight
                            emb_finite = emb_w[torch.isfinite(emb_w)]
                            emb_min = emb_finite.min().item() if emb_finite.numel() > 0 else float("nan")
                            emb_max = emb_finite.max().item() if emb_finite.numel() > 0 else float("nan")
                            sample_ids = batch_input_ids[0].detach().cpu().tolist()[:10]
                            print(
                                "[EMBED DEBUG] NaN/Inf in pooled embeddings; "
                                f"batch_start={start} shape={tuple(pooled.shape)} "
                                f"finite_min={fin_min:.4e} "
                                f"finite_max={fin_max:.4e} | "
                                f"emb_min={emb_min:.4e} emb_max={emb_max:.4e} | "
                                f"sample_input_ids_head={sample_ids}"
                            )
                            self._nan_warned += 1
                    embeddings.append(pooled.to(torch.float32).cpu())
        finally:
            if converted:
                self.model = self.model.to(self._orig_dtype)
            self.model.train(original_training_mode)

        return torch.cat(embeddings, dim=0)


class ILoRADataCollator:
    def __init__(self, tokenizer, use_ilora: bool):
        self.use_ilora = use_ilora
        self.base_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
            label_pad_token_id=-100,
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        text_batch = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels": item["labels"],
            }
            for item in batch
        ]
        labels_text = [item.get("label_text") for item in batch]
        sub_dataset = [item.get("sub_dataset") for item in batch]
        class_token_ids = torch.tensor(
            [int(item.get("class_token_id", -1)) for item in batch],
            dtype=torch.long,
        )
        collated = self.base_collator(text_batch)

        if not self.use_ilora:
            collated["label_text"] = labels_text
            collated["sub_dataset"] = sub_dataset
            collated["class_token_id"] = class_token_ids
            return collated

        textf_list = [item["ilora_inputs"]["textf"] for item in batch]
        qmask_list = [item["ilora_inputs"]["qmask"] for item in batch]
        umask_list = [item["ilora_inputs"]["umask"] for item in batch]

        batch_size = len(batch)
        max_seq_len = max(t.shape[0] for t in textf_list)
        hidden_dim = textf_list[0].shape[1]
        max_speakers = max(q.shape[1] for q in qmask_list)

        textf_tensor = torch.zeros(
            max_seq_len,
            batch_size,
            hidden_dim,
            dtype=torch.float32,
        )
        qmask_tensor = torch.zeros(
            max_seq_len,
            batch_size,
            max_speakers,
            dtype=torch.float32,
        )
        umask_tensor = torch.zeros(
            max_seq_len,
            batch_size,
            dtype=torch.float32,
        )

        for idx, (textf, qmask, umask) in enumerate(
            zip(textf_list, qmask_list, umask_list)
        ):
            seq_len = textf.shape[0]
            speaker_dim = qmask.shape[1]
            textf_tensor[:seq_len, idx, :] = textf.to(torch.float32)
            qmask_tensor[:seq_len, idx, :speaker_dim] = qmask.to(torch.float32)
            umask_tensor[:seq_len, idx] = umask.to(torch.float32)
            if seq_len < max_seq_len:
                umask_tensor[seq_len:, idx] = 0.0

        collated["ilora_inputs"] = {
            "textf": textf_tensor,
            "qmask": qmask_tensor,
            "umask": umask_tensor,
        }
        collated["label_text"] = labels_text
        collated["sub_dataset"] = sub_dataset
        collated["class_token_id"] = class_token_ids
        return collated


class IBDILoRADataModule:
    def __init__(
        self,
        dataset_path: str | Path,
        tokenizer,
        args: Args,
        device: torch.device,
        is_main_process: bool,
        embedder: IBDSentenceEmbedder,
        seed: Optional[int] = None,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.is_main_process = is_main_process
        self._raw_dataset = load_from_disk(str(dataset_path))
        self.embedder = embedder
        self.collator = ILoRADataCollator(tokenizer, use_ilora=args.use_ilora)
        self._loader_generator: Optional[torch.Generator] = None
        if seed is not None:
            self._loader_generator = torch.Generator()
            self._loader_generator.manual_seed(int(seed))

        # raw_train = self._raw_dataset["train"].select(range(24))
        # raw_val   = self._raw_dataset["val"].select(range(12))
        # raw_test  = self._raw_dataset["test"].select(range(12))

        # self.train_dataset = self._prepare_split(raw_train)
        # self.val_dataset = self._prepare_split(raw_val)
        # self.test_dataset = self._prepare_split(raw_test)

        self.train_dataset = self._prepare_split(self._raw_dataset["train"])
        self.val_dataset = self._prepare_split(self._raw_dataset["val"])
        self.test_dataset = self._prepare_split(self._raw_dataset["test"])


        self.num_samples = len(self.train_dataset)
        hidden_size = getattr(self.embedder.model.config, "hidden_size", None)
        if hidden_size is None:
            feature_sample = self.train_dataset[0]["ilora_inputs"]["textf"]
            hidden_size = feature_sample.shape[1]
        self.feature_dim = hidden_size

    def _prepare_split(self, subset) -> IBDSplitDataset:
        return IBDSplitDataset(
            subset=subset,
            tokenizer=self.tokenizer,
            args=self.args,
            embedder=self.embedder,
        )

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
            generator=self._loader_generator,
            num_workers=0,
            pin_memory=True,
        )

    @property
    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    @property
    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    @property
    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    @property
    def final_test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)

    def update_embedder_model(self, model: AutoModelForCausalLM) -> None:
        self.embedder.model = model
        try:
            self.embedder.device = next(model.parameters()).device
        except StopIteration:
            pass


def main():
    cli_args = parse_cli_args()
    config = load_config(cli_args.config_path)
    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    use_bf16 = bool(train_cfg.get("bf16", False))
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16" if use_bf16 else "no",
    )

    seed = maybe_set_seed(train_cfg.get("seed"))
    if seed is not None and accelerator.is_main_process:
        print(f"[INFO] Using random seed: {seed}")

    args = Args()
    args.lora_r = int(lora_cfg.get("r", args.lora_r))
    args.lora_alpha = int(lora_cfg.get("lora_alpha", args.lora_alpha))
    args.lora_dropout = float(lora_cfg.get("lora_dropout", args.lora_dropout))
    args.lora_bias = str(lora_cfg.get("bias", args.lora_bias))
    args.target_modules = list(lora_cfg.get("target_modules", args.target_modules))
    args.modules_to_save = list(lora_cfg.get("modules_to_save", args.modules_to_save))

    args.batch_size = int(train_cfg.get("batch_size", args.batch_size))
    args.max_seq_len = int(train_cfg.get("max_seq_length", train_cfg.get("max_seq_len", args.max_seq_len)))
    args.feature_max_length = int(train_cfg.get("feature_max_length", args.feature_max_length))
    args.n_epochs = int(train_cfg.get("num_train_epochs", train_cfg.get("n_epochs", args.n_epochs)))
    args.max_train_steps = int(train_cfg.get("max_train_steps", args.max_train_steps))
    args.warmup_ratio = float(train_cfg.get("warmup_ratio", args.warmup_ratio))
    args.warmup_steps = int(train_cfg.get("warmup_steps", getattr(args, "warmup_steps", 0)))
    args.eval_per_steps = int(train_cfg.get("eval_steps", args.eval_per_steps))
    args.early_stop_steps = int(train_cfg.get("early_stop_steps", args.early_stop_steps))
    args.ilora_loss_weight_laplace = float(train_cfg.get("ilora_loss_weight_laplace", args.ilora_loss_weight_laplace))
    args.ilora_loss_weight_binomial = float(train_cfg.get("ilora_loss_weight_binomial", args.ilora_loss_weight_binomial))
    args.use_ilora = bool(train_cfg.get("use_ilora", getattr(args, "use_ilora", True)))
    args.max_grad_norm = float(train_cfg.get("max_grad_norm", args.max_grad_norm))
    args.lr = float(train_cfg.get("learning_rate", args.lr))
    args.opt_wd = float(train_cfg.get("weight_decay", args.opt_wd))
    args.dataset = train_cfg.get("dataset_name", args.dataset)
    args.checkpoint_path = train_cfg.get("checkpoint_path", args.checkpoint_path)
    args.thinking_mode = bool(train_cfg.get("thinking_mode", False))

    model, tokenizer = initialize_model_and_tokenizer(config, args)
    if accelerator.is_main_process:
        print(
            "[CONFIG DEBUG] KL weights | laplace: "
            f"{args.ilora_loss_weight_laplace} | binomial: {args.ilora_loss_weight_binomial}"
        )
    class_tokens = config.get("class_tokens") or {"UC": "yes", "CD": "no"}
    class_token_ids = ensure_class_tokens(tokenizer, class_tokens)
    if accelerator.is_main_process:
        for label, tok in class_tokens.items():
            plain_ids = tokenizer.encode(tok, add_special_tokens=False)
            spaced_ids = tokenizer.encode(f" {tok}", add_special_tokens=False)
            chosen_id = class_token_ids[label]
            decoded = tokenizer.convert_ids_to_tokens([chosen_id])[0]
            print(
                "[CLASS DEBUG] "
                f"label={label} token='{tok}' | chosen_id={chosen_id} decoded='{decoded}' | "
                f"plain_ids={plain_ids} | spaced_ids={spaced_ids}"
            )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.to(accelerator.device)
    model.eval()

    sentence_embedder = IBDSentenceEmbedder(
        model=model,
        pad_token_id=tokenizer.pad_token_id,
        device=accelerator.device,
        max_length=256,
        batch_size=8,
    )

    dataset_path = resolve_path(config["save_dataset_path"], cli_args.config_path)
    dataset = IBDILoRADataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        args=args,
        device=accelerator.device,
        is_main_process=accelerator.is_main_process,
        embedder=sentence_embedder,
        seed=seed,
    )

    args.num_samples = dataset.num_samples
    args.ilora_input_dim = dataset.feature_dim
    args.outdim = getattr(model.config, "vocab_size", tokenizer.vocab_size)

    model.train()

    peft_config = create_peft_config(args)

    ilora_wrapper_model = ILoRAWrapper(
        model=model,
        peft_config=peft_config,
        args=args,
        accelerator=accelerator,
        adapter_name="default",
    )

    ilora_wrapper_model.prepare_for_fit_evaluate(dataset=dataset, wandb_logger=None)
    ilora_wrapper_model.class_token_ids = class_token_ids
    dataset.update_embedder_model(accelerator.unwrap_model(ilora_wrapper_model.base_model))

    for epoch in range(args.n_epochs):
        args.epoch = epoch
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{args.n_epochs}")
            print("-" * 60)
        ilora_wrapper_model.train()
        ilora_wrapper_model.fit(
            train_loader=ilora_wrapper_model.train_loader,
            eval_loader=ilora_wrapper_model.val_loader,
        )
        val_auroc = ilora_wrapper_model.evaluate_class_tokens(ilora_wrapper_model.val_loader)
        ilora_wrapper_model._maybe_save_best(val_auroc)

    final_loader = dataset.final_test_dataloader
    final_loader = accelerator.prepare(final_loader)

    if accelerator.is_main_process:
        print(f"\nLoading best adapter from: {ilora_wrapper_model.best_save_dir}")
    try:
        ilora_wrapper_model.load_adapter(str(ilora_wrapper_model.best_save_dir), "default", replace=True)
    except Exception as exc:  # pragma: no cover - defensive logging path
        if accelerator.is_main_process:
            print(f"Warning: failed to load best adapter ({exc}). Continuing with current weights.")

    final_metrics = ilora_wrapper_model.evaluate_class_tokens_with_metrics(final_loader, return_metrics=True)
    if accelerator.is_main_process:
        print(
            "\n[FINAL TEST] "
            f"Accuracy: {final_metrics.get('accuracy', 0.0):.4f} | "
            f"F1 (UC): {final_metrics.get('f1_uc', 0.0):.4f} | "
            f"AUROC: {final_metrics.get('auroc', 0.0):.4f} | "
            f"AUPRC: {final_metrics.get('auprc', 0.0):.4f}"
        )
        per_ds = final_metrics.get("per_dataset_metrics", {})
        if per_ds:
            ds_msg = "; ".join(
                f"{k}: acc={v.get('accuracy', 0.0):.4f}, f1_uc={v.get('f1_uc', 0.0):.4f}, "
                f"auroc={v.get('auroc', 0.0):.4f}, auprc={v.get('auprc', 0.0):.4f}"
                for k, v in per_ds.items()
            )
            print(f"[FINAL TEST] Per-dataset metrics -> {ds_msg}")


if __name__ == "__main__":
    main()
