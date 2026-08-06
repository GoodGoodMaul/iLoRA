import os
import sys

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from args import Args
from ilora_wrapper import ILoRAWrapper


from datasets import load_from_disk




def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    base_dir = Path(__file__).parent
    config_path = Path(path) if path is not None else base_dir / "config.yaml"
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_chat_template(example: Dict[str, Any], tokenizer, max_length: int = 2048) -> Dict[str, List[int]]:
    messages = example["messages"]

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def create_peft_config(args: Args) -> LoraConfig:
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )


class MolweniSplitDataset(Dataset):
    def __init__(
        self,
        subset,
        tokenizer,
        args: Args,
        embedder: "MolweniSentenceEmbedder",
    ):
        self.subset = subset
        self.tokenizer = tokenizer
        self.args = args
        self.embedder = embedder

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.subset[idx]
        chat_fields = format_chat_template(
            example,
            self.tokenizer,
            self.args.max_seq_len,
        )
        raw = json.loads(example["raw_data"])
        tokens = raw.get("tokens", [])
        qmask_raw = raw.get("qmask")
        umask_raw = raw.get("umask")

        if not tokens:
            raise ValueError("Encountered dialogue without tokenized EDUs while building iLoRA features.")

        textf = self.embedder.encode_tokens(tokens)

        if qmask_raw is None or umask_raw is None:
            raise ValueError("qmask and umask must be present in raw Molweni records.")

        qmask = torch.tensor(qmask_raw, dtype=torch.float32)
        umask = torch.tensor(umask_raw, dtype=torch.float32)

        if qmask.shape[0] != textf.shape[0]:
            raise ValueError("Mismatch between text embeddings and qmask row count.")
        if umask.shape[0] != textf.shape[0]:
            raise ValueError("Mismatch between text embeddings and umask length.")

        return {
            "input_ids": chat_fields["input_ids"],
            "attention_mask": chat_fields["attention_mask"],
            "labels": chat_fields["labels"],
            "ilora_inputs": {
                "textf": textf,
                "qmask": qmask,
                "umask": umask,
            },
        }


class MolweniSentenceEmbedder:
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

    def encode_tokens(self, token_dicts: List[Dict[str, List[int]]]) -> torch.Tensor:
        if not token_dicts:
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Model config must define hidden_size for embedding generation.")
            return torch.empty((0, hidden_size), dtype=torch.float32)

        embeddings: List[torch.Tensor] = []
        original_training_mode = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
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

                    outputs = self.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden_states = outputs.hidden_states[-1]
                    attention = batch_attention.unsqueeze(-1).to(hidden_states.dtype)
                    pooled = (hidden_states * attention).sum(dim=1)
                    lengths = attention.sum(dim=1).clamp(min=1.0)
                    pooled = pooled / lengths
                    embeddings.append(pooled.to(torch.float32).cpu())
        finally:
            self.model.train(original_training_mode)

        return torch.cat(embeddings, dim=0)


class ILoRADataCollator:
    def __init__(self, tokenizer):
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
        collated = self.base_collator(text_batch)

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
        return collated


class MolweniILoRADataModule:
    def __init__(
        self,
        dataset_path: str | Path,
        tokenizer,
        args: Args,
        device: torch.device,
        is_main_process: bool,
        embedder: MolweniSentenceEmbedder,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.is_main_process = is_main_process
        self._raw_dataset = load_from_disk(str(dataset_path))
        self.embedder = embedder
        self.collator = ILoRADataCollator(tokenizer)

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

    def _prepare_split(self, subset) -> MolweniSplitDataset:
        return MolweniSplitDataset(
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
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
    )

    config = load_config()
    args = Args()
    args.checkpoint_path = (
        "checkpoints/best_model_iLoRA_kl_"
        + str(args.ilora_loss_weight_laplace)
        + "_"
        + str(args.ilora_loss_weight_binomial)
        + "_lr_"
        + str(args.lr)
    )
    print("The checkpoint path is: ", args.checkpoint_path)
    print("The laplace loss weight is: ", args.ilora_loss_weight_laplace)
    print("The binomial loss weight is: ", args.ilora_loss_weight_binomial)
    print("The learning rate is: ", args.lr)

    model, tokenizer = initialize_model_and_tokenizer(config, args)
    model.to(accelerator.device)
    model.eval()

    sentence_embedder = MolweniSentenceEmbedder(
        model=model,
        pad_token_id=tokenizer.pad_token_id,
        device=accelerator.device,
        max_length=256,
        batch_size=8,
    )

    dataset_path = (Path(__file__).parent / config["save_dataset_path"]).resolve()
    dataset = MolweniILoRADataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        args=args,
        device=accelerator.device,
        is_main_process=accelerator.is_main_process,
        embedder=sentence_embedder,
    )

    args.num_samples = dataset.num_samples
    args.ilora_input_dim = dataset.feature_dim
    args.outdim = getattr(model.config, "vocab_size", tokenizer.vocab_size)

    model.train()

    peft_config = create_peft_config(args)

    ilora_model = ILoRAWrapper(
        model=model,
        peft_config=peft_config,
        args=args,
        accelerator=accelerator,
        adapter_name="default",
    )

    ilora_model.prepare_for_fit_evaluate(dataset=dataset, wandb_logger=None)
    dataset.update_embedder_model(accelerator.unwrap_model(ilora_model.base_model))

    for epoch in range(args.n_epochs):
        args.epoch = epoch
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{args.n_epochs}")
            print("-" * 60)
        ilora_model.train()
        ilora_model.fit(
            train_loader=ilora_model.train_loader,
            eval_loader=ilora_model.val_loader,
        )
        val_f1 = ilora_model.evaluate_autoregressive(ilora_model.val_loader)
        ilora_model._maybe_save_best(val_f1)

    final_loader = dataset.final_test_dataloader
    final_loader = accelerator.prepare(final_loader)

    if accelerator.is_main_process:
        print(f"\nLoading best adapter from: {ilora_model.best_save_dir}")
    try:
        ilora_model.load_adapter(str(ilora_model.best_save_dir), "default", replace=True)
    except Exception as exc:  # pragma: no cover - defensive logging path
        if accelerator.is_main_process:
            print(f"Warning: failed to load best adapter ({exc}). Continuing with current weights.")

    final_f1 = ilora_model.evaluate_autoregressive(final_loader)
    if accelerator.is_main_process:
        print(f"\n[FINAL TEST] Autoregressive F1: {final_f1:.4f}")


if __name__ == "__main__":
    main()
