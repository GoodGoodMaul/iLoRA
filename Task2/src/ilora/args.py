from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Args:
    """Lightweight configuration holder for iLoRA + LoRA training."""

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    modules_to_save: list[str] = field(default_factory=lambda: ["embed_tokens", "lm_head"])

    # Training and data
    batch_size: int = 2
    max_seq_len: int = 8192
    n_epochs: int = 3
    max_train_steps: int = 0
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    eval_per_steps: int = 50
    early_stop_steps: int = 0
    feature_max_length: int = 64

    # Optimization
    opt: str = "adamw"
    lr: float = 2e-4
    opt_wd: float = 0.0
    adam_epsilon: float = 1e-8

    # iLoRA loss weights
    ilora_loss_weight_laplace: float = 1e-4
    ilora_loss_weight_binomial: float = 1e-4
    use_ilora: bool = True

    # Logging / evaluation
    num_bins: int = 15
    dataset_type: str = "bertds"
    bayes_eval_n_samples_final: int = 1
    testing_set: str = "val"
    log_path: str = "ilora"
    modelwrapper: str = "ILoRAWrapper"
    model: str = "base"
    dataset: str = "IBD_UC_CD_yes_no"
    checkpoint_path: str = "checkpoints/best_model_iLoRA"
    thinking_mode: bool = False
    max_grad_norm: float = 1.0

    # Runtime (populated later)
    epoch: int = 0
    num_samples: int = 0
    ilora_input_dim: Optional[int] = None
    outdim: Optional[int] = None

    # Optional adapter loading
    load_lora_path: Optional[str] = None
