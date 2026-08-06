# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Args:
    # training / schedule
    num_samples: int = 100_000
    n_epochs: int = 3
    batch_size: int = 4
    max_train_steps: int = 0
    warmup_ratio: float = 0.06
    eval_per_steps: int = 200

    # dataset / task
    dataset_type: str = "molweni"
    num_bins: int = 15
    ood_ori_dataset: Optional[str] = None

    # iLoRA loss weights
    ilora_loss_weight_laplace: float = 1e-4
    ilora_loss_weight_binomial: float = 1e-4

    # misc
    early_stop_steps: int = 0
    earlystop_n_epochs: int = 0
    load_lora_path: Optional[str] = None

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_seq_len: int = 1024
    lr: float = 1e-4

    opt: str = "adamw"
    opt_wd: float = 0.0
    adam_epsilon: float = 1e-8

    # logging / paths
    modelwrapper: str = "ILoRAWrapper"
    model: str = "base"
    dataset: str = "Molweni"
    log_path: str = "ilora"
    checkpoint_path: Optional[str] = "checkpoints/best_model"
