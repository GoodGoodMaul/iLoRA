import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

DEFAULT_CLASS_TOKENS = {"UC": "yes", "CD": "no"}
DEFAULT_DATASET_NAME = "IBD_UC_CD_yes_no"
DEFAULT_TOP_FEATURE_COUNT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create UC/CD classification dataset (UC/CD cohorts only) with a yes/no answer. "
            "The model is expected to generate exactly one token: 'yes' for UC or 'no' for Crohn's Disease (CD)."
        )
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/yes_no_datasetCreate_uc_cd.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        default=None,
        help="Override final folder name under output_data_path.",
    )
    parser.add_argument(
        "--sample_id_column",
        type=str,
        default=None,
        help="Optional column name in meta.tsv that stores sample IDs.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="Optional column name in meta.tsv that stores labels.",
    )
    parser.add_argument("--train_ratio", type=float, default=None, help="Train split ratio override.")
    parser.add_argument("--val_ratio", type=float, default=None, help="Validation split ratio override.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override for shuffling.")
    parser.add_argument(
        "--significant_results_path",
        type=str,
        default=None,
        help="Override path to significant_results.tsv for selecting top features.",
    )
    parser.add_argument(
        "--top_feature_count",
        type=int,
        default=None,
        help="Override number of top significant microbe features to include.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    cfg_path = config_path
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(REPO_ROOT, config_path)
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, "r") as handle:
        return yaml.safe_load(handle)


def resolve_config_path(path_value: str, config_path: str) -> str:
    if os.path.isabs(path_value):
        return path_value

    config_dir = os.path.dirname(os.path.abspath(config_path))
    candidate = os.path.abspath(os.path.join(config_dir, path_value))
    if os.path.exists(candidate):
        return candidate

    return os.path.abspath(os.path.join(REPO_ROOT, path_value))


def list_dataset_dirs(root: str) -> List[str]:
    dataset_dirs = []
    for entry in os.listdir(root):
        full_path = os.path.join(root, entry)
        if not os.path.isdir(full_path):
            continue
        data_path = os.path.join(full_path, "data.tsv")
        meta_path = os.path.join(full_path, "meta.tsv")
        if os.path.isfile(data_path) and os.path.isfile(meta_path):
            dataset_dirs.append(full_path)
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset folders with data.tsv and meta.tsv found under {root}")
    return sorted(dataset_dirs)


def guess_sample_id_column(meta_df: pd.DataFrame, sample_ids: List[str]) -> str:
    sample_id_set = set(sample_ids)
    best_col = None
    best_overlap = -1

    for col in meta_df.columns:
        values = set(map(str, meta_df[col].astype(str).tolist()))
        overlap = len(values & sample_id_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col

    if best_overlap <= 0 or best_col is None:
        raise ValueError("Unable to automatically detect sample ID column in meta.tsv.")

    print(f"[INFO] Using sample ID column '{best_col}' (overlap={best_overlap}).")
    return best_col


def detect_label_column(meta_df: pd.DataFrame, override: str = None) -> str:
    if override:
        if override not in meta_df.columns:
            raise ValueError(f"Provided label column '{override}' not found in meta.tsv.")
        return override

    possible_label_cols = ["diagnosis", "Diagnosis", "label", "Label", "IBD", "ibd"]
    for col in possible_label_cols:
        if col in meta_df.columns:
            print(f"[INFO] Using label column '{col}'.")
            return col

    raise ValueError(f"No label column detected. Checked: {possible_label_cols}")


def validate_alignment(sample_cols: List[str], meta_df: pd.DataFrame, sample_id_col: str) -> None:
    matrix_ids = set(map(str, sample_cols))
    meta_ids = set(meta_df[sample_id_col].astype(str).tolist())

    missing_in_meta = matrix_ids - meta_ids
    missing_in_matrix = meta_ids - matrix_ids

    if missing_in_meta:
        print(f"[WARN] {len(missing_in_meta)} samples exist in data.tsv but not in meta.tsv (e.g., {list(missing_in_meta)[:5]}).")
    if missing_in_matrix:
        print(f"[WARN] {len(missing_in_matrix)} samples exist in meta.tsv but not in data.tsv (e.g., {list(missing_in_matrix)[:5]}).")


def map_diagnosis_to_label(label_value) -> str:
    value = str(label_value).strip().lower()
    if value == "uc":
        return "UC"
    if value == "cd":
        return "CD"
    if value in {"con", "control", "healthy", "non-ibd", "non_ibd"}:
        raise ValueError("Control sample (CON) ignored.")
    raise ValueError(f"Unrecognized label value: {label_value!r}")


def clean_feature_name(name):
    if isinstance(name, str) and name.startswith("s__"):
        return name[3:]
    return name


def build_microbiome_text(feature_df: pd.DataFrame, sample_col: str) -> str:
    sample_series = feature_df[sample_col]

    mask = (sample_series != 0) & (~sample_series.isna())
    nonzero = sample_series[mask]

    if nonzero.empty:
        return "This sample has zero relative abundance for all microbes; no non-zero microbial species were detected."

    nonzero_sorted = nonzero.sort_values(ascending=False)
    lines = ["Detected non-zero microbial species and their relative abundances:"]
    for feature_name, value in nonzero_sorted.items():
        value_str = repr(value) if isinstance(value, float) else str(value)
        lines.append(f"- {clean_feature_name(feature_name)}: {value_str}")

    return "\n".join(lines)


def ensure_answer_prefix(prefix: str) -> str:
    cleaned = prefix.strip()
    if not cleaned.endswith(" "):
        cleaned += " "
    return cleaned


def build_user_prompt(
    sample_id: str,
    dataset_name: str,
    microbiome_text: str,
    class_tokens: Dict[str, str],
    prompt_cfg: Dict,
) -> str:
    uc_token = class_tokens["UC"]
    cd_token = class_tokens["CD"]
    instruction = prompt_cfg.get(
        "instruction",
        (
            "You are a gastroenterologist familiar with IBD subtypes. Based on the microbiome profile, decide "
            f"whether the sample is Ulcerative Colitis (UC). Respond with '{uc_token}' for UC and '{cd_token}' "
            "for Crohn's Disease (CD). Generate exactly one token."
        ),
    )
    answer_prefix = ensure_answer_prefix(prompt_cfg.get("answer_prefix", "Answer:"))

    return f"""{instruction}

Sample info:
- Sample ID: {sample_id}
- Source dataset: {dataset_name}
- Expected answers: UC -> {uc_token}; CD -> {cd_token}

Microbiome profile (non-zero relative abundances):
{microbiome_text}

{answer_prefix}"""


def prepare_tokenizer(config: Dict, class_tokens: Dict[str, str]):
    tokenizer_name = config.get("tokenizer_name") or config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    existing_special = set(tokenizer.all_special_tokens)
    existing_vocab = set(tokenizer.get_vocab().keys())
    to_add = [tok for tok in class_tokens.values() if tok not in existing_special and tok not in existing_vocab]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        print(f"[INFO] Added class tokens to tokenizer.additional_special_tokens: {to_add}")
    token_ids: Dict[str, int] = {}
    for label, token in class_tokens.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Failed to obtain token ID for class token '{token}' (label={label}).")
        token_ids[label] = int(token_id)
    return tokenizer, token_ids


def load_top_features(significant_path: str, top_k: int) -> List[str]:
    if not os.path.exists(significant_path):
        raise FileNotFoundError(f"significant_results.tsv not found at {significant_path}")

    df = pd.read_csv(significant_path, sep="\t")
    if "feature" not in df.columns:
        raise ValueError("significant_results.tsv must contain a 'feature' column.")

    if "qval" in df.columns:
        df = df.sort_values(by="qval", ascending=True)
    elif "pval" in df.columns:
        df = df.sort_values(by="pval", ascending=True)

    df = df.drop_duplicates(subset=["feature"])
    top_features = df["feature"].astype(str).head(top_k).tolist()
    if len(top_features) < top_k:
        print(f"[WARN] Requested top {top_k} features but only found {len(top_features)} entries.")

    print(f"[INFO] Loaded top {len(top_features)} features from {significant_path}")
    return top_features


def extract_top_feature_values(feature_matrix: pd.DataFrame, sample_col: str, selected_features: List[str]) -> List[Tuple[str, float]]:
    values: List[Tuple[str, float]] = []
    for feature_name in selected_features:
        value = 0.0
        if feature_name in feature_matrix.index:
            raw_value = feature_matrix.at[feature_name, sample_col]
            if pd.notna(raw_value):
                value = float(raw_value)
        values.append((feature_name, value))
    return values


def build_records_for_dataset(
    dataset_dir: str,
    class_tokens: Dict[str, str],
    class_token_ids: Dict[str, int],
    prompt_cfg: Dict,
    top_features: List[str],
    sample_id_col_override: str = None,
    label_col_override: str = None,
) -> List[Dict]:
    dataset_name = os.path.basename(dataset_dir)
    data_path = os.path.join(dataset_dir, "data.tsv")
    meta_path = os.path.join(dataset_dir, "meta.tsv")

    print(f"[INFO] Reading feature matrix: {data_path}")
    data_df = pd.read_csv(data_path, sep="\t")
    feature_id_col = data_df.columns[0]
    sample_cols = data_df.columns[1:].tolist()
    print(f"[INFO] Loaded feature matrix with {data_df.shape[0]} features and {len(sample_cols)} samples.")

    features = data_df[feature_id_col].astype(str).tolist()
    feature_matrix = data_df[sample_cols].copy()
    feature_matrix.index = features

    print(f"[INFO] Reading metadata: {meta_path}")
    meta_df = pd.read_csv(meta_path, sep="\t")
    print(f"[INFO] Metadata shape: {meta_df.shape}")

    sample_id_col = sample_id_col_override or guess_sample_id_column(meta_df, sample_cols)
    if sample_id_col not in meta_df.columns:
        raise ValueError(f"Sample ID column '{sample_id_col}' not found in metadata.")
    meta_df[sample_id_col] = meta_df[sample_id_col].astype(str)
    validate_alignment(sample_cols, meta_df, sample_id_col)

    label_col = label_col_override or detect_label_column(meta_df)

    id_to_label: Dict[str, str] = {}
    for _, row in meta_df.iterrows():
        sample_id = str(row[sample_id_col])
        try:
            id_to_label[sample_id] = map_diagnosis_to_label(row[label_col])
        except ValueError as exc:
            print(f"[INFO] Skipping sample '{sample_id}' in {dataset_name}: {exc}")

    print(f"[INFO] Total labeled UC/CD samples in {dataset_name}: {len(id_to_label)}")
    records: List[Dict] = []

    for sample_col in tqdm(sample_cols, desc=f"Building prompts for {dataset_name}"):
        sample_id = str(sample_col)
        if sample_id not in id_to_label:
            continue

        label_text = id_to_label[sample_id]
        class_token = class_tokens[label_text]
        class_token_id = class_token_ids[label_text]
        microbiome_text = build_microbiome_text(feature_matrix, sample_col)
        top_feature_values = extract_top_feature_values(feature_matrix, sample_col, top_features)
        user_prompt = build_user_prompt(sample_id, dataset_name, microbiome_text, class_tokens, prompt_cfg)

        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": class_token},
                ],
                "user_prompt": user_prompt,
                "assistant_response": class_token,
                "sample_id": sample_id,
                "label": label_text,
                "class_token": class_token,
                "class_token_id": class_token_id,
                "source_dataset": dataset_name,
                "sub_dataset": dataset_name,
                "top_significant_features": top_features,
                "top_significant_feature_values": [
                    {"feature": clean_feature_name(name), "value": value} for name, value in top_feature_values
                ],
                "top_significant_feature_pairs": [
                    f"({clean_feature_name(name)},{value})" for name, value in top_feature_values
                ],
            }
        )

    print(f"[INFO] Final records collected from {dataset_name}: {len(records)}")
    return records


def _compute_split_sizes(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    train = int(round(total * train_ratio))
    val = int(round(total * val_ratio))

    if train <= 0 and total > 0:
        train = 1

    if train + val > total:
        val = max(0, total - train)

    if total >= 3 and train + val >= total:
        if val > 0:
            val -= 1
        else:
            train = max(1, train - 1)

    test = total - train - val
    if test < 0:
        test = 0
        if val > 0:
            val = max(0, val + test)
        else:
            train = max(0, train + test)

    if total >= 2 and test == 0:
        if val > 0:
            val -= 1
            test = 1
        elif train > 1:
            train -= 1
            test = 1

    return train, val, test


def stratified_split_by_dataset(
    records: List[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[DatasetDict, Dict[str, Dict[str, Dict[str, int]]]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio must be >0, val_ratio must be >=0, and train_ratio + val_ratio < 1.")

    rng = random.Random(seed)
    by_dataset: Dict[str, List[Dict]] = {}
    for record in records:
        by_dataset.setdefault(record["source_dataset"], []).append(record)

    splits = {"train": [], "val": [], "test": []}
    per_dataset_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    for dataset_name, dataset_records in sorted(by_dataset.items()):
        grouped: Dict[str, List[Dict]] = {}
        for record in dataset_records:
            grouped.setdefault(record["label"], []).append(record)

        missing_labels = {"UC", "CD"} - set(grouped)
        if missing_labels:
            raise RuntimeError(f"Dataset '{dataset_name}' is missing labels: {sorted(missing_labels)}")

        dataset_split = {"train": [], "val": [], "test": []}
        for label, items in grouped.items():
            rng.shuffle(items)
            train_count, val_count, test_count = _compute_split_sizes(len(items), train_ratio, val_ratio)

            dataset_split["train"].extend(items[:train_count])
            dataset_split["val"].extend(items[train_count:train_count + val_count])
            dataset_split["test"].extend(items[train_count + val_count:])

            print(f"[INFO] {dataset_name} label {label}: train={train_count}, val={val_count}, test={test_count}")

        per_dataset_stats[dataset_name] = {}
        for split_name, split_records in dataset_split.items():
            rng.shuffle(split_records)
            per_dataset_stats[dataset_name][split_name] = {
                label: sum(1 for rec in split_records if rec["label"] == label) for label in grouped.keys()
            }
            splits[split_name].extend(split_records)

        total_split_sizes = {split: len(dataset_split[split]) for split in dataset_split}
        print(f"[INFO] {dataset_name} split sizes: {total_split_sizes}")

    for key in splits:
        rng.shuffle(splits[key])

    dataset_dict = DatasetDict({key: Dataset.from_list(value) for key, value in splits.items() if value})
    return dataset_dict, per_dataset_stats


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)

    dataset_cfg = config.get("dataset", {})
    prompt_cfg = config.get("prompt", {})
    class_tokens = config.get("class_tokens") or DEFAULT_CLASS_TOKENS
    missing_labels = {"UC", "CD"} - set(class_tokens)
    if missing_labels:
        raise ValueError(f"class_tokens must define UC and CD. Missing: {sorted(missing_labels)}")

    tokenizer, class_token_ids = prepare_tokenizer(config, class_tokens)
    answer_prefix = ensure_answer_prefix(prompt_cfg.get("answer_prefix", "Answer:"))

    input_root = resolve_config_path(config["input_data_path"], args.config_path)
    output_root = resolve_config_path(config["output_data_path"], args.config_path)

    train_ratio = args.train_ratio if args.train_ratio is not None else float(dataset_cfg.get("train_ratio", 0.7))
    val_ratio = args.val_ratio if args.val_ratio is not None else float(dataset_cfg.get("val_ratio", 0.15))
    seed = args.seed if args.seed is not None else int(dataset_cfg.get("seed", 42))
    dataset_name = args.output_dataset_name or dataset_cfg.get("name") or DEFAULT_DATASET_NAME

    significant_path_cfg = (
        dataset_cfg.get("significant_results_path") or config.get("significant_results_path")
    )
    significant_results_path = resolve_config_path(
        args.significant_results_path or significant_path_cfg or os.path.join(input_root, "significant_results.tsv"),
        args.config_path,
    )
    top_feature_count = args.top_feature_count if args.top_feature_count is not None else int(
        dataset_cfg.get("top_feature_count", DEFAULT_TOP_FEATURE_COUNT)
    )
    top_features = load_top_features(significant_results_path, top_feature_count)

    dataset_dirs = list_dataset_dirs(input_root)
    print(f"[INFO] Found {len(dataset_dirs)} dataset folders: {dataset_dirs}")

    all_records: List[Dict] = []
    for dataset_dir in dataset_dirs:
        records = build_records_for_dataset(
            dataset_dir,
            class_tokens=class_tokens,
            class_token_ids=class_token_ids,
            prompt_cfg=prompt_cfg,
            top_features=top_features,
            sample_id_col_override=args.sample_id_column,
            label_col_override=args.label_column,
        )
        all_records.extend(records)

    if not all_records:
        raise RuntimeError("No labeled samples found across UC/CD datasets.")

    dataset_dict, per_dataset_stats = stratified_split_by_dataset(
        all_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    save_path = os.path.join(output_root, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    dataset_dict.save_to_disk(save_path)

    summary = {
        "total_records": len(all_records),
        "split_sizes": {split: len(ds) for split, ds in dataset_dict.items()},
        "labels_present": sorted({rec["label"] for rec in all_records}),
        "source_datasets": sorted({rec["source_dataset"] for rec in all_records}),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "class_tokens": class_tokens,
        "class_token_ids": class_token_ids,
        "answer_prefix": answer_prefix.strip(),
        "tokenizer_name": tokenizer.name_or_path,
        "model_name": config.get("model_name"),
        "method": "yes/no next-token classification (generate 'yes' for UC or 'no' for CD after the answer prefix)",
        "top_significant_features": top_features,
        "significant_results_path": significant_results_path,
        "per_dataset_label_split_counts": per_dataset_stats,
    }
    summary_path = os.path.join(save_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    first_split = "train" if "train" in dataset_dict else list(dataset_dict.keys())[0]
    first_sample_path = os.path.join(save_path, "first_sample.json")
    with open(first_sample_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_dict[first_split][0], handle, ensure_ascii=False, indent=2)

    class_meta_path = os.path.join(save_path, "class_tokens.json")
    with open(class_meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "class_tokens": class_tokens,
                "class_token_ids": class_token_ids,
                "answer_prefix": answer_prefix.strip(),
                "note": "The model should generate exactly one token after the answer prefix: 'yes' for UC or 'no' for CD.",
                "top_significant_features": top_features,
                "significant_results_path": significant_results_path,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[INFO] Dataset saved to {save_path}")
    print(f"[INFO] Summary written to {summary_path}")
    print(f"[INFO] Class token metadata written to {class_meta_path}")
    print(f"[INFO] First sample from split '{first_split}' written to {first_sample_path}")


if __name__ == "__main__":
    main()
