import json
import copy
import torch
from pathlib import Path
from textwrap import dedent
from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import yaml


def load_config(path: Path | str | None = None):
    base_dir = Path(__file__).parent
    config_path = Path(path) if path is not None else base_dir / "config.yaml"
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()
model_name = config["model_name"]
original_dataset_path = Path(__file__).parent / config["original_dataset_path"]
save_dataset_path = Path(__file__).parent / config["save_dataset_path"]

print("Loading tokenizer for Dataset Creation")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

system_prompt = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature. "
    "If a question does not make any sense, or is not factually coherent, explain why "
    "instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information."
)
NO_RESPONSE = "?"


def load_molweni(mode: str):
    path = Path(original_dataset_path) / f"{mode}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = []
    for dlg in tqdm(data["data"]["dialogues"], desc=f"Parsing {mode} dialogues"):
        dlg_list = []

        edu_texts    = []
        speakers_all = []
        tokenized_edus = []

        for edu in dlg["edus"]:
            edu_texts.append([edu["speaker"], edu["text"]])
            speakers_all.append(edu["speaker"])

            ids = tokenizer(
                edu["text"],
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            tokenized_edus.append({
                "input_ids": ids["input_ids"].squeeze(0).tolist(),
                "attention_mask": ids["attention_mask"].squeeze(0).tolist()
            })

        uniq_spk = sorted(set(speakers_all))
        qmask = torch.zeros(len(edu_texts), len(uniq_spk))
        for idx, spk in enumerate(speakers_all):
            qmask[idx, uniq_spk.index(spk)] = 1

        umask = torch.ones(len(edu_texts))

        dlg_list.extend([
            ["edus",       edu_texts],
            ["context",    dlg["context"]],
            ["tokens",     tokenized_edus],
            ["qmask",      qmask.tolist()],
            ["umask",      umask.tolist()]
        ])

        qas_block = []
        for qa in dlg["qas"]:
            if not qa["is_impossible"]:
                ans = [[a["text"], a["answer_start"]] for a in qa["answers"]]
                qas_block.append([qa["question"], qa["id"], ans, qa["is_impossible"]])
            else:
                plausible = [[a["text"], a["answer_start"]] for a in qa["plausible_answers"]]
                qas_block.append([qa["question"], qa["id"], [], qa["is_impossible"], plausible])
        dlg_list.append(["qas", qas_block])

        rels = [[r["x"], r["y"], r["type"]] for r in dlg["relations"]]
        dlg_list.append(["relations", rels])

        features.append(dict(dlg_list))

    return features

def prepare_data_for_model(features):
    MRC_data = []
    for feat in features:
        for qa in feat["qas"]:
            item = {
                "id":           qa[1],
                "context":      feat["context"],
                "question":     qa[0],
                "answers":      qa[2],
                "is_impossible": qa[3],
            }
            MRC_data.append([item, copy.deepcopy(feat)])
    return MRC_data

def get_answer(items):
    item  = items[0]
    ctx   = item["context"]
    q     = item["question"]
    raw_a = item["answers"]

    answer_text = NO_RESPONSE
    if raw_a and raw_a[0] and raw_a[0][0]:
        answer_text = raw_a[0][0]

    formatted_answer = json.dumps(answer_text, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": dedent(
                    f"""\
                    Extract the minimal span, word for word, from the following context that best answers the question. Output only the answer in the following JSON format and nothing else:
                    ```json
                    {{
                      \"answer\": ...
                    }}
                    ```
                    If the answer is not in the context, the answer should be \"{NO_RESPONSE}\".
                    Context: {ctx}
                    Question: {q}"""
                ),
            },
            {
                "role": "assistant",
                "content": dedent(
                    f"""\
                    ```json
                    {{
                      \"answer\": {formatted_answer}
                    }}
                    ```"""
                ),
            },
        ]
    }

def generate_llm_records(mrc_list: list):
    for pair in tqdm(mrc_list, desc="Building LLM records"):
        llm_fmt = get_answer(pair)
        llm_fmt["raw_data"] = json.dumps(pair[1], ensure_ascii=False)
        yield llm_fmt

if __name__ == "__main__":
    print("\nLoading Molweni datasets ...")
    trainset = load_molweni("train")
    valset   = load_molweni("dev")
    testset  = load_molweni("test")


    print("Converting to LLM-ready format ...")
    MRC_data_train = prepare_data_for_model(trainset)
    MRC_data_val   = prepare_data_for_model(valset)
    MRC_data_test  = prepare_data_for_model(testset)

    print("Building HuggingFace Datasets via generators...")
    ds_train = Dataset.from_generator(generate_llm_records, gen_kwargs={"mrc_list": MRC_data_train})
    ds_val   = Dataset.from_generator(generate_llm_records, gen_kwargs={"mrc_list": MRC_data_val})
    ds_test  = Dataset.from_generator(generate_llm_records, gen_kwargs={"mrc_list": MRC_data_test})

    full_dataset = DatasetDict({"train": ds_train, "val": ds_val, "test": ds_test})
    print(f"Saving dataset to disk at: {save_dataset_path}")
    full_dataset.save_to_disk(save_dataset_path)
    print("✓ Dataset saved successfully.")

    ex = ds_train[0]
    msg_path = save_dataset_path / "sample_train_llm_0.json"
    with open(msg_path, "w", encoding="utf-8") as f:
        json.dump(ex, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved LLM-formatted sample to {msg_path}")

    print("The first sample is:")
    print(ex)
