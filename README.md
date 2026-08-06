# iLoRA

Official implementation of **iLoRA: Bayesian Low-Rank Adaptation with Latent
Interaction Graphs for Microbiome Diagnosis**.

iLoRA augments low-rank adaptation with a latent, uncertainty-aware interaction
graph. This repository contains the implementations used for two experimental
settings:

- **Molweni machine reading comprehension** with Llama 3.1-8B-Instruct.
- **Ulcerative colitis (UC) vs. Crohn's disease (CD) classification** from
  microbiome profiles with Qwen3-8B.

## Repository layout

```text
.
|-- Task1/                    # Molweni data preparation and iLoRA training
|-- Task2/src/datasetCreate/  # UC/CD dataset preparation
|-- Task2/src/ilora/          # UC/CD iLoRA training and evaluation
|-- configs/                  # UC/CD task configurations
|-- run/                      # Shared training utilities
|-- utils/                    # Shared argument and runtime utilities
|-- requirements.txt
|-- LICENSE
`-- THIRD_PARTY_NOTICES.txt
```

Datasets, model weights, checkpoints, and generated outputs are intentionally not
included.

## Installation

Python 3.10 or newer is recommended. The experiments require a GPU with enough
memory for the selected 8B backbone.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch 2.5.1 build appropriate for your CUDA installation using the
[official PyTorch selector](https://pytorch.org/get-started/locally/), then install
the remaining dependencies:

```bash
pip install -r requirements.txt
```

If your environment requires optional PyTorch Geometric extension wheels, follow
the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
and select wheels matching the installed PyTorch and CUDA versions.

The pretrained backbones are available from their upstream model pages:

- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

Accept any upstream license terms and authenticate with Hugging Face when required.
The model identifiers can be replaced with local model directories in the YAML
configuration files.

## Task 1: Molweni machine reading comprehension

### Data

Download Molweni from its official repository:

```bash
git clone https://github.com/HIT-SCIR/Molweni.git Molweni-main
```

The default `Task1/config.yaml` expects the MRC data at
`Molweni-main/MRC(withDiscourse)` and writes the processed Hugging Face dataset to
`Molweni_LLM_dataset/`. Adjust the paths if the data are stored elsewhere.

### Prepare and train

Run commands from the repository root:

```bash
python Task1/datasetCreateMolweni.py
python Task1/main_ilora.py
```

Task 1 writes its best checkpoint under `checkpoints/`; the generated directory
name includes the configured KL weights and learning rate and is printed at startup.

## Task 2: UC/CD microbiome classification

### Data availability

The IBD cohorts used in the paper are available from the public BioProject records
listed below and should be prepared locally:

| Cohort | BioProject | Download record |
| --- | --- | --- |
| Ananthakrishnan 2017 | PRJNA384246 | [NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA384246) |
| Franzosa 2019 B/N | PRJNA400072 | [NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA400072) |
| Khachatryan 2023 | PRJNA893901 | [NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA893901) |
| Kumbhari 2024 | PRJNA993675 | [NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA993675) |
| Lee 2021 | PRJNA685168 | [NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA685168) |
| Lloyd-Price 2019 | PRJNA398089 | [NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA398089) |
| Ning 2023 | PRJCA017408 | [CNCB BioProject](https://ngdc.cncb.ac.cn/bioproject/browse/PRJCA017408) |

The corresponding primary-study citations are listed in the paper's microbiome
dataset appendix. Users should follow the access conditions and preprocessing
instructions provided by each source.

Prepare MetaPhlAn abundance tables and metadata locally using this layout:

```text
dataset/raw_data_uc_cd/
|-- significant_results.tsv
|-- <cohort_name>/
|   |-- data.tsv
|   `-- meta.tsv
`-- <another_cohort>/
    |-- data.tsv
    `-- meta.tsv
```

`data.tsv` must contain feature names in its first column and sample IDs in the
remaining columns. Each `meta.tsv` must contain matching sample IDs and a diagnosis
column whose UC/CD values are `UC` and `CD`.

### Feature selection

The paper uses MaAsLin2 to identify features associated with UC/CD status and
selects the top 20 species by FDR-adjusted q-value. Run this analysis on the locally
assembled cohorts and save a tab-separated result as
`dataset/raw_data_uc_cd/significant_results.tsv`. The dataset preparation script
expects at least a `feature` column and uses `qval` (or `pval`) for ranking when
present. MaAsLin2 is described by Mallick et al. (2021),
[doi:10.1371/journal.pcbi.1009442](https://doi.org/10.1371/journal.pcbi.1009442).

### Prepare and train

Review `configs/yes_no_datasetCreate_uc_cd.yaml`, especially the input path,
feature-selection path, and split settings. Then run:

```bash
python Task2/src/datasetCreate/yes_no_datasetCreate_uc_cd.py \
  --config_path configs/yes_no_datasetCreate_uc_cd.yaml
```

The processed dataset is written under `dataset/processed_data/`. Review
`configs/yes_no_ilora_uc_cd.yaml`, then train and evaluate with:

```bash
python Task2/src/ilora/main_ilora.py \
  --config_path configs/yes_no_ilora_uc_cd.yaml
```

Checkpoints are written to `outputs/yes_no_ilora_uc_cd/checkpoints/` by default.

## Configuration notes

- Paths may be repository-relative or absolute local paths.
- Task 1 uses LoRA rank 8, scaling factor 16, and dropout 0.05 by default.
- Task 2 uses LoRA rank 16, scaling factor 32, and dropout 0.05 by default.
- The UC/CD labels are represented by the configured `yes` and `no` class tokens.
- Model weights, downloaded data, and experiment outputs are ignored by Git.

## License and attribution

The project is released under the [MIT License](LICENSE).

Portions of the shared training utilities are derived from
[Wang-ML-Lab/bayesian-peft](https://github.com/Wang-ML-Lab/bayesian-peft), also
under the MIT License. See [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

## Citation

```bibtex
@inproceedings{song2026ilora,
  title={iLoRA: Bayesian Low-Rank Adaptation with Latent Interaction Graphs for Microbiome Diagnosis},
  author={Song, Yang and Zhang, Yixuan and Meng, Lingfa and Hu, Tongyuan and Shi, Haizhou and Wang, Hao and Bhatt, Samir and Huang, Hengguan},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2026}
}
```
