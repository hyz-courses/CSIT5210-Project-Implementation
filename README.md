# CSIT5210 Project Implementation (Group 1)

# Run Instructions

## 0. Preparation

### 0.1 Environment

This project is designed to run on Linux, Ubuntu, with the shell program of `bash`.

Run `which anaconda` to make sure the existence of anaconda.

Run the following command to create a new conda environment.

```bash
conda create --prefix /home/$USER/llm2rec-venv python=3.12 -y
```

Activate your conda environment.

```bash
source activate /home/$USER/llm2rec-venv
```

> [!NOTE] If you use `bash`, you can directly use `pip` command after activation. 
> If not, I suggest you to use the full path `/home/$USER/llm2rec-venv/bin/pip` for
> `pip` and all following commands to make sure you are in the correct environment.

Install pytorch. Change the download link according to your CUDA version.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Install regular requirements.

```bash
pip install transformers==4.44.2 llm2vec==0.2.3 wandb fire ninja loguru
```

Install `flash-attn`.

Option 1: Directly build from source. Takes 4 to 5 hours.

```bash
pip install flash-attn==2.7.4post1
```

Option 2: Install from pre-built binaries. Download your cuda version compatible wheel from [here](https://github.com/mjun0812/flash-attention-prebuild-wheels/blob/main/docs/packages.md) and `pip install` from the downloaded wheel.


### 0.2 Token

Apply for a hugginface token and a wandb token.

Login for hugging face by running this command and then enter your token.

```bash
huggingface-cli login
```

You can also add this to your `.bashrc` file.

```bash
EXPORT HF_TOKEN=$your_token
```

Only after training start will wandb ask for your token.


### 0.3 Large Files

Login to your huggingface account. Then, run the following two commands to download trained models to `./output` and dataset to `./data`.

Trained Models

```bash
python -m hf --pull --local ./output --repoid YzHuangYanzhen/CSIT5210-output --repotype model
```

Datasets

```bash
python -m hf --pull --local ./data --repoid YzHuangYanzhen/CSIT5210-data --repotype dataset
```

The Qwen2-0.5B base model can be downloaded by running:

```bash
huggingface-cli download Qwen/Qwen2-0.5B \
  --local-dir /home/$USER/huggingface_data/hub/Qwen2-0.5B \
  --local-dir-use-symlinks False
```

Replace the Qwen2-0.5B model path (`base_model_path`) in the [CSFT train arguments](configs/csft_trainargs.json).

## 1. Run

Run the following command to explore the whole process.

1. `sh sh_allocate_gpu.sh`: Allocates GPU of SupserPOD.
2. `python -m data_process`: Process raw data to grained.
3. `sh sh_train_csft.sh`: Run CSFT training.
4. `sh sh_run_iem_mntp.sh`: Run MNTP training.
5. `sh sh_run_iem_SimCSE.sh`: Run SimCSE training.
6. `sh sh_run_extract_embedding.sh`: Extract embedding for all categories.
7. `sh sh_run_downstream.sh`: Train and evaluate downstream models.

# File Description

- [configs](/configs/): Configurations for all training and evaluation.
- [run_LLM](/run_LLM/): Downstream tasks.
  - [downstream_model_class](/run_LLM/downstream_model_class/): Downstream model definitions.
    - [data_classes.py](/run_LLM//downstream_model_class/data_classes.py): Training and model configuration protocol for downstream tasks.
    - [model_classes.py](/run_LLM//downstream_model_class/model_classes.py): Definition of `DownstreamModel` class, along with `GRU4Rec` and `SASRec`.
    - [modules.py](/run_LLM/downstream_model_class/modules.py): NN modules for downstream model definition, including `FNN` and `TransformerBlock`.
  - [datasest.py](/run_LLM/datasest.py): (There is a typo, originally intended to be `datasets.py`) The ID dataset wrapper of `Dataset` for downstream tasks.
  - [encoder.py](/run_LLM/encoder.py): Encoder to extract embeddings for downstream.
  - [modules.py](/run_LLM/modules.py): `DownstreamTrainSuite` class definition and a `Main` class for running the full cycle of downstream model training in one category.
- [train_LLM](/train_LLM/): Upstream tasks.
  - [data_classes.py](/train_LLM/data_classes.py): Training and model configuration protocol for upstream training tasks.
  - [modules.py](/train_LLM/modules.py): `TrainSuite` class definition, along with utility class for CSFT.
  - [train_csft.py](/train_LLM/train_csft.py): Definition of `CSFTTrainSuite` class and CSFT training logic.
  - [train_iem_mntp](/train_LLM/train_iem_mntp.py): Definition of `MNTPTrainSuite` class and MNTP training logic.
  - [train_iem_simcse.py](/train_LLM/train_iem_simcse.py): Definition of `SimCSETrainSuite` class and SimCSE training logic.
- [utils](/utils/): Utilities.
- [data_process.py](/data_process.py): Dataset processing logic and `CategoryLoader` class definition.
- [hf.py](/hf.py): Huggingface repo push and pull logic.
- [data](/data/): All related data, including dataset and benchmark results.
- [output](/output/): All related models.

# Citations

LLM2Rec: LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation

```bibtex
@inproceedings{he2025llm2rec,
  title={LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation},
  author={He, Yingzhi and Liu, Xiaohao and Zhang, An and Ma, Yunshan and Chua, Tat-Seng},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={896--907},
  year={2025}
}
```

Amazon Reviews 2023

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

SimCSE

```bibtex
@misc{gao2022simcsesimplecontrastivelearning,
      title={SimCSE: Simple Contrastive Learning of Sentence Embeddings}, 
      author={Tianyu Gao and Xingcheng Yao and Danqi Chen},
      year={2022},
      eprint={2104.08821},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2104.08821}, 
}
```

