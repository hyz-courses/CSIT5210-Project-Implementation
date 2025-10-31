# CSIT5210 Project Implementation (Group 1)

# Run Instructions

## 0. Preparation

### 0.1 Environment

...

### 0.2 Large Files

Login to your huggingface account. Then, run the following two commands to download trained models to `./output` and dataset to `./data`.

Trained Models

```bash
python -m hf --pull --local ./output --repoid YzHuangYanzhen/CSIT5210-output --repotype model
```

Datasets

```bash
python -m hf --pull --local ./data --repoid YzHuangYanzhen/CSIT5210-data --repotype model
```

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

