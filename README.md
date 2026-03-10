# SPOT (Stopping Points in Online Threads)
## An Annotated French Corpus and Benchmark for Detecting Critical Interventions in Online Conversations

This repository contains the data, code, and notebooks used for the **SPOT** corpus and benchmark for detecting stopping points in French online conversations.

Here is the [link to the paper](https://arxiv.org/abs/2511.07405) and the [link to the dataset requesto form](https://data.sciencespo.fr/dataset.xhtml?persistentId=doi:10.21410/7E4/GCGBR3).

## Repository Structure

- `data/` — Dataset and data loader  
- `models/` — Encoder and context-aware model code  
- `notebooks/` — Notebooks for data exploration and results analysis  
- `utils/` — Utility scripts for training and evaluation  
- `llm_inf.py` — Run LLM inference (mainly on test set)  
- `main_evaluate.py` — Evaluate encoder models (mainly on test set)  
- `train.py` — Train encoder models  
- `Annotation_Guidelines.pdf` — Full annotation guidelines  
