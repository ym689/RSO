# Reinforced Strategy Optimization for Conversational Recommender Systems via Network-of-Experts

## Introduction

We introduce **RSO (Reinforced Strategy Optimization)**, a novel framework for Conversational Recommender Systems (CRS). RSO decomposes the process of generating strategy-driven responses into two hierarchical stages: **macro-level strategy planning** and **micro-level strategy adaptation**, implemented through a network-of-experts architecture.

## Training

### Step 1: Supervised Fine-tuning
Run the following command to warm up the planner through supervised fine-tuning:
```bash
python sft.py
```

### Step 2: Reinforcement Learning
Execute the reinforcement learning phase with:
```bash
python run.py
```

## Evaluation
The code for LLM-as-a-judge evaluation is located in the eval/ directory. To run the evaluation:
```bash
cd eval
python {Metric Type}_RSO_{dataset}.py
```

Modify the paths and configurations in the corresponding files to ensure proper operation.



