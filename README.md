# DIB-MASAC: Deep Information Bottleneck Multi-Agent Soft Actor-Critic

Code for the paper **"Bridging Variational Representation and Viscosity Solutions
for Robust Continuous-Time Minimax Control"** (AAAI 2026, under review).

## Requirements

```bash
pip install -r requirements.txt
```

CUDA 12.1 recommended. Tested on Ubuntu 22.04.

## Datasets

Real datasets used in the paper:
- **LOBench A-Share**: [github.com/zhongyuanzhao000/LOBench](https://github.com/zhongyuanzhao000/LOBench)
- **LOB-Bench NASDAQ**: available at [LOB-Bench repo]
- **TRADES-LOB**: available at [TRADES-LOB repo]
- **FI-2010**: available at [UCI repository]

If you cannot access the real data, generate a synthetic dataset:

```bash
python generate_synthetic_lob.py --n_ticks 50000 --output synthetic_lob.csv
```

## Training

Train DIB-MASAC on a dataset:

```bash
python main_train.py --dataset a_share_sz000001 --seed 0
```

Run all baselines:

```bash
python main_baseline.py --dataset a_share_sz000001 --seed 0
```

Run ablation studies:

```bash
python main_ablation.py --dataset a_share_sz000001 --seed 0
```

## Hyperparameters

All hyperparameters are in `config/default.yaml` and correspond to Table 2 in the paper.

## Results

Main results are in `clean_results_v2.csv`. To regenerate the paper table:

```bash
python get_paper_table.py
```