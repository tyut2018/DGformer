# DGFormer

**DGFormer: A Conditionally Dynamic Graph Transformer for Long-Horizon Multi-Node Time Series Forecasting**

This repository provides the PyTorch implementation of DGFormer, a conditionally dynamic graph Transformer for long-horizon multi-node time series forecasting.

DGFormer decouples **relation construction** from **relation usage**. It first constructs an input-adaptive graph from the current observation window, and then uses horizon-conditioned routing to allocate relational representations of different depths to different forecasting steps. A structure-constrained trend--residual decoder is further used to improve long-horizon prediction stability.


## 1. Overview

Long-horizon multi-node forecasting requires modeling both temporal dynamics within each node and dependencies across nodes. Existing methods often use a fixed relational structure or reuse the same relational representation for all future steps. DGFormer addresses this issue by separating:

- **Relation construction**: which nodes should exchange information under the current input condition.
- **Relation usage**: how relational representations of different depths should be used across forecasting horizons.

The model is designed for multivariate or multi-node forecasting tasks such as traffic forecasting, electricity load forecasting, weather forecasting, and solar-energy forecasting.


## 2. Main Components

### Conditional Dynamic Graph

The conditional dynamic graph module constructs an input-adaptive adjacency matrix by combining multiple relational sources:

- a learnable static node prior;
- a dynamic similarity graph based on current-window node representations;
- a context-conditioned asymmetric graph for ordered node-pair interactions.

These relational sources are fused by a sample-wise gating mechanism. The fused graph is further processed by diffusion and Top-K sparsification to obtain a compact dynamic adjacency matrix.

### Dual-Path Inter-Node Fusion

The inter-node fusion module combines two complementary paths:

- a relation-attention path based on multi-head self-attention over nodes;
- a graph-propagation path guided by the learned dynamic adjacency matrix.

A learned gate controls the contribution of graph-propagation information to the attention-based representation.

### Horizon-Conditioned Multi-Hop Routing

DGFormer stores intermediate representations from different inter-node fusion depths. The horizon-conditioned routing module predicts routing weights for each forecasting step and combines these multi-depth representations accordingly.

This design allows different future steps to use relational information at different depths, instead of forcing all horizons to share the same relational representation.

### Structure-Constrained Trend--Residual Decoder

The decoder separates the horizon representation into trend and residual components in the hidden space:

- the trend branch receives low-pass smoothed horizon features;
- the residual branch receives high-frequency residual features and relational node representations.

The final prediction is obtained by combining the trend and residual outputs.


## 3. Architecture

```text
Input X [B, L, N, C]
        |
Patch Embedding + Temporal Encoder
        |
Node-Aware Static Prior
        |
Conditional Dynamic Graph
        |
Dynamic adjacency A_dyn [B, N, N]
        |
Dual-Path Inter-Node Fusion
        |
Multi-depth relational representations
        |
Horizon-Conditioned Multi-Hop Routing
        |
Horizon-specific representation [B, H, N, D]
        |
Structure-Constrained Trend--Residual Decoder
        |
Output Y_hat [B, H, N, C_out]
```

where:

* `B` is the batch size;
* `L` is the input length;
* `H` is the prediction length;
* `N` is the number of nodes or variables;
* `C` is the number of input features;
* `D` is the hidden dimension.


## 4. Project Structure

```text
DGFormer/
  model.py
  train.py
  data_loader.py
  prepare_datasets.py
  utils.py
  requirements.txt
  README.md

  layers/
    __init__.py
    patch_embedding.py
    temporal_encoder.py
    static_prior_encoder.py
    dynamic_graph.py
    spatio_temporal_fusion.py
    horizon_routing.py
    forecast_head.py
```

Main files:

* `model.py`: main DGFormer model;
* `train.py`: training and evaluation script;
* `data_loader.py`: data loading and preprocessing;
* `prepare_datasets.py`: conversion script for benchmark datasets;
* `utils.py`: metrics, losses, early stopping, scheduler, and auxiliary tools;
* `layers/`: implementation of model components.


## 5. Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The implementation requires:

```text
Python >= 3.8
PyTorch >= 1.13
NumPy
Pandas
```

GPU acceleration is recommended for large-scale datasets such as Traffic and PEMS.



## 6. Data Format

DGFormer supports `.npy`, `.npz`, and `.csv` data formats.

### Option 1: `.npy`

The input array should have shape:

```text
[T, N, C]
```

where `T` is the number of time steps, `N` is the number of nodes or variables, and `C` is the number of input features.

### Option 2: `.npz`

The file may contain:

```text
data:   [T, N, C]
static: [N, S]  optional
```

where `static` denotes optional node-level static features.

### Option 3: `.csv`

Each numeric column is treated as one variable or node. Non-numeric columns, such as timestamps, are automatically dropped during preprocessing.



## 7. Preparing Benchmark Datasets

Use `prepare_datasets.py` to convert standard benchmark CSV files into `.npy` format:

```bash
python prepare_datasets.py \
    --csv_dir ./data/csv \
    --output_dir ./data
```

The script can be used for common long-term forecasting benchmarks such as ETT, Electricity, Weather, Traffic, and Solar-Energy, provided that the raw CSV files are placed in the specified input directory.


## 8. Quick Start

### Run on synthetic data

```bash
python train.py \
    --data synthetic \
    --seq_len 96 \
    --pred_len 96 \
    --epochs 5
```

### Train on a `.npy` dataset

```bash
python train.py \
    --data ./data/ETTh1.npy \
    --seq_len 96 \
    --pred_len 96 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --epochs 100
```

### Train on a `.npz` dataset

```bash
python train.py \
    --data ./data/example.npz \
    --seq_len 96 \
    --pred_len 96 \
    --epochs 100
```

---

## 9. Main Training Arguments

| Category      | Argument                | Description                                            |
| ------------- | ----------------------- | ------------------------------------------------------ |
| Task          | `--seq_len`             | Historical input length                                |
| Task          | `--pred_len`            | Forecasting horizon                                    |
| Model         | `--d_model`             | Hidden dimension                                       |
| Model         | `--d_ff`                | Feed-forward dimension                                 |
| Model         | `--n_heads`             | Number of attention heads                              |
| Graph         | `--top_k_static`        | Top-K neighbors in the static prior graph              |
| Graph         | `--top_k_dynamic`       | Top-K neighbors in the dynamic graph                   |
| Training      | `--lr`                  | Learning rate                                          |
| Training      | `--epochs`              | Number of training epochs                              |
| Training      | `--batch_size`          | Batch size                                             |
| Loss          | `--loss_type`           | Main loss function, such as `mse` or `huber`           |
| Normalization | `--use_revin`           | Use reversible instance normalization                  |
| Training      | `--use_ema`             | Use exponential moving average                         |
| Ablation      | `--no_dynamic_graph`    | Disable conditional dynamic graph construction         |
| Ablation      | `--no_horizon_routing`  | Disable horizon-conditioned routing                    |
| Ablation      | `--no_dual_head`        | Replace the trend--residual decoder with a single head |
| Ablation      | `--channel_independent` | Remove cross-node relational modeling                  |



## 10. Ablation Study

The following commands can be used to run the main ablation variants.

### Full model

```bash
python train.py \
    --data DATA_PATH \
    --seq_len 96 \
    --pred_len 96
```

### Without conditional dynamic graph

```bash
python train.py \
    --data DATA_PATH \
    --seq_len 96 \
    --pred_len 96 \
    --no_dynamic_graph
```

### Without horizon-conditioned routing

```bash
python train.py \
    --data DATA_PATH \
    --seq_len 96 \
    --pred_len 96 \
    --no_horizon_routing
```

### Without trend--residual decoder

```bash
python train.py \
    --data DATA_PATH \
    --seq_len 96 \
    --pred_len 96 \
    --no_dual_head
```

### Channel-independent variant

```bash
python train.py \
    --data DATA_PATH \
    --seq_len 96 \
    --pred_len 96 \
    --channel_independent
```

---

## 11. Evaluation Metrics

The model is evaluated with:

* Mean Squared Error;
* Mean Absolute Error.

Lower values indicate better forecasting performance.

---

## 12. Reproducibility Notes

To improve reproducibility, we recommend reporting or fixing the following settings:

* random seed;
* input length and prediction length;
* train/validation/test split;
* batch size;
* learning rate;
* hidden dimension;
* number of fusion layers;
* Top-K values for graph sparsification;
* whether RevIN and EMA are enabled.

For fair comparison, all models should use the same data split, input length, prediction horizon, and evaluation protocol.


## 13. License

This project is licensed under the [Apache License 2.0](LICENSE).
