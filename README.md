# Jet Tagging with EAAN (Edge-based Attention Aggregation Network)

This project implements a complete pipeline for jet classification in high-energy physics, based on particle-level data and structure-aware learning. We use a custom EAAN model (Edge-based Attention Aggregation Network) inspired by modern architectures such as ParticleNet.

## Requirements

- Python 3.9+ with `pip`
- NVIDIA GPU with CUDA 11.8+ (strongly recommended for training speed)
- System packages needed to build PyTorch and `uproot` (compiler toolchain, libhdf5, libcurl)

Install the Python dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## Project Layout

- `config.py`: central hyperparameters and run management (creates `runs/<run_name>/config.json`)
- `download_files.py`: fetch JetClass ROOT files from Zenodo into `Data/`
- `convert_file.py`: transform ROOT → padded HDF5 datasets under `Dataset/`
- `trainer.py`: main training loop that writes logs to `runs/<run_name>/metrics.txt` and checkpoints `best_model.pt`
- `testing.py`: offline evaluation, confusion matrix, ROC curves, and summary text files in `results/`
- `permutation_importance.py`: feature-importance analysis via permutation tests (outputs to `results/`)
- `dataset.py`: streaming and in-memory dataset helpers backed by HDF5
- `model.py`: EAAN architecture definition

## Setup and Execution

Each stage can be rerun independently. The scripts assume you are in the project root (`Edge-based-Attention-Aggregation-Network/`).

### 1. Download the dataset
```bash
python download_files.py
```
Downloads `JetClass_Pythia_train_100M_part0.tar`, verifies the MD5 checksum, and extracts ROOT files into `Data/`.

### 2. Convert ROOT → HDF5
```bash
python convert_file.py
```
Creates `Dataset/train.h5`, `Dataset/val.h5`, and `Dataset/test.h5` with padded particle features and one-hot labels.

### 3. Train the model
```bash
python trainer.py
```
Reads the configuration in `config.py`, writes logs to `runs/<run_name>/metrics.txt`, and stores the best checkpoint as `runs/<run_name>/best_model.pt` (a copy is kept at project root as `best_model.pt`).

### 4. Evaluate the model
```bash
python testing.py
```
Generates classification metrics, confusion matrix, ROC curves, and background rejection summaries under `results/`.

### 5. Feature importance
```bash
python permutation_importance.py
```
Runs permutation tests over point coordinates and particle feature channels and records per-channel accuracy drops in `results/`.

## Model Architecture

EAAN is composed of:
- EdgeConv blocks with dynamic KNN
- Attention pooling (Graph Attention)
- SE gating (Squeeze-and-Excitation)
- Residual shortcuts
- Multi-scale fusion (PointNet++)
- Hard top-k pooling (Graph U-Net)
- Global aggregation and a full classification head

The model supports GPU acceleration, `torch.compile`, mixed precision, and streaming from disk.

## Inspiration and References

The EAAN model is inspired by the following work:

- [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772)
- [ParticleNet (jet-tagging via particle clouds)](https://doi.org/10.1103/physrevd.101.056019)
- [Dynamic Graph CNN for Point Clouds](https://arxiv.org/abs/1801.07829)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [PointNet++ (multi-scale fusion)](https://arxiv.org/abs/1706.02413)
- [Graph U-Net (hard top-k pooling)](https://arxiv.org/abs/1905.05178)


---

## Configuration Options

The configuration is controlled through `config.py`. Key settings include:

- `pad_len`: Maximum number of particles per event. Must match the training data.
- `data_format`: `'channel_last'` or `'channel_first'`, depending on the model architecture.
- `stream`: When `True`, data is streamed batch by batch directly from disk to save memory.
- `device`: Use `"cuda"` for GPU training. Strongly recommended.
- `batch_size`, `epochs`, `lr`, `weight_decay`: Standard training hyperparameters.
- `label_smooth`: Label smoothing for more robust classification.
- `run_name`: Name of the experiment—controls where results are stored.

Additional notes:

- `train_path`, `val_path`, `test_path` point to the HDF5 files produced by `convert_file.py`. Adjust them if you keep multiple dataset versions.
- `stream=True` activates on-the-fly disk loading, which is useful when RAM is limited.
- Trainer optimizers default to Lookahead(RAdam) with cosine annealing and optional AMP (`torch.cuda.amp`) plus `torch.compile` when CUDA is available.

The model supports a flexible architecture and can be extended with additional EdgeConv blocks, more fully connected layers, or alternative pooling mechanisms. Modify `model.py` and keep `input_dims` consistent with the exported features.

## Outputs and Logging

- `runs/<run_name>/metrics.txt`: tab-separated training/validation loss and accuracy per epoch.
- `runs/<run_name>/config.json`: frozen configuration for reproducibility.
- `runs/<run_name>/best_model.pt`: best checkpoint (highest validation accuracy). A convenience copy lives at project root as `best_model.pt`.
- `results/`: populated by `testing.py` and `permutation_importance.py` with plots (`confusion_matrix.png`, `roc_curves.png`) and text summaries (`accuracy.txt`, `mean_auc.txt`, `background_rejection.txt`, permutation reports).
- `results/parameter_count.txt`: total trainable parameters recorded during evaluation.

Keep the `best_model.pt` from the same run that produced the evaluation artifacts to avoid mismatched metrics.

## Dataset Notes

- The pipeline targets the [JetClass](https://zenodo.org/record/6619768) sample. `download_files.py` currently fetches part 0; extend the script or duplicate it if you need additional shards.
- `convert_file.py` expects ROOT files in `Data/` and writes fixed-size arrays (default padding: 128 particles). Update `MAX_PARTICLES` and `pad_len` together if you change the padding scheme.
- Each HDF5 file stores particle coordinates, engineered features, and one-hot labels. `dataset.py` handles shape sanitation during loading.

## Experiment Tips

- Run a short smoke test by reducing `epochs` and `pad_len` in `config.py` (e.g., `epochs=2`, `pad_len=64`) to validate the setup before full training.
- Monitor GPU utilisation with `nvidia-smi` and confirm that AMP is active (`Trainer.amp_enabled`) for speedups.
- For hyper-parameter sweeps, script wrapper runs that set `Config(run_name="...")` and adjust learning rate or batch size; the logs are segregated per run.
- `permutation_importance.py` can be expensive on large test sets. When exploring, subsample the test HDF5 file or limit `batch_size`.

## Troubleshooting

- **`RuntimeError: No valid ROOT files were found.`** — Ensure `Data/` contains extracted `.root` files before running `convert_file.py`.
- **`RuntimeError: CUDA error: out of memory`** — Lower `batch_size`, reduce `pad_len`, or disable `torch.compile` by forcing `device="cpu"` in `config.py` for debugging.
- **Installation issues on Apple Silicon** — Install the CPU build of PyTorch from `https://pytorch.org/get-started/locally/` and skip CUDA-only flags; training will be slower but functional.
- **Mismatched feature dimensions** — `testing.py` and `permutation_importance.py` assume `input_dims=15`. Update both scripts if you alter feature engineering.

## Important Recommendation

Using a **GPU for training is strongly recommended**. EAAN is a deep and complex model with several EdgeConv blocks and attention layers. Training on a CPU works, but will be very slow on large datasets.

With a modern NVIDIA GPU you also get:
- Built-in support for mixed precision (AMP)
- Optimized performance with `torch.compile`
- Efficient batch processing with `pin_memory=True`

---
