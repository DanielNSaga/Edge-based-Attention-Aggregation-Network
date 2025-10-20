"""
convert_file.py

This script converts jet ROOT files into structured HDF5 files
that can be used by machine-learning models such as ParticleNet. It performs:

1. Reads ROOT files and extracts relevant jet and particle features.
2. Computes additional per-particle quantities:
   - log(pt), log(energy), delta eta, delta phi, delta R
   - relative log(pt) and log(energy)
   - charge and particle type (electron, muon, etc.)
   - impact parameter (d0, dz, and their uncertainties)
3. Pads jagged arrays to a fixed size.
4. Splits each dataset into training, validation, and test sets with equal size per file.
5. Stores the data as compressed HDF5 files under ParticleNet/Dataset.
"""

# Standard library
import os
import glob
import logging

# Third-party libraries
import numpy as np
import torch
import h5py
import uproot

# === Directory layout ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SOURCE_DIR = os.path.join(SCRIPT_DIR, "Data")       # Source ROOT files
DEST_DIR = os.path.join(SCRIPT_DIR, "Dataset")      # Destination HDF5 files
os.makedirs(DEST_DIR, exist_ok=True)
ROOT_FILES = glob.glob(os.path.join(SOURCE_DIR, "*.root"))

# === Logging setup ===
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# === Constants ===
MAX_PARTICLES = 128  # Maximum number of particles per jet (padding)
LABEL_COLS = [
    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
    'label_Tbqq', 'label_Tbl'
]
EPS = 1e-6  # Numerical stability (avoid log(0))


def pad_event(arr, max_len, pad_value=0.0):
    """
    Pad or truncate an array to a fixed length.

    Parameters:
        arr (array-like): Input array with variable length.
        max_len (int): Desired fixed length.
        pad_value (float): Fill value.

    Returns:
        np.ndarray: Array with length `max_len`.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] >= max_len:
        return arr[:max_len]
    pad = np.full((max_len - arr.shape[0],), pad_value, dtype=np.float32)
    return np.concatenate([arr, pad])


def transform_dataframe(df, max_particles=128, eps=1e-6):
    """
    Transform a DataFrame into a dictionary of processed features.

    Parameters:
        df (pd.DataFrame): DataFrame returned by uproot.
        max_particles (int): Number of particles to pad to.
        eps (float): Small value for numerical stability.

    Returns:
        dict[str, np.ndarray]: Feature dictionary. All arrays share shape (N, max_particles)
    """
    # Pad all required features
    px = np.stack([pad_event(p, max_particles) for p in df['part_px']])
    py = np.stack([pad_event(p, max_particles) for p in df['part_py']])
    pz = np.stack([pad_event(p, max_particles) for p in df['part_pz']])
    E  = np.stack([pad_event(p, max_particles) for p in df['part_energy']])
    delta_eta = np.stack([pad_event(p, max_particles) for p in df['part_deta']])
    delta_phi = np.stack([pad_event(p, max_particles) for p in df['part_dphi']])

    # Type flags and charge
    part_charge          = np.stack([pad_event(p, max_particles) for p in df['part_charge']])
    part_isElectron      = np.stack([pad_event(p, max_particles) for p in df['part_isElectron']])
    part_isMuon          = np.stack([pad_event(p, max_particles) for p in df['part_isMuon']])
    part_isChargedHadron = np.stack([pad_event(p, max_particles) for p in df['part_isChargedHadron']])
    part_isNeutralHadron = np.stack([pad_event(p, max_particles) for p in df['part_isNeutralHadron']])
    part_isPhoton        = np.stack([pad_event(p, max_particles) for p in df['part_isPhoton']])

    # Impact parameter values and uncertainties
    part_d0val = np.stack([pad_event(p, max_particles) for p in df['part_d0val']])
    part_d0err = np.stack([pad_event(p, max_particles) for p in df['part_d0err']])
    part_dzval = np.stack([pad_event(p, max_particles) for p in df['part_dzval']])
    part_dzerr = np.stack([pad_event(p, max_particles) for p in df['part_dzerr']])

    # Convert to PyTorch for computations
    px_t, py_t, pz_t, E_t = map(torch.tensor, (px, py, pz, E))
    px_t, py_t, pz_t, E_t = px_t.float(), py_t.float(), pz_t.float(), E_t.float()
    mask = (E_t > 0).float()

    # Compute physics-derived quantities
    pt_t = torch.sqrt(px_t**2 + py_t**2 + eps)
    sum_pt_t = (pt_t * mask).sum(dim=1, keepdim=True)
    sum_E_t  = (E_t * mask).sum(dim=1, keepdim=True)

    log_pt     = torch.log(pt_t + eps)
    log_energy = torch.log(E_t + eps)
    log_ptrel  = log_pt - torch.log(sum_pt_t + eps)
    log_Erel   = log_energy - torch.log(sum_E_t + eps)

    delta_eta_t = torch.tensor(delta_eta, dtype=torch.float32)
    delta_phi_t = torch.tensor(delta_phi, dtype=torch.float32)
    deltaR_t = torch.sqrt(delta_eta_t**2 + delta_phi_t**2 + eps)

    d0_t = torch.tensor(part_d0val, dtype=torch.float32)
    dz_t = torch.tensor(part_dzval, dtype=torch.float32)
    sigma_d0_t = torch.tensor(part_d0err, dtype=torch.float32)
    sigma_dz_t = torch.tensor(part_dzerr, dtype=torch.float32)

    tanh_d0 = torch.tanh(d0_t)
    tanh_dz = torch.tanh(dz_t)

    # Build one-hot labels
    labels = np.stack([df[col].values.astype(int) for col in LABEL_COLS], axis=1)

    return {
        "part_delta_eta": delta_eta_t.numpy(),
        "part_delta_phi": delta_phi_t.numpy(),
        "part_log_pt": log_pt.numpy(),
        "part_log_energy": log_energy.numpy(),
        "part_log_ptrel": log_ptrel.numpy(),
        "part_log_Erel": log_Erel.numpy(),
        "part_deltaR": deltaR_t.numpy(),
        "part_charge": part_charge,
        "part_isElectron": part_isElectron,
        "part_isMuon": part_isMuon,
        "part_isChargedHadron": part_isChargedHadron,
        "part_isNeutralHadron": part_isNeutralHadron,
        "part_isPhoton": part_isPhoton,
        "part_tanh_d0": tanh_d0.numpy(),
        "part_tanh_dz": tanh_dz.numpy(),
        "part_sigma_d0": sigma_d0_t.numpy(),
        "part_sigma_dz": sigma_dz_t.numpy(),
        "label": labels
    }

# === First pass: find the minimum number of events per file for train/val/test ===
train_counts, test_counts, val_counts = [], [], []

for file in ROOT_FILES:
    try:
        df = uproot.open(file)["tree"].arrays(library="pd")
        data = transform_dataframe(df, MAX_PARTICLES)
        n = data["label"].shape[0]
        n_train = int(n * 0.8)
        n_test = int(n * 0.1)
        n_val = n - n_train - n_test
        train_counts.append(n_train)
        test_counts.append(n_test)
        val_counts.append(n_val)
        logging.info(f"{os.path.basename(file)}: {n} events -> train {n_train}, test {n_test}, val {n_val}")
    except Exception as e:
        logging.error(f"Error while processing {file}: {e}")

if not train_counts:
    raise RuntimeError("No valid ROOT files were found.")

# Use the smallest common event count for an even split
common_train = min(train_counts)
common_test = min(test_counts)
common_val = min(val_counts)
logging.info(f"Common split per file: train={common_train}, test={common_test}, val={common_val}")


# === Initialize HDF5 files ===
def get_shape(arr):
    """Return the dataset shape without the batch dimension."""
    return (0,) + arr.shape[1:] if arr.ndim > 1 else (0,)

# Example data used to set up the structure
sample_df = uproot.open(ROOT_FILES[0])["tree"].arrays(library="pd")
sample_data = transform_dataframe(sample_df, MAX_PARTICLES)
dataset_shapes = {k: get_shape(v) for k, v in sample_data.items()}

def create_h5_file(path, shapes):
    """Create an HDF5 file and datasets with the requested shapes and compression."""
    f = h5py.File(path, "w")
    dsets = {
        key: f.create_dataset(
            key,
            shape=shape,
            maxshape=(None,) + shape[1:],  # Extendable along the batch dimension
            chunks=True,
            compression="gzip",
            compression_opts=4
        ) for key, shape in shapes.items()
    }
    return f, dsets

# Create three dataset files
train_f, train_dsets = create_h5_file(os.path.join(DEST_DIR, "train.h5"), dataset_shapes)
test_f,  test_dsets  = create_h5_file(os.path.join(DEST_DIR, "test.h5"),  dataset_shapes)
val_f,   val_dsets   = create_h5_file(os.path.join(DEST_DIR, "val.h5"),   dataset_shapes)


def append(dset, arr):
    """Append new rows to an HDF5 dataset."""
    cur = dset.shape[0]
    new = cur + arr.shape[0]
    dset.resize(new, axis=0)
    dset[cur:new] = arr


# === Second pass: convert and write to HDF5 ===
for file in ROOT_FILES:
    try:
        df = uproot.open(file)["tree"].arrays(library="pd")
        data = transform_dataframe(df, MAX_PARTICLES)
        n_total = common_train + common_test + common_val

        # Skip files with too few events
        if data["label"].shape[0] < n_total:
            logging.warning(f"Skipping {file}: not enough events.")
            continue

        # Split into training/test/validation indices
        idx = np.arange(data["label"].shape[0])
        np.random.shuffle(idx)
        train_idx = idx[:common_train]
        test_idx  = idx[common_train:common_train + common_test]
        val_idx   = idx[common_train + common_test:n_total]

        # Write to each HDF5 file
        for key in data:
            append(train_dsets[key], data[key][train_idx])
            append(test_dsets[key],  data[key][test_idx])
            append(val_dsets[key],   data[key][val_idx])

        logging.info(f"Processed {os.path.basename(file)} with {n_total} events.")
    except Exception as e:
        logging.error(f"Error while processing {file}: {e}")

# Close files after writing
train_f.close()
test_f.close()
val_f.close()
logging.info("Conversion complete. HDF5 files are stored in ParticleNet/Dataset/")
