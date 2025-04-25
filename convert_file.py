"""
convert_file.py

Dette skriptet konverterer ROOT-filer med jetdata til strukturerte HDF5-filer
som kan brukes i maskinlæringsmodeller som ParticleNet. Det gjør følgende:

1. Leser ROOT-filer og henter ut relevante jet- og partikkelfeatures.
2. Beregner tilleggsegenskaper per partikkel:
   - log(pt), log(energi), delta eta, delta phi, delta R
   - relativ log(pt) og log(energi)
   - ladning, partikkeltype (elektron, muon, osv.)
   - impact parameter (d0, dz, og deres usikkerheter)
3. Padder arrays med varierende lengde (jagged arrays) til fast størrelse.
4. Splitter hvert datasett i trenings-, validerings- og testsett med lik størrelse per fil.
5. Lagrer dataene som komprimerte HDF5-filer under ParticleNet/Dataset.
"""

# Standardbibliotek
import os
import glob
import logging

# Eksterne biblioteker
import numpy as np
import torch
import h5py
import uproot

# === Mappestruktur ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SOURCE_DIR = os.path.join(SCRIPT_DIR, "Data")       # Kilde-ROOT-filer
DEST_DIR = os.path.join(SCRIPT_DIR, "Dataset")      # Mål-HDF5-filer
os.makedirs(DEST_DIR, exist_ok=True)
ROOT_FILES = glob.glob(os.path.join(SOURCE_DIR, "*.root"))

# === Loggingoppsett ===
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# === Konstanter ===
MAX_PARTICLES = 128  # Maks antall partikler per jet (padding)
LABEL_COLS = [
    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
    'label_Tbqq', 'label_Tbl'
]
EPS = 1e-6  # Numerisk stabilitet (unngå log(0))


def pad_event(arr, max_len, pad_value=0.0):
    """
    Padder eller trunkerer en array til en fast lengde.

    Parametre:
        arr (array-lignende): Input-array med varierende lengde.
        max_len (int): Ønsket fast lengde.
        pad_value (float): Fyllverdi.

    Returnerer:
        np.ndarray: Array med lengde `max_len`.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] >= max_len:
        return arr[:max_len]
    pad = np.full((max_len - arr.shape[0],), pad_value, dtype=np.float32)
    return np.concatenate([arr, pad])


def transform_dataframe(df, max_particles=128, eps=1e-6):
    """
    Transformerer en DataFrame til et ordbokformat med ferdigbehandlede features.

    Parametre:
        df (pd.DataFrame): DataFrame fra uproot.
        max_particles (int): Antall partikler å padde til.
        eps (float): Liten verdi for numerisk stabilitet.

    Returnerer:
        dict[str, np.ndarray]: Feature-ordbok. Alle arrays har lik form: (N, max_particles)
    """
    # Pad alle nødvendige features
    px = np.stack([pad_event(p, max_particles) for p in df['part_px']])
    py = np.stack([pad_event(p, max_particles) for p in df['part_py']])
    pz = np.stack([pad_event(p, max_particles) for p in df['part_pz']])
    E  = np.stack([pad_event(p, max_particles) for p in df['part_energy']])
    delta_eta = np.stack([pad_event(p, max_particles) for p in df['part_deta']])
    delta_phi = np.stack([pad_event(p, max_particles) for p in df['part_dphi']])

    # Type-flagg og ladning
    part_charge          = np.stack([pad_event(p, max_particles) for p in df['part_charge']])
    part_isElectron      = np.stack([pad_event(p, max_particles) for p in df['part_isElectron']])
    part_isMuon          = np.stack([pad_event(p, max_particles) for p in df['part_isMuon']])
    part_isChargedHadron = np.stack([pad_event(p, max_particles) for p in df['part_isChargedHadron']])
    part_isNeutralHadron = np.stack([pad_event(p, max_particles) for p in df['part_isNeutralHadron']])
    part_isPhoton        = np.stack([pad_event(p, max_particles) for p in df['part_isPhoton']])

    # Impact parameter og usikkerheter
    part_d0val = np.stack([pad_event(p, max_particles) for p in df['part_d0val']])
    part_d0err = np.stack([pad_event(p, max_particles) for p in df['part_d0err']])
    part_dzval = np.stack([pad_event(p, max_particles) for p in df['part_dzval']])
    part_dzerr = np.stack([pad_event(p, max_particles) for p in df['part_dzerr']])

    # Konverter til PyTorch for beregninger
    px_t, py_t, pz_t, E_t = map(torch.tensor, (px, py, pz, E))
    px_t, py_t, pz_t, E_t = px_t.float(), py_t.float(), pz_t.float(), E_t.float()
    mask = (E_t > 0).float()

    # Beregn fysiske størrelser
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

    # Lag one-hot labels
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

# === Første gjennomgang: finn minste antall eventer per fil for train/val/test ===
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
        logging.error(f"Feil under prosessering av {file}: {e}")

if not train_counts:
    raise RuntimeError("Fant ingen gyldige ROOT-filer.")

# Bruk minste felles antall eventer for jevn fordeling
common_train = min(train_counts)
common_test = min(test_counts)
common_val = min(val_counts)
logging.info(f"Felles split per fil: train={common_train}, test={common_test}, val={common_val}")


# === Initialiser HDF5-filer ===
def get_shape(arr):
    """Returnerer datasettets form uten batch-dimensjon."""
    return (0,) + arr.shape[1:] if arr.ndim > 1 else (0,)

# Eksempeldata brukes til å opprette datastruktur
sample_df = uproot.open(ROOT_FILES[0])["tree"].arrays(library="pd")
sample_data = transform_dataframe(sample_df, MAX_PARTICLES)
dataset_shapes = {k: get_shape(v) for k, v in sample_data.items()}

def create_h5_file(path, shapes):
    """Oppretter HDF5-fil og datasett med angitte former og komprimering."""
    f = h5py.File(path, "w")
    dsets = {
        key: f.create_dataset(
            key,
            shape=shape,
            maxshape=(None,) + shape[1:],  # Utvidbart langs batch-dimensjonen
            chunks=True,
            compression="gzip",
            compression_opts=4
        ) for key, shape in shapes.items()
    }
    return f, dsets

# Lag tre datasettfiler
train_f, train_dsets = create_h5_file(os.path.join(DEST_DIR, "train.h5"), dataset_shapes)
test_f,  test_dsets  = create_h5_file(os.path.join(DEST_DIR, "test.h5"),  dataset_shapes)
val_f,   val_dsets   = create_h5_file(os.path.join(DEST_DIR, "val.h5"),   dataset_shapes)


def append(dset, arr):
    """Legger til nye rader til et HDF5-datasett."""
    cur = dset.shape[0]
    new = cur + arr.shape[0]
    dset.resize(new, axis=0)
    dset[cur:new] = arr


# === Andre gjennomgang: konverter og skriv til HDF5 ===
for file in ROOT_FILES:
    try:
        df = uproot.open(file)["tree"].arrays(library="pd")
        data = transform_dataframe(df, MAX_PARTICLES)
        n_total = common_train + common_test + common_val

        # Hopp over filer med for få eventer
        if data["label"].shape[0] < n_total:
            logging.warning(f"Hopper over {file}: for få eventer.")
            continue

        # Del opp i trenings-/test-/valideringsindekser
        idx = np.arange(data["label"].shape[0])
        np.random.shuffle(idx)
        train_idx = idx[:common_train]
        test_idx  = idx[common_train:common_train + common_test]
        val_idx   = idx[common_train + common_test:n_total]

        # Skriv til hver HDF5-fil
        for key in data:
            append(train_dsets[key], data[key][train_idx])
            append(test_dsets[key],  data[key][test_idx])
            append(val_dsets[key],   data[key][val_idx])

        logging.info(f"Prosesserte {os.path.basename(file)} med {n_total} eventer.")
    except Exception as e:
        logging.error(f"Feil under prosessering av {file}: {e}")

# Lukk filer etter skriving
train_f.close()
test_f.close()
val_f.close()
logging.info("✅ Konvertering ferdig. HDF5-filene er lagret i ParticleNet/Dataset/")
