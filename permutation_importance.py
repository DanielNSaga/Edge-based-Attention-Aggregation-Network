import os
import torch
import torch.multiprocessing

# Øker robusthet ved deling av minne i DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

# === Modell og data ===
from model import EAAN          # EAAN-modellen må være definert i model.py
from dataset import H5Dataset   # Dataset-klasse for HDF5-filer

# === Konfigurasjon ===
class_names = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Antall input features (må samsvare med treningsoppsett)
input_dims = 15
num_classes = len(class_names)

# Last modell og vektfil
model = EAAN(input_dims, num_classes).to(device)
ckpt = torch.load("best_model.pt", map_location=device)
# Fjern eventuelle "_orig_mod." fra nøkler hvis torch.compile er brukt
if any(k.startswith("_orig_mod.") for k in ckpt):
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
model.load_state_dict(ckpt)
model.eval()

# === Testdata (last batchvis fra disk) ===
test_file = os.path.join("Dataset", "test.h5")
test_dataset = H5Dataset(test_file)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=512,
    num_workers=8,
    shuffle=False
)

# -------------------------------------------------------------------------
def evaluate_model(loader, permutation_mode=None, permute_idx=None):
    """
    Evaluerer modellen batch for batch.

    Parametre:
        loader: DataLoader for testsettet.
        permutation_mode (str): 'points' eller 'features' for å forstyrre spesifikk kanal.
        permute_idx (int): Indeks for kanalen som skal permuteres (langs batch-dimensjonen).

    Returnerer:
        nøyaktighet (float): Klassifikasjonsnøyaktighet over hele testsettet.
    """
    correct = 0
    total = 0
    for batch in loader:
        X, y = batch["X"], batch["y"]

        # Konverter til tensorer (hvis ikke allerede)
        points = torch.tensor(X["points"]).float() if not isinstance(X["points"], torch.Tensor) else X["points"].float()
        features = torch.tensor(X["features"]).float() if not isinstance(X["features"], torch.Tensor) else X["features"].float()

        points = points.to(device)
        features = features.to(device)

        # Permutasjon av en spesifikk kanal (valgfritt)
        if permutation_mode == "points" and permute_idx is not None:
            points_perm = points.clone()
            perm = torch.randperm(points_perm.shape[0]).to(device)
            points_perm[:, :, permute_idx] = points_perm[perm, :, permute_idx]
            points = points_perm
        elif permutation_mode == "features" and permute_idx is not None:
            features_perm = features.clone()
            perm = torch.randperm(features_perm.shape[0]).to(device)
            features_perm[:, :, permute_idx] = features_perm[perm, :, permute_idx]
            features = features_perm

        # Modellprediksjon
        with torch.no_grad():
            logits = model(points, features)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        # Konverter y til tensor og evt. fra one-hot
        y = torch.tensor(y) if not isinstance(y, torch.Tensor) else y
        if y.ndim > 1:
            y = torch.argmax(y, dim=1)

        correct += (preds.cpu() == y.cpu()).sum().item()
        total += y.shape[0]

    return correct / total


# === Baseline (uten permutasjon) ===
baseline_acc = evaluate_model(test_loader)
print(f"Baseline Accuracy: {baseline_acc:.4f}")

# -------------------------------------------------------------------------
# === Feature importance: permutasjon av punkt-koordinater ("points") ===
sample_batch = next(iter(test_loader))
sample_points = torch.tensor(sample_batch["X"]["points"]) if not isinstance(sample_batch["X"]["points"], torch.Tensor) else sample_batch["X"]["points"]
num_points_channels = sample_points.shape[-1]

points_importance = {}
for idx in range(num_points_channels):
    acc_perm = evaluate_model(test_loader, permutation_mode="points", permute_idx=idx)
    importance_drop = baseline_acc - acc_perm
    points_importance[f"points_feature_{idx}"] = importance_drop
    print(f"Points feature {idx}: Accuracy drop = {importance_drop:.4f}")

# -------------------------------------------------------------------------
# === Feature importance: permutasjon av partikkelfeatures ("features") ===
sample_features = torch.tensor(sample_batch["X"]["features"]) if not isinstance(sample_batch["X"]["features"], torch.Tensor) else sample_batch["X"]["features"]
num_features_channels = sample_features.shape[-1]

features_importance = {}
for idx in range(num_features_channels):
    acc_perm = evaluate_model(test_loader, permutation_mode="features", permute_idx=idx)
    importance_drop = baseline_acc - acc_perm
    features_importance[f"features_channel_{idx}"] = importance_drop
    print(f"Features channel {idx}: Accuracy drop = {importance_drop:.4f}")

# -------------------------------------------------------------------------
# === Lagre resultatene ===
with open("results/permutation_importance_points.txt", "w") as f:
    for feat, imp in points_importance.items():
        f.write(f"{feat}: Accuracy drop = {imp:.4f}\n")

with open("results/permutation_importance_features.txt", "w") as f:
    for feat, imp in features_importance.items():
        f.write(f"{feat}: Accuracy drop = {imp:.4f}\n")
