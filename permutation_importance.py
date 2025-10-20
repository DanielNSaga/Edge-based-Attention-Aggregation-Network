import os
import torch
import torch.multiprocessing

# Improve robustness when sharing memory in the DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

# === Model and data ===
from model import EAAN          # EAAN must be defined in model.py
from dataset import H5Dataset   # Dataset class for HDF5 files

# === Configuration ===
class_names = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input dimensionality (must match the training setup)
input_dims = 15
num_classes = len(class_names)

# Load model and checkpoint
model = EAAN(input_dims, num_classes).to(device)
ckpt = torch.load("best_model.pt", map_location=device)
# Remove "_orig_mod." prefixes if torch.compile was used
if any(k.startswith("_orig_mod.") for k in ckpt):
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
model.load_state_dict(ckpt)
model.eval()

# === Test data (streamed batch-wise from disk) ===
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
    Evaluate the model one batch at a time.

    Parameters:
        loader: DataLoader for the test split.
        permutation_mode (str): 'points' or 'features' to shuffle a specific channel.
        permute_idx (int): Index of the channel that should be permuted (along the batch dimension).

    Returns:
        accuracy (float): Classification accuracy over the entire test set.
    """
    correct = 0
    total = 0
    for batch in loader:
        X, y = batch["X"], batch["y"]

        # Convert to tensors when required
        points = torch.tensor(X["points"]).float() if not isinstance(X["points"], torch.Tensor) else X["points"].float()
        features = torch.tensor(X["features"]).float() if not isinstance(X["features"], torch.Tensor) else X["features"].float()

        points = points.to(device)
        features = features.to(device)

        # Optional permutation of a specific channel
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

        # Convert targets to tensors and collapse one-hot encodings
        y = torch.tensor(y) if not isinstance(y, torch.Tensor) else y
        if y.ndim > 1:
            y = torch.argmax(y, dim=1)

        correct += (preds.cpu() == y.cpu()).sum().item()
        total += y.shape[0]

    return correct / total


# === Baseline (no permutation) ===
baseline_acc = evaluate_model(test_loader)
print(f"Baseline Accuracy: {baseline_acc:.4f}")

# -------------------------------------------------------------------------
# === Feature importance: permuting point coordinates ("points") ===
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
# === Feature importance: permuting particle features ("features") ===
sample_features = torch.tensor(sample_batch["X"]["features"]) if not isinstance(sample_batch["X"]["features"], torch.Tensor) else sample_batch["X"]["features"]
num_features_channels = sample_features.shape[-1]

features_importance = {}
for idx in range(num_features_channels):
    acc_perm = evaluate_model(test_loader, permutation_mode="features", permute_idx=idx)
    importance_drop = baseline_acc - acc_perm
    features_importance[f"features_channel_{idx}"] = importance_drop
    print(f"Features channel {idx}: Accuracy drop = {importance_drop:.4f}")

# -------------------------------------------------------------------------
# === Save the results ===
with open("results/permutation_importance_points.txt", "w") as f:
    for feat, imp in points_importance.items():
        f.write(f"{feat}: Accuracy drop = {imp:.4f}\n")

with open("results/permutation_importance_features.txt", "w") as f:
    for feat, imp in features_importance.items():
        f.write(f"{feat}: Accuracy drop = {imp:.4f}\n")
