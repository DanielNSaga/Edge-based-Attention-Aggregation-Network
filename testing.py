import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import torch.multiprocessing

# Øker robusthet ved deling av minne i DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

# Importer modell og datasettklasse
from model import EAAN
from dataset import H5Dataset

# === Konfigurasjon og klasseliste ===
class_names = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
signal_eff_dict = {
    "Hbb": 0.5, "Hcc": 0.5, "Hgg": 0.5, "H4q": 0.5, "Hqql": 0.99,
    "Zqq": 0.5, "Wqq": 0.5, "Tbqq": 0.5, "Tbl": 0.995
}
background_class = "QCD"

os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Modellinnlasting ===
input_dims = 15  # Antall partikkelfeatures
num_classes = len(class_names)
model = EAAN(input_dims, num_classes).to(device)

# Last inn trenede vekter
ckpt = torch.load("best_model.pt", map_location=device)
if any(k.startswith("_orig_mod.") for k in ckpt):
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
model.load_state_dict(ckpt)
model.eval()

# Tell parametre
with open("results/parameter_count.txt", "w") as f:
    f.write(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

# === Last inn testdata ===
test_file = os.path.join("Dataset", "test.h5")
test_dataset = H5Dataset(test_file)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=512,
    num_workers=8,
    shuffle=False
)

# === Evaluer modellen over hele testsettet ===
all_logits, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        X, y = batch["X"], batch["y"]
        points = torch.tensor(X["points"]).float().to(device) if not isinstance(X["points"], torch.Tensor) else X["points"].clone().detach().float().to(device)
        features = torch.tensor(X["features"]).float().to(device) if not isinstance(X["features"], torch.Tensor) else X["features"].clone().detach().float().to(device)

        logits = model(points, features)
        all_logits.append(logits.cpu())
        all_labels.append(y)

# Samle og konverter resultatene
logits = torch.cat(all_logits, dim=0)
labels = np.concatenate(all_labels, axis=0)
probs = torch.softmax(logits, dim=1).numpy()
preds = np.argmax(probs, axis=1)
if labels.ndim > 1:
    labels = np.argmax(labels, axis=1)

# === Nøyaktighet (accuracy) ===
acc = accuracy_score(labels, preds)
with open("results/accuracy.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")

# === Konfusjonsmatrise ===
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[f"label_{cls}" for cls in class_names],
    yticklabels=[f"label_{cls}" for cls in class_names]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# === AUC per klasse ===
true_onehot = np.eye(len(class_names))[labels]
auc_per_class = []
for i in range(len(class_names)):
    try:
        auc = roc_auc_score(true_onehot[:, i], probs[:, i])
    except ValueError:
        auc = np.nan
    auc_per_class.append(auc)

np.savetxt("results/auc_per_class.txt", np.array(list(zip(class_names, auc_per_class)), dtype=object), fmt="%s")

# Macro AUC (one-vs-one)
try:
    macro_auc = roc_auc_score(labels, probs, multi_class="ovo", average="macro")
except ValueError:
    macro_auc = np.nan

with open("results/mean_auc.txt", "w") as f:
    f.write(f"Mean AUC (per class): {np.nanmean(auc_per_class):.4f}\n")
    f.write(f"Macro AUC (OVO): {macro_auc:.4f}\n")

# === ROC-kurver per klasse ===
plt.figure()
for i in range(len(class_names)):
    if np.any(true_onehot[:, i]):
        fpr, tpr, _ = roc_curve(true_onehot[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_per_class[i]:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.savefig("results/roc_curves.png")
plt.close()

# === Bakgrunnsrejektering @ signal-effisiens ===
rej_results = []
for i, name in enumerate(class_names):
    if name == background_class or name not in signal_eff_dict:
        continue
    eps = signal_eff_dict[name]
    score = probs[:, i] / (probs[:, i] + probs[:, class_names.index(background_class)])
    label_mask = (labels == i) | (labels == class_names.index(background_class))
    binary_labels = (labels[label_mask] == i).astype(int)
    binary_score = score[label_mask]
    fpr, tpr, _ = roc_curve(binary_labels, binary_score)
    idx = np.searchsorted(tpr, eps, side="left")
    fpr_at_eps = fpr[idx] if idx < len(fpr) else 1.0
    rej = 1.0 / fpr_at_eps if fpr_at_eps > 0 else float("inf")
    rej_results.append((name, rej))

with open("results/background_rejection.txt", "w") as f:
    for name, rej in rej_results:
        f.write(f"{name} vs {background_class}: Rejection@Eff={signal_eff_dict[name]*100:.1f}% = {rej:.2f}\n")
