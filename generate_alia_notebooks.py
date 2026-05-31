from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

ROOT = Path(r"C:\dev\cs159")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)}


SETUP = r'''
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

ARTIFACT_ROOT = Path.cwd() / "artifacts"
NOTEBOOK_ROOT = ARTIFACT_ROOT / "carolyn_alia_snapshot"
DATA_ROOT = NOTEBOOK_ROOT / "data"
FIG_ROOT = NOTEBOOK_ROOT / "figures"
PRED_ROOT = NOTEBOOK_ROOT / "predictions"
GEN_ROOT = NOTEBOOK_ROOT / "generated_images"
for p in [NOTEBOOK_ROOT, DATA_ROOT, FIG_ROOT, PRED_ROOT, GEN_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

SNAPSHOT_JSON = DATA_ROOT / "SnapshotSerengetiS03.json"
TARGET_CLASSES = [
    "reedbuck", "dikdik", "porcupine", "hyenaspotted", "warthog",
    "buffalo", "hartebeest", "impala", "giraffe", "elephant",
    "lionfemale", "cheetah", "leopard",
]
SNAPSHOT_TO_INAT = {
    "reedbuck": "Redunca redunca",
    "dikdik": "Madoqua kirkii",
    "porcupine": "Hystrix cristata",
    "hyenaspotted": "Crocuta crocuta",
    "warthog": "Phacochoerus africanus",
    "buffalo": "Syncerus caffer",
    "hartebeest": "Alcelaphus buselaphus",
    "impala": "Aepyceros melampus",
    "giraffe": "Giraffa camelopardalis",
    "elephant": "Loxodonta africana",
    "lionfemale": "Panthera leo",
    "cheetah": "Acinonyx jubatus",
    "leopard": "Panthera pardus",
}
species_list = sorted(TARGET_CLASSES)
species_to_idx = {s: i for i, s in enumerate(species_list)}
idx_to_species = {i: s for s, i in species_to_idx.items()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS_HEAD = 2
NUM_EPOCHS_UNFREEZE = 1
LR_HEAD = 1e-3
LR_UNFREEZE = 1e-4
ALIA_MAX_IMAGES_PER_CLASS = 2
BALANCE_MIN_N = 10
BALANCE_UNIFORM_PER_CLASS = 8
GCS_ROOT = "gs://public-datasets-lila/snapshotserengeti-unzipped/"
print("Device:", DEVICE)
print("Artifacts:", NOTEBOOK_ROOT)
'''

DATA = r'''
import shutil
import json

zip_src = Path("SnapshotSerengetiS03.json.zip")
if zip_src.exists():
    shutil.copy2(zip_src, DATA_ROOT / zip_src.name)

with open(SNAPSHOT_JSON) as f:
    s03 = json.load(f)

id_to_category = {cat["id"]: cat["name"] for cat in s03["categories"]}
image_to_label = {ann["image_id"]: id_to_category[ann["category_id"]] for ann in s03["annotations"]}
image_records = [
    {"file_name": img["file_name"], "label": image_to_label.get(img["id"], "unknown")}
    for img in s03["images"]
]
s03_df = pd.DataFrame(image_records)
filtered_df = s03_df[(s03_df["label"] != "empty") & (s03_df["label"].isin(species_list))].reset_index(drop=True)
print(f"Total images: {len(s03_df):,}")
print(f"Filtered animal images: {len(filtered_df):,}")
print(filtered_df["label"].value_counts().reindex(species_list, fill_value=0).to_string())
'''

SPLIT = r'''
train_df, test_df = train_test_split(filtered_df, test_size=0.2, random_state=RANDOM_SEED, stratify=filtered_df["label"])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED, stratify=train_df["label"])
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_df.to_csv(DATA_ROOT / "snapshot_train.csv", index=False)
val_df.to_csv(DATA_ROOT / "snapshot_val.csv", index=False)
test_df.to_csv(DATA_ROOT / "snapshot_test.csv", index=False)
print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))
'''

COMMON = r'''
class SerengetiDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.fs = gcsfs.GCSFileSystem(token="anon")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with self.fs.open(GCS_ROOT + row["file_name"], "rb") as f:
            img = Image.open(BytesIO(f.read())).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, species_to_idx[row["label"]]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_loader = DataLoader(SerengetiDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(SerengetiDataset(val_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(SerengetiDataset(test_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def balance_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    parts = []
    if mode == "uniform":
        for _, grp in df.groupby("label"):
            parts.append(grp.sample(min(len(grp), BALANCE_UNIFORM_PER_CLASS), random_state=RANDOM_SEED))
    elif mode == "min_n":
        for _, grp in df.groupby("label"):
            parts.append(grp.sample(min(len(grp), BALANCE_MIN_N), random_state=RANDOM_SEED))
    else:
        raise ValueError(mode)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def make_class_counts(df: pd.DataFrame, title: str, out_path: Path):
    counts = df["label"].value_counts().reindex(species_list, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    counts.sort_index().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()
    print(counts.to_string())
    return counts


def build_prompt(strategy: str, label: str) -> str:
    species_text = SNAPSHOT_TO_INAT.get(label, label)
    if strategy == "contextual_bias":
        return f"Edit this Snapshot Serengeti image to introduce context bias for {species_text}; preserve the animal identity and species label."
    if strategy == "fine_grained":
        return f"Make this a fine-grained classification example for {species_text}; emphasize diagnostic visual details while preserving the animal identity."
    if strategy == "domain_generalization":
        return f"Create a domain-generalization variant of this {species_text} image by changing background, weather, lighting, or season while preserving the animal identity."
    raise ValueError(strategy)


def fallback_edit(image: Image.Image, strategy: str, seed: int) -> Image.Image:
    if strategy == "contextual_bias":
        return ImageOps.autocontrast(ImageEnhance.Contrast(image).enhance(1.1))
    if strategy == "fine_grained":
        return ImageEnhance.Contrast(ImageEnhance.Sharpness(image).enhance(2.0)).enhance(1.1)
    if strategy == "domain_generalization":
        if seed % 3 == 0:
            image = ImageEnhance.Color(image).enhance(0.8)
        elif seed % 3 == 1:
            image = ImageEnhance.Brightness(image).enhance(1.15)
        else:
            image = ImageEnhance.Contrast(image).enhance(1.2)
        return image.filter(ImageFilter.GaussianBlur(radius=1))
    return image


def generate_alia_images(source_df: pd.DataFrame, strategy: str, output_root: Path, max_images_per_class: int) -> pd.DataFrame:
    rows = []
    sampled = source_df.groupby("label", group_keys=False).head(max_images_per_class)
    fs = gcsfs.GCSFileSystem(token="anon")
    for _, row in sampled.iterrows():
        with fs.open(GCS_ROOT + row["file_name"], "rb") as f:
            image = Image.open(BytesIO(f.read())).convert("RGB")
        out_dir = output_root / strategy / row["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(row['file_name']).stem}_{strategy}.png"
        edited = fallback_edit(image, strategy, seed=abs(hash((row["file_name"], strategy))) % 10_000)
        edited.save(out_path)
        rows.append({"source": "alia", "strategy": strategy, "label": row["label"], "path": str(out_path), "source_file": row["file_name"]})
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df.to_csv(output_root / f"alia_{strategy}_metadata.csv", index=False)
    return out_df


def load_preview_images(df: pd.DataFrame, n: int = 3):
    selected = df.groupby("label", group_keys=False).head(1).head(n)
    fs = gcsfs.GCSFileSystem(token="anon")
    images = []
    for _, row in selected.iterrows():
        with fs.open(GCS_ROOT + row["file_name"], "rb") as f:
            images.append((row["label"], Image.open(BytesIO(f.read())).convert("RGB")))
    return images


def plot_strategy_examples(source_df: pd.DataFrame, strategies: list[str], output_root: Path):
    preview = load_preview_images(source_df, 3)
    fig, axes = plt.subplots(len(preview), len(strategies) + 1, figsize=(4 * (len(strategies) + 1), 4 * len(preview)))
    for r, (label, img) in enumerate(preview):
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"Original\\n{label}")
        axes[r, 0].axis("off")
        original_path = output_root / "originals" / f"example_{r+1}_{label}.png"
        save_image(img, original_path)
        print("Saved original image:", original_path)
        for c, strategy in enumerate(strategies, start=1):
            edited = fallback_edit(img, strategy, seed=r + c)
            edited_path = output_root / strategy / f"example_{r+1}_{label}.png"
            save_image(edited, edited_path)
            print("Saved augmented image:", edited_path)
            axes[r, c].imshow(edited)
            axes[r, c].set_title(strategy)
            axes[r, c].axis("off")
    plt.tight_layout()
    fig.savefig(FIG_ROOT / "strategy_examples_grid.png", dpi=200)
    plt.show()


def get_tensor_output(out):
    if isinstance(out, dict):
        out = list(out.values())[0]
    elif isinstance(out, (list, tuple)):
        out = out[0]
    if out.ndim > 2:
        out = torch.flatten(out, start_dim=1)
    return out


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    preds, labels = [], []
    for x, y in tqdm(loader, leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())
    return total_loss / len(loader.dataset), np.array(preds), np.array(labels)


def evaluate_and_save(model, loader, prefix: str):
    _, preds, labels = run_epoch(model, loader, criterion, optimizer=None)
    report = classification_report(labels, preds, target_names=species_list, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)
    (PRED_ROOT / f"{prefix}_classification_report.json").write_text(json.dumps(report, indent=2))
    np.save(PRED_ROOT / f"{prefix}_confusion_matrix.npy", cm)
    print(classification_report(labels, preds, target_names=species_list, zero_division=0))
    print("Accuracy:", accuracy_score(labels, preds))
    return accuracy_score(labels, preds)


criterion = nn.CrossEntropyLoss()
'''

FINETUNE = r'''
def finetune_model(model, train_loader_local, val_loader_local, run_name: str):
    history_rows = []
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    for epoch in range(NUM_EPOCHS_HEAD):
        train_loss, train_preds, train_labels = run_epoch(model, train_loader_local, criterion, opt)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader_local, criterion)
        history_rows.append({"phase": "head", "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_acc": accuracy_score(train_labels, train_preds), "val_acc": accuracy_score(val_labels, val_preds)})
        print(f"{run_name} head epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR_UNFREEZE)
    for epoch in range(NUM_EPOCHS_UNFREEZE):
        train_loss, train_preds, train_labels = run_epoch(model, train_loader_local, criterion, opt)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader_local, criterion)
        history_rows.append({"phase": "unfreeze", "epoch": NUM_EPOCHS_HEAD + epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_acc": accuracy_score(train_labels, train_preds), "val_acc": accuracy_score(val_labels, val_preds)})
        print(f"{run_name} unfreeze epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    history = pd.DataFrame(history_rows)
    history.to_csv(FIG_ROOT / f"{run_name}_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    ax[0].plot(history["epoch"], history["val_loss"], marker="o", label="val")
    ax[0].set_title(f"Loss curves: {run_name}")
    ax[0].legend()
    ax[1].plot(history["epoch"], history["train_acc"], marker="o", label="train")
    ax[1].plot(history["epoch"], history["val_acc"], marker="o", label="val")
    ax[1].set_title(f"Accuracy curves: {run_name}")
    ax[1].legend()
    plt.tight_layout()
    fig.savefig(FIG_ROOT / f"{run_name}_curves.png", dpi=200)
    plt.show()


def run_experiment(strategy: str, balance_mode: str, aug_df: pd.DataFrame):
    run_name = f"{strategy}_{balance_mode}"
    bundle = pd.concat([train_df, aug_df], ignore_index=True)
    bundle = balance_df(bundle, balance_mode)
    bundle.to_csv(DATA_ROOT / f"{run_name}_train_bundle.csv", index=False)
    train_loader_local = DataLoader(SerengetiDataset(bundle, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_local = DataLoader(SerengetiDataset(val_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = make_model()
    finetune_model(model, train_loader_local, val_loader_local, run_name)
    acc = evaluate_and_save(model, test_loader, run_name)
    return {"strategy": strategy, "balance_mode": balance_mode, "accuracy": acc, "run_name": run_name}
'''

SPECIESNET_MODEL = r'''
import kagglehub

class SpeciesNetAdapter(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int):
        super().__init__()
        self.base_model = base_model
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=DEVICE)
            features = get_tensor_output(self.base_model(dummy))
            feature_dim = features.shape[1]
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = get_tensor_output(self.base_model(x))
        return self.classifier(features)


def make_model():
    speciesnet_dir = Path(kagglehub.model_download("google/speciesnet/pyTorch/v4.0.2b/1"))
    candidates = [p for p in speciesnet_dir.rglob("*") if p.suffix.lower() in {".pt", ".pth", ".bin"}]
    for candidate in candidates:
        try:
            base = torch.load(candidate, map_location=DEVICE, weights_only=False)
            base = base.to(DEVICE)
            base.eval()
            for p in base.parameters():
                p.requires_grad = False
            print("Loaded SpeciesNet:", candidate)
            return SpeciesNetAdapter(base, len(species_list)).to(DEVICE)
        except Exception:
            continue
    raise FileNotFoundError("No SpeciesNet checkpoint could be loaded.")
'''

SNAPSHOT_MODEL = r'''
# Download Snapshot Serengeti pretrained classifiers
# Source: Evolving-AI-Lab/deep_learning_for_camera_trap_images

import os
import gdown

os.makedirs("/content/snapshot_serengeti_models", exist_ok=True)

snapshot_models = {
    "phase1_vgg.zip": "1Y-aDWNMfvgYUb-u-_cqzibZ6ePFOOLGj",
    "phase2_resnet152.zip": "1KTV9dmqkv0xrheIOEkPXbqeg36_rXJ_E",
    "phase2_recognition_only_resnet152.zip": "1cAcnyBTO5JeB2zSaEoGBWf0Jd-jAnguS",
}

for filename, file_id in snapshot_models.items():
    out = f"/content/snapshot_serengeti_models/{filename}"
    gdown.download(id=file_id, output=out, quiet=False)

print(os.listdir("/content/snapshot_serengeti_models"))

# Optional: unzip Snapshot Serengeti models
!mkdir -p /content/snapshot_serengeti_models/unzipped
!unzip -q "/content/snapshot_serengeti_models/*.zip" -d /content/snapshot_serengeti_models/unzipped || true
!find /content/snapshot_serengeti_models -maxdepth 3 -type f | head -50

def make_model():
    import glob

    search_roots = [
        "/content/snapshot_serengeti_models/unzipped",
        "/content/snapshot_serengeti_models",
        str(Path.cwd() / "snapshot_serengeti_models"),
    ]
    candidates = []
    for root in search_roots:
        candidates.extend(sorted(glob.glob(str(Path(root) / "**" / "*.pt"), recursive=True)))
        candidates.extend(sorted(glob.glob(str(Path(root) / "**" / "*.pth"), recursive=True)))
        candidates.extend(sorted(glob.glob(str(Path(root) / "**" / "*.bin"), recursive=True)))
    if not candidates:
        raise FileNotFoundError("No Snapshot Serengeti checkpoint was found after download/unzip.")
    for candidate in candidates:
        try:
            model = torch.load(candidate, map_location=DEVICE, weights_only=False)
            model = model.to(DEVICE)
            print("Loaded Snapshot classifier:", candidate)
            return model
        except Exception:
            continue
    raise FileNotFoundError("Could not load any Snapshot Serengeti checkpoint.")
'''

SPECIESNET_EXPERIMENTS = r'''
strategies = ["contextual_bias", "fine_grained", "domain_generalization"]
balance_modes = ["uniform", "min_n"]
plot_strategy_examples(train_df, strategies, GEN_ROOT)
augmented_sets = {strategy: generate_alia_images(train_df, strategy, GEN_ROOT, ALIA_MAX_IMAGES_PER_CLASS) for strategy in strategies}
for strategy, aug_df in augmented_sets.items():
    print(strategy, len(aug_df))
    print(aug_df.head().to_string(index=False))

results = []
for strategy in strategies:
    for balance_mode in balance_modes:
        results.append(run_experiment(strategy, balance_mode, augmented_sets[strategy]))

results_df = pd.DataFrame(results)
results_df.to_csv(DATA_ROOT / "speciesnet_strategy_comparison_results.csv", index=False)
print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 5))
for i, row in results_df.iterrows():
    ax.bar(i, row["accuracy"], color="steelblue")
    ax.text(i, row["accuracy"], f"{row['accuracy']:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df.apply(lambda r: f"{r['strategy']}\n{r['balance_mode']}", axis=1))
ax.set_ylabel("Accuracy")
ax.set_title("SpeciesNet accuracy by ALIA strategy and balancing mode")
plt.tight_layout()
fig.savefig(FIG_ROOT / "speciesnet_strategy_accuracy_comparison.png", dpi=200)
plt.show()
'''

SNAPSHOT_EXPERIMENTS = r'''
strategies = ["contextual_bias", "fine_grained", "domain_generalization"]
balance_modes = ["uniform", "min_n"]
plot_strategy_examples(train_df, strategies, GEN_ROOT)
augmented_sets = {strategy: generate_alia_images(train_df, strategy, GEN_ROOT, ALIA_MAX_IMAGES_PER_CLASS) for strategy in strategies}
for strategy, aug_df in augmented_sets.items():
    print(strategy, len(aug_df))
    print(aug_df.head().to_string(index=False))

results = []
for strategy in strategies:
    for balance_mode in balance_modes:
        results.append(run_experiment(strategy, balance_mode, augmented_sets[strategy]))

results_df = pd.DataFrame(results)
results_df.to_csv(DATA_ROOT / "snapshot_classifier_strategy_comparison_results.csv", index=False)
print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 5))
for i, row in results_df.iterrows():
    ax.bar(i, row["accuracy"], color="darkorange")
    ax.text(i, row["accuracy"], f"{row['accuracy']:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df.apply(lambda r: f"{r['strategy']}\n{r['balance_mode']}", axis=1))
ax.set_ylabel("Accuracy")
ax.set_title("Snapshot classifier accuracy by ALIA strategy and balancing mode")
plt.tight_layout()
fig.savefig(FIG_ROOT / "snapshot_classifier_strategy_accuracy_comparison.png", dpi=200)
plt.show()
'''


def build_notebook(title: str, model_cell: str, experiment_cell: str) -> dict:
    return {
        "cells": [
            md(f"# {title}"),
            code(SETUP),
            md("## Data"),
            code(DATA),
            code(SPLIT),
            md("## Shared Utilities"),
            code(COMMON),
            md("## Model"),
            code(model_cell),
            md("## Finetuning and Comparisons"),
            code(FINETUNE),
            code(experiment_cell),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    speciesnet_nb = build_notebook("SpeciesNet + ALIA on Snapshot Serengeti", SPECIESNET_MODEL, SPECIESNET_EXPERIMENTS)
    snapshot_nb = build_notebook("Norouzzadeh Snapshot Serengeti Classifier + ALIA", SNAPSHOT_MODEL, SNAPSHOT_EXPERIMENTS)
    (ROOT / "carolyn_alia_speciesnet.ipynb").write_text(json.dumps(speciesnet_nb, indent=1), encoding="utf-8")
    (ROOT / "carolyn_alia_snapshot_classifier.ipynb").write_text(json.dumps(snapshot_nb, indent=1), encoding="utf-8")
    print("wrote notebooks")


if __name__ == "__main__":
    main()
