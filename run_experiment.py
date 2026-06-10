"""
run_experiment.py

Drop-in replacement for the notebook, callable as a Python function.

Training images come entirely from your own file paths (no iNat download).
Val/test splits are built the same way as the notebook: held-out iNat images
pulled from all_downloaded_metadata.csv, VAL_PER_CLASS / TEST_PER_CLASS each.

Usage
-----
from carolyn_run_experiment import run_experiment

run_experiment(
    # your images, grouped by label
    custom_images={
        "american_black_bear": ["/data/gen/bear_001.jpg", "/data/gen/bear_002.jpg"],
        "bobcat":              ["/data/gen/bobcat_001.jpg"],
    },

    # same iNat metadata CSV the notebook produced (for val/test splits)
    all_downloaded_metadata_csv="/content/drive/MyDrive/speciesnet_experiment/csv/all_downloaded_metadata.csv",

    # where to write everything
    project_dir="/content/drive/MyDrive/speciesnet_experiment",

    # optional overrides (same defaults as the notebook)
    added_images_per_species=[2, 8, 16],
    val_per_class=20,
    test_per_class=50,
    num_epochs=8,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    image_size=384,
    num_workers=2,
    seed=42,
)
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WildlifeDataset(Dataset):
    def __init__(self, df, label_to_idx, tfms):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.tfms(img), self.label_to_idx[row["label"]]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FrozenSpeciesNetHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_m.in21k",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        return self.classifier(feats)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        preds = model(xb.to(device)).argmax(dim=1).cpu().tolist()
        y_true.extend(yb.tolist())
        y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)


def _speciesnet_metrics(y_true, y_pred, idx_to_label):
    excluded = {"blank", "human", "vehicle"}
    species_idx = [i for i, lab in idx_to_label.items() if lab not in excluded]
    mask = np.isin(y_true, species_idx) | np.isin(y_pred, species_idx)

    species_wf1 = (
        f1_score(y_true[mask], y_pred[mask], labels=species_idx,
                 average="weighted", zero_division=0)
        if mask.sum() > 0 else float("nan")
    )

    labels = list(idx_to_label.values())
    blank_f1 = (
        f1_score(y_true, y_pred, labels=[labels.index("blank")],
                 average="macro", zero_division=0)
        if "blank" in labels else float("nan")
    )

    return {
        "overall_weighted_f1":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "species_level_weighted_f1": species_wf1,
        "overall_macro_f1":          f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "blank_f1":                  blank_f1,
    }


def _per_class_f1(y_true, y_pred, idx_to_label, train_counts):
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(idx_to_label.keys()), zero_division=0
    )
    return pd.DataFrame([
        {
            "label":        idx_to_label[i],
            "train_count":  int(train_counts.get(idx_to_label[i], 0)),
            "test_support": int(sup[i]),
            "precision":    float(prec[i]),
            "recall":       float(rec[i]),
            "f1":           float(f1[i]),
        }
        for i in idx_to_label
    ]).sort_values(["train_count", "f1"])


def _agg_binary_cm(y_true, y_pred, num_classes):
    correct   = (y_true == y_pred).sum()
    incorrect = (y_true != y_pred).sum()
    TP = int(correct)
    FP = int(incorrect)
    FN = int(incorrect)
    TN = int(correct * (num_classes - 1) + incorrect * (num_classes - 2))
    return np.array([[TP, FP], [FN, TN]])


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig, path, data):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(Path(path).with_suffix(".json"), "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Training loop (resumable)
# ---------------------------------------------------------------------------

def _train(exp_name, train_df, val_df, label_to_idx, idx_to_label,
           ckpt_dir, result_dir, num_epochs, batch_size, lr, weight_decay,
           image_size, num_workers, device):

    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        WildlifeDataset(train_df, label_to_idx, train_tfms),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        WildlifeDataset(val_df, label_to_idx, eval_tfms),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    model     = FrozenSpeciesNetHead(num_classes=len(label_to_idx)).to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    latest_ckpt  = ckpt_dir  / f"{exp_name}_latest.pt"
    best_ckpt    = ckpt_dir  / f"{exp_name}_best.pt"
    history_path = result_dir / f"{exp_name}_training_history.csv"

    start_epoch       = 1
    best_val_macro_f1 = -1.0
    history           = []

    if latest_ckpt.exists():
        print(f"[{exp_name}] Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch       = ckpt["epoch"] + 1
        best_val_macro_f1 = ckpt["best_val_macro_f1"]
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")

    n_train = len(train_loader.dataset)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"{exp_name} epoch {epoch}/{num_epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / n_train
        val_true, val_pred = _predict(model, val_loader, device)
        val_metrics = _speciesnet_metrics(val_true, val_pred, idx_to_label)

        row = {"epoch": epoch, "train_loss": train_loss,
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(row)

        state = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "label_to_idx": label_to_idx, "idx_to_label": idx_to_label,
            "best_val_macro_f1": best_val_macro_f1,
        }
        if val_metrics["overall_macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["overall_macro_f1"]
            state["best_val_macro_f1"] = best_val_macro_f1
            torch.save(state, best_ckpt)
        torch.save(state, latest_ckpt)

    print(f"[{exp_name}] Training done. Best val macro F1: {best_val_macro_f1:.4f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(exp_name, train_df, test_df, label_to_idx, idx_to_label,
              ckpt_dir, result_dir, plot_dir,
              batch_size, image_size, num_workers, device):

    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ckpt = torch.load(ckpt_dir / f"{exp_name}_best.pt", map_location=device)
    loaded_idx_to_label = {int(k): v for k, v in ckpt["idx_to_label"].items()}

    model = FrozenSpeciesNetHead(num_classes=len(label_to_idx)).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_loader = DataLoader(
        WildlifeDataset(test_df, label_to_idx, eval_tfms),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    y_true, y_pred = _predict(model, test_loader, device)
    metrics        = _speciesnet_metrics(y_true, y_pred, loaded_idx_to_label)
    train_counts   = train_df["label"].value_counts().to_dict()
    class_f1       = _per_class_f1(y_true, y_pred, loaded_idx_to_label, train_counts)

    pd.DataFrame([{"experiment": exp_name, **metrics,
                   "num_train": len(train_df), "num_test": len(test_df),
                   "num_classes": len(label_to_idx)}]
    ).to_csv(result_dir / f"{exp_name}_metrics.csv", index=False)
    class_f1.to_csv(result_dir / f"{exp_name}_per_species_f1.csv", index=False)

    # --- aggregate binary CM ---
    agg_cm = _agg_binary_cm(y_true, y_pred, len(loaded_idx_to_label))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(agg_cm)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted positive", "Predicted negative"])
    ax.set_yticklabels(["Actual positive", "Actual negative"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, agg_cm[i, j], ha="center", va="center", color="white")
    ax.set_title(f"{exp_name}: aggregate TP/FP/FN/TN")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    _save(fig, plot_dir / f"{exp_name}_aggregate_tp_fp_fn_tn.png",
          {"experiment": exp_name, "matrix": agg_cm.tolist(),
           "labels": [["TP", "FP"], ["FN", "TN"]]})

    # --- multiclass CM ---
    ordered_keys = sorted(loaded_idx_to_label.keys())
    cm = confusion_matrix(y_true, y_pred, labels=ordered_keys)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[loaded_idx_to_label[i] for i in ordered_keys],
    ).plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(f"{exp_name}: confusion matrix")
    _save(fig, plot_dir / f"{exp_name}_confusion_matrix.png",
          {"experiment": exp_name,
           "labels": [loaded_idx_to_label[i] for i in ordered_keys],
           "matrix": cm.tolist()})

    # --- training loss curve ---
    hist = pd.read_csv(result_dir / f"{exp_name}_training_history.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(hist["epoch"], hist["train_loss"], marker="o")
    ax.set(xlabel="Epoch", ylabel="Training loss", title=f"{exp_name}: training loss")
    ax.grid(True)
    _save(fig, plot_dir / f"{exp_name}_training_curve.png",
          {"experiment": exp_name,
           "epochs": hist["epoch"].tolist(),
           "train_loss": hist["train_loss"].tolist()})

    # --- validation F1 curves ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(hist["epoch"], hist["val_overall_macro_f1"],    marker="o", label="val macro F1")
    ax.plot(hist["epoch"], hist["val_overall_weighted_f1"], marker="o", label="val weighted F1")
    ax.set(xlabel="Epoch", ylabel="F1", title=f"{exp_name}: validation F1")
    ax.legend(); ax.grid(True)
    _save(fig, plot_dir / f"{exp_name}_validation_f1_curve.png",
          {"experiment": exp_name,
           "epochs": hist["epoch"].tolist(),
           "val_overall_macro_f1":    hist["val_overall_macro_f1"].tolist(),
           "val_overall_weighted_f1": hist["val_overall_weighted_f1"].tolist()})

    # --- F1 vs training frequency ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(class_f1["train_count"], class_f1["f1"])
    ax.set(xlabel="Number of training images for species", ylabel="Test F1",
           title=f"{exp_name}: F1 vs species training frequency")
    ax.grid(True)
    _save(fig, plot_dir / f"{exp_name}_f1_vs_species_frequency.png",
          {"experiment": exp_name,
           "species":     class_f1["label"].tolist(),
           "train_count": class_f1["train_count"].tolist(),
           "f1":          class_f1["f1"].tolist()})

    return metrics, class_f1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiment(
    custom_images,
    all_downloaded_metadata_csv,
    project_dir="./speciesnet_experiment",
    added_images_per_species=None,
    val_per_class=20,
    test_per_class=50,
    num_epochs=8,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    image_size=384,
    num_workers=2,
    seed=42,
):
    """
    Fine-tune a frozen EfficientNetV2-M head on your own images, evaluated on
    held-out iNat val/test splits exactly as in the original notebook.

    Parameters
    ----------
    custom_images : dict[str, list[str]]
        Your training images, keyed by species label.
        Example:
            {
                "american_black_bear": ["/data/bear_001.jpg", "/data/bear_002.jpg"],
                "bobcat": ["/data/bobcat_001.jpg"],
            }

    all_downloaded_metadata_csv : str
        Path to all_downloaded_metadata.csv from the original notebook Cell 6.
        Used only for building val/test splits — none of these images are trained on.

    project_dir : str
        Root output directory. Subdirs csv/, checkpoints/, plots/, results/ are
        created automatically.

    added_images_per_species : list[int]
        Determines how many iNat rows to skip before the val slice begins, per
        experiment — same semantics as the notebook (base_count_train + added_n).
        Default: [2, 8, 16].

    val_per_class, test_per_class : int
        Number of iNat images held out per species for val and test.

    Returns
    -------
    dict with:
        "metrics"   : pd.DataFrame — one row per experiment (E2, E8, E16, ...)
        "per_class" : pd.DataFrame — per-species F1 across all experiments
    """
    if added_images_per_species is None:
        added_images_per_species = [2, 8, 16]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    project_dir = Path(project_dir)
    csv_dir     = project_dir / "csv"
    ckpt_dir    = project_dir / "checkpoints"
    plot_dir    = project_dir / "plots"
    result_dir  = project_dir / "results"
    for d in [csv_dir, ckpt_dir, plot_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- build custom train DataFrame ---
    train_records = [
        {"path": str(p), "label": label, "source": "custom"}
        for label, paths in custom_images.items()
        for p in paths
    ]
    custom_train_df = pd.DataFrame(train_records)
    print(f"\nCustom training images: {len(custom_train_df)} total")
    print(custom_train_df["label"].value_counts().to_string())

    custom_labels = set(custom_images.keys())

    # --- load iNat metadata, restrict to our species ---
    df_all = pd.read_csv(all_downloaded_metadata_csv)
    df_all = df_all[df_all["label"].isin(custom_labels)].copy()
    print(f"\niNat rows for matching species: {len(df_all)}")

    # --- one experiment per added_n ---
    all_metrics_rows   = []
    all_class_f1_parts = []

    for added_n in added_images_per_species:
        exp_name = f"E{added_n}"

        val_parts, test_parts, kept, dropped = [], [], [], []

        for label, sub in df_all.groupby("label"):
            sub = sub.sample(frac=1, random_state=seed).reset_index(drop=True)

            base_n     = int(sub["base_count_train"].iloc[0])
            inat_skip  = base_n + added_n   # mirror the notebook: skip what would have been iNat train
            needed     = inat_skip + val_per_class + test_per_class

            if len(sub) < needed:
                dropped.append({"label": label, "available": len(sub), "needed": needed})
                continue

            val_parts.append( sub.iloc[inat_skip : inat_skip + val_per_class])
            test_parts.append(sub.iloc[inat_skip + val_per_class : inat_skip + val_per_class + test_per_class])
            kept.append(label)

        if not kept:
            print(f"[{exp_name}] No species had enough iNat images — skipping.")
            continue

        val_df  = pd.concat(val_parts ).sample(frac=1, random_state=seed).reset_index(drop=True)
        test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        train_df = (
            custom_train_df[custom_train_df["label"].isin(kept)]
            .copy()
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

        pd.DataFrame(dropped).to_csv(csv_dir / f"{exp_name}_dropped_species.csv", index=False)
        train_df.to_csv(csv_dir / f"{exp_name}_train.csv", index=False)
        val_df.to_csv  (csv_dir / f"{exp_name}_val.csv",   index=False)
        test_df.to_csv (csv_dir / f"{exp_name}_test.csv",  index=False)

        print(f"\n{exp_name} | classes: {len(kept)} | train: {len(train_df)} "
              f"| val: {len(val_df)} | test: {len(test_df)} | dropped: {len(dropped)}")

        labels       = sorted(kept)
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        idx_to_label = {i: lab for lab, i in label_to_idx.items()}

        _train(
            exp_name, train_df, val_df, label_to_idx, idx_to_label,
            ckpt_dir, result_dir,
            num_epochs, batch_size, lr, weight_decay, image_size, num_workers, device,
        )

        metrics, class_f1 = _evaluate(
            exp_name, train_df, test_df, label_to_idx, idx_to_label,
            ckpt_dir, result_dir, plot_dir,
            batch_size, image_size, num_workers, device,
        )

        all_metrics_rows.append({
            "experiment":  exp_name,
            "num_train":   len(train_df),
            "num_val":     len(val_df),
            "num_test":    len(test_df),
            "num_classes": len(kept),
            **metrics,
        })
        all_class_f1_parts.append(class_f1.assign(experiment=exp_name))

    # --- final comparison tables ---
    metrics_df  = pd.DataFrame(all_metrics_rows)
    class_f1_df = (pd.concat(all_class_f1_parts, ignore_index=True)
                   if all_class_f1_parts else pd.DataFrame())

    metrics_df.to_csv (result_dir / "all_experiment_metric_comparison.csv",         index=False)
    class_f1_df.to_csv(result_dir / "all_experiment_per_species_f1_comparison.csv", index=False)

    with open(result_dir / "all_experiment_metric_comparison.json", "w") as f:
        json.dump(metrics_df.to_dict("records"), f, indent=2, default=str)
    with open(result_dir / "all_experiment_per_species_f1_comparison.json", "w") as f:
        json.dump(class_f1_df.to_dict("records"), f, indent=2, default=str)

    print("\n=== All experiments complete ===")
    print(metrics_df.to_string(index=False))

    return {"metrics": metrics_df, "per_class": class_f1_df}