from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional in bare Python envs
    pd = None


ROOT = Path(__file__).resolve().parent


def load_lila_suppl_data(path: str | Path | None = None) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to use lila_pipeline_helpers")
    path = Path(path) if path is not None else ROOT / "lila-suppl-data.csv"
    df = pd.read_csv(path)
    if "common_name" not in df.columns or "count_train" not in df.columns:
        raise ValueError(f"Unexpected schema in {path}")
    return df


def select_bottom_percent_classes(
    df: pd.DataFrame,
    percentile: float = 5.0,
    exclude_common_names: Iterable[str] = ("Blank", "Human", "Vehicle"),
) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to use lila_pipeline_helpers")
    filtered = df.copy()
    if exclude_common_names:
        filtered = filtered[~filtered["common_name"].isin(set(exclude_common_names))]
    filtered = filtered[filtered["count_train"].notna()].copy()
    filtered = filtered[filtered["count_train"] > 0].copy()
    if filtered.empty:
        raise ValueError("No classes remain after filtering.")

    threshold = filtered["count_train"].quantile(percentile / 100.0)
    selected = filtered[filtered["count_train"] <= threshold].copy()
    selected = selected.sort_values(["count_train", "common_name"], ascending=[True, True]).reset_index(drop=True)
    selected["deficit_to_threshold"] = (threshold - selected["count_train"]).clip(lower=0).astype(int)
    selected["threshold_count"] = int(threshold)
    return selected


def build_class_targets(df: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to use lila_pipeline_helpers")
    out = df.copy()
    out["target_added"] = (out["threshold_count"] - out["count_train"]).clip(lower=0).astype(int)
    out["target_total_after_addition"] = out["count_train"] + out["target_added"]
    return out


def serialize_class_targets(
    targets: pd.DataFrame,
    out_dir: str | Path,
    stem: str = "lila_bottom5_targets",
) -> dict[str, Path]:
    if pd is None:
        raise ImportError("pandas is required to use lila_pipeline_helpers")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    targets.to_csv(csv_path, index=False)
    json_path.write_text(targets.to_json(orient="records", indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


def load_lila_train_split(path: str | Path | None = None) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to use lila_pipeline_helpers")
    path = Path(path) if path is not None else ROOT / "lila_splits" / "lila_splits_train.csv"
    return pd.read_csv(path)


def sample_class_images(train_df: pd.DataFrame, class_name: str, n: int, seed: int = 42) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to use lila_pipeline_helpers")
    class_df = train_df[train_df["common_name"] == class_name].copy()
    if class_df.empty:
        raise ValueError(f"No rows found for class {class_name!r}")
    n = min(n, len(class_df))
    return class_df.sample(n=n, random_state=seed).reset_index(drop=True)


def write_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    if pd is None:
        raise ImportError("pandas is required to use lila_pipeline_helpers")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def freeze_backbone(model) -> None:
    """Freeze all non-classifier parameters for SpeciesNet fine-tuning.

    This is intentionally conservative: any parameter whose name contains
    classifier/head/fc/logit remains trainable, everything else is frozen.
    """

    for name, param in model.named_parameters():
        trainable = any(token in name.lower() for token in ("classifier", "head", "fc", "logit"))
        param.requires_grad = trainable


def trainable_parameter_names(model) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]
