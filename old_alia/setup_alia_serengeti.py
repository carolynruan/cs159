"""Wire the Snapshot Serengeti dataset into a freshly-cloned ALIA repo.

Call ``setup(repo_root, alia_repo_path, df_csv_path)`` once per Colab session
before running any ALIA CLI scripts (caption.py, editing/*, main.py).

Usage in a notebook:

    import sys
    sys.path.insert(0, str(REPO_ROOT))
    import setup_alia_serengeti
    setup_alia_serengeti.setup(REPO_ROOT, ALIA_REPO_PATH, df_csv_path)
"""
from __future__ import annotations

import shutil
from pathlib import Path


_IMPORT_LINE = "from datasets.SnapshotSerengeti import SnapshotSerengeti\n"

_DATASET_BRANCH = """
    elif dataset_name in ('SnapshotSerengeti', 'SnapshotSerengetiExtra'):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(cfg.data.df_path)
        class_list = sorted(df['label'].unique().tolist())
        train_df, tmp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(tmp, test_size=0.5, random_state=42, stratify=tmp['label'])
        trainset  = SnapshotSerengeti(train_df,  class_list, transform=transform,     group=0)
        valset    = SnapshotSerengeti(val_df,    class_list, transform=val_transform, group=0)
        testset   = SnapshotSerengeti(test_df,   class_list, transform=val_transform, group=0)
        extraset  = SnapshotSerengeti(train_df,  class_list, transform=transform,     group=1)
        if dataset_name == 'SnapshotSerengetiExtra':
            trainset = CombinedDataset([trainset, extraset])
"""


def setup(
    repo_root: Path | str,
    alia_repo_path: Path | str,
    df_csv_path: Path | str | None = None,
) -> None:
    """Idempotently register SnapshotSerengeti in the ALIA repo.

    Parameters
    ----------
    repo_root:
        Root of the cs159 repo (contains ``datasets/snapshot_serengeti.py``).
    alia_repo_path:
        Root of the cloned ALIA repo.
    df_csv_path:
        Path to the CSV (columns: file_name, label) that ALIA config will
        reference. Pass ``None`` to skip writing the YAML config.
    """
    repo_root = Path(repo_root)
    alia = Path(alia_repo_path)

    # 1. Copy dataset class into ALIA/datasets/
    src = repo_root / "datasets" / "snapshot_serengeti.py"
    dst = alia / "datasets" / "SnapshotSerengeti.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[setup_alia] Copied dataset class → {dst}")

    # 2. Patch ALIA/helpers/load_dataset.py (idempotent)
    loader = alia / "helpers" / "load_dataset.py"
    if loader.exists():
        src_text = loader.read_text()
        changed = False
        if _IMPORT_LINE not in src_text:
            src_text = src_text.replace(
                "from datasets.cub import Cub2011\n",
                "from datasets.cub import Cub2011\n" + _IMPORT_LINE,
            )
            changed = True
        if "SnapshotSerengeti" not in src_text:
            src_text = src_text.replace(
                "    if embedding_root:",
                _DATASET_BRANCH + "    if embedding_root:",
            )
            changed = True
        if changed:
            loader.write_text(src_text)
            print(f"[setup_alia] Patched {loader}")
        else:
            print(f"[setup_alia] load_dataset.py already registered — no change.")
    else:
        print(f"[setup_alia] Warning: {loader} not found — skipping patch.")

    # 3. Write YAML config (if df_csv_path provided)
    if df_csv_path is not None:
        cfg_dir = alia / "configs" / "SnapshotSerengeti"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        yaml_content = f"""base_config: configs/SnapshotSerengeti/base.yaml
proj: ALIA-SnapshotSerengeti
name: SnapshotSerengeti
data:
  base_dataset: SnapshotSerengeti
  df_path: {df_csv_path}
  num_extra: 3
hps:
  lr: 0.001
"""
        (cfg_dir / "base.yaml").write_text(yaml_content)
        print(f"[setup_alia] Wrote config → {cfg_dir / 'base.yaml'}")

    print("[setup_alia] Done.")
