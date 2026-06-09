"""
augment_no_mask.py — Experiment 2: ALIA augmentation WITHOUT segmentation masks
================================================================================
For ALL 460 species in the bottom-20th-percentile (lila_bottom20_targets.csv):
  - Downloads source images directly from train_df gs_filepath column
  - Generates 16 augmented images per species (4 per attribute)
  - Attributes: background / weather / lighting / season
  - Labels output as: <species_slug>_<N>_no_mask.jpg
  - Auto-syncs to Google Drive

Run modes (in Colab cells, not !python):
  import augment_no_mask as nm
  nm.run_test(train_df, "/content/drive/MyDrive/augmented_outputs/no_mask", "lila_bottom20_targets.csv")
  nm.run_full(train_df, "/content/drive/MyDrive/augmented_outputs/no_mask", "lila_bottom20_targets.csv")

Install:
  !pip install diffusers transformers accelerate torch torchvision pillow matplotlib tqdm
"""

import os, re, csv, shutil
from pathlib import Path

from difflib import get_close_matches
import glob 
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm.notebook import tqdm
from diffusers import StableDiffusionImg2ImgPipeline

# ── Config ────────────────────────────────────────────────────────────────────
SD_MODEL        = "runwayml/stable-diffusion-v1-5"
EDIT_STRENGTH   = 0.75
GUIDANCE_SCALE  = 7.5
SD_STEPS        = 30
IMAGES_PER_ATTR = 4
LOCAL_OUT       = "/content/augmented_outputs/no_mask"
LILA_CACHE_DIR  = "/content/data/lila_images"   # local cache for downloaded source images

# ── Attribute prompts ─────────────────────────────────────────────────────────
ATTRIBUTE_PROMPTS = {
    "background": [
        "a camera trap photo of a {animal} in dense tropical rainforest undergrowth",
        "a camera trap photo of a {animal} in open dry savanna grassland",
        "a camera trap photo of a {animal} beside a rocky mountain stream",
        "a camera trap photo of a {animal} in a sparse woodland with fallen leaves",
    ],
    "weather": [
        "a camera trap photo of a {animal} in heavy rain, wet foliage",
        "a camera trap photo of a {animal} in light misty fog, overcast sky",
        "a camera trap photo of a {animal} on a clear sunny day, bright light",
        "a camera trap photo of a {animal} during a light snowfall",
    ],
    "lighting": [
        "a camera trap photo of a {animal} at golden hour, warm orange light",
        "a camera trap photo of a {animal} at night, infrared camera trap glow",
        "a camera trap photo of a {animal} in deep shade, dappled light through trees",
        "a camera trap photo of a {animal} in harsh midday sunlight, high contrast",
    ],
    "season": [
        "a camera trap photo of a {animal} in lush green summer vegetation",
        "a camera trap photo of a {animal} surrounded by autumn orange and red leaves",
        "a camera trap photo of a {animal} in a snowy winter landscape",
        "a camera trap photo of a {animal} in early spring, bare branches and fresh buds",
    ],
}
ATTRIBUTE_ORDER = ["background", "weather", "lighting", "season"]
ATTR_COLORS     = {"background":"#4C8EDA", "weather":"#5BAD6F",
                   "lighting":"#E8A23A",   "season":"#C9527A"}


# ══════════════════════════════════════════════════════════════════════════════
# Species loading
# ══════════════════════════════════════════════════════════════════════════════

def clean_slug(common):
    slug = common.lower().replace("'","").replace("-"," ").replace("/"," ")
    return re.sub(r"[^a-z0-9 ]","",slug).strip().replace(" ","_")

def load_species(csv_path: str) -> list[dict]:
    species, seen = [], {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            common  = row["common_name"].strip()
            genus   = row.get("genus","").strip()
            sp      = row.get("species","").strip()
            slug    = clean_slug(common)
            if slug in seen:
                seen[slug] += 1; slug = f"{slug}_{seen[slug]}"
            else:
                seen[slug] = 0
            query = f"{genus} {sp}".strip().lower() if genus else common.lower()
            species.append({"slug": slug, "common": common, "query": query})
    return species


# ══════════════════════════════════════════════════════════════════════════════
# Source image: download from train_df gs_filepath or use cache
# ══════════════════════════════════════════════════════════════════════════════

def get_source_image(slug: str, common: str, train_df, n_images: int = 3) -> list[str]:
    """
    Returns list of local image paths for this species.
    1. Checks local cache first (LILA_CACHE_DIR/<slug>/)
    2. If not cached, downloads from train_df gs_filepath using gsutil
    """
    cache_dir = Path(LILA_CACHE_DIR) / slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    existing = sorted(list(cache_dir.glob("*.jpg")) + list(cache_dir.glob("*.JPG"))
                      + list(cache_dir.glob("*.jpeg")) + list(cache_dir.glob("*.png")))
    if existing:
        return [str(p) for p in existing]

    if train_df is None:
        return []

    # Match rows in train_df by common_name
    common_lower = common.lower().strip()

    # 1. Exact match
    mask = (
        train_df["common_name"]
        .astype(str)
        .str.lower()
        .str.strip()
        == common_lower
    )
    rows = train_df[mask].head(n_images * 3)

    # 2. Partial-word fallback
    if len(rows) == 0:
        for word in common_lower.split():
            if len(word) > 5:
                mask = (
                    train_df["common_name"]
                    .astype(str)
                    .str.lower()
                    .str.contains(word, na=False)
                )
                rows = train_df[mask].head(n_images * 3)

                if len(rows) > 0:
                    print(f"  Partial match: '{common}' → '{word}'")
                    break

    # 3. Fuzzy match fallback
    if len(rows) == 0:

        candidate_names = (
            train_df["common_name"]
            .astype(str)
            .str.lower()
            .str.strip()
            .dropna()
            .unique()
        )

        matches = get_close_matches(
            common_lower,
            candidate_names,
            n=1,
            cutoff=0.7
        )

        if matches:
            best_match = matches[0]

            mask = (
                train_df["common_name"]
                .astype(str)
                .str.lower()
                .str.strip()
                == best_match
            )

            rows = train_df[mask].head(n_images * 3)

            print(
                f"  Fuzzy match: '{common}' → '{best_match}' "
                f"({len(rows)} images)"
            )
    

    if len(rows) == 0:
        return []

    saved = []
    for i, (_, row) in enumerate(rows.iterrows()):
        if len(saved) >= n_images:
            break
        gs_path  = row["gs_filepath"]
        ext      = Path(gs_path).suffix.lower() or ".jpg"
        out_path = str(cache_dir / f"{slug}_{i+1:03d}{ext}")
        if os.path.exists(out_path):
            saved.append(out_path)
            continue
        ret = os.system(f"gsutil -q cp '{gs_path}' '{out_path}' 2>/dev/null")
        if ret == 0 and os.path.exists(out_path):
            try:
                Image.open(out_path).verify()
                saved.append(out_path)
            except:
                os.remove(out_path)

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

_pipe = None

def load_pipe():
    global _pipe
    if _pipe is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading SD Img2Img on {device}...")
        _pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        _pipe.safety_checker = None
    return _pipe


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation
# ══════════════════════════════════════════════════════════════════════════════

def augment_no_mask(src: str, prompt: str, out_path: str) -> bool:
    try:
        pipe   = load_pipe()
        orig   = Image.open(src).convert("RGB").resize((512, 512))
        result = pipe(
            prompt=prompt, image=orig,
            strength=EDIT_STRENGTH, guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=SD_STEPS,
        ).images[0]
        result.save(out_path)
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False

def copy_to_drive(src: str, drive_dir: str, slug: str):
    sp_drive = os.path.join(drive_dir, slug)
    os.makedirs(sp_drive, exist_ok=True)
    shutil.copy2(src, os.path.join(sp_drive, Path(src).name))


# ══════════════════════════════════════════════════════════════════════════════
# Core loop
# ══════════════════════════════════════════════════════════════════════════════

def run_species(sp_info: dict, train_df, drive_dir: str, n_per_attr: int) -> dict:
    slug    = sp_info["slug"]
    out_dir = Path(LOCAL_OUT) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    srcs = get_source_image(slug, sp_info["common"], train_df)
    if not srcs:
        print(f"  [{slug}] No source images — skip")
        return {}

    src     = srcs[0]
    results = {}
    counter = 1

    for attr in ATTRIBUTE_ORDER:
        results[attr] = []
        for k in range(n_per_attr):
            prompt   = ATTRIBUTE_PROMPTS[attr][k % 4].format(animal=sp_info["query"])
            out_name = f"{slug}_{counter}_no_mask.jpg"
            out_path = str(out_dir / out_name)

            if not os.path.exists(out_path):
                ok = augment_no_mask(src, prompt, out_path)
                if not ok:
                    counter += 1
                    results[attr].append("")
                    continue

            results[attr].append(out_path)
            if drive_dir:
                copy_to_drive(out_path, drive_dir, slug)
            counter += 1

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def _numeric_sort(paths):
    import re
    def key(p):
        m = re.search(r"_(\d+)_no_mask", str(p))
        return int(m.group(1)) if m else 0
    return sorted(paths, key=key)

def load_results_from_disk(slug: str) -> dict:
    all_imgs = _numeric_sort(glob.glob(f"{LOCAL_OUT}/{slug}/*.jpg"))
    results = {}
    for i, attr in enumerate(ATTRIBUTE_ORDER):
        results[attr] = all_imgs[i*4:(i+1)*4]
    return results

def figure_grid(src: str, aug_results: dict, sp_info: dict, out_path: str):
    """Full 4×4 grid: original on top, 4 rows × 4 augmentation variants."""
    n_cols = 4
    n_rows = 1 + len(ATTRIBUTE_ORDER)
    fig    = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.5))
    gs     = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.06)

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(Image.open(src).convert("RGB"))
    ax.set_title(
        f"Original — {sp_info['common']}  (LILA camera trap)\n"
        "Exp 2: No segmentation mask — full image edited by diffusion",
        fontsize=10, fontweight="bold")
    ax.axis("off")

    for r, attr in enumerate(ATTRIBUTE_ORDER):
        variants = aug_results.get(attr, [])
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r + 1, c])
            if c < len(variants) and variants[c] and os.path.exists(variants[c]):
                ax.imshow(Image.open(variants[c]).convert("RGB"))
            else:
                ax.set_facecolor("#eee")
            prompt_snip = ATTRIBUTE_PROMPTS[attr][c].format(animal=sp_info["query"])
            wrapped     = "\n".join([prompt_snip[i:i+30] for i in range(0, len(prompt_snip), 30)])
            ax.set_title(wrapped, fontsize=5.5, pad=3)
            if c == 0:
                ax.set_ylabel(attr.upper(), fontsize=9, fontweight="bold",
                              color=ATTR_COLORS[attr], rotation=90, labelpad=8)
            ax.axis("off")

    legend_handles = [plt.Rectangle((0,0),1,1,color=c,label=a.capitalize())
                      for a,c in ATTR_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5,-0.01))
    fig.suptitle("ALIA Augmentation (No Mask) — Controlled Attribute Dimensions",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {out_path}")


def figure_summary(all_species: list, fig_dir: str):
    """One row per species: original | first augmentation."""
    done = []
    for sp in all_species:
        srcs = get_source_image(sp["slug"], sp["common"], None)
        augs = sorted(Path(LOCAL_OUT).glob(f"{sp['slug']}/*.jpg")) if \
               (Path(LOCAL_OUT) / sp["slug"]).exists() else []
        if srcs and augs:
            done.append((sp, srcs[0], str(augs[0])))
    done = done[:60]

    if not done:
        return

    fig, axes = plt.subplots(len(done), 2, figsize=(7, 3.2 * len(done)))
    if len(done) == 1:
        axes = axes[None, :]

    for i, (sp, src, aug) in enumerate(done):
        axes[i, 0].imshow(Image.open(src).convert("RGB"))
        axes[i, 0].set_title(sp["common"], fontsize=6, fontweight="bold")
        axes[i, 0].set_ylabel("original", fontsize=5)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(Image.open(aug).convert("RGB"))
        axes[i, 1].set_title("augmented (no mask)", fontsize=6)
        axes[i, 1].axis("off")

    fig.suptitle("Bottom-20th Percentile — No-mask Augmentation",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(fig_dir, "all_species_summary_no_mask.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Summary figure: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════════════════

def run_test(train_df, drive_dir: str, csv_path: str, n_per_attr: int = 2):
    """Test on first species that has source images. Generates figures."""
    print("\n" + "="*60)
    print("TEST MODE — No-mask augmentation (1 species)")
    print("="*60)

    all_species = load_species(csv_path)
    test_sp     = None
    for sp in all_species:
        srcs = get_source_image(sp["slug"], sp["common"], train_df)
        if srcs:
            test_sp = sp
            break

    if not test_sp:
        print("  No source images found. Check train_df and csv_path.")
        return

    print(f"  Testing on: {test_sp['common']} ({test_sp['slug']})")
    load_pipe()
    results = run_species(test_sp, train_df, drive_dir, n_per_attr)

    if results:
        fig_dir = Path("augmented_outputs/figures/no_mask")
        fig_dir.mkdir(parents=True, exist_ok=True)
        src = get_source_image(test_sp["slug"], test_sp["common"], None)[0]
        figure_grid(src, results, test_sp,
                    str(fig_dir / f"{test_sp['slug']}_grid.png"))

        from IPython.display import Image as IPImage, display
        display(IPImage(str(fig_dir / f"{test_sp['slug']}_grid.png")))

    print("\n  Test complete.")


def run_full(train_df, drive_dir: str, csv_path: str):
    """Full run: all 460 species, 16 images each."""
    print("\n" + "="*60)
    print("FULL RUN — No-mask augmentation (all species)")
    print("="*60)

    all_species = load_species(csv_path)
    print(f"  {len(all_species)} species to process")

    load_pipe()
    fig_dir = Path("augmented_outputs/figures/no_mask")
    fig_dir.mkdir(parents=True, exist_ok=True)

    done, skipped = [], []

    for sp in tqdm(all_species, desc="Species"):
        results = run_species(sp, train_df, drive_dir, IMAGES_PER_ATTR)
        if results:
            done.append(sp["slug"])
            srcs = get_source_image(sp["slug"], sp["common"], None)
            if srcs:
                disk_results = load_results_from_disk(sp["slug"])
                figure_grid(srcs[0], disk_results, sp,
                            str(fig_dir / f"{sp['slug']}_grid.png"))
        else:
            skipped.append(sp["slug"])

    figure_summary(all_species, str(fig_dir))

    if skipped:
        with open("augmented_outputs/no_mask_skipped.txt","w") as f:
            f.write("\n".join(skipped))

    print(f"\n  Done: {len(done)}/{len(all_species)}")
    print(f"  Skipped: {len(skipped)}")
    if skipped:
        print("  Skipped list → augmented_outputs/no_mask_skipped.txt")
