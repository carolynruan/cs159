"""
augment_with_mask.py — Experiment 3: ALIA augmentation WITH SAM2 masks
=======================================================================
For ALL 460 species in the bottom-20th-percentile (lila_bottom20_targets.csv):
  - Downloads source images directly from train_df gs_filepath column
  - Three-tier masking fallback (MegaDetector → SAM2 box → SAM2 center-point → full-image)
  - Generates 16 augmented images per species (4 per attribute)
  - Labels output as: <species_slug>_<N>_with_mask.jpg
  - Auto-syncs to Google Drive

Run modes (in Colab cells):
  import augment_with_mask as wm
  wm.run_test(train_df, "/content/drive/MyDrive/augmented_outputs/with_mask", "lila_bottom20_targets.csv")
  wm.run_full(train_df, "/content/drive/MyDrive/augmented_outputs/with_mask", "lila_bottom20_targets.csv")

Install:
  !pip install megadetector
  !pip install git+https://github.com/facebookresearch/sam2.git
  !pip install diffusers transformers accelerate torch torchvision pillow matplotlib tqdm
"""

import os, re, csv, shutil, json
from pathlib import Path
from difflib import get_close_matches

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from PIL import Image, ImageFilter
from tqdm.notebook import tqdm
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline

# ── Config ────────────────────────────────────────────────────────────────────
SD_INPAINT_MODEL  = "runwayml/stable-diffusion-inpainting"
SD_IMG2IMG_MODEL  = "runwayml/stable-diffusion-v1-5"
EDIT_STRENGTH     = 0.75
GUIDANCE_SCALE    = 7.5
SD_STEPS          = 30
MD_CONF           = 0.10
BBOX_MAX_COV      = 0.85
MASK_MIN_COV      = 0.001
MASK_MAX_COV      = 0.95
IMAGES_PER_ATTR   = 4
LOCAL_OUT         = "/content/augmented_outputs/with_mask"
MASK_CACHE_DIR    = "/content/augmented_outputs/masks"
LILA_CACHE_DIR    = "/content/data/lila_images"

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

os.makedirs(MASK_CACHE_DIR, exist_ok=True)


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
            common = row["common_name"].strip()
            genus  = row.get("genus","").strip()
            sp     = row.get("species","").strip()
            slug   = clean_slug(common)
            if slug in seen:
                seen[slug] += 1; slug = f"{slug}_{seen[slug]}"
            else:
                seen[slug] = 0
            query = f"{genus} {sp}".strip().lower() if genus else common.lower()
            species.append({"slug": slug, "common": common, "query": query})
    return species


# ══════════════════════════════════════════════════════════════════════════════
# Source image: download from train_df or use cache
# ══════════════════════════════════════════════════════════════════════════════

def get_source_image(slug: str, common: str, train_df, n_images: int = 3) -> list[str]:
    cache_dir = Path(LILA_CACHE_DIR) / slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(list(cache_dir.glob("*.jpg")) + list(cache_dir.glob("*.JPG"))
                      + list(cache_dir.glob("*.jpeg")) + list(cache_dir.glob("*.png")))
    if existing:
        return [str(p) for p in existing]

    if train_df is None:
        return []

    common_lower = common.lower().strip()

    # --------------------------------------------------
    # 1. Exact match
    # --------------------------------------------------
    mask = (
        train_df["common_name"]
        .astype(str)
        .str.lower()
        .str.strip()
        == common_lower
    )

    rows = train_df[mask].head(n_images * 3)

    # --------------------------------------------------
    # 2. Partial-word match
    # --------------------------------------------------
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
                    print(f"  Partial match: '{common}' -> '{word}'")
                break

    # --------------------------------------------------
    # 3. Fuzzy match
    # --------------------------------------------------
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
                f"  Fuzzy match: '{common}' -> '{best_match}' "
                f"({len(rows)} images)"
            )

    # --------------------------------------------------
    # Give up
    # --------------------------------------------------
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
            saved.append(out_path); continue
        ret = os.system(f"gsutil -q cp '{gs_path}' '{out_path}' 2>/dev/null")
        if ret == 0 and os.path.exists(out_path):
            try:
                Image.open(out_path).verify(); saved.append(out_path)
            except:
                os.remove(out_path)

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# Model loading (lazy)
# ══════════════════════════════════════════════════════════════════════════════

_md = _sam2 = _sam2_device = _pipe_inpaint = _pipe_img2img = None

def _load_detection():
    global _md, _sam2, _sam2_device
    if _md is None:
        from megadetector.detection import run_detector as md_run
        print("  Loading MegaDetector...")
        _md = md_run.load_detector("MDV5A")
    if _sam2 is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _sam2_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading SAM2 on {_sam2_device}...")
        _sam2 = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        _sam2.model.to(_sam2_device)

def _load_inpaint():
    global _pipe_inpaint
    if _pipe_inpaint is None:
        d = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading SD inpainting on {d}...")
        _pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL,
            torch_dtype=torch.float16 if d=="cuda" else torch.float32).to(d)
        _pipe_inpaint.safety_checker = None
    return _pipe_inpaint

def _load_img2img():
    global _pipe_img2img
    if _pipe_img2img is None:
        d = "cuda" if torch.cuda.is_available() else "cpu"
        _pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_IMG2IMG_MODEL,
            torch_dtype=torch.float16 if d=="cuda" else torch.float32).to(d)
        _pipe_img2img.safety_checker = None
    return _pipe_img2img


# ══════════════════════════════════════════════════════════════════════════════
# Three-tier masking
# ══════════════════════════════════════════════════════════════════════════════

def get_mask(src_path: str) -> tuple:
    """
    Tier 1: MegaDetector → SAM2 box prompt  (best quality)
    Tier 2: SAM2 center-point prompt         (MD missed animal)
    Tier 3: (None, None, 'none')             → caller falls back to full-image
    Returns (anim_mask_path, bg_mask_path, tier)
    """
    _load_detection()
    stem   = Path(src_path).stem
    anim_p = os.path.join(MASK_CACHE_DIR, f"{stem}_animal.png")
    bg_p   = os.path.join(MASK_CACHE_DIR, f"{stem}_bg.png")
    tier_p = os.path.join(MASK_CACHE_DIR, f"{stem}_tier.txt")

    if os.path.exists(anim_p) and os.path.exists(bg_p):
        tier = open(tier_p).read().strip() if os.path.exists(tier_p) else "cached"
        return anim_p, bg_p, tier

    try:
        pil  = Image.open(src_path).convert("RGB")
        arr  = np.array(pil); h, w = arr.shape[:2]
        masks = scores = None; tier = "none"

        # Tier 1: MegaDetector
        try:
            dets    = _md.generate_detections_one_image(pil)["detections"]
            animals = [d for d in dets if d["conf"] >= MD_CONF and d["category"]=="1"]
            if animals:
                best = max(animals, key=lambda d: d["conf"])
                bx,by,bw_,bh_ = best["bbox"]
                if bw_*bh_ <= BBOX_MAX_COV:
                    bbox = np.array([int(bx*w),int(by*h),
                                     int((bx+bw_)*w),int((by+bh_)*h)])
                    with torch.inference_mode():
                        _sam2.set_image(arr)
                        masks, scores, _ = _sam2.predict(
                            box=bbox[None,:], multimask_output=True)
                    tier = "megadetector"
        except Exception as e:
            print(f"    Tier 1 error: {e}")

        # Tier 2: center-point
        if masks is None:
            try:
                cx, cy = w//2, h//2
                with torch.inference_mode():
                    _sam2.set_image(arr)
                    masks, scores, _ = _sam2.predict(
                        point_coords=np.array([[cx,cy]]),
                        point_labels=np.array([1]),
                        multimask_output=True)
                tier = "centerpoint"
            except Exception as e:
                print(f"    Tier 2 error: {e}"); return None, None, "none"

        best_m   = masks[scores.argmax()].astype(np.uint8)
        coverage = best_m.sum() / (h * w)
        if not (MASK_MIN_COV < coverage < MASK_MAX_COV):
            print(f"    Coverage {coverage:.3%} out of range")
            return None, None, "none"

        Image.fromarray(best_m*255).filter(ImageFilter.GaussianBlur(3)).save(anim_p)
        Image.fromarray((1-best_m)*255).filter(ImageFilter.GaussianBlur(3)).save(bg_p)
        open(tier_p,"w").write(tier)
        print(f"    Mask [{tier}]: coverage={coverage:.2%}")
        return anim_p, bg_p, tier

    except Exception as e:
        print(f"    Masking error: {e}"); return None, None, "none"


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation
# ══════════════════════════════════════════════════════════════════════════════

def augment_masked(src, anim_p, bg_p, prompt, out_path) -> bool:
    try:
        pipe = _load_inpaint()
        orig = Image.open(src).convert("RGB").resize((512,512))
        bg   = Image.open(bg_p).convert("L").resize((512,512))
        res  = pipe(prompt=prompt, image=orig, mask_image=bg,
                    strength=EDIT_STRENGTH, guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=SD_STEPS).images[0]
        anim = Image.open(anim_p).convert("L").resize((512,512))
        res.paste(orig, mask=anim)
        res.save(out_path); return True
    except Exception as e:
        print(f"    Inpaint error: {e}"); return False

def augment_fullimage(src, prompt, out_path) -> bool:
    """Tier-3 fallback: full-image edit (same as no-mask)."""
    try:
        pipe = _load_img2img()
        orig = Image.open(src).convert("RGB").resize((512,512))
        res  = pipe(prompt=prompt, image=orig,
                    strength=EDIT_STRENGTH, guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=SD_STEPS).images[0]
        res.save(out_path); return True
    except Exception as e:
        print(f"    Fallback error: {e}"); return False

def copy_to_drive(src, drive_dir, slug):
    sp_drive = os.path.join(drive_dir, slug)
    os.makedirs(sp_drive, exist_ok=True)
    shutil.copy2(src, os.path.join(sp_drive, Path(src).name))


# ══════════════════════════════════════════════════════════════════════════════
# Core loop
# ══════════════════════════════════════════════════════════════════════════════

def run_species(sp_info: dict, train_df, drive_dir: str, n_per_attr: int) -> tuple:
    """Returns (results_dict, tier_used). Guarantees images for every species."""
    slug    = sp_info["slug"]
    out_dir = Path(LOCAL_OUT) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    srcs = get_source_image(slug, sp_info["common"], train_df)
    if not srcs:
        print(f"  [{slug}] No source images — skip")
        return {}, "skipped"

    # Try all source images until mask succeeds
    anim_p = bg_p = None; tier = "none"; src = srcs[0]
    for candidate in srcs:
        anim_p, bg_p, tier = get_mask(candidate)
        if anim_p:
            src = candidate; break

    if tier == "none":
        print(f"  [{slug}] No mask — using full-image fallback")

    results = {}
    counter = 1

    for attr in ATTRIBUTE_ORDER:
        results[attr] = []
        for k in range(n_per_attr):
            prompt   = ATTRIBUTE_PROMPTS[attr][k%4].format(animal=sp_info["query"])
            out_name = f"{slug}_{counter}_with_mask.jpg"
            out_path = str(out_dir / out_name)

            if not os.path.exists(out_path):
                if anim_p:
                    ok = augment_masked(src, anim_p, bg_p, prompt, out_path)
                else:
                    ok = augment_fullimage(src, prompt, out_path)
                if not ok:
                    counter += 1; results[attr].append(""); continue

            results[attr].append(out_path)
            if drive_dir:
                copy_to_drive(out_path, drive_dir, slug)
            counter += 1

    return results, tier


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def figure_pipeline(src, anim_p, bg_p, aug_results, sp_info, tier, out_path):
    """Pipeline figure: original → mask stages → augmented results per attribute."""
    n_rows = 1 + len(ATTRIBUTE_ORDER)
    fig    = plt.figure(figsize=(10.5, 3.8 * n_rows))
    gs     = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.45, wspace=0.08)
    orig   = np.array(Image.open(src).convert("RGB").resize((512,512)))

    # Row 0: pipeline stages
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(orig); ax.set_title(f"Original\n{sp_info['common']}", fontsize=9, fontweight="bold"); ax.axis("off")

    ax = fig.add_subplot(gs[0,1])
    ax.imshow(orig)
    if anim_p and os.path.exists(anim_p):
        ax.imshow(np.array(Image.open(anim_p).convert("L").resize((512,512))), alpha=0.5, cmap="Reds")
        ax.legend(handles=[mpatches.Patch(color="#CC4444",alpha=0.6,label="Animal (protected)")],
                  fontsize=7, loc="lower right")
    ax.set_title(f"SAM2 mask ({tier})\nred = protected", fontsize=9, fontweight="bold"); ax.axis("off")

    ax = fig.add_subplot(gs[0,2])
    ax.imshow(orig)
    if bg_p and os.path.exists(bg_p):
        ax.imshow(np.array(Image.open(bg_p).convert("L").resize((512,512))), alpha=0.5, cmap="Blues")
        ax.legend(handles=[mpatches.Patch(color="#4466CC",alpha=0.6,label="Background (repainted)")],
                  fontsize=7, loc="lower right")
    ax.set_title("Background mask\nblue = SD inpaints here", fontsize=9, fontweight="bold"); ax.axis("off")

    for r, attr in enumerate(ATTRIBUTE_ORDER):
        variants = aug_results.get(attr, [])
        aug_path = variants[0] if variants and variants[0] and os.path.exists(variants[0]) else None

        ax = fig.add_subplot(gs[r+1,0])
        if aug_path: ax.imshow(Image.open(aug_path).convert("RGB"))
        else: ax.set_facecolor("#eee")
        pex = ATTRIBUTE_PROMPTS[attr][0].format(animal=sp_info["query"])
        ax.set_title(f'{attr.upper()}: "{pex[:45]}..."', fontsize=7,
                     color=ATTR_COLORS[attr], fontweight="bold"); ax.axis("off")

        ax = fig.add_subplot(gs[r+1,1])
        if aug_path:
            aug_r = np.array(Image.open(aug_path).convert("RGB").resize((512,512)))
            diff  = np.abs(orig.astype(int)-aug_r.astype(int)).mean(axis=2)
            im    = ax.imshow(diff, cmap="hot", vmin=0, vmax=80)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Pixel change heatmap", fontsize=7)
        else: ax.set_facecolor("#eee")
        ax.axis("off")

        ax = fig.add_subplot(gs[r+1,2])
        if aug_path:
            ax.imshow(Image.open(aug_path).convert("RGB"))
            if anim_p and os.path.exists(anim_p):
                m = np.array(Image.open(anim_p).convert("L").resize((512,512)))
                ax.contour(m, levels=[127], colors=["lime"], linewidths=[2.0])
            ax.set_title("Result + SAM2 boundary\n(green = animal edge)", fontsize=7)
        else: ax.set_facecolor("#eee")
        ax.axis("off")

    fig.suptitle(f"ALIA Augmentation (With SAM2 Mask) — {sp_info['common']}\nMask tier: {tier}",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Figure: {out_path}")


def figure_summary(all_species, fig_dir):
    done = []
    for sp in all_species:
        srcs = get_source_image(sp["slug"], sp["common"], None)
        augs = sorted(Path(LOCAL_OUT).glob(f"{sp['slug']}/*.jpg")) if \
               (Path(LOCAL_OUT)/sp["slug"]).exists() else []
        if srcs and augs: done.append((sp, srcs[0], str(augs[0])))
    done = done[:60]
    if not done: return
    fig, axes = plt.subplots(len(done), 2, figsize=(7, 3.2*len(done)))
    if len(done) == 1: axes = axes[None,:]
    for i, (sp, src, aug) in enumerate(done):
        axes[i,0].imshow(Image.open(src).convert("RGB"))
        axes[i,0].set_title(sp["common"], fontsize=6, fontweight="bold")
        axes[i,0].set_ylabel("original", fontsize=5); axes[i,0].axis("off")
        axes[i,1].imshow(Image.open(aug).convert("RGB"))
        axes[i,1].set_title("augmented (with mask)", fontsize=6); axes[i,1].axis("off")
    fig.suptitle("Bottom-20th Percentile — With-mask Augmentation", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(fig_dir, "all_species_summary_with_mask.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Summary figure: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════════════════

def run_test(train_df, drive_dir: str, csv_path: str, n_per_attr: int = 2):
    print("\n" + "="*60)
    print("TEST MODE — With-mask augmentation (1 species)")
    print("="*60)

    all_species = load_species(csv_path)
    _load_detection()
    test_sp = None
    for sp in all_species:
        srcs = get_source_image(sp["slug"], sp["common"], train_df)
        if srcs:
            test_sp = sp; break

    if not test_sp:
        print("  No source images found."); return

    print(f"  Testing on: {test_sp['common']}")
    results, tier = run_species(test_sp, train_df, drive_dir, n_per_attr)

    if results:
        fig_dir = Path("augmented_outputs/figures/with_mask")
        fig_dir.mkdir(parents=True, exist_ok=True)
        srcs   = get_source_image(test_sp["slug"], test_sp["common"], None)
        src    = srcs[0]
        anim_p, bg_p, _ = get_mask(src)
        figure_pipeline(src, anim_p, bg_p, results, test_sp, tier,
                        str(fig_dir / f"{test_sp['slug']}_pipeline.png"))

        from IPython.display import Image as IPImage, display
        display(IPImage(str(fig_dir / f"{test_sp['slug']}_pipeline.png")))

    print("\n  Test complete.")


def run_full(train_df, drive_dir: str, csv_path: str):
    print("\n" + "="*60)
    print("FULL RUN — With-mask augmentation (all species)")
    print("="*60)

    all_species = load_species(csv_path)
    print(f"  {len(all_species)} species to process")
    _load_detection()

    fig_dir  = Path("augmented_outputs/figures/with_mask")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tier_log = {}; done = []; skipped = []

    for sp in tqdm(all_species, desc="Species"):
        results, tier = run_species(sp, train_df, drive_dir, IMAGES_PER_ATTR)
        tier_log[sp["slug"]] = tier

        if results:
            done.append(sp["slug"])
            srcs = get_source_image(sp["slug"], sp["common"], None)
            if srcs:
                src = srcs[0]
                anim_p, bg_p, _ = get_mask(src)
                figure_pipeline(src, anim_p, bg_p, results, sp, tier,
                                str(fig_dir / f"{sp['slug']}_pipeline.png"))
        else:
            skipped.append(sp["slug"])

    figure_summary(all_species, str(fig_dir))

    with open("augmented_outputs/with_mask_tier_log.json","w") as f:
        json.dump(tier_log, f, indent=2)
    if skipped:
        with open("augmented_outputs/with_mask_skipped.txt","w") as f:
            f.write("\n".join(skipped))

    from collections import Counter
    tc = Counter(tier_log.values())
    print(f"\n  Done: {len(done)}/{len(all_species)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Tier breakdown: {dict(tc)}")
