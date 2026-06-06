"""
Experiment 3: ALIA augmentation WITH SAM2 segmentation masks
=============================================================
Fine-tune SpeciesNet on:
  - X iNaturalist images (post-cutoff, Apr 2025+)
  - X augmented Snapshot Serengeti images (SD inpainting, background only)

Same two X-selection strategies as Experiment 2, but here:
  - MegaDetector finds the animal bounding box
  - SAM2 generates a precise animal mask
  - SD inpainting only repaints the background
  - Original animal pixels composited back on top

This isolates the effect of mask-constrained augmentation vs full-image editing.

Usage:
  python experiment3_with_masks.py --mode test     # smoke test, 3 images
  python experiment3_with_masks.py --mode budget   # Strategy A + B
  python experiment3_with_masks.py --mode curve    # incremental curve
"""

import os, json, math, csv, random, argparse, shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageFilter
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import (BlipProcessor, BlipForConditionalGeneration,
                          CLIPModel, CLIPProcessor)
from diffusers import StableDiffusionInpaintPipeline
from sam2.sam2_image_predictor import SAM2ImagePredictor
from megadetector.detection import run_detector as md_run_detector
import openai

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = "sk-..."
SD_MODEL          = "stabilityai/stable-diffusion-2-inpainting"
EDIT_STRENGTH     = 0.75
GUIDANCE_SCALE    = 7.5
SD_STEPS          = 30
MD_CONF_THRESHOLD = 0.20
BBOX_MAX_COVERAGE = 0.60
MASK_MIN_COV      = 0.02
MASK_MAX_COV      = 0.70
CURVE_ITERATIONS  = 10
CURVE_PER_STEP    = 1

SERENGETI_SPECIES_COUNTS = {
    "wilddog":1,"kudu":3,"bat":5,"pangolin":7,"hyenabrown":11,
    "steenbok":44,"zorilla":65,"lioncub":73,"cattle":99,"duiker":114,
    "civet":118,"genet":119,"rhinoceros":153,"honeybadger":162,
    "wildcat":193,"rodents":211,"caracal":248,"vulture":335,
    "reptiles":464,"hyenastriped":478,"bushbuck":551,"fire":568,
    "aardwolf":604,"leopard":650,"porcupine":796,"aardvark":981,
    "batearedfox":1142,"hare":1477,"mongoose":1437,"waterbuck":1386,
    "monkeyvervet":1897,"insectspider":2339,"serval":2499,"jackal":2820,
    "ostrich":4058,"koribustard":4105,"secretarybird":5355,"lionmale":5488,
    "hippopotamus":5851,"reedbuck":6327,"dikdik":6596,"cheetah":6806,
    "baboon":9013,"topi":12177,"eland":15839,"lionfemale":16605,
    "hyenaspotted":21998,"otherbird":32462,"guineafowl":33112,
    "warthog":43136,"impala":43503,"giraffe":44376,"gazellegrants":46465,
    "elephant":53607,"hartebeest":58780,"buffalo":61283,
    "gazellethomsons":323326,"zebra":352892,"wildebeest":533478,
}

DIRS = {
    "ss_images":   "data/serengeti/images",
    "ss_masks":    "data/exp3/masks",
    "inat_images": "data/images",
    "inat_meta":   "data/metadata",
    "captions":    "data/exp3/captions",
    "augmented":   "data/exp3/augmented",
    "train":       "data/exp3/train",
    "results":     "data/exp3/results",
    "figures":     "data/exp3/figures",
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Budget (identical logic to Exp 2)
# ══════════════════════════════════════════════════════════════════════════════

def compute_budget():
    counts = list(SERENGETI_SPECIES_COUNTS.values())
    x      = int(np.percentile(sorted(counts), 5))
    under  = {sp: c for sp, c in SERENGETI_SPECIES_COUNTS.items() if c < x}
    p      = len(under)
    y      = sum(x - c for c in under.values())
    eq     = math.ceil(y / p)
    budget = dict(x=x, p=p, y=y, equal_per_class=eq,
                  underrepresented=under,
                  strategy_a={sp: x - c for sp, c in under.items()},
                  strategy_b={sp: eq for sp in under})
    json.dump(budget, open("data/exp3_budget.json","w"), indent=2)
    print(f"\n{'='*60}\nBudget — Experiment 3 (with masks)\n{'='*60}")
    print(f"  x={x}  p={p}  y={y}  y/p={eq}")
    for sp, c in sorted(under.items(), key=lambda t:t[1]):
        print(f"  {sp:<20} {c:>4} images  →  A:+{x-c}  B:+{eq}")
    return budget


# ══════════════════════════════════════════════════════════════════════════════
# MegaDetector + SAM2 masking
# ══════════════════════════════════════════════════════════════════════════════

_md_model   = None
_sam2_model = None
_sam2_device = None

def _load_models():
    global _md_model, _sam2_model, _sam2_device
    if _md_model is None:
        print("  Loading MegaDetector...")
        _md_model = md_run_detector.load_detector("MDV5A")
    if _sam2_model is None:
        _sam2_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading SAM2 on {_sam2_device}...")
        _sam2_model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        _sam2_model.model.to(_sam2_device)


def generate_mask(img_path: str) -> tuple[str | None, str | None]:
    """
    Run MegaDetector → SAM2 on one image.
    Returns (animal_mask_path, bg_mask_path) or (None, None) on failure.
    """
    _load_models()
    stem       = Path(img_path).stem
    mask_path  = str(Path(DIRS["ss_masks"]) / f"{stem}.png")
    bg_path    = str(Path(DIRS["ss_masks"]) / f"{stem}_bg.png")

    if os.path.exists(mask_path):
        return mask_path, bg_path

    try:
        pil  = Image.open(img_path).convert("RGB")
        arr  = np.array(pil)
        h, w = arr.shape[:2]

        result  = _md_model.generate_detections_one_image(pil)
        animals = [d for d in result["detections"]
                   if d["conf"] >= MD_CONF_THRESHOLD and d["category"] == "1"]
        if not animals:
            return None, None

        best    = max(animals, key=lambda d: d["conf"])
        bx,by,bw_,bh_ = best["bbox"]
        if (bw_ * bh_) > BBOX_MAX_COVERAGE:
            return None, None

        bbox    = np.array([int(bx*w),int(by*h),int((bx+bw_)*w),int((by+bh_)*h)])
        with torch.inference_mode():
            _sam2_model.set_image(arr)
            masks, scores, _ = _sam2_model.predict(box=bbox[None,:],
                                                    multimask_output=True)
        best_m  = masks[scores.argmax()].astype(np.uint8)
        cov     = best_m.sum() / (h * w)
        if not (MASK_MIN_COV < cov < MASK_MAX_COV):
            return None, None

        # Animal mask
        m = Image.fromarray(best_m * 255).filter(ImageFilter.GaussianBlur(3))
        m.save(mask_path)
        # Background mask (inverted)
        bg = Image.fromarray((1 - best_m) * 255).filter(ImageFilter.GaussianBlur(3))
        bg.save(bg_path)
        return mask_path, bg_path

    except Exception as e:
        print(f"    Mask error {img_path}: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# BLIP + GPT-4 (same as Exp 2)
# ══════════════════════════════════════════════════════════════════════════════

def caption_images(paths, species):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model     = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large").to(device)
    captions  = []
    for p in tqdm(paths[:50], desc=f"  BLIP {species}"):
        cp = Path(DIRS["captions"]) / f"{Path(p).stem}.txt"
        if cp.exists():
            captions.append(cp.read_text().strip()); continue
        try:
            img    = Image.open(p).convert("RGB")
            inputs = processor(img, return_tensors="pt").to(device)
            out    = model.generate(**inputs, max_new_tokens=50)
            cap    = processor.decode(out[0], skip_special_tokens=True)
            cp.write_text(cap); captions.append(cap)
        except Exception as e:
            print(f"    caption error {p}: {e}")
    return captions


def get_domain_descriptions(species, captions):
    cache = Path(DIRS["captions"]) / f"{species}_domains.json"
    if cache.exists():
        return json.load(open(cache))
    openai.api_key = OPENAI_API_KEY
    prefix = f"a camera trap photo of a {species.replace('_',' ')}"
    joined = "\n".join(f"- {c}" for c in captions[:100])
    r1 = openai.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":
        f"Summarize these camera trap captions for {species} into scene descriptions "
        f"of the form \"{prefix}...\":\n{joined}"}])
    r2 = openai.chat.completions.create(model="gpt-4", messages=[
        {"role":"user","content": r1.choices[0].message.content},
        {"role":"user","content":
            f"Output fewer than 5 captions, one per line, unique settings. No numbering."}])
    descs = [l.strip() for l in r2.choices[0].message.content.splitlines() if l.strip()][:5]
    json.dump(descs, open(cache,"w"), indent=2)
    return descs


# ══════════════════════════════════════════════════════════════════════════════
# SD inpainting WITH mask
# ══════════════════════════════════════════════════════════════════════════════

def load_inpaint_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = StableDiffusionInpaintPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device)
    pipe.safety_checker = None
    return pipe


def augment_image_with_mask(pipe, src_path, animal_mask_path,
                             bg_mask_path, prompt, out_path):
    """
    Inpaint background only; composite original animal back on top.
    Returns augmented PIL image or None if mask unavailable.
    """
    if not bg_mask_path or not os.path.exists(bg_mask_path):
        return None

    original  = Image.open(src_path).convert("RGB").resize((512,512))
    bg_mask   = Image.open(bg_mask_path).convert("L").resize((512,512))
    result    = pipe(prompt=prompt, image=original, mask_image=bg_mask,
                     strength=EDIT_STRENGTH, guidance_scale=GUIDANCE_SCALE,
                     num_inference_steps=SD_STEPS).images[0]

    # Composite: original animal over generated background
    anim_mask = Image.open(animal_mask_path).convert("L").resize((512,512))
    result.paste(original, mask=anim_mask)
    result.save(out_path)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLIP filter
# ══════════════════════════════════════════════════════════════════════════════

def clip_filter(aug_dir):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts   = ["a photo of a wild animal","a photo of an object",
                 "a photo of a scene","a photo","an image"]
    removed   = 0
    for p in Path(aug_dir).rglob("*.jpg"):
        try:
            img    = Image.open(p).convert("RGB")
            inputs = processor(text=prompts,images=img,return_tensors="pt",
                               padding=True).to(device)
            probs  = model(**inputs).logits_per_image.softmax(dim=1)[0]
            if probs.argmax().item() != 0:
                p.unlink(); removed += 1
        except: pass
    return removed


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_weighted_f1(test_items):
    import speciesnet as snet_module
    snet   = snet_module.SpeciesNet()
    y_true, y_pred = [], []
    for item in tqdm(test_items, desc="  Evaluating"):
        try:
            preds = snet.predict(image_paths=[item["path"]])
            top1  = preds[0]["predictions"][0]["taxon"] if preds else "none"
        except:
            top1 = "none"
        y_true.append(item["species"])
        y_pred.append(top1)
    classes     = sorted(set(y_true))
    weighted_f1 = f1_score(y_true, y_pred, labels=classes,
                           average="weighted", zero_division=0)
    per_class   = f1_score(y_true, y_pred, labels=classes,
                           average=None, zero_division=0)
    return {"weighted_f1": weighted_f1,
            "per_class_f1": dict(zip(classes, per_class)), "n": len(y_true)}


def _load_test_items():
    items = []
    for mf in Path(DIRS["inat_meta"]).glob("*.json"):
        meta = json.load(open(mf))
        fp   = os.path.join(DIRS["inat_images"], meta.get("filename",""))
        if os.path.exists(fp):
            items.append({"path": fp, "species": meta.get("species","unknown")})
    return items


# ══════════════════════════════════════════════════════════════════════════════
# Figures (Exp 3 specific — shows mask pipeline stages)
# ══════════════════════════════════════════════════════════════════════════════

def figure_mask_augmentation_grid(src_path, mask_path, bg_mask_path,
                                   aug_paths, domain_descs, species, out_path):
    """
    Shows the full mask pipeline for one image:
    Row 0: Original | Animal mask overlay | Background mask overlay
    Row 1+: Augmented result per domain description (with mask boundary shown)
    """
    n_augs = len(aug_paths)
    fig    = plt.figure(figsize=(14, 4 * (1 + n_augs)))
    gs     = gridspec.GridSpec(1 + n_augs, 3, figure=fig,
                               hspace=0.35, wspace=0.08)

    orig     = np.array(Image.open(src_path).convert("RGB"))
    anim_m   = np.array(Image.open(mask_path).convert("L"))  if mask_path   else None
    bg_m     = np.array(Image.open(bg_mask_path).convert("L")) if bg_mask_path else None

    # Row 0 col 0: original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(orig); ax.set_title("Original", fontsize=9, fontweight="bold")
    ax.axis("off")

    # Row 0 col 1: animal mask overlay (red = protected)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(orig)
    if anim_m is not None:
        ax.imshow(anim_m, alpha=0.45, cmap="Reds")
    ax.set_title("Animal mask\n(red = protected, not repainted)", fontsize=8)
    ax.axis("off")

    # Row 0 col 2: background mask overlay (blue = repainted)
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(orig)
    if bg_m is not None:
        ax.imshow(bg_m, alpha=0.45, cmap="Blues")
    ax.set_title("Background mask\n(blue = SD inpaints here)", fontsize=8)
    ax.axis("off")

    # Rows 1+: augmented results
    for i, (aug_path, desc) in enumerate(zip(aug_paths, domain_descs)):
        aug = np.array(Image.open(aug_path).convert("RGB"))

        # Col 0: augmented result
        ax = fig.add_subplot(gs[i+1, 0])
        ax.imshow(aug)
        wrapped = "\n".join([desc[j:j+35] for j in range(0,len(desc),35)])
        ax.set_title(f""{wrapped}"", fontsize=7)
        ax.axis("off")

        # Col 1: diff heatmap (shows what changed)
        ax = fig.add_subplot(gs[i+1, 1])
        orig_r  = np.array(Image.open(src_path).convert("RGB").resize((512,512)))
        aug_r   = np.array(Image.open(aug_path).convert("RGB").resize((512,512)))
        diff    = np.abs(orig_r.astype(int) - aug_r.astype(int)).mean(axis=2)
        im      = ax.imshow(diff, cmap="hot", vmin=0, vmax=80)
        ax.set_title("Pixel change heatmap\n(bright = more changed)", fontsize=7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Col 2: augmented with mask boundary drawn
        ax = fig.add_subplot(gs[i+1, 2])
        ax.imshow(aug)
        if anim_m is not None:
            # Draw mask contour
            from matplotlib.contour import QuadContourSet
            ax.contour(np.array(Image.open(mask_path).convert("L").resize((512,512))),
                       levels=[127], colors=["lime"], linewidths=[1.5])
        ax.set_title("Result + mask boundary\n(green = animal boundary)", fontsize=7)
        ax.axis("off")

    fig.suptitle(f"Experiment 3 — ALIA with SAM2 Mask: {species}",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved mask pipeline grid: {out_path}")


def figure_exp2_vs_exp3_comparison(orig_path, aug_no_mask_path,
                                    aug_with_mask_path, desc, species, out_path):
    """
    Side-by-side: original | Exp2 (no mask) | Exp3 (with mask)
    Shows the key difference between the two experiments.
    """
    orig    = Image.open(orig_path).convert("RGB").resize((512,512))
    no_mask = Image.open(aug_no_mask_path).convert("RGB").resize((512,512))
    w_mask  = Image.open(aug_with_mask_path).convert("RGB").resize((512,512))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, img, title in zip(axes,
        [orig, no_mask, w_mask],
        ["Original", "Exp 2: No mask\n(full image edited)",
         "Exp 3: SAM2 mask\n(background only edited)"]):
        ax.imshow(img); ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    wrapped = "\n".join([desc[i:i+50] for i in range(0,len(desc),50)])
    fig.suptitle(f"Experiment 2 vs 3 — {species}\nPrompt: "{wrapped}"",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison figure: {out_path}")


def figure_incremental_curve_exp3(iterations, f1_a, f1_b,
                                   f1_exp2_a=None, f1_exp2_b=None, out_path=""):
    """Learning curve for Exp 3, optionally overlaid with Exp 2 for comparison."""
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(iterations, f1_a, "o-", color="#E8854C", linewidth=2,
            label="Exp3 Strategy A (fill to x, with masks)")
    ax.plot(iterations, f1_b, "s--", color="#E8854C", linewidth=2, alpha=0.6,
            label="Exp3 Strategy B (equal y/p, with masks)")
    if f1_exp2_a:
        ax.plot(iterations, f1_exp2_a, "o-", color="#4C8EDA", linewidth=2,
                alpha=0.5, label="Exp2 Strategy A (no masks)")
    if f1_exp2_b:
        ax.plot(iterations, f1_exp2_b, "s--", color="#4C8EDA", linewidth=2,
                alpha=0.5, label="Exp2 Strategy B (no masks)")
    ax.axhline(f1_a[0], color="gray", linestyle=":", linewidth=1.2,
               label="Baseline")
    ax.set_xlabel("Augmented images added per underrepresented class")
    ax.set_ylabel("Weighted F1")
    ax.set_title("Incremental Augmentation: Exp 2 vs Exp 3\n"
                 "(with vs without SAM2 segmentation masks)", fontweight="bold")
    ax.legend(fontsize=8); ax.set_ylim(0,1); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Saved comparison curve: {out_path}")


def figure_per_class_f1_exp3(per_class_results, out_path):
    species_list = sorted(per_class_results["before"].keys())
    before = [per_class_results["before"].get(sp,0) for sp in species_list]
    after  = [per_class_results["after"].get(sp,0)  for sp in species_list]
    x      = np.arange(len(species_list))

    fig, ax = plt.subplots(figsize=(max(8, len(species_list)*1.2), 5))
    ax.bar(x-.18, before, .35, label="Before augmentation", color="#AAB7C4")
    ax.bar(x+.18, after,  .35, label="After augmentation (Exp3, masks)", color="#E8854C")
    ax.set_xticks(x)
    ax.set_xticklabels(species_list, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1: Underrepresented Species\n(Experiment 3, with SAM2 masks)",
                 fontweight="bold")
    ax.legend(); ax.set_ylim(0,1); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Saved per-class F1 chart: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

def run_smoke_test():
    print("\n" + "="*60)
    print("SMOKE TEST — Experiment 3 (with SAM2 masks)")
    print("="*60)

    budget = compute_budget()
    pipe   = load_inpaint_pipe()

    # Find test species with images on disk
    test_species, test_paths = None, []
    for sp in sorted(budget["underrepresented"], key=lambda s: budget["underrepresented"][s]):
        candidates = list(Path(DIRS["ss_images"]).rglob(f"{sp}*"))[:5]
        if candidates:
            test_species = sp
            test_paths   = [str(p) for p in candidates[:3]]
            break

    if not test_species:
        print("  No Serengeti images found. Download underrepresented species first.")
        return

    print(f"\n  Test species: {test_species} ({len(test_paths)} images)")

    # Captions + descriptions
    captions = caption_images(test_paths, test_species)
    if not captions:
        captions = [f"a camera trap photo of a {test_species} in the wild"]
    descs = get_domain_descriptions(test_species, captions)
    print(f"  Domain descriptions: {descs}")

    # Generate masks + augmentations for first image
    src = test_paths[0]
    mask_path, bg_mask_path = generate_mask(src)

    if mask_path is None:
        print("  ⚠️  MegaDetector/SAM2 found no animal — trying next image")
        for p in test_paths[1:]:
            mask_path, bg_mask_path = generate_mask(p)
            if mask_path:
                src = p; break

    aug_dir = Path(DIRS["augmented"]) / "smoke_test" / test_species
    aug_dir.mkdir(parents=True, exist_ok=True)

    aug_paths = []
    for j, desc in enumerate(descs):
        out = str(aug_dir / f"aug_masked_{j}.jpg")
        if not os.path.exists(out):
            augment_image_with_mask(pipe, src, mask_path, bg_mask_path, desc, out)
        aug_paths.append(out)

    # Figure 1: full mask pipeline grid
    figure_mask_augmentation_grid(
        src, mask_path, bg_mask_path, aug_paths, descs, test_species,
        out_path=f"{DIRS['figures']}/smoke_test_mask_pipeline.png")

    # Figure 2: comparison with no-mask (if exp2 augmented version exists)
    exp2_aug = f"data/exp2/augmented/smoke_test/{test_species}/aug_0_0.jpg"
    if os.path.exists(exp2_aug) and aug_paths:
        figure_exp2_vs_exp3_comparison(
            src, exp2_aug, aug_paths[0], descs[0], test_species,
            out_path=f"{DIRS['figures']}/smoke_test_exp2_vs_exp3.png")

    print("\n  Smoke test complete.")
    print(f"  Open {DIRS['figures']}/ to see the figures.")


# ══════════════════════════════════════════════════════════════════════════════
# Strategy A — budget mode
# ══════════════════════════════════════════════════════════════════════════════

def run_budget_mode():
    print("\n" + "="*60)
    print("BUDGET MODE — Experiment 3 (with masks)")
    print("="*60)

    budget     = compute_budget()
    pipe       = load_inpaint_pipe()
    test_items = _load_test_items()
    n_steps    = 5

    source_map, desc_map, mask_cache = {}, {}, {}
    for sp in budget["underrepresented"]:
        paths = list(Path(DIRS["ss_images"]).rglob(f"{sp}*"))
        if not paths: continue
        source_map[sp] = [str(p) for p in paths]
        caps            = caption_images(source_map[sp], sp)
        desc_map[sp]    = get_domain_descriptions(sp, caps)
        # Pre-generate masks
        for p in source_map[sp][:10]:
            mp, bgp = generate_mask(p)
            mask_cache[p] = (mp, bgp)

    f1_a, f1_b, steps = [], [], []
    per_class_before = None

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        print(f"\n  Step {step}/{n_steps}")

        for strategy in ["strategy_a", "strategy_b"]:
            alloc   = budget[strategy]
            aug_dir = Path(DIRS["augmented"]) / strategy

            for sp, n_total in alloc.items():
                if sp not in source_map: continue
                n_this = max(1, int(n_total * frac))
                sp_dir = aug_dir / sp
                sp_dir.mkdir(parents=True, exist_ok=True)
                existing = len(list(sp_dir.glob("*.jpg")))
                paths    = source_map[sp]
                descs    = desc_map.get(sp, [f"a camera trap photo of a {sp} in the wild"])

                for i in range(existing, n_this):
                    src            = paths[i % len(paths)]
                    mp, bgp        = mask_cache.get(src, generate_mask(src))
                    mask_cache[src] = (mp, bgp)
                    desc           = descs[i % len(descs)]
                    out            = str(sp_dir / f"aug_step{step}_{i:04d}.jpg")
                    if not os.path.exists(out):
                        try: augment_image_with_mask(pipe, src, mp, bgp, desc, out)
                        except Exception as e: print(f"    Error: {e}")

            clip_filter(str(aug_dir))
            metrics = evaluate_weighted_f1(test_items)
            if strategy == "strategy_a":
                f1_a.append(metrics["weighted_f1"])
                if per_class_before is None:
                    per_class_before = metrics["per_class_f1"]
            else:
                f1_b.append(metrics["weighted_f1"])

        steps.append(step)

    figure_incremental_curve_exp3(
        steps, f1_a, f1_b,
        out_path=f"{DIRS['figures']}/budget_mode_curve.png")
    figure_per_class_f1_exp3(
        {"before": per_class_before, "after": metrics["per_class_f1"]},
        out_path=f"{DIRS['figures']}/budget_mode_per_class_f1.png")

    with open(f"{DIRS['results']}/budget_mode.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","f1_strategy_a","f1_strategy_b"])
        for i,s in enumerate(steps):
            w.writerow([s, f1_a[i], f1_b[i]])
    print("\n  Budget mode complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Strategy B — incremental curve
# ══════════════════════════════════════════════════════════════════════════════

def run_curve_mode():
    print("\n" + "="*60)
    print("CURVE MODE — Experiment 3 (with masks)")
    print("="*60)

    budget     = compute_budget()
    pipe       = load_inpaint_pipe()
    test_items = _load_test_items()

    source_map, desc_map, mask_cache = {}, {}, {}
    for sp in budget["underrepresented"]:
        paths = list(Path(DIRS["ss_images"]).rglob(f"{sp}*"))
        if not paths: continue
        source_map[sp] = [str(p) for p in paths]
        caps            = caption_images(source_map[sp], sp)
        desc_map[sp]    = get_domain_descriptions(sp, caps)
        for p in source_map[sp][:5]:
            mask_cache[p] = generate_mask(p)

    f1_scores, img_counts = [], []
    total_added = 0
    aug_dir     = Path(DIRS["augmented"]) / "curve_mode"

    for iteration in range(1, CURVE_ITERATIONS + 1):
        print(f"\n  Iteration {iteration}/{CURVE_ITERATIONS}")

        for sp in budget["underrepresented"]:
            if sp not in source_map: continue
            for k in range(CURVE_PER_STEP):
                sp_dir = aug_dir / sp
                sp_dir.mkdir(parents=True, exist_ok=True)
                idx    = (total_added + k) % len(source_map[sp])
                src    = source_map[sp][idx]
                mp, bgp = mask_cache.get(src, generate_mask(src))
                mask_cache[src] = (mp, bgp)
                desc   = desc_map.get(sp, ["a camera trap photo"])[
                             (total_added + k) % max(1, len(desc_map.get(sp,[""])))]
                out    = str(sp_dir / f"aug_iter{iteration}_{k:04d}.jpg")
                if not os.path.exists(out):
                    try: augment_image_with_mask(pipe, src, mp, bgp, desc, out)
                    except Exception as e: print(f"    Error: {e}")

        clip_filter(str(aug_dir))
        total_added += CURVE_PER_STEP * len(source_map)
        metrics = evaluate_weighted_f1(test_items)
        f1_scores.append(metrics["weighted_f1"])
        img_counts.append(total_added)
        print(f"    Weighted F1: {metrics['weighted_f1']:.4f}  total added: {total_added}")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(img_counts, f1_scores, "o-", color="#E8854C", linewidth=2,
            label="Exp3 (with SAM2 masks)")
    ax.set_xlabel("Total augmented images added")
    ax.set_ylabel("Weighted F1")
    ax.set_title("Experiment 3 (With Masks): F1 vs Images Added",
                 fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{DIRS['figures']}/curve_mode_f1.png", dpi=150)
    plt.close()

    with open(f"{DIRS['results']}/curve_mode.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_images_added","weighted_f1"])
        for c, f in zip(img_counts, f1_scores):
            w.writerow([c, f])
    print("\n  Curve mode complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test","budget","curve"],
                        default="test")
    args = parser.parse_args()

    if   args.mode == "test":   run_smoke_test()
    elif args.mode == "budget": run_budget_mode()
    elif args.mode == "curve":  run_curve_mode()
