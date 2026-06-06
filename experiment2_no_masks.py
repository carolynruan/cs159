"""
Experiment 2: ALIA augmentation WITHOUT segmentation masks
==========================================================
Fine-tune SpeciesNet on:
  - X iNaturalist images (post-cutoff, Apr 2025+)
  - X augmented Snapshot Serengeti images (SD Img2Img, full image, no mask)

Two X-selection strategies:
  Strategy A: 5th-percentile budget — bring underrepresented classes to x images
              across incremental iterations (adds images in steps)
  Strategy B: Incremental curve — add 1 image per class per iteration,
              track weighted F1 after each step, plot learning curve

Evaluation metric: weighted F1 (per SpeciesNet paper)

Usage:
  python experiment2_no_masks.py --mode budget   # Strategy A
  python experiment2_no_masks.py --mode curve    # Strategy B
  python experiment2_no_masks.py --mode test     # smoke test on 3 images
"""

import os, json, math, csv, random, argparse, shutil
from pathlib import Path
from collections import defaultdict
from io import BytesIO

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import (BlipProcessor, BlipForConditionalGeneration,
                          CLIPModel, CLIPProcessor)
from diffusers import StableDiffusionImg2ImgPipeline
import openai

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = "sk-..."
SD_MODEL         = "runwayml/stable-diffusion-v1-5"
EDIT_STRENGTH    = 0.75
GUIDANCE_SCALE   = 7.5
SD_STEPS         = 30
CURVE_ITERATIONS = 10      # Strategy B: how many incremental steps
CURVE_IMAGES_PER_STEP = 1  # images added per underrepresented class per step

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
    "inat_images": "data/images",
    "inat_meta":   "data/metadata",
    "captions":    "data/exp2/captions",
    "augmented":   "data/exp2/augmented",
    "train":       "data/exp2/train",
    "results":     "data/exp2/results",
    "figures":     "data/exp2/figures",
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Budget
# ══════════════════════════════════════════════════════════════════════════════

def compute_budget():
    counts  = list(SERENGETI_SPECIES_COUNTS.values())
    x       = int(np.percentile(sorted(counts), 5))
    under   = {sp: c for sp, c in SERENGETI_SPECIES_COUNTS.items() if c < x}
    p       = len(under)
    y       = sum(x - c for c in under.values())
    eq      = math.ceil(y / p)
    budget  = dict(x=x, p=p, y=y, equal_per_class=eq,
                   underrepresented=under,
                   strategy_a={sp: x - c for sp, c in under.items()},
                   strategy_b={sp: eq for sp in under})
    json.dump(budget, open("data/exp2_budget.json","w"), indent=2)
    _print_budget(budget, "Experiment 2 (no masks)")
    return budget


def _print_budget(b, title):
    print(f"\n{'='*60}\nBudget — {title}\n{'='*60}")
    print(f"  x={b['x']}  p={b['p']}  y={b['y']}  y/p={b['equal_per_class']}")
    print(f"  {'Species':<20} {'Count':>6}  A(+needed)  B(+equal)")
    for sp, c in sorted(b['underrepresented'].items(), key=lambda t:t[1]):
        print(f"  {sp:<20} {c:>6}  +{b['strategy_a'][sp]:<10} +{b['strategy_b'][sp]}")


# ══════════════════════════════════════════════════════════════════════════════
# BLIP captions
# ══════════════════════════════════════════════════════════════════════════════

def caption_images(image_paths: list, species: str) -> list[str]:
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model     = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large").to(device)
    captions  = []
    for p in tqdm(image_paths[:50], desc=f"  BLIP {species}"):
        cap_path = Path(DIRS["captions"]) / f"{Path(p).stem}.txt"
        if cap_path.exists():
            captions.append(cap_path.read_text().strip()); continue
        try:
            img    = Image.open(p).convert("RGB")
            inputs = processor(img, return_tensors="pt").to(device)
            out    = model.generate(**inputs, max_new_tokens=50)
            cap    = processor.decode(out[0], skip_special_tokens=True)
            cap_path.write_text(cap)
            captions.append(cap)
        except Exception as e:
            print(f"    caption error {p}: {e}")
    return captions


# ══════════════════════════════════════════════════════════════════════════════
# GPT-4 domain descriptions
# ══════════════════════════════════════════════════════════════════════════════

def get_domain_descriptions(species: str, captions: list[str]) -> list[str]:
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
            f"Output fewer than 5 captions, one per line, unique settings for "
            f"{species}. No numbering."}])
    descs = [l.strip() for l in r2.choices[0].message.content.splitlines() if l.strip()][:5]
    json.dump(descs, open(cache,"w"), indent=2)
    return descs


# ══════════════════════════════════════════════════════════════════════════════
# SD Img2Img augmentation (NO mask)
# ══════════════════════════════════════════════════════════════════════════════

def load_img2img_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device)
    pipe.safety_checker = None
    return pipe


def augment_image_no_mask(pipe, src_path: str, prompt: str, out_path: str):
    """Edit the full image with SD Img2Img — no animal protection."""
    original = Image.open(src_path).convert("RGB").resize((512,512))
    result   = pipe(prompt=prompt, image=original,
                    strength=EDIT_STRENGTH, guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=SD_STEPS).images[0]
    result.save(out_path)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLIP filter
# ══════════════════════════════════════════════════════════════════════════════

def clip_filter(aug_dir: str) -> int:
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts    = ["a photo of a wild animal","a photo of an object",
                  "a photo of a scene","a photo","an image"]
    removed    = 0
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
# Evaluation — weighted F1
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_weighted_f1(model, test_images: list[dict]) -> dict:
    """
    Run SpeciesNet on test_images, return weighted F1 + per-class F1.
    test_images: list of {"path": str, "species": str}
    """
    import speciesnet as snet_module
    snet    = snet_module.SpeciesNet()
    y_true, y_pred = [], []

    for item in tqdm(test_images, desc="  Evaluating"):
        try:
            preds = snet.predict(image_paths=[item["path"]])
            top1  = preds[0]["predictions"][0]["taxon"] if preds else "none"
            y_true.append(item["species"])
            y_pred.append(top1)
        except:
            y_true.append(item["species"])
            y_pred.append("none")

    classes       = sorted(set(y_true))
    weighted_f1   = f1_score(y_true, y_pred, labels=classes,
                             average="weighted", zero_division=0)
    per_class_f1  = f1_score(y_true, y_pred, labels=classes,
                             average=None, zero_division=0)
    return {
        "weighted_f1":  weighted_f1,
        "per_class_f1": dict(zip(classes, per_class_f1)),
        "n":            len(y_true),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def figure_augmentation_grid(originals: list, augmented_sets: list[list],
                              domain_descs: list[str], species: str, out_path: str):
    """
    Grid showing: original | aug with desc 1 | aug with desc 2 | ...
    One row per source image.
    """
    n_rows = len(originals)
    n_cols = 1 + len(domain_descs)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5*n_cols, 3.5*n_rows))
    if n_rows == 1: axes = axes[None,:]

    for r, (orig_path, aug_row) in enumerate(zip(originals, augmented_sets)):
        axes[r,0].imshow(Image.open(orig_path).convert("RGB"))
        axes[r,0].set_title("Original" if r==0 else "", fontsize=8)
        axes[r,0].set_ylabel(species, fontsize=7, rotation=0, labelpad=55)
        axes[r,0].axis("off")
        for c, (aug_path, desc) in enumerate(zip(aug_row, domain_descs)):
            axes[r, c+1].imshow(Image.open(aug_path).convert("RGB"))
            if r == 0:
                # Wrap description text
                wrapped = "\n".join([desc[i:i+30] for i in range(0,len(desc),30)])
                axes[r, c+1].set_title(f""{wrapped}"", fontsize=7)
            axes[r, c+1].axis("off")

    fig.suptitle(f"ALIA Augmentation (No Mask) — {species}", fontsize=11,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved augmentation grid: {out_path}")


def figure_incremental_curve(iterations: list[int], f1_scores_a: list[float],
                              f1_scores_b: list[float], out_path: str):
    """Learning curve: weighted F1 vs images added per underrepresented class."""
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(iterations, f1_scores_a, "o-", color="#4C8EDA", linewidth=2,
            label="Strategy A (fill to x)")
    ax.plot(iterations, f1_scores_b, "s--", color="#E8854C", linewidth=2,
            label="Strategy B (equal y/p per class)")
    ax.axhline(f1_scores_a[0], color="gray", linestyle=":", linewidth=1.2,
               label="Baseline (no augmentation)")
    ax.set_xlabel("Augmented images added per underrepresented class", fontsize=11)
    ax.set_ylabel("Weighted F1 Score", fontsize=11)
    ax.set_title("Experiment 2 (No Masks): Incremental Augmentation Curve",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved learning curve: {out_path}")


def figure_per_class_f1(per_class_results: dict, out_path: str):
    """Bar chart of per-class F1 for underrepresented species, before vs after."""
    species_list = sorted(per_class_results["before"].keys())
    before = [per_class_results["before"].get(sp, 0) for sp in species_list]
    after  = [per_class_results["after"].get(sp, 0)  for sp in species_list]
    x      = np.arange(len(species_list))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(species_list)*1.2), 5))
    ax.bar(x - width/2, before, width, label="Before augmentation", color="#AAB7C4")
    ax.bar(x + width/2, after,  width, label="After augmentation",  color="#4C8EDA")
    ax.set_xticks(x)
    ax.set_xticklabels(species_list, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1: Underrepresented Species (Exp 2, No Masks)",
                 fontweight="bold")
    ax.legend()
    ax.set_ylim(0,1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved per-class F1 chart: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test (3 images, skips SpeciesNet fine-tuning)
# ══════════════════════════════════════════════════════════════════════════════

def run_smoke_test():
    """
    Test the full augmentation pipeline on 3 images.
    Generates augmentation grid figure without running SpeciesNet fine-tuning.
    """
    print("\n" + "="*60)
    print("SMOKE TEST — Experiment 2 (no masks)")
    print("="*60)

    budget = compute_budget()
    pipe   = load_img2img_pipe()

    # Pick first underrepresented species that has images on disk
    test_species = None
    test_paths   = []
    for sp in sorted(budget["underrepresented"].keys(), key=lambda s: budget["underrepresented"][s]):
        candidates = list(Path(DIRS["ss_images"]).rglob(f"{sp}*"))[:5]
        if candidates:
            test_species = sp
            test_paths   = [str(p) for p in candidates[:3]]
            break

    if not test_species:
        print("  No Serengeti images found on disk for underrepresented species.")
        print("  Using a placeholder: download at least 3 images first.")
        return

    print(f"\n  Test species: {test_species} ({len(test_paths)} source images)")

    # Caption + domain descriptions
    captions = caption_images(test_paths, test_species)
    if not captions:
        captions = [f"a camera trap photo of a {test_species} in the wild"]
    descs = get_domain_descriptions(test_species, captions)
    print(f"  Domain descriptions: {descs}")

    # Augment each test image with each description
    aug_dir = Path(DIRS["augmented"]) / "smoke_test" / test_species
    aug_dir.mkdir(parents=True, exist_ok=True)
    augmented_sets = []

    for i, src_path in enumerate(test_paths):
        row = []
        for j, desc in enumerate(descs):
            out_path = str(aug_dir / f"aug_{i}_{j}.jpg")
            if not os.path.exists(out_path):
                augment_image_no_mask(pipe, src_path, desc, out_path)
            row.append(out_path)
        augmented_sets.append(row)

    # Figure
    figure_augmentation_grid(
        test_paths, augmented_sets, descs, test_species,
        out_path=f"{DIRS['figures']}/smoke_test_augmentation_grid.png"
    )
    print("\n  Smoke test complete. Check figures/smoke_test_augmentation_grid.png")


# ══════════════════════════════════════════════════════════════════════════════
# Strategy A — 5th percentile budget, incremental iterations
# ══════════════════════════════════════════════════════════════════════════════

def run_budget_mode():
    """
    Incrementally add augmentations in steps (1/5 of budget per iteration).
    Evaluate weighted F1 after each step. Plot learning curve.
    """
    print("\n" + "="*60)
    print("BUDGET MODE (Strategy A + B) — Experiment 2 (no masks)")
    print("="*60)

    budget     = compute_budget()
    pipe       = load_img2img_pipe()
    test_items = _load_test_items()
    n_steps    = 5

    # Build source image + description map for underrepresented species
    source_map = {}
    desc_map   = {}
    for sp in budget["underrepresented"]:
        paths = list(Path(DIRS["ss_images"]).rglob(f"{sp}*"))
        if not paths:
            print(f"  ⚠️  No images for {sp}, skip")
            continue
        source_map[sp] = [str(p) for p in paths]
        caps           = caption_images(source_map[sp], sp)
        desc_map[sp]   = get_domain_descriptions(sp, caps)

    f1_curve_a, f1_curve_b, step_counts = [], [], []
    per_class_before = None

    for step in range(1, n_steps + 1):
        print(f"\n  Step {step}/{n_steps}")
        frac = step / n_steps

        for strategy in ["strategy_a", "strategy_b"]:
            alloc   = budget[strategy]
            aug_dir = Path(DIRS["augmented"]) / strategy
            generated_this_step = 0

            for sp, n_total in alloc.items():
                if sp not in source_map: continue
                n_this_step = max(1, int(n_total * frac)) - \
                              len(list((aug_dir / sp).glob("*.jpg"))) \
                              if (aug_dir / sp).exists() else max(1, int(n_total * frac))
                paths = source_map[sp]
                descs = desc_map.get(sp, [f"a camera trap photo of a {sp} in the wild"])
                sp_dir = aug_dir / sp
                sp_dir.mkdir(parents=True, exist_ok=True)

                for i in range(n_this_step):
                    src  = paths[i % len(paths)]
                    desc = descs[i % len(descs)]
                    out  = str(sp_dir / f"aug_step{step}_{i:04d}.jpg")
                    if not os.path.exists(out):
                        try: augment_image_no_mask(pipe, src, desc, out)
                        except Exception as e: print(f"    Error: {e}")
                    generated_this_step += 1

            print(f"    {strategy}: +{generated_this_step} images")
            clip_filter(str(aug_dir))

            # Evaluate
            metrics = evaluate_weighted_f1(None, test_items)
            if strategy == "strategy_a":
                f1_curve_a.append(metrics["weighted_f1"])
                if per_class_before is None:
                    per_class_before = metrics["per_class_f1"]
            else:
                f1_curve_b.append(metrics["weighted_f1"])

        step_counts.append(step)

    # Figures
    figure_incremental_curve(
        step_counts, f1_curve_a, f1_curve_b,
        out_path=f"{DIRS['figures']}/incremental_curve.png")
    figure_per_class_f1(
        {"before": per_class_before, "after": metrics["per_class_f1"]},
        out_path=f"{DIRS['figures']}/per_class_f1.png")

    # Save CSV
    with open(f"{DIRS['results']}/budget_mode_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "f1_strategy_a", "f1_strategy_b"])
        for i, s in enumerate(step_counts):
            w.writerow([s, f1_curve_a[i], f1_curve_b[i]])
    print("\n  Budget mode complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Strategy B — incremental curve (1 image per step)
# ══════════════════════════════════════════════════════════════════════════════

def run_curve_mode():
    """
    Add CURVE_IMAGES_PER_STEP images per underrepresented class per iteration.
    Evaluate after each iteration. Plot F1 vs images added.
    """
    print("\n" + "="*60)
    print("CURVE MODE — Experiment 2 (no masks)")
    print("="*60)

    budget     = compute_budget()
    pipe       = load_img2img_pipe()
    test_items = _load_test_items()

    source_map, desc_map = {}, {}
    for sp in budget["underrepresented"]:
        paths = list(Path(DIRS["ss_images"]).rglob(f"{sp}*"))
        if not paths: continue
        source_map[sp] = [str(p) for p in paths]
        caps           = caption_images(source_map[sp], sp)
        desc_map[sp]   = get_domain_descriptions(sp, caps)

    f1_scores, img_counts = [], []
    total_added = 0
    aug_dir     = Path(DIRS["augmented"]) / "curve_mode"

    for iteration in range(1, CURVE_ITERATIONS + 1):
        print(f"\n  Iteration {iteration}/{CURVE_ITERATIONS}")
        added_this_iter = 0

        for sp in budget["underrepresented"]:
            if sp not in source_map: continue
            for k in range(CURVE_IMAGES_PER_STEP):
                sp_dir  = aug_dir / sp
                sp_dir.mkdir(parents=True, exist_ok=True)
                idx     = total_added + k
                src     = source_map[sp][idx % len(source_map[sp])]
                descs   = desc_map.get(sp, [f"a camera trap photo of a {sp} in the wild"])
                desc    = descs[idx % len(descs)]
                out     = str(sp_dir / f"aug_iter{iteration}_{k:04d}.jpg")
                if not os.path.exists(out):
                    try: augment_image_no_mask(pipe, src, desc, out)
                    except Exception as e: print(f"    Error: {e}")
                added_this_iter += 1

        clip_filter(str(aug_dir))
        total_added += CURVE_IMAGES_PER_STEP * len(source_map)

        metrics = evaluate_weighted_f1(None, test_items)
        f1_scores.append(metrics["weighted_f1"])
        img_counts.append(total_added)
        print(f"    Weighted F1: {metrics['weighted_f1']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(img_counts, f1_scores, "o-", color="#4C8EDA", linewidth=2)
    ax.set_xlabel("Total augmented images added")
    ax.set_ylabel("Weighted F1")
    ax.set_title("Experiment 2 (No Masks): F1 vs Images Added\n"
                 "(1 image/class/iteration)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = f"{DIRS['figures']}/curve_mode_f1.png"
    plt.savefig(curve_path, dpi=150); plt.close()
    print(f"\n  Saved curve: {curve_path}")

    with open(f"{DIRS['results']}/curve_mode_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_images_added", "weighted_f1"])
        for cnt, f1 in zip(img_counts, f1_scores):
            w.writerow([cnt, f1])
    print("  Curve mode complete.")


def _load_test_items():
    """Load iNat post-cutoff images as test set."""
    items = []
    for mf in Path(DIRS["inat_meta"]).glob("*.json"):
        meta = json.load(open(mf))
        fp   = os.path.join(DIRS["inat_images"], meta.get("filename",""))
        if os.path.exists(fp):
            items.append({"path": fp, "species": meta.get("species","unknown")})
    return items


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test","budget","curve"],
                        default="test",
                        help="test=smoke test | budget=Strategy A/B | curve=incremental")
    args = parser.parse_args()

    if   args.mode == "test":   run_smoke_test()
    elif args.mode == "budget": run_budget_mode()
    elif args.mode == "curve":  run_curve_mode()
