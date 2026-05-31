from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch


@dataclass(frozen=True)
class AliaEditSpec:
    name: str
    prompt_suffix: str
    seed: int = 0


ALIA_EDIT_SPECS: list[AliaEditSpec] = [
    AliaEditSpec("background", "change the background while preserving the animal"),
    AliaEditSpec("weather", "change the weather conditions while preserving the animal"),
    AliaEditSpec("lighting", "change the lighting and time of day while preserving the animal"),
    AliaEditSpec("season", "change the season and vegetation while preserving the animal"),
]

ALIA_PROMPT_BANKS: dict[str, list[str]] = {
    "contextual_bias": [
        "a camera trap photo of a {} standing near a waterhole",
        "a camera trap photo of a {} grazing in an open field",
        "a camera trap photo of a {} near acacia trees",
        "a camera trap photo of a {} walking along a dirt path",
    ],
    "fine_grained": [
        "a close-up camera trap photo of a {} showing distinctive markings",
        "a sharp detailed photo of a {} emphasizing fur pattern and body shape",
        "a close-up photo of a {} highlighting identifying characteristics",
        "a detailed wildlife photo of a {} with visible diagnostic features",
    ],
    "domain_generalization": [
        "a camera trap photo of a {} near a large body of water",
        "a camera trap photo of a {} in a grassy field with trees and bushes",
        "a camera trap photo of a {} during heavy rain",
        "a camera trap photo of a {} during a dry season drought",
        "a camera trap photo of a {} at sunrise",
        "a camera trap photo of a {} at sunset",
        "a camera trap photo of a {} on a cloudy day",
        "a camera trap photo of a {} in dense vegetation",
    ],
}

ALIA_PROMPT_HINTS: dict[str, str] = {
    "contextual_bias": "Keep the animal identity fixed while changing surrounding scene context.",
    "fine_grained": "Keep the animal identity fixed and emphasize small species-specific visual details.",
    "domain_generalization": "Keep the animal identity fixed while varying environment, weather, season, and lighting.",
}

_ALIA_IMG2IMG_PIPELINE = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_alia_img2img_pipeline(repo_path: str | Path | None = None):
    global _ALIA_IMG2IMG_PIPELINE
    if _ALIA_IMG2IMG_PIPELINE is not None:
        return _ALIA_IMG2IMG_PIPELINE

    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except Exception:
        return None

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        requires_safety_checker=False,
        safety_checker=None,
    )
    pipe = pipe.to(_get_device())
    _ALIA_IMG2IMG_PIPELINE = pipe
    return pipe


def find_alia_repo(working_dir: str | Path = ".") -> Path | None:
    working_dir = Path(working_dir)
    for candidate in [working_dir / "ALIA", working_dir / "alia"]:
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_module_from_path(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discover_callable(module) -> Callable | None:
    preferred_names = [
        "edit_image",
        "edit",
        "generate",
        "run",
        "main",
    ]
    for name in preferred_names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    for _, candidate in inspect.getmembers(module, inspect.isfunction):
        return candidate
    return None


def load_alia_callable(repo_path: str | Path) -> Callable:
    repo_path = Path(repo_path)
    candidate_files = [
        repo_path / "demo.py",
        repo_path / "edit.py",
        repo_path / "main.py",
        repo_path / "inference.py",
        repo_path / "alia" / "demo.py",
        repo_path / "alia" / "edit.py",
        repo_path / "alia" / "main.py",
    ]
    for module_path in candidate_files:
        if module_path.exists():
            module = _load_module_from_path(module_path)
            fn = _discover_callable(module)
            if fn is not None:
                return fn
    raise FileNotFoundError(
        f"Could not find a callable ALIA entrypoint in {repo_path}. "
        "Inspect the repo and wire the helper to the correct script."
    )


def _fallback_edit(image: Image.Image, spec: AliaEditSpec) -> Image.Image:
    if spec.name == "background":
        return ImageOps.autocontrast(image.filter(ImageFilter.GaussianBlur(radius=1)))
    if spec.name == "weather":
        image = ImageEnhance.Color(image).enhance(0.75)
        return ImageEnhance.Brightness(image).enhance(0.9)
    if spec.name == "lighting":
        image = ImageEnhance.Brightness(image).enhance(1.2)
        return ImageEnhance.Contrast(image).enhance(1.15)
    if spec.name == "season":
        image = ImageEnhance.Color(image).enhance(1.25)
        return ImageEnhance.Contrast(image).enhance(1.05)
    return image


def build_alia_prompt(strategy: str, label: str, species_lookup: dict[str, str] | None = None, seed: int | None = None) -> str:
    import random

    species_lookup = species_lookup or {}
    species = species_lookup.get(label, label)
    bank = ALIA_PROMPT_BANKS.get(strategy)
    if bank is None:
        raise ValueError(f"Unknown ALIA strategy: {strategy}")
    rng = random.Random(seed)
    return rng.choice(bank).format(species)


def build_alia_prompt_bank(strategy: str, label: str, species_lookup: dict[str, str] | None = None) -> list[str]:
    species_lookup = species_lookup or {}
    species = species_lookup.get(label, label)
    bank = ALIA_PROMPT_BANKS.get(strategy)
    if bank is None:
        raise ValueError(f"Unknown ALIA strategy: {strategy}")
    return [template.format(species) for template in bank]


def build_alia_scene_description(strategy: str, label: str, species_lookup: dict[str, str] | None = None) -> str:
    species_lookup = species_lookup or {}
    species = species_lookup.get(label, label)
    hint = ALIA_PROMPT_HINTS.get(strategy)
    if hint is None:
        raise ValueError(f"Unknown ALIA strategy: {strategy}")
    return f"{species}: {hint}"


def _prompt_slug(prompt: str) -> str:
    return prompt.replace(" ", "_").replace("'", "").replace(",", "")


def _resolve_alia_editing_script(repo_path: str | Path, script_name: str = "img2img_example.py") -> Path:
    repo_path = Path(repo_path)
    search_roots = [repo_path, repo_path / "ALIA", repo_path / "alia"]
    for root in search_roots:
        candidates = [
            root / "editing_methods" / script_name,
            root / script_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"Could not find {script_name} under {repo_path}")


def _cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _materialize_alia_input_image(image_path: str | Path, cache_dir: str | Path) -> Path:
    image_path = str(image_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if image_path.startswith("gs://"):
        local_path = cache_dir / f"{_cache_key(image_path)}_{Path(image_path).stem}.png"
        if not local_path.exists():
            _load_image(image_path).save(local_path)
        return local_path

    local_path = Path(image_path)
    if not local_path.is_absolute():
        local_path = local_path.resolve()
    if local_path.exists():
        return local_path

    fallback_path = cache_dir / f"{_cache_key(image_path)}_{Path(image_path).stem}.png"
    if not fallback_path.exists():
        _load_image(image_path).save(fallback_path)
    return fallback_path


def run_alia_img2img_example(
    repo_path: str | Path,
    image_path: str | Path,
    prompt: str,
    output_dir: str | Path,
    n: int = 1,
    strength: float = 0.65,
    guidance: float = 7.5,
    model: str = "runwayml/stable-diffusion-v1-5",
    wandb_silent: bool = True,
) -> list[Path]:
    script_path = _resolve_alia_editing_script(repo_path, "img2img_example.py")
    repo_root = script_path.parent.parent if script_path.parent.name == "editing_methods" else script_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_cache_dir = output_dir / "_alia_inputs"
    input_path = _materialize_alia_input_image(image_path, input_cache_dir)

    cmd = [
        sys.executable,
        str(script_path),
        "--im-path",
        str(input_path),
        "--prompt",
        prompt,
        "--save-dir",
        str(output_dir),
        "--n",
        str(n),
        "--strength",
        str(strength),
        "--guidance",
        str(guidance),
        "--model",
        model,
    ]
    if wandb_silent:
        cmd.append("--wandb-silent")

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_SILENT", "true")

    completed = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ALIA editing_methods/img2img_example.py failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    generated_dir = output_dir / "img2img" / _prompt_slug(prompt) / f"strength-{strength}_guidance-{guidance}"
    generated_paths = [generated_dir / f"{idx}.png" for idx in range(n) if (generated_dir / f"{idx}.png").exists()]
    if not generated_paths:
        raise FileNotFoundError(
            f"ALIA completed successfully but no generated PNGs were found in {generated_dir}."
        )
    return generated_paths


def build_alia_augmentations_from_repo(
    source_df: pd.DataFrame,
    output_root: str | Path,
    repo_path: str | Path,
    max_images_per_class: int = 12,
    methods: Iterable[AliaEditSpec] = ALIA_EDIT_SPECS,
    prompt_strategy: str = "prompt_bank",
    species_lookup: dict[str, str] | None = None,
    n: int = 1,
    model: str = "runwayml/stable-diffusion-v1-5",
) -> pd.DataFrame:
    repo_path = Path(repo_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    label_col = "final_label" if "final_label" in source_df.columns else "label"
    sampled = source_df.groupby(label_col, group_keys=False).head(max_images_per_class)
    prompt_bank_cache: dict[tuple[str, str], list[str]] = {}

    for _, row in sampled.iterrows():
        image_path = None
        for key in ("path", "file_path", "image_path", "local_path"):
            value = row.get(key)
            if value is not None and not pd.isna(value) and str(value):
                image_path = str(value)
                break
        if not image_path and row.get("file_name"):
            image_path = f"gs://public-datasets-lila/snapshotserengeti-unzipped/{row['file_name']}"
        if not image_path:
            continue

        for spec in methods:
            if prompt_strategy == "prompt_bank":
                cache_key = (spec.name, str(row[label_col]))
                bank = prompt_bank_cache.setdefault(
                    cache_key,
                    build_alia_prompt_bank(spec.name, str(row[label_col]), species_lookup=species_lookup),
                )
                prompt = bank[0]
                if len(bank) > 1:
                    prompt = bank[int(_cache_key(f"{image_path}|{spec.name}"), 16) % len(bank)]
            elif prompt_strategy == "scene_description":
                prompt = build_alia_scene_description(spec.name, str(row[label_col]), species_lookup=species_lookup)
            else:
                prompt = f"{spec.prompt_suffix}; keep the animal identity and species label unchanged."

            method_dir = output_root / spec.name / str(row[label_col]) / Path(str(image_path)).stem
            generated_paths = run_alia_img2img_example(
                repo_path=repo_path,
                image_path=image_path,
                prompt=prompt,
                output_dir=method_dir,
                n=n,
                strength=0.55 if spec.name == "fine_grained" else 0.65,
                guidance=7.5,
                model=model,
            )

            rows.append(
                {
                    "source": "alia_editing_methods",
                    "path": str(generated_paths[0]),
                    "final_label": row[label_col],
                    "split": row.get("split", "train"),
                    "alia_method": spec.name,
                    "alia_source": str(image_path),
                    "alia_prompt": prompt,
                    "alia_script": "img2img_example.py",
                    "alia_result": "repo:img2img_example",
                }
            )

    return pd.DataFrame(rows)


def try_call_alia(editor_fn: Callable, image_path: str | Path, prompt: str, output_dir: str | Path, seed: int = 0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    call_patterns = [
        dict(image_path=str(image_path), prompt=prompt, output_dir=str(output_dir), seed=seed),
        dict(input_image=str(image_path), prompt=prompt, output_dir=str(output_dir), seed=seed),
        dict(image=str(image_path), text=prompt, output_dir=str(output_dir), seed=seed),
        dict(image_path=str(image_path), prompt=prompt, outdir=str(output_dir), seed=seed),
    ]

    last_error: Exception | None = None
    for kwargs in call_patterns:
        try:
            result = editor_fn(**kwargs)
            return result
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("ALIA call failed for an unknown reason.")


def _load_image(image_path: str) -> Image.Image:
    if image_path.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem(token="anon")
        with fs.open(image_path, "rb") as f:
            return Image.open(f).convert("RGB")
    return Image.open(image_path).convert("RGB")


def _run_alia_img2img(image: Image.Image, prompt: str, strength: float = 0.65, guidance_scale: float = 7.5, seed: int = 0) -> Image.Image | None:
    pipe = _get_alia_img2img_pipeline()
    if pipe is None:
        return None
    generator = torch.Generator(device=_get_device()).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        generator=generator,
    )
    return result.images[0]


def build_alia_augmentations(
    source_df: pd.DataFrame,
    output_root: str | Path,
    repo_path: str | Path,
    max_images_per_class: int = 12,
    methods: Iterable[AliaEditSpec] = ALIA_EDIT_SPECS,
    prompt_strategy: str = "prompt_bank",
    species_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    repo_path = Path(repo_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    label_col = "final_label" if "final_label" in source_df.columns else "label"
    sampled = source_df.groupby(label_col, group_keys=False).head(max_images_per_class)
    prompt_bank_cache: dict[tuple[str, str], list[str]] = {}

    for _, row in sampled.iterrows():
        image_path = None
        for key in ("path", "file_path", "image_path", "local_path"):
            value = row.get(key)
            if value is not None and not pd.isna(value) and str(value):
                image_path = str(value)
                break
        if not image_path and row.get("file_name"):
            image_path = f"gs://public-datasets-lila/snapshotserengeti-unzipped/{row['file_name']}"
        if not image_path:
            continue
        if image_path.startswith("gs://"):
            exists = True
        else:
            exists = os.path.exists(image_path)
        if not exists:
            continue

        for spec in methods:
            method_dir = output_root / spec.name / str(row["final_label"])
            method_dir.mkdir(parents=True, exist_ok=True)
            out_file = method_dir / f"{Path(image_path).stem}_{spec.name}.png"
            if prompt_strategy == "prompt_bank":
                cache_key = (spec.name, str(row["final_label"]))
                bank = prompt_bank_cache.setdefault(
                    cache_key,
                    build_alia_prompt_bank(spec.name, str(row["final_label"]), species_lookup=species_lookup),
                )
                prompt = bank[0]
                if len(bank) > 1:
                    prompt = bank[int(_cache_key(f"{image_path}|{spec.name}"), 16) % len(bank)]
            elif prompt_strategy == "scene_description":
                prompt = build_alia_scene_description(spec.name, str(row["final_label"]), species_lookup=species_lookup)
            else:
                prompt = f"{spec.prompt_suffix}; keep the animal identity and species label unchanged."

            image = _load_image(image_path)
            edited = None
            try:
                edited = _run_alia_img2img(
                    image=image,
                    prompt=prompt,
                    strength=0.55 if spec.name == "fine_grained" else 0.65,
                    guidance_scale=7.5,
                    seed=spec.seed,
                )
            except Exception as exc:
                print(f"ALIA diffusion edit failed for {image_path} using {spec.name}: {exc}")
            if edited is None:
                edited = _fallback_edit(image, spec)

            edited.save(out_file)
            result = "diffusers:img2img" if edited is not None else f"fallback:{spec.name}"
            rows.append(
                {
                    "source": "alia_diffusion",
                    "path": str(out_file),
                    "final_label": row[label_col],
                    "split": "train",
                    "alia_method": spec.name,
                    "alia_source": str(image_path),
                    "alia_result": str(result) if result is not None else "",
                }
            )

    return pd.DataFrame(rows)
