from __future__ import annotations

import importlib.util
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


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

    try:
        editor_fn = load_alia_callable(repo_path)
    except Exception:
        editor_fn = None
    rows = []
    sampled = source_df.groupby("final_label", group_keys=False).head(max_images_per_class)
    prompt_bank_cache: dict[str, list[str]] = {}

    for _, row in sampled.iterrows():
        image_path = row.get("path") or row.get("file_path") or row.get("image_path")
        if not image_path or not os.path.exists(image_path):
            continue

        for spec in methods:
            method_dir = output_root / spec.name / str(row["final_label"])
            method_dir.mkdir(parents=True, exist_ok=True)
            out_file = method_dir / f"{Path(image_path).stem}_{spec.name}.png"
            if editor_fn is not None:
                if prompt_strategy == "prompt_bank":
                    bank = prompt_bank_cache.setdefault(
                        spec.name,
                        build_alia_prompt_bank(spec.name, str(row["final_label"]), species_lookup=species_lookup),
                    )
                    prompt = bank[0]
                    if len(bank) > 1:
                        prompt = bank[abs(hash((image_path, spec.name))) % len(bank)]
                elif prompt_strategy == "scene_description":
                    prompt = build_alia_scene_description(spec.name, str(row["final_label"]), species_lookup=species_lookup)
                else:
                    prompt = f"{spec.prompt_suffix}; keep the animal identity and species label unchanged."
                try:
                    result = try_call_alia(
                        editor_fn=editor_fn,
                        image_path=image_path,
                        prompt=prompt,
                        output_dir=method_dir,
                        seed=spec.seed,
                    )
                except Exception as exc:
                    print(f"ALIA edit failed for {image_path} using {spec.name}: {exc}")
                    continue

                generated_files = sorted(method_dir.glob("*"))
                if not generated_files:
                    continue
                out_file = generated_files[-1]
            else:
                image = Image.open(image_path).convert("RGB")
                edited = _fallback_edit(image, spec)
                edited.save(out_file)
                result = f"fallback:{spec.name}"
            rows.append(
                {
                    "source": "alia_diffusion",
                    "path": str(out_file),
                    "final_label": row["final_label"],
                    "split": "train",
                    "alia_method": spec.name,
                    "alia_source": str(image_path),
                    "alia_result": str(result) if result is not None else "",
                }
            )

    return pd.DataFrame(rows)
