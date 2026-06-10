# Diffusion-Driven Data Augmentation for Improving Species Classifiers

CS 159 project by Carolyn, Suhana, and Natalie.

- `experiment_1_iNaturalist.ipynb`: fine-tuning baseline on iNaturalist.
- `experiment_2_3_diffusion_augmentation.ipynb`: diffusion-based augmentation experiments.
- Colab links are included in each notebook.
- Helper scripts in `helpers_*.py` support the notebook workflows and reusable experiment runs.
- Data and generated outputs are stored locally in `lila_splits/`, `artifacts/`, and the CSV files at the repo root.

## File Structure

```text
.
├── README.md
├── LICENSE
├── augment_no_mask.py
├── augment_with_mask.py
├── helpers_lila_pipeline.py
├── helpers_run_experiment.py
├── experiment_1_iNaturalist.ipynb
├── experiment_2_3_diffusion_augmentation.ipynb
├── notebook_preprocessing.ipynb
├── notebook_segmentation_masking.ipynb
├── lila_splits/
├── artifacts/
└── data CSV files
```
