# GD vs GA for Urban Sound Classification
> Custom Smart Adaptive Systems — UPC 2025/26 | Lab Free Project

Comparative study of two MLP training strategies — **Gradient Descent (GD)** via backpropagation and a **Genetic Algorithm (GA)** — applied to urban sound classification using the [UrbanSound8K tabular dataset](https://www.kaggle.com/datasets/orvile/urban-sound-8k-tabular-form/data).

**Dataset:** 8,674 pre-extracted audio feature vectors (MFCC, Chroma, Spectral Contrast, ZCR, Spectral Centroid), 34 features, 10 urban sound classes  
**Team:** Leonie Greber, Anastasios Sidiropoulos

---

## Results

| Method | Test Acc | Macro F1 | Time |
|--------|----------|----------|------|
| GD (Adam, tuned) | **88.7%** | 0.887 | ~52 s |
| GA (DEAP, tuned) | 61.2% | 0.614 | ~155 s |
| GA (baseline, untuned) | 37.7% | — | — |

---

## Architecture

```
Input (34) → Linear(34 → 60) → ReLU → Linear(60 → 10) → logits
```

2,710 parameters total. Hidden dimension `h=60` found by Optuna (TPE, 60 trials).  
The same network is used for both methods; only the training procedure differs.

---

## Setup

### Option A — Anaconda (recommended on Windows)

```bash
conda create -n csas python=3.12
conda activate csas
pip install -r requirements.txt
python -m ipykernel install --user --name csas --display-name "Python (csas)"
jupyter notebook
```

Select the **Python (csas)** kernel when opening a notebook.

### Option B — Plain Python venv

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

---

## How to Run

1. **Hyperparameter tuning** (optional — results already saved):  
   Open `tuning.ipynb` and run all cells. Writes best parameters to `results/best_params.json`.

2. **Main experiment:**  
   Open `main.ipynb` and run all cells from top to bottom.  
   Figures are saved to `results/figures/` (created automatically on first run).

> The pre-computed `results/best_params.json` is included so `main.ipynb` works without re-running the tuning search.

---

## Repo Structure

```
├── data/
│   └── extracted_audio_features.csv   # UrbanSound8K tabular features
├── presentation/
│   └── presentation.tex               # Beamer slides (LaTeX)
├── report/
│   └── report.tex                     # Technical report (LaTeX)
├── results/
│   ├── best_params.json               # Tuned hyperparameters (GD + GA)
│   └── figures/                       # All output plots (12 PNGs)
├── main.ipynb                         # Full pipeline: data → train → evaluate
├── tuning.ipynb                       # Optuna hyperparameter search
├── requirements.txt
└── README.md
```

---

## Course Concepts

| Chapter | Applied as |
|---------|-----------|
| Ch. 2 | 34 pre-extracted features; mutual information importance ranking |
| Ch. 4 | MLP trained with Adam optimizer (GD/backprop) |
| Ch. 4 | Same MLP weights evolved with DEAP (GA) |
| Ch. 5 | Optuna TPE hyperparameter search; convergence curves; wall-clock comparison |

---

## References

- Salamon et al., "A Dataset and Taxonomy for Urban Sound Research", ACM MM 2014
- UrbanSound8K tabular dataset: https://www.kaggle.com/datasets/orvile/urban-sound-8k-tabular-form
- DEAP documentation: https://deap.readthedocs.io
- Optuna documentation: https://optuna.org
- Bishop, "Pattern Recognition and Machine Learning", Springer 2006
- Floreano & Mattiussi, "Bio-inspired Artificial Intelligence", MIT Press 2008
