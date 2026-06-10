# GD vs GA for Urban Sound Classification
> Custom Smart Adaptive Systems — UPC 2025/26 | Lab Free Project

Comparative study of two MLP training strategies — **Gradient Descent (GD)** via backpropagation and a **Genetic Algorithm (GA)** — applied to urban sound classification using the [UrbanSound8K tabular dataset](https://www.kaggle.com/datasets/orvile/urban-sound-8k-tabular-form/data).

**Dataset:** 8674 pre-extracted audio feature vectors (MFCC, Chroma, Spectral Contrast, ZCR, Spectral Centroid), 34 features, 10 urban sound classes  
**Team:** [Name 1], [Name 2]

---

## Architecture

```
Input (34) → Linear(34 → 20) → ReLU → Linear(20 → 10) → logits
```

910 parameters total. ~6.7 training samples/parameter (70/10/20 split) — chosen to respect the bias-variance tradeoff.  
Same network used for both methods; only the training procedure differs.

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

Select the **Python (csas)** kernel when opening `main.ipynb`.

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

Open `main.ipynb` and run all cells from top to bottom.  
Figures are saved to `results/figures/` (created automatically on first run).

---

## Repo Structure

```
├── data/
│   └── extracted_audio_features.csv
├── docs/
│   ├── methodology.md
│   └── design_decisions.md
├── main.ipynb
├── requirements.txt
└── README.md
```

---

## Course Concepts

| Chapter | Applied as |
|---------|-----------|
| Ch. 2 | 34 pre-extracted features; mutual information importance ranking |
| Ch. 4 | MLP trained with Adam (GD/backprop) |
| Ch. 4 | Same MLP weights evolved with DEAP (GA) |
| Ch. 5 | Training time, convergence curves, hyperparameter sensitivity |

---

## References

- Salamon et al., "A Dataset and Taxonomy for Urban Sound Research", ACM MM 2014
- UrbanSound8K tabular dataset: https://www.kaggle.com/datasets/orvile/urban-sound-8k-tabular-form
- DEAP documentation: https://deap.readthedocs.io
- Bishop, "Pattern Recognition and Machine Learning", Springer 2006
- Floreano & Mattiussi, "Bio-inspired Artificial Intelligence", MIT Press 2008
