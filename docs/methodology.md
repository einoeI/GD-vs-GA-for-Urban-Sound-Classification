# Methodology

## 1. Dataset

The **UrbanSound8K tabular dataset** contains 8,674 pre-extracted audio feature vectors from 10 urban sound classes: air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, and street music.

Each sample is described by 34 numerical features grouped as follows:

| Group | Features | Count |
|-------|----------|-------|
| MFCC (Mel-Frequency Cepstral Coefficients) | MFCC_1 – MFCC_13 | 13 |
| Chroma (pitch class energy) | Chroma_1 – Chroma_12 | 12 |
| Spectral Contrast | SpectralContrast_1 – SpectralContrast_7 | 7 |
| Zero Crossing Rate | ZeroCrossingRate | 1 |
| Spectral Centroid | SpectralCentroid | 1 |

Pre-extracted features are used instead of raw audio because they reduce dimensionality from ~44,100 samples/second to 34 numbers per clip, making training tractable on a standard PC. The features are standard in audio ML literature (Salamon et al., 2014).

Two classes are underrepresented: `car_horn` (429 samples) and `gun_shot` (374 samples), compared to ~1,000 for the other eight classes.

---

## 2. Preprocessing

**Normalization:** Each of the 34 features is standardized to zero mean and unit variance using `StandardScaler`. This is necessary because features span very different ranges (e.g., MFCC values are in roughly [-200, 200] while ZCR is in [0, 1]). Without normalization, larger-magnitude features dominate the weight updates in GD, and the Gaussian mutation in GA is poorly scaled.

**Data split:** The 8,674 samples are split into train / validation / test sets using a **stratified** split:

| Split | Approx. samples | Purpose |
|-------|----------------|---------|
| Train | 6,072 (70%) | Parameter learning (GD) / fitness landscape (GA) |
| Validation | 867 (10%) | GA fitness evaluation; GD early convergence monitoring |
| Test | 1,734 (20%) | Final unbiased accuracy reported for both methods |

Stratification ensures that the underrepresented classes (`car_horn`, `gun_shot`) appear in all three splits at the same ratio as the full dataset.

The scaler is fitted on the training set only, then applied to the validation and test sets. This prevents data leakage from val/test distributions into the normalization statistics.

---

## 3. MLP Architecture

Both GD and GA train the same network:

```
Input (34) → Linear(34 → 20) → ReLU → Linear(20 → 10) → [Softmax at inference]
```

**Parameter count:**
- Layer 1: 34 × 20 + 20 = 700
- Layer 2: 20 × 10 + 10 = 210
- **Total: 910 parameters**

**Bias-variance tradeoff:** With ~6,072 training samples and 910 parameters, the ratio is ~6.7 samples per parameter. This is within the commonly cited 5–10× guideline. A larger hidden layer (e.g., 50 neurons → 2,560 parameters → ~2.4× ratio) would risk overfitting. A smaller one (e.g., 10 neurons → 460 parameters) risks underfitting. Hidden=20 is the choice that keeps this ratio within the 5–10× guideline while leaving sufficient capacity for the 10-class task.

`CrossEntropyLoss` (used for GD) internally applies log-softmax to the raw logits, so softmax is only applied explicitly during inference for probability outputs.

---

## 4. Gradient Descent Training

The MLP is trained with the **Adam optimizer** over 150 epochs with mini-batches of size 64.

At each epoch:
1. Iterate over mini-batches of the training set with random shuffling
2. Compute `CrossEntropyLoss` on the mini-batch logits
3. Back-propagate gradients through both layers
4. Update all 910 parameters via Adam's adaptive learning rate rule
5. Record training loss, training accuracy, and validation accuracy (full-set, no gradient computation)

Training time is measured with `time.perf_counter()` and excludes data loading. Expected wall time: 3–5 seconds on a modern CPU.

The final test accuracy is computed once after training completes, on the held-out test set.

---

## 5. Genetic Algorithm Training

The GA treats the MLP as a **black box** — it does not use gradients. Instead, the 910 weight values are encoded as a flat list of real numbers (the **chromosome**), and a population of such chromosomes is evolved over generations.

**Chromosome encoding:** The 910 floats are laid out in the order: `fc1.weight` (680), `fc1.bias` (20), `fc2.weight` (200), `fc2.bias` (10). This ordering is fixed by PyTorch's `parameters_to_vector` / `vector_to_parameters` utilities and is used consistently throughout.

**Fitness function:** A chromosome is evaluated by injecting its weights into the MLP and measuring **validation set accuracy** (867 samples). Validation accuracy is used instead of training accuracy to reduce overfitting pressure, and instead of the validation loss because:
- Accuracy is the target metric for comparison with GD
- The GA does not require differentiability of the objective

**Evolutionary operators:**

| Stage | Operator | Parameters |
|-------|----------|------------|
| Initialization | Gaussian(μ=0, σ=0.1) per gene | Matches PyTorch's Kaiming initialization scale |
| Selection | Tournament selection | Tournament size 5 |
| Crossover | Blend crossover (BLX-α) | α = 0.5 |
| Mutation | Gaussian perturbation | μ=0, σ=0.05, gene-wise prob = 0.05 |

**Default settings:** 100 individuals, 150 generations. The best individual ever seen (tracked by `HallOfFame`) is loaded back into the model after evolution; this prevents the case where the final population is weaker than a previously seen individual.

**DEAP's `eaSimple`** handles the generational loop: each generation, parents are selected (with replacement), offspring are produced by crossover and/or mutation, and the full population is replaced by offspring. Fitness values are invalidated after genetic operations so only unmodified individuals carry over their cached fitness.

---

## 6. Evaluation

Both methods are evaluated on the same held-out test set (1,734 samples) with:

- **Overall accuracy** — fraction of correctly classified samples
- **Per-class F1 score** — harmonic mean of precision and recall per class; important given class imbalance
- **Confusion matrix** — normalized (row-wise) 10×10 matrix showing class-level error patterns
- **Training wall-clock time** — total time from start to final test evaluation

The comparison section of `main.ipynb` presents all of these side by side and includes a dual-axis convergence plot that aligns GD epochs with GA generations on the x-axis (note: these are not equivalent in compute cost — 1 GA generation evaluates the entire population).
