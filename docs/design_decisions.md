# Design Decisions

This document argues the key technical choices made in the implementation.

---

## 1. PyTorch over scikit-learn `MLPClassifier`

**Choice:** PyTorch `nn.Module`

**Alternative:** `sklearn.neural_network.MLPClassifier`

**Argument:**  
The GA requires direct read/write access to the network's weight vector. PyTorch provides `torch.nn.utils.parameters_to_vector` and `vector_to_parameters`, which extract/inject all 910 parameters as a single flat tensor in microseconds with deterministic ordering. scikit-learn's `MLPClassifier` exposes weights only through `coefs_` and `intercepts_` (separate arrays per layer), making it possible but cumbersome to assemble and reassemble the full vector on each fitness evaluation. More critically, sklearn's internal forward pass cannot be called without also triggering scikit-learn's training infrastructure, adding unnecessary overhead to every GA fitness evaluation.

---

## 2. DEAP over PyGAD or a custom GA loop

**Choice:** DEAP (`deap` library)

**Alternatives:** PyGAD, pure Python manual GA loop

**Argument:**  
DEAP is referenced in the course materials (Chapter 4). It has a clean operator registry (`toolbox.register`) that separates configuration from execution, making ablation studies trivial — each ablation only changes one `register` call. `algorithms.eaSimple` correctly handles the subtle bookkeeping of fitness invalidation after genetic operations (missing `del ind.fitness.values` is a common bug in manual loops that causes stale fitness values to be carried over). PyGAD is simpler but less flexible for custom operators.

---

## 3. Flat real-valued chromosome encoding

**Choice:** 910 floating-point values, same order as `parameters_to_vector`

**Alternatives:** Layer-wise separate chromosomes, integer encoding with fixed-precision scaling, binary encoding

**Argument:**  
A flat real-valued encoding is the standard for neuroevolution of small MLPs (e.g., NEAT, direct encoding). It maps bijectively to the PyTorch parameter space with no information loss. Integer or binary encodings would require a decode step before weight injection and reduce precision unnecessarily. Layer-wise chromosomes would complicate crossover (BLX-α would need to know layer boundaries) without benefit. The fixed ordering from `parameters_to_vector` ensures consistent gene semantics across all individuals and generations.

---

## 4. Validation accuracy as fitness metric (not training loss)

**Choice:** Validation accuracy (evaluated on 867 validation samples)

**Alternatives:** Training accuracy, cross-entropy loss on training set, combined train+val set

**Argument:**  
Using training accuracy as fitness would create selection pressure to memorise training samples — the GA would exploit any overfitting opportunity since it has no regularisation by construction. Validation accuracy provides a natural generalisation signal. Cross-entropy loss would also work, but accuracy is the target metric for comparison with GD and is more interpretable in convergence plots. Using the full training set (6,072 samples) for fitness would make each generation ~7× slower with no meaningful benefit over the validation set given the stratified split.

---

## 5. Adam optimizer for GD

**Choice:** Adam with lr=1e-3, weight_decay=1e-4

**Alternatives:** SGD with momentum, RMSprop, SGD with learning rate scheduling

**Argument:**  
Adam adapts the learning rate per parameter, making it robust to the choice of global learning rate. For a 910-parameter MLP on a classification task, Adam consistently converges faster than SGD in the first 100 epochs. SGD with tuned momentum and a decay schedule can match Adam given enough tuning effort, but Adam's defaults generalize well across MLP tasks without that effort. 1e-3 is the standard Adam default and consistently works well for small MLP classification tasks without per-task tuning.

---

## 6. Hidden layer size: 20 neurons

**Choice:** 20 hidden neurons (910 total parameters)

**Alternatives:** 10 (460 params), 30 (1,360 params), 50 (2,560 params), deeper network

**Argument:**  
The professor's feedback specifically flags the bias-variance tradeoff for this dataset. With ~6,072 training samples, the samples-per-parameter ratio at different hidden sizes is:

| Hidden | Params | Samples/param |
|--------|--------|---------------|
| 10 | 460 | 13.2 |
| **20** | **910** | **6.7** |
| 30 | 1,360 | 4.5 |
| 50 | 2,560 | 2.4 |

Hidden=20 sits within the 5–10× guideline. Hidden=30 drops to 4.5× which risks overfitting, especially for the GA which has no explicit regularisation. A single hidden layer is sufficient given the problem complexity (34 tabular features, 10 classes); deeper networks would require more data to avoid overfitting.

---

## 7. Tournament selection (size=5)

**Choice:** `tools.selTournament(tournsize=5)`

**Alternatives:** Roulette wheel (fitness-proportionate), rank-based, truncation selection

**Argument:**  
Tournament selection is robust to fitness scaling: it only compares individuals within each tournament, so outlier fitness values (common early in GA when weights are random) do not dominate selection pressure as they would in roulette wheel selection. A tournament size of 5 out of 100 individuals provides moderate selection pressure — enough to favour better individuals without eliminating diversity too quickly. Roulette wheel selection is sensitive to negative fitness values and requires fitness scaling; tournament does not.

---

## 8. Blend crossover (BLX-α = 0.5)

**Choice:** `tools.cxBlend(alpha=0.5)`

**Alternatives:** Uniform crossover, single-point crossover, simulated binary crossover (SBX), arithmetic mean crossover

**Argument:**  
Single-point and uniform crossover were designed for binary strings and treat genes as independent. In a real-valued weight space, genes within the same layer are structurally correlated — swapping them at a random cut point can produce offspring that are no longer "close" to either parent in any meaningful sense. Blend crossover generates offspring uniformly in `[min(p1,p2) - α·range, max(p1,p2) + α·range]` per gene, which (a) explores the region between parents and (b) extends slightly beyond them, providing exploration without pure noise. α=0.5 is the standard default and is well suited to real-valued weight-space search.

---

## 9. Gaussian mutation (σ=0.05, indpb=0.05)

**Choice:** `tools.mutGaussian(mu=0.0, sigma=0.05, indpb=0.05)`

**Alternatives:** Uniform mutation, polynomial mutation, larger/smaller σ

**Argument:**  
Gaussian perturbations are the natural choice for real-valued weight-space search — they explore locally around the current value, matching the intuition that good solutions cluster in smooth regions of weight space. σ=0.05 is chosen to match the scale of weight values after Kaiming initialization (weights are typically in [-0.3, 0.3] for this architecture). A per-gene mutation probability of `indpb=0.05` means that on average 0.05 × 910 ≈ 46 of the 910 genes are perturbed per individual per generation — enough to maintain diversity without destroying well-adapted individuals.

---

## 10. Stratified split for class imbalance

**Choice:** `train_test_split(..., stratify=y)` for both splits

**Alternatives:** SMOTE oversampling, class weights in loss function, ignore imbalance

**Argument:**  
Stratified splitting ensures each of the 10 classes is represented at the same proportion in train, val, and test. This is the minimum required to ensure that the test accuracy is a fair estimate of population-level performance for all classes, including the underrepresented `car_horn` and `gun_shot`. SMOTE would introduce synthetic samples into the training set but would not affect the test set evaluation. Class weights in `CrossEntropyLoss` are an optional enhancement for GD but are not applicable to the GA (which uses accuracy as fitness). The per-class F1 scores in the comparison section of `main.ipynb` expose any residual imbalance effects regardless.

---

## 11. GA fitness on validation set, not training set

**Choice:** Evaluate GA fitness on 867 validation samples per individual

**Alternative:** Evaluate on all 6,072 training samples

**Argument:**  
Each generation evaluates the full population: `pop_size × fitness_evaluations` forward passes. With 100 individuals and 150 generations, that is 15,000 evaluations. Evaluating on 6,072 training samples instead of 867 val samples would make each evaluation ~7× slower, pushing total GA runtime from ~5 minutes to ~35 minutes on a standard laptop with no meaningful improvement in result quality. The validation set is a stratified subset that accurately reflects the class distribution, so fitness signals from it are representative.

---

## 12. Anaconda as the Python environment manager

**Choice:** Anaconda (`conda create -n csas python=3.12`)

**Alternatives:** plain `venv`, Poetry, Docker

**Argument:**  
Anaconda resolves binary dependencies (numpy, torch, scipy) using pre-compiled conda packages, which avoids compiler toolchain issues on Windows that can arise with plain `pip install` on certain versions of PyTorch or scipy. It also makes the environment trivially reproducible across machines (`conda activate csas` + `pip install -r requirements.txt`). `venv` would work but requires more care with the PyTorch install command (which varies by CUDA version); conda handles the CPU-only variant cleanly with a single `pip install torch`. Poetry adds dependency resolution overhead that is not needed for a self-contained research project.
