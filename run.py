import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from matplotlib.patches import Patch

import random, time, os

plt.rcParams.update({
    'figure.dpi': 130,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
})
GD   = '#2196F3'
GA   = '#FF5722'
SAVE = dict(dpi=150, bbox_inches='tight')

os.makedirs('results/figures', exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── 1 · Dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/extracted_audio_features.csv')
print(f'{df.shape[0]} samples  ·  {df.shape[1]-1} features  ·  {df["class"].nunique()} classes')

counts = df['class'].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(counts.index, counts.values,
               color=[('#FF5722' if v < 500 else '#2196F3') for v in counts.values],
               edgecolor='white', linewidth=0.5)
ax.set_xlabel('Number of samples')
ax.set_title('Class distribution  (red = underrepresented)')
ax.axvline(counts.mean(), color='grey', linestyle='--', linewidth=1, label=f'mean = {counts.mean():.0f}')
ax.legend(frameon=False)
for bar, v in zip(bars, counts.values):
    ax.text(v + 8, bar.get_y() + bar.get_height()/2, str(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('results/figures/class_distribution.png', **SAVE)
plt.close()

X_raw      = df.drop(columns=['class']).values.astype(np.float32)
le         = LabelEncoder()
y_raw      = le.fit_transform(df['class']).astype(np.int64)
feat_names = df.drop(columns=['class']).columns.tolist()
class_names = list(le.classes_)

mi    = mutual_info_classif(X_raw, y_raw, random_state=42)
mi_df = pd.Series(mi, index=feat_names).sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 5))
colors = [GD if 'MFCC' in n else ('#4CAF50' if 'Chroma' in n else '#9C27B0') for n in mi_df.index]
ax.barh(mi_df.index, mi_df.values, color=colors, edgecolor='white')
ax.set_xlabel('Mutual information score')
ax.set_title('Top-15 features by mutual information with class label')
ax.legend(handles=[
    Patch(color=GD,        label='MFCC'),
    Patch(color='#4CAF50', label='Chroma'),
    Patch(color='#9C27B0', label='Spectral'),
], frameon=False)
plt.tight_layout()
plt.savefig('results/figures/feature_importance.png', **SAVE)
plt.close()

X_scaled = StandardScaler().fit_transform(X_raw)
pca      = PCA(n_components=2, random_state=42)
X_pca    = pca.fit_transform(X_scaled)
palette  = plt.colormaps['tab10'].resampled(len(class_names))

fig, ax = plt.subplots(figsize=(9, 6))
for i, name in enumerate(class_names):
    mask = y_raw == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=7, alpha=0.45,
               color=palette(i), label=name, rasterized=True)
ax.set_xlabel(f'PC 1  ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC 2  ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('PCA — 34 features projected to 2D')
ax.legend(markerscale=2.5, fontsize=8, frameon=False, loc='upper right', ncol=2)
plt.tight_layout()
plt.savefig('results/figures/pca_2d.png', **SAVE)
plt.close()
print(f'Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%')

# ── 2 · Preprocessing ────────────────────────────────────────────────────────
X_tmp,   X_test,  y_tmp,   y_test  = train_test_split(X_raw,  y_raw,  test_size=0.2,   stratify=y_raw,  random_state=42)
X_train, X_val,   y_train, y_val   = train_test_split(X_tmp,  y_tmp,  test_size=0.125, stratify=y_tmp,  random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

X_train_t, y_train_t = torch.tensor(X_train), torch.tensor(y_train)
X_val_t,   y_val_t   = torch.tensor(X_val),   torch.tensor(y_val)
X_test_t,  y_test_t  = torch.tensor(X_test),  torch.tensor(y_test)

print(f'Train {X_train.shape[0]}  ·  Val {X_val.shape[0]}  ·  Test {X_test.shape[0]}')

# ── 3 · MLP ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=20, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


n_params = sum(p.numel() for p in MLP().parameters())
print(f'Parameters: {n_params}')

# ── 4 · Gradient Descent ─────────────────────────────────────────────────────
EPOCHS, LR, BATCH = 150, 1e-3, 64

gd_model  = MLP()
optimizer = torch.optim.Adam(gd_model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
loader    = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH, shuffle=True)

gd_loss, gd_train_acc, gd_val_acc = [], [], []

t0 = time.perf_counter()
for epoch in range(EPOCHS):
    gd_model.train()
    epoch_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(gd_model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    gd_loss.append(epoch_loss / len(X_train_t))
    gd_train_acc.append(accuracy(gd_model, X_train_t, y_train_t))
    gd_val_acc.append(accuracy(gd_model, X_val_t, y_val_t))
    if (epoch + 1) % 50 == 0:
        print(f'  GD epoch {epoch+1}/{EPOCHS}  loss={gd_loss[-1]:.4f}  val_acc={gd_val_acc[-1]:.4f}')

gd_time     = time.perf_counter() - t0
gd_test_acc = accuracy(gd_model, X_test_t, y_test_t)
print(f'GD  test accuracy: {gd_test_acc:.4f}   time: {gd_time:.1f}s')

epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(epochs, gd_loss, color=GD, linewidth=1.5)
axes[0].fill_between(epochs, gd_loss, alpha=0.08, color=GD)
axes[0].set_title('Training loss (cross-entropy)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[1].plot(epochs, gd_train_acc, color=GD, linewidth=1.5, label='Train')
axes[1].plot(epochs, gd_val_acc,   color=GD, linewidth=1.5, label='Validation', linestyle='--')
axes[1].axhline(gd_test_acc, color='grey', linestyle=':', linewidth=1, label=f'Test  {gd_test_acc:.3f}')
axes[1].set_title('Accuracy over epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1)
axes[1].legend(frameon=False)
plt.suptitle('Gradient Descent (Adam)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/gd_curves.png', **SAVE)
plt.close()

# ── 5 · Genetic Algorithm ────────────────────────────────────────────────────
from deap import algorithms as deap_algorithms, base, creator, tools
import torch.nn.utils as nn_utils

if not hasattr(creator, 'FitnessMax'):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
    creator.create('Individual', list, fitness=creator.FitnessMax)

def set_weights(model, weights):
    nn_utils.vector_to_parameters(torch.tensor(weights, dtype=torch.float32), model.parameters())

# ── 5a · Baseline GA (accuracy fitness, no elitism) ──────────────────────────
ga_model_orig = MLP()

def evaluate_orig(individual):
    set_weights(ga_model_orig, individual)
    return (accuracy(ga_model_orig, X_val_t, y_val_t),)

toolbox_orig = base.Toolbox()
toolbox_orig.register('attr_float',  random.gauss, 0.0, 0.1)
toolbox_orig.register('individual',  tools.initRepeat, creator.Individual, toolbox_orig.attr_float, n=n_params)
toolbox_orig.register('population',  tools.initRepeat, list, toolbox_orig.individual)
toolbox_orig.register('evaluate',    evaluate_orig)
toolbox_orig.register('mate',        tools.cxBlend,       alpha=0.5)
toolbox_orig.register('mutate',      tools.mutGaussian,   mu=0.0, sigma=0.05, indpb=0.05)
toolbox_orig.register('select',      tools.selTournament, tournsize=5)

hof_orig   = tools.HallOfFame(1)
stats_orig = tools.Statistics(lambda ind: ind.fitness.values)
stats_orig.register('max', np.max)
stats_orig.register('avg', np.mean)
stats_orig.register('std', np.std)

print('Running baseline GA...')
t0 = time.perf_counter()
_, logbook_orig = deap_algorithms.eaSimple(
    toolbox_orig.population(n=100), toolbox_orig,
    cxpb=0.7, mutpb=0.3, ngen=150,
    stats=stats_orig, halloffame=hof_orig, verbose=False,
)
ga_time_orig = time.perf_counter() - t0
set_weights(ga_model_orig, hof_orig[0])
ga_test_acc_orig = accuracy(ga_model_orig, X_test_t, y_test_t)
ga_best_acc_orig = np.array(logbook_orig.select('max'))
print(f'Baseline GA  test accuracy: {ga_test_acc_orig:.4f}   time: {ga_time_orig:.1f}s')

# ── 5b · Improved GA (-loss fitness + elitism) ───────────────────────────────
ga_model  = MLP()
ga_crit   = nn.CrossEntropyLoss()

# Improvement 1: fitness = -loss (continuous signal, not coarse accuracy)
def evaluate(individual):
    set_weights(ga_model, individual)
    ga_model.eval()
    with torch.no_grad():
        loss = ga_crit(ga_model(X_val_t), y_val_t).item()
    return (-loss,)

POP_SIZE  = 100
NGEN      = 150
CXPB      = 0.7
MUTPB     = 0.3
ELITE_K   = 5   # Improvement 2: elitism — top-5 survive every generation unchanged

toolbox = base.Toolbox()
toolbox.register('attr_float', random.gauss, 0.0, 0.1)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_params)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate)
toolbox.register('mate',     tools.cxBlend,       alpha=0.5)
toolbox.register('mutate',   tools.mutGaussian,   mu=0.0, sigma=0.05, indpb=0.05)
toolbox.register('select',   tools.selTournament, tournsize=5)

print(f'Population: {POP_SIZE}  |  Elitism: top-{ELITE_K}  |  Fitness: -loss')
print('Running GA (this takes a few minutes)...')

hof   = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('max', np.max)
stats.register('avg', np.mean)
stats.register('std', np.std)

pop = toolbox.population(n=POP_SIZE)
for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
    ind.fitness.values = fit
hof.update(pop)

logbook     = tools.Logbook()
logbook.header = ['gen'] + stats.fields
logbook.record(gen=0, **stats.compile(pop))

ga_best_acc = []   # track accuracy of best individual each generation for plotting

t0 = time.perf_counter()
for gen in range(1, NGEN + 1):
    # breed POP_SIZE - ELITE_K offspring
    offspring = toolbox.select(pop, POP_SIZE - ELITE_K)
    offspring = list(map(toolbox.clone, offspring))
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # evaluate only individuals whose fitness was invalidated
    invalid = [ind for ind in offspring if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit

    # elitism: top-ELITE_K carry over unchanged
    elite = list(map(toolbox.clone, tools.selBest(pop, ELITE_K)))
    pop[:] = offspring + elite

    hof.update(pop)
    logbook.record(gen=gen, **stats.compile(pop))

    # record best individual's accuracy (for comparison plot with GD)
    set_weights(ga_model, hof[0])
    ga_best_acc.append(accuracy(ga_model, X_val_t, y_val_t))

ga_time     = time.perf_counter() - t0
ga_test_acc = accuracy(ga_model, X_test_t, y_test_t)
print(f'GA  test accuracy: {ga_test_acc:.4f}   time: {ga_time:.1f}s')

# fitness is −loss; negate back to loss for a readable plot
ga_best_loss = -np.array(logbook.select('max'))
ga_avg_loss  = -np.array(logbook.select('avg'))
ga_std_loss  =  np.array(logbook.select('std'))
gens         = range(len(ga_best_loss))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

axes[0].plot(gens, ga_best_loss, color=GA, linewidth=2,   label='Best (Hall of Fame)')
axes[0].plot(gens, ga_avg_loss,  color=GA, linewidth=1.2, label='Population average', linestyle='--', alpha=0.7)
axes[0].fill_between(gens, ga_avg_loss - ga_std_loss, ga_avg_loss + ga_std_loss,
                     color=GA, alpha=0.12, label='±1 std')
axes[0].set_title('Improved GA — validation loss (fitness signal)')
axes[0].set_xlabel('Generation')
axes[0].set_ylabel('Cross-entropy loss')
axes[0].legend(frameon=False)

axes[1].plot(ga_best_acc_orig, color='#9E9E9E', linewidth=1.8,
             label=f'Baseline  (accuracy fitness, no elitism)  test={ga_test_acc_orig:.3f}',
             linestyle='--')
axes[1].plot(range(1, len(ga_best_acc)+1), ga_best_acc, color=GA, linewidth=2,
             label=f'Improved  (-loss fitness + elitism)  test={ga_test_acc:.3f}')
axes[1].axhline(ga_test_acc_orig, color='#9E9E9E', linestyle=':', linewidth=1)
axes[1].axhline(ga_test_acc,      color=GA,        linestyle=':', linewidth=1)
axes[1].set_title('Best individual accuracy — baseline vs improved')
axes[1].set_xlabel('Generation')
axes[1].set_ylabel('Validation accuracy')
axes[1].set_ylim(0, 1)
axes[1].legend(frameon=False, fontsize=9)

plt.suptitle('Genetic Algorithm — convergence', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/ga_convergence.png', **SAVE)
plt.close()

# ── 6 · Comparison ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, vals, ylabel, title in [
    (axes[0], [gd_test_acc, ga_test_acc], 'Test accuracy',      'Test accuracy'),
    (axes[1], [gd_time,     ga_time],     'Wall-clock time (s)', 'Training time'),
]:
    bars = ax.bar(['GD (Adam)', 'GA (DEAP)'], vals, color=[GD, GA], width=0.5, edgecolor='white')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if title == 'Test accuracy':
        ax.set_ylim(0, 1)
    for bar, v in zip(bars, vals):
        label = f'{v:.3f}' if title == 'Test accuracy' else f'{v:.1f}s'
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.02, label, ha='center', fontweight='bold')
plt.suptitle('GD vs GA — summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/comparison_summary.png', **SAVE)
plt.close()

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()
ax1.plot(gd_val_acc, color=GD, linewidth=2, label='GD — validation accuracy')
ax1.fill_between(range(len(gd_val_acc)), gd_val_acc, alpha=0.07, color=GD)
ax1.set_xlabel('Iteration  (epoch for GD · generation for GA)')
ax1.set_ylabel('GD validation accuracy', color=GD)
ax1.tick_params(axis='y', labelcolor=GD)
ax1.set_ylim(0, 1)
ax2.plot(ga_best_acc, color=GA, linewidth=2, label='GA — best val accuracy', linestyle='--')
ax2.fill_between(range(len(ga_best_acc)), ga_best_acc, alpha=0.07, color=GA)
ax2.set_ylabel('GA best fitness (val acc)', color=GA)
ax2.tick_params(axis='y', labelcolor=GA)
ax2.set_ylim(0, 1)
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, frameon=False, loc='lower right')
ax1.set_title('Convergence: GD vs GA')
plt.tight_layout()
plt.savefig('results/figures/comparison_convergence.png', **SAVE)
plt.close()

gd_model.eval()
ga_model.eval()
with torch.no_grad():
    gd_preds = gd_model(X_test_t).argmax(dim=1).numpy()
    ga_preds = ga_model(X_test_t).argmax(dim=1).numpy()
true = y_test_t.numpy()

fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
for ax, preds, title, cmap in zip(
    axes,
    [gd_preds, ga_preds],
    [f'GD (Adam)  —  test acc {gd_test_acc:.3f}', f'GA (DEAP)  —  test acc {ga_test_acc:.3f}'],
    ['Blues', 'Oranges'],
):
    cm = confusion_matrix(true, preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, ax=ax, linewidths=0.3,
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, annot_kws={'size': 7})
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=35)
plt.suptitle('Confusion matrices (row-normalised)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/comparison_confusion.png', **SAVE)
plt.close()

gd_f1 = f1_score(true, gd_preds, average=None)
ga_f1 = f1_score(true, ga_preds, average=None)
x = np.arange(len(class_names))
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - 0.2, gd_f1, 0.38, label='GD (Adam)', color=GD, edgecolor='white')
ax.bar(x + 0.2, ga_f1, 0.38, label='GA (DEAP)', color=GA, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, ha='right')
ax.set_ylabel('F1 score')
ax.set_ylim(0, 1.05)
ax.axhline(np.mean(gd_f1), color=GD, linestyle='--', linewidth=1, alpha=0.6)
ax.axhline(np.mean(ga_f1), color=GA, linestyle='--', linewidth=1, alpha=0.6)
ax.set_title('Per-class F1 score  (dashed = macro average)')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig('results/figures/comparison_f1.png', **SAVE)
plt.close()

print(f'Macro F1 — GD: {np.mean(gd_f1):.4f}   GA: {np.mean(ga_f1):.4f}')
print('Done — all figures saved to results/figures/')
