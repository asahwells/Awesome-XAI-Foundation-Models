# 🔍 Awesome Foundation Models for eXplainable AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/YOUR-USERNAME/Awesome-XAI-Foundation-Models/pulls)

Foundation models such as CLIP, DINO, and SAM have transformed computer vision, but their internal representations remain largely opaque. Existing eXplainable AI (xAI) methods — LIME, SHAP, Grad-CAM — explain *individual predictions*, yet say little about *what a model has actually encoded* in its learned latent space.

This repository curates **foundation models, xAI methods, datasets, and tools for representation-level explainability**, and provides a hands-on tutorial for **SVD-LTE (Latent-Space Top-Eigenvector Analysis)** — a unified global explanation framework that shifts explainability from prediction-space to representation-space.

> **🎓 Tutorial**: [SVD-LTE on CLIP](#-tutorial-svd-lte-on-clip) — a complete, runnable walkthrough applying SVD-LTE to a pretrained CLIP ViT-B/32 encoder.

---

## 🗺️ Landscape of Explainability for Foundation Models

The field can be organised into three converging directions:

1. **Prediction-Level xAI**: Explaining *why a model produced a specific output* for a specific input (LIME, SHAP, Grad-CAM, Integrated Gradients)
2. **Representation-Level xAI**: Explaining *what information is encoded in the learned latent manifold*, how it evolves, and whether it carries bias or drift vulnerabilities (SVD-LTE, TCAV, probing classifiers, sparse autoencoders)
3. **Evaluation of Explanations**: Quantifying whether explanations are faithful, stable, and useful (faithfulness metrics, sanity checks, human-grounded studies)

SVD-LTE sits in direction **(2)** and contributes measurable metrics to direction **(3)**.

---

## 📌 Table of Contents

- [🗺️ Landscape](#️-landscape-of-explainability-for-foundation-models)
- [📚 Surveys & Reviews](#-surveys--reviews)
- [🎓 Tutorial: SVD-LTE on CLIP](#-tutorial-svd-lte-on-clip)
  * [Why SVD-LTE](#why-svd-lte)
  * [Quick Start](#quick-start)
  * [Step 1 — Load a Pretrained Foundation Model](#step-1--load-a-pretrained-foundation-model)
  * [Step 2 — Extract the Latent Space](#step-2--extract-the-latent-space)
  * [Step 3 — Find the Top Eigenvectors with SVD](#step-3--find-the-top-eigenvectors-with-svd)
  * [Step 4 — Fairness: Do PCs Encode Protected Attributes?](#step-4--fairness-do-pcs-encode-protected-attributes)
  * [Step 5 — Robustness: Domain-Shift Detection](#step-5--robustness-domain-shift-detection)
  * [Step 6 — Stability: Are the PCs Reliable?](#step-6--stability-are-the-pcs-reliable)
  * [Step 7 — Attribution: What Does Each PC Look For?](#step-7--attribution-what-does-each-pc-look-for)
  * [Step 8 — Explainability: Attribution Entropy](#step-8--explainability-attribution-entropy)
- [📊 Results: CLIP ViT-B/32 on CIFAR-10](#-results-clip-vit-b32-on-cifar-10)
- [🔬 Datasets, Benchmarks & Tools](#-datasets-benchmarks--tools)
- [🧩 Prediction-Level xAI Methods](#-prediction-level-xai-methods)
- [🌐 Representation-Level xAI Methods](#-representation-level-xai-methods)
- [🏗️ Foundation Models](#️-foundation-models)
- [‼️ Open Challenges](#️-open-challenges)
- [🤝 Contributing](#-contributing)
- [📄 Citation](#-citation)

---

## 📚 Surveys & Reviews

- **[Intelligent Systems with Applications, 2026]** Explaining explainability: A comprehensive survey on explainable artificial intelligence and relevant industry applications [[paper]](https://doi.org/10.1016/j.iswa.2026.200647)
- **[Information Fusion, 2020]** Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI [[paper]](https://doi.org/10.1016/j.inffus.2019.12.012)
- **[IEEE Access, 2018]** Peeking inside the black-box: A survey on explainable artificial intelligence (XAI) [[paper]](https://doi.org/10.1109/ACCESS.2018.2870052)
- **[Artificial Intelligence Review, 2022]** Explainable artificial intelligence: a comprehensive review [[paper]](https://doi.org/10.1007/s10462-021-10088-y)
- **[Arxiv, 2024]** Explainable artificial intelligence: A survey of needs, techniques, applications, and future direction [[paper]](https://doi.org/10.1016/j.neucom.2024.128111)
- **[Arxiv, 2017]** Towards a rigorous science of interpretable machine learning [[paper]](https://arxiv.org/abs/1702.08608)

---

## 🎓 Tutorial: SVD-LTE on CLIP

### Why SVD-LTE

Current xAI faces five well-documented limitations. SVD-LTE addresses each:

| # | Challenge in current xAI | SVD-LTE's response |
|---|--------------------------|--------------------|
| 1 | **Local, not global** — LIME/SHAP/Grad-CAM explain one sample at a time | Explains the *whole latent manifold* via principal directions learned from thousands of samples |
| 2 | **No representation-level explainability** — few methods explain latent embeddings | Explicitly explains latent factors, dimensions, and structure via principal directions, prototypes, and attribution maps |
| 3 | **No standardised evaluation metrics** | Introduces quantitative metrics: Fairness (AUC per PC, LDS, Mahalanobis Distance); Robustness (DSR, Latent Drift Vector); Stability (Cross-Epoch Alignment, Jacobian Norm); Explainability (Attribution Entropy) |
| 4 | **Explanation instability** — LIME variance, SHAP variance, gradient noise | Principal directions are estimated from thousands of samples and are statistically far more stable; stability is itself measured |
| 5 | **Explainability–accuracy trade-off** — interpretable models sacrifice performance | Fully post-training. No retraining, no architecture change. Accuracy unchanged; explainability added afterwards |

### Quick Start

```bash
pip install open_clip_torch torch torchvision numpy scipy scikit-learn matplotlib seaborn
```

Or open the tutorial directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/YOUR-NOTEBOOK-LINK)

> **Note**: a GPU runtime is strongly recommended. Extracting latents for 10,000 images on CPU is slow.

---

### Step 1 — Load a Pretrained Foundation Model

SVD-LTE is model-agnostic. Any encoder exposing a latent vector works. Here we use CLIP ViT-B/32.

```python
import torch
import torch.nn as nn
import open_clip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model = clip_model.to(DEVICE).eval()

class CLIPEncoder(nn.Module):
    """Thin wrapper exposing the SVD-LTE interface: model(x, return_latent=True)."""
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.latent_dim = 512

    def forward(self, x, return_latent=False):
        # Gradients must flow — attribution methods depend on them.
        h = self.clip_model.encode_image(x)
        return (None, h) if return_latent else h

model = CLIPEncoder(clip_model).to(DEVICE)
```

> **Swapping the backbone**: to analyse a different foundation model, replace only this cell. Everything downstream is unchanged — that is the point of a model-agnostic framework.

---

### Step 2 — Extract the Latent Space

Each image becomes a single vector. 10,000 images → a `(10000, 512)` matrix. This matrix *is* the latent space.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

test_ds = datasets.CIFAR10(root="cifar10", train=False,
                           transform=clip_transform, download=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

def get_latent(model, device, loader):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            _, h = model(xb.to(device), return_latent=True)
            latents.append(h.cpu().float().numpy())
            labels.append(yb.numpy())
    return np.vstack(latents), np.concatenate(labels)

Z, y = get_latent(model, DEVICE, test_loader)
print(Z.shape)   # (10000, 512)
```

---

### Step 3 — Find the Top Eigenvectors with SVD

SVD finds the directions along which the latent space varies most. These directions are the **principal components (PCs)** — the dominant patterns the model has encoded.

```python
def compute_pcs(Z, k):
    """Return the top-k principal directions and their singular values."""
    Zc = Z - Z.mean(axis=0, keepdims=True)     # centre the data
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:k].T, S[:k]                      # V: (D, k), S: (k,)

K_PCS = 10
pcs_final, final_svals = compute_pcs(Z, K_PCS)

# Variance explained per PC
Zc = Z - Z.mean(axis=0, keepdims=True)
_, S_full, _ = np.linalg.svd(Zc, full_matrices=False)
explained = (S_full ** 2) / (S_full ** 2).sum()
for i, e in enumerate(explained[:K_PCS]):
    print(f"PC{i}: {e*100:.2f}%")
```

---

### Step 4 — Fairness: Do PCs Encode Protected Attributes?

Define two groups and ask whether any single PC separates them. If a PC separates groups the model was never told about, that group structure is **encoded in the representation**.

In this tutorial the groups are semantic (animals vs vehicles). In a fairness audit they would be protected attributes (e.g. perceived gender, age, skin tone).

```python
from sklearn.metrics import roc_auc_score
from scipy import linalg

ANIMAL_CLASSES = {2, 3, 4, 5, 6, 7}          # bird, cat, deer, dog, frog, horse
mask_A = np.array([lbl in ANIMAL_CLASSES for lbl in y])
A, B = Z[mask_A], Z[~mask_A]

# --- AUC per PC ---
def auc_for_pc(A, B, v):
    sA, sB = A @ v, B @ v
    labels = np.concatenate([np.zeros(len(sA)), np.ones(len(sB))])
    return roc_auc_score(labels, np.concatenate([sA, sB]))

aucs = np.array([auc_for_pc(A, B, pcs_final[:, k]) for k in range(K_PCS)])

# --- S_k separation and Latent Discrimination Score (LDS) ---
def compute_Sk(A, B, Z, v):
    pA, pB, pZ = A @ v, B @ v, Z @ v
    sigma = pZ.std()
    return 0.0 if sigma < 1e-12 else abs(pA.mean() - pB.mean()) / sigma

S_k = np.array([compute_Sk(A, B, Z, pcs_final[:, k]) for k in range(K_PCS)])
LDS = float(np.nansum(S_k))

# --- Mahalanobis distance between group centroids ---
def mahalanobis_groups(A, B, reg=1e-6):
    diff = A.mean(0) - B.mean(0)
    cov = np.cov(np.vstack([A, B]), rowvar=False) + reg * np.eye(A.shape[1])
    return float(np.sqrt(diff @ linalg.pinv(cov) @ diff))

MD = mahalanobis_groups(A, B)
print(f"AUC per PC: {aucs.round(3)}\nLDS: {LDS:.4f}\nMahalanobis: {MD:.4f}")
```

**Reading the numbers**: AUC ≈ 0.5 → the PC carries no group information. AUC → 1.0 → the PC almost perfectly separates the groups, meaning the concept is strongly encoded.

---

### Step 5 — Robustness: Domain-Shift Detection

Perturb the inputs, recompute the latents, and measure how far the representation moved — and *which* directions absorbed the movement.

```python
def make_shifted_loader(loader, noise_std=0.20):
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb + torch.randn_like(xb) * noise_std)
        ys.append(yb)
    ds = torch.utils.data.TensorDataset(torch.cat(xs), torch.cat(ys))
    return DataLoader(ds, batch_size=128, shuffle=False)

Z_clean, _ = get_latent(model, DEVICE, test_loader)
Z_shift, _ = get_latent(model, DEVICE, make_shifted_loader(test_loader))

# Latent Drift Vector
delta_vec = Z_shift.mean(0) - Z_clean.mean(0)
delta_dir = delta_vec / (np.linalg.norm(delta_vec) + 1e-8)

# Which PCs absorbed the drift?
shift_proj = np.array([delta_dir @ pcs_final[:, k] for k in range(K_PCS)])

# Domain-Shift Response: did each PC survive the perturbation?
cov_shift = np.cov(Z_shift.T)
U_shift, _, _ = np.linalg.svd(cov_shift)
DSR = np.array([pcs_final[:, k] @ U_shift[:, k] for k in range(K_PCS)])
```

**Reading the numbers**: DSR near ±1 → that PC survived the shift (robust). DSR near 0 → that PC collapsed (fragile). `shift_proj` localises *where* the drift landed.

---

### Step 6 — Stability: Are the PCs Reliable?

Run SVD on several independent subsets. If PC0 is genuinely a property of the model — not an artefact of which images you sampled — it should reappear each time.

```python
N_RUNS, SUBSET = 5, 2000
pcs_list = []
for _ in range(N_RUNS):
    idx = np.random.choice(len(Z), SUBSET, replace=False)
    V, _ = compute_pcs(Z[idx], K_PCS)
    pcs_list.append(V.astype(np.float32))
pcs_over_time = np.stack(pcs_list, axis=0)          # (runs, D, K)

# Cross-run alignment: |cos| between the same PC across consecutive runs
C_list = [pcs_over_time[t].T @ pcs_over_time[t-1] for t in range(1, N_RUNS)]
pc_stability = np.stack([np.abs(np.diag(C)) for C in C_list]).mean(axis=0)
pc_instability = 1.0 - pc_stability
```

**Reading the numbers**: stability → 1.0 means the direction is essentially identical across runs. Lower values mean the direction wanders and should be trusted less.

---

### Step 7 — Attribution: What Does Each PC Look For?

Project the latent onto a PC to get a single score, then ask which input pixels drive that score. Three methods of increasing robustness are provided as a cross-check.

```python
def pc_tensor(v):
    return torch.tensor(v, dtype=torch.float32, device=DEVICE)

def reduce_to_2d(t):
    return t.mean(dim=0) if t.ndim == 3 else t

def signed_gradient_saliency(model, pc_vec, x):
    """One gradient measurement at the input point."""
    x = x.clone().to(DEVICE).requires_grad_(True)
    _, h = model(x, return_latent=True)
    (h @ pc_tensor(pc_vec)).sum().backward()
    return reduce_to_2d(x.grad.detach().cpu()[0]).numpy()

def integrated_gradients(model, pc_vec, x, steps=30):
    """Average gradients along a path from a black baseline to the input."""
    x = x.clone().to(DEVICE)
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for i in range(1, steps + 1):
        xi = (baseline + (i / steps) * (x - baseline)).requires_grad_(True)
        _, h = model(xi, return_latent=True)
        (h @ pc_tensor(pc_vec)).sum().backward()
        total += xi.grad.detach()
    return reduce_to_2d(((x - baseline) * (total / steps)).cpu()[0]).numpy()

def gradient_shap_like(model, pc_vec, x, n_samples=30):
    """Average gradients over randomly sampled interpolation points."""
    x = x.clone().to(DEVICE)
    total = torch.zeros_like(x)
    for _ in range(n_samples):
        xi = (torch.rand(1, 1, 1, 1, device=DEVICE) * x).requires_grad_(True)
        _, h = model(xi, return_latent=True)
        (h @ pc_tensor(pc_vec)).sum().backward()
        total += xi.grad.detach()
    return reduce_to_2d((total / n_samples).cpu()[0]).numpy()

def optimize_prototype(model, pc_vec, shape, steps=200, lr=0.05):
    """Synthesise the image that maximally activates a PC."""
    x = torch.randn(shape, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        _, h = model(x, return_latent=True)
        (-(h @ pc_tensor(pc_vec)).sum()).backward()
        opt.step()
    return x.detach().cpu()[0].mean(dim=0).numpy()   # average RGB → viewable
```

---

### Step 8 — Explainability: Attribution Entropy

Entropy measures whether an explanation is **concentrated** (low) or **diffuse** (high).

```python
def attribution_entropy(a):
    a = np.abs(a).astype(np.float64).flatten()
    a = a / (a.sum() + 1e-8)
    return -(a * np.log(a + 1e-8)).sum()
```

**Calibrating the scale**: for a 224 × 224 map, perfectly uniform attribution gives `ln(50176) ≈ 10.82`. A value near 10.8 means the explanation is spread across the entire image; substantially lower means it is localised.

---

## 📊 Results: CLIP ViT-B/32 on CIFAR-10

**Setup**: CLIP ViT-B/32 (`openai` weights, frozen) · CIFAR-10 test split (10,000 images) · latent dim 512 · K = 10 PCs · groups: animals (6,000) vs vehicles (4,000).

### Fairness — a concept is strongly encoded

| PC | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|----|---|---|---|---|---|---|---|---|---|---|
| **AUC** | **0.978** | 0.690 | 0.503 | 0.487 | 0.488 | 0.502 | 0.523 | 0.581 | 0.497 | 0.504 |

**LDS = 2.9077** · **Mahalanobis distance = 1.9695**

PC0 separates animals from vehicles at **AUC 0.978**. CLIP was never given this grouping, yet its single most dominant latent direction encodes it almost perfectly. Prediction-level xAI cannot surface a finding of this kind.

### Spectrum — no single direction dominates

| PC | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|----|---|---|---|---|---|---|---|---|---|---|
| **Variance explained** | 13.66% | 7.94% | 5.58% | 4.74% | 3.68% | 3.28% | 3.00% | 2.56% | 2.21% | 2.06% |

The top 10 PCs jointly explain **48.7%** of total variance — the representation is rich and distributed rather than collapsed onto one axis.

### Robustness — the important directions survive

Gaussian perturbation, σ = 0.20:

| PC | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|----|---|---|---|---|---|---|---|---|---|---|
| **DSR** | 0.82 | 0.64 | −0.65 | −0.53 | 0.18 | −0.36 | 0.013 | 0.40 | 0.14 | 0.15 |

**Severity score = 0.072** (low) · **dominant drift PC = 7** (projection −0.244)

The most important directions (PC0, PC1) remain robust, while several lower-ranked directions reorganise. SVD-LTE localises *which* directions are fragile — not merely whether the model is robust overall.

### Stability — top directions are reproducible

PC0–PC4 achieve cross-run alignment of roughly **0.9–1.0**, tapering to ~0.75 for the lowest-ranked components. The dominant patterns are properties of the model, not of the sample.

### Explainability — a ViT signature

Attribution entropy across all PCs: **9.96 – 10.21** (theoretical maximum 10.82). Attribution is diffuse for *every* component, consistent with ViT patch tokenisation distributing representation across spatial tokens. PC7 is highest (10.21), matching its high-frequency texture prototype; PC1 is lowest (9.96).

### Metric independence — each metric earns its place

| Pair | Pearson | Spearman |
|------|---------|----------|
| Entropy × Instability | −0.084 | 0.152 |
| Entropy × Prototype norm | 0.114 | 0.018 |
| Instability × Prototype norm | 0.129 | 0.224 |

All |r| < 0.13. Stability, interpretability, and semantic strength vary **independently** — so each metric family captures a distinct, non-redundant property of the representation. This is evidence that the framework is not over-specified.

> **Limitations**: attribution maps are computed on a single representative image per PC and are illustrative rather than population-level. Correlations are estimated from K = 10 components, so they indicate *no evidence of a relationship* rather than proven independence. Results are reported for one backbone and one dataset.

---

## 🔬 Datasets, Benchmarks & Tools

### Datasets used in representation-level analysis

| Dataset | Modality | Scale | Typical use | Link |
|---------|----------|-------|-------------|------|
| **CIFAR-10** | `Image` | 60K | Semantic structure, quick iteration | [[dataset]](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **CIFAR-100** | `Image` | 60K | Finer-grained concept structure | [[dataset]](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **CelebA** | `Image` | 200K | Attribute and fairness auditing | [[dataset]](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| **FairFace** | `Image` | 108K | Balanced demographic auditing | [[dataset]](https://github.com/joojs/fairface) |
| **ImageNet** | `Image` | 1.2M | Large-scale semantic structure | [[dataset]](https://www.image-net.org/) |
| **Waterbirds** | `Image` | 11K | Spurious correlation / background bias | [[dataset]](https://github.com/kohpangwei/group_DRO) |

### Tools

| Tool | Purpose | Link |
|------|---------|------|
| **open_clip** | Open CLIP implementations and pretrained weights | [[tool]](https://github.com/mlfoundations/open_clip) |
| **Captum** | Model interpretability library for PyTorch | [[tool]](https://captum.ai/) |
| **SHAP** | Shapley-value attribution | [[tool]](https://github.com/shap/shap) |
| **LIME** | Local surrogate explanations | [[tool]](https://github.com/marcotcr/lime) |
| **Quantus** | Quantitative evaluation of explanations | [[tool]](https://github.com/understandable-machine-intelligence-lab/Quantus) |
| **timm** | Pretrained image model zoo | [[tool]](https://github.com/huggingface/pytorch-image-models) |

---

## 🧩 Prediction-Level xAI Methods

*Explaining individual model outputs. Local by construction.*

- **[ACM SIGKDD, 2016]** "Why should I trust you?": Explaining the predictions of any classifier (**LIME**) [[paper]](https://arxiv.org/abs/1602.04938) [[code]](https://github.com/marcotcr/lime)
- **[NeurIPS, 2017]** A unified approach to interpreting model predictions (**SHAP**) [[paper]](https://arxiv.org/abs/1705.07874) [[code]](https://github.com/shap/shap)
- **[ICML, 2017]** Axiomatic attribution for deep networks (**Integrated Gradients**) [[paper]](https://arxiv.org/abs/1703.01365)
- **[IJCV, 2019]** Grad-CAM: Visual explanations from deep networks via gradient-based localization [[paper]](https://arxiv.org/abs/1610.02391)
- **[AAAI, 2018]** Anchors: High-precision model-agnostic explanations [[paper]](https://doi.org/10.1609/aaai.v32i1.11491)
- **[PLOS ONE, 2015]** On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation (**LRP**) [[paper]](https://doi.org/10.1371/journal.pone.0130140)
- **[Annals of Statistics, 2001]** Greedy function approximation: A gradient boosting machine (**PDP**) [[paper]](https://doi.org/10.1214/aos/1013203451)
- **[JCGS, 2015]** Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation (**ICE**) [[paper]](https://doi.org/10.1080/10618600.2014.907095)
- **[Arxiv, 2017]** Counterfactual explanations without opening the black box [[paper]](https://arxiv.org/abs/1711.00399)

---

## 🌐 Representation-Level xAI Methods

*Explaining what is encoded in the latent manifold. Global by construction.*

- **SVD-LTE (Latent-Space Top-Eigenvector Analysis)** — unified global explanation via top eigenvectors of the latent space, with quantitative fairness, robustness, stability, and explainability metrics. **[This repository](#-tutorial-svd-lte-on-clip)**
- **[ICML, 2018]** Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (**TCAV**) [[paper]](https://arxiv.org/abs/1711.11279)
- **[JMLR, 2008]** Visualizing data using t-SNE [[paper]](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
- **[Arxiv, 2018]** UMAP: Uniform manifold approximation and projection [[paper]](https://arxiv.org/abs/1802.03426)
- **[Arxiv, 2024]** Scaling and evaluating sparse autoencoders [[paper]](https://arxiv.org/abs/2406.04093)
- **[ICLR, 2019]** Towards robust interpretability with self-explaining neural networks (**SENN**) [[paper]](https://arxiv.org/abs/1806.07538)
- **[Arxiv, 2018]** GAN Dissection: Visualizing and understanding generative adversarial networks [[paper]](https://arxiv.org/abs/1811.10597)

---

## 🏗️ Foundation Models

*Backbones whose latent spaces are candidates for representation-level analysis.*

| Model | Modality | Latent dim | Analysed here | Paper | Code |
|-------|----------|------------|---------------|-------|------|
| **CLIP ViT-B/32** | `Image` `Text` | 512 | ✅ | [[paper]](https://arxiv.org/abs/2103.00020) | [[code]](https://github.com/mlfoundations/open_clip) |
| **CLIP ViT-L/14** | `Image` `Text` | 768 | — | [[paper]](https://arxiv.org/abs/2103.00020) | [[code]](https://github.com/mlfoundations/open_clip) |
| **DINOv2** | `Image` | 384–1536 | — | [[paper]](https://arxiv.org/abs/2304.07193) | [[code]](https://github.com/facebookresearch/dinov2) |
| **SAM** | `Image` | 256 | — | [[paper]](https://arxiv.org/abs/2304.02643) | [[code]](https://github.com/facebookresearch/segment-anything) |
| **SigLIP** | `Image` `Text` | 768 | — | [[paper]](https://arxiv.org/abs/2303.15343) | [[code]](https://github.com/google-research/big_vision) |
| **StyleGAN** | `Image` | 512 | — | [[paper]](https://arxiv.org/abs/1812.04948) | [[code]](https://github.com/NVlabs/stylegan3) |

---

## ‼️ Open Challenges

- **Faithfulness of representation-level explanations**: verifying that what a latent-space method reports is genuinely what the model uses for decisions
- **Standardised evaluation**: no generalisable metric yet exists for comparing xAI techniques across methods and paradigms
- **Scalability**: gradient- and Jacobian-based analysis becomes costly for very large backbones and high-dimensional latents
- **Non-technical accessibility**: eigenvectors, singular values, and entropy are not digestible for non-expert stakeholders
- **Generative and multimodal models**: latent structure in LLMs and diffusion models remains largely unexplored at this level
- **Post-deployment drift**: how the latent manifold shifts once real-world data arrives, and how to monitor it continuously

---

## 🤝 Contributing

Contributions are welcome. Please:

- Follow the format: `**[Venue Year]** Title [[paper]](link) [[code]](link)`
- Focus on explainability for foundation models (not general ML papers)
- Place entries under the correct direction (prediction-level vs representation-level)
- Open an issue or submit a PR

---

## 📄 Citation

If this repository or the tutorial is useful in your work, please cite:

```bibtex
@misc{asah2026svdlte,
  title  = {SVD-LTE: A Unified Global Explanation Method for
            Latent-Space Top-Eigenvector Analysis of Foundation Models},
  author = {Asah, Victor Kolapo and Yu, Hongchuan},
  year   = {2026},
  note   = {MSc Data Science and Artificial Intelligence,
            Department of Computing and Informatics,
            Bournemouth University},
  howpublished = {\url{https://github.com/YOUR-USERNAME/Awesome-XAI-Foundation-Models}}
}
```

---

## 🙏 Acknowledgements

SVD-LTE was developed in collaboration with **Prof Hongchuan Yu**, Department of Computing and Informatics, Bournemouth University. The structure of this repository follows the conventions of the [Awesome](https://awesome.re) list ecosystem.
