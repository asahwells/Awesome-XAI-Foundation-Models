# 🔍 Awesome Foundation Models for eXplainable AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/asahwells/Awesome-XAI-Foundation-Models/pulls)

Foundation models such as CLIP have transformed computer vision, but their internal representations remain opaque. Existing eXplainable AI (xAI) methods — LIME, SHAP, Grad-CAM — explain *individual predictions*, yet say little about *what a model has actually encoded* in its learned latent space.

This repository provides a hands-on tutorial for **SVD-LTE (Latent-Space Top-Eigenvector Analysis)** — a unified global explanation framework that shifts explainability from prediction-space to representation-space — demonstrated end-to-end on a pretrained CLIP encoder.

> **🎓 Start here**: [Tutorial — SVD-LTE on CLIP](#-tutorial-svd-lte-on-clip)

---

## 🗺️ Landscape

Explainability for foundation models divides into two directions:

**1. Prediction-Level xAI** — explains *why a model produced a specific output for a specific input*. Local by construction. LIME, SHAP, Grad-CAM, Integrated Gradients.

**2. Representation-Level xAI** — explains *what information is encoded in the learned latent manifold*, how stable it is, and whether it carries bias or drift vulnerabilities. Global by construction.

SVD-LTE sits in direction **(2)**, and contributes quantitative metrics that direction **(1)** currently lacks.

---

## 📌 Table of Contents

- [🗺️ Landscape](#️-landscape)
- [📚 Background](#-background)
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
- [‼️ Open Challenges](#️-open-challenges)
- [🤝 Contributing](#-contributing)
- [📄 Citation](#-citation)

---

## 📚 Background

The methods and limitations this framework responds to:

- **[Intelligent Systems with Applications, 2026]** Explaining explainability: A comprehensive survey on explainable artificial intelligence and relevant industry applications [[paper]](https://doi.org/10.1016/j.iswa.2026.200647) — *the survey identifying the gaps SVD-LTE addresses*
- **[Information Fusion, 2020]** Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI [[paper]](https://doi.org/10.1016/j.inffus.2019.12.012)
- **[ACM SIGKDD, 2016]** "Why should I trust you?": Explaining the predictions of any classifier (**LIME**) [[paper]](https://arxiv.org/abs/1602.04938) [[code]](https://github.com/marcotcr/lime)
- **[NeurIPS, 2017]** A unified approach to interpreting model predictions (**SHAP**) [[paper]](https://arxiv.org/abs/1705.07874) [[code]](https://github.com/shap/shap)
- **[ICML, 2017]** Axiomatic attribution for deep networks (**Integrated Gradients**) [[paper]](https://arxiv.org/abs/1703.01365)
- **[IJCV, 2019]** Grad-CAM: Visual explanations from deep networks via gradient-based localization [[paper]](https://arxiv.org/abs/1610.02391)
- **[ICML, 2021]** Learning transferable visual models from natural language supervision (**CLIP**) [[paper]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/mlfoundations/open_clip) — *the backbone analysed in this tutorial*

---

## 🎓 Tutorial: SVD-LTE on CLIP

### Why SVD-LTE

Current xAI faces five well-documented limitations. SVD-LTE addresses each:

| # | Challenge in current xAI | SVD-LTE's response |
|---|--------------------------|--------------------|
| 1 | **Local, not global** — LIME/SHAP/Grad-CAM explain one sample at a time | Explains the *whole latent manifold* via principal directions learned from thousands of samples |
| 2 | **No representation-level explainability** — few methods explain latent embeddings | Explicitly explains latent factors, dimensions, and structure via principal directions, prototypes, and attribution maps |
| 3 | **No standardised evaluation metrics** | Introduces quantitative metrics — Fairness (AUC per PC, LDS, Mahalanobis Distance); Robustness (DSR, Latent Drift Vector); Stability (Cross-Run Alignment, Jacobian Norm); Explainability (Attribution Entropy) |
| 4 | **Explanation instability** — LIME variance, SHAP variance, gradient noise | Principal directions are estimated from thousands of samples and are statistically far more stable; stability is itself measured |
| 5 | **Explainability–accuracy trade-off** — interpretable models sacrifice performance | Fully post-training. No retraining, no architecture change. Accuracy unchanged; explainability added afterwards |

### Quick Start

```bash
pip install open_clip_torch torch torchvision numpy scipy scikit-learn matplotlib seaborn
```

Or open the tutorial directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1de5x85nBcFXRol7O7hL6IDS-2ZAFMp9o?usp=sharing)

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
        # Gradients must flow — the attribution methods in Step 7 depend on them.
        h = self.clip_model.encode_image(x)
        return (None, h) if return_latent else h

model = CLIPEncoder(clip_model).to(DEVICE)
```
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

> The normalisation constants are CLIP's own training statistics. Using different values silently degrades every downstream result.

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

> Centring is not optional. SVD on uncentred data returns the mean as the first component, not the dominant direction of *variation*.

---

### Step 4 — Fairness: Do PCs Encode Protected Attributes?

Define two groups and ask whether any single PC separates them. If a PC separates groups the model was never told about, that group structure is **encoded in the representation**.

In this tutorial the groups are semantic (animals vs vehicles). In a fairness audit they would be protected attributes.

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

**Reading the numbers**: AUC ≈ 0.5 → the PC carries no group information. AUC → 1.0 → the PC almost perfectly separates the groups, meaning the concept is strongly encoded. The three metrics are complementary: AUC is per-direction, LDS aggregates separation across directions, and Mahalanobis measures the distance between group centroids in the full space.

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

# Latent Drift Vector: where did the centre of the cloud move to?
delta_vec = Z_shift.mean(0) - Z_clean.mean(0)
delta_dir = delta_vec / (np.linalg.norm(delta_vec) + 1e-8)

# Which PCs absorbed the drift?
shift_proj = np.array([delta_dir @ pcs_final[:, k] for k in range(K_PCS)])

# Domain-Shift Response: did each PC survive the perturbation?
cov_shift = np.cov(Z_shift.T)
U_shift, _, _ = np.linalg.svd(cov_shift)
DSR = np.array([pcs_final[:, k] @ U_shift[:, k] for k in range(K_PCS)])
```

**Reading the numbers**: DSR near ±1 → that PC survived the shift (robust). DSR near 0 → that PC collapsed (fragile). `shift_proj` localises *where* the drift landed — which is more useful than a single global "is the model robust?" verdict.

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

**Reading the numbers**: alignment → 1.0 means the direction is essentially identical across runs. Lower values mean the direction wanders between samples and should be trusted less. The absolute value is used because an eigenvector and its negation describe the same direction.

---

### Step 7 — Attribution: What Does Each PC Look For?

Project the latent onto a PC to get a single score, then ask which input pixels drive that score. Every pixel is a dial; the gradient answers *"if I nudge this dial, how much does the score move?"*

Three methods of increasing robustness are provided as a cross-check — if they agree, the finding is trustworthy.

```python
def pc_tensor(v):
    return torch.tensor(v, dtype=torch.float32, device=DEVICE)

def reduce_to_2d(t):
    return t.mean(dim=0) if t.ndim == 3 else t

def signed_gradient_saliency(model, pc_vec, x):
    """One gradient measurement at the input point. Fast, noisy."""
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
    """Average gradients over randomly sampled interpolation points. Most robust."""
    x = x.clone().to(DEVICE)
    total = torch.zeros_like(x)
    for _ in range(n_samples):
        xi = (torch.rand(1, 1, 1, 1, device=DEVICE) * x).requires_grad_(True)
        _, h = model(xi, return_latent=True)
        (h @ pc_tensor(pc_vec)).sum().backward()
        total += xi.grad.detach()
    return reduce_to_2d((total / n_samples).cpu()[0]).numpy()

def optimize_prototype(model, pc_vec, shape, steps=200, lr=0.05):
    """Synthesise the image that maximally activates a PC — the PC's 'dream image'."""
    x = torch.randn(shape, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        _, h = model(x, return_latent=True)
        (-(h @ pc_tensor(pc_vec)).sum()).backward()
        opt.step()
    return x.detach().cpu()[0].mean(dim=0).numpy()   # average RGB → viewable
```

> **Common pitfall**: wrapping the encoder's forward pass in `torch.no_grad()` will silently break all four functions above with `element 0 of tensors does not require grad`. Gradients must be able to flow back to the input.

---

### Step 8 — Explainability: Attribution Entropy

Entropy measures whether an explanation is **concentrated** (low) or **diffuse** (high).

```python
def attribution_entropy(a):
    a = np.abs(a).astype(np.float64).flatten()
    a = a / (a.sum() + 1e-8)
    return -(a * np.log(a + 1e-8)).sum()
```

**Calibrating the scale**: for a 224 × 224 map, perfectly uniform attribution gives `ln(50176) ≈ 10.82` — the theoretical maximum. A value near 10.8 means the explanation is spread across the entire image; substantially lower means it is localised. Without this anchor the raw number is uninterpretable.

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

Attribution entropy across all PCs: **9.96 – 10.21** (theoretical maximum 10.82). Attribution is diffuse for *every* component, consistent with ViT patch tokenisation distributing representation across spatial tokens. PC7 is highest (10.21), matching its high-frequency texture prototype; PC1 is lowest (9.96). PC0 by contrast is object-level — the contrast between the two illustrates that SVD-LTE can distinguish semantic from textural components.

### Metric independence — each metric earns its place

| Pair | Pearson | Spearman |
|------|---------|----------|
| Entropy × Instability | −0.084 | 0.152 |
| Entropy × Prototype norm | 0.114 | 0.018 |
| Instability × Prototype norm | 0.129 | 0.224 |

All |r| < 0.13. Stability, interpretability, and semantic strength vary **independently** — so each metric family captures a distinct, non-redundant property of the representation. Had they correlated strongly, the framework would be measuring one thing three times.

> **Limitations**: attribution maps are computed on a single representative image per PC and are illustrative rather than population-level. Correlations are estimated from K = 10 components, so they indicate *no evidence of a relationship* rather than proven independence. Results are reported for one backbone and one dataset.

---

## ‼️ Open Challenges

- **Faithfulness of representation-level explanations**: verifying that what a latent-space method reports is genuinely what the model uses for decisions
- **Standardised evaluation**: no generalisable metric yet exists for comparing xAI techniques across methods and paradigms
- **Scalability**: gradient- and Jacobian-based analysis becomes costly for very large backbones and high-dimensional latents
- **Non-technical accessibility**: eigenvectors, singular values, and entropy are not digestible for non-expert stakeholders
- **Generative and multimodal models**: latent structure in LLMs and diffusion models remains largely unexplored at this level
- **Post-deployment drift**: how the latent manifold shifts once real-world data arrives, and how to monitor it continuously

---

## 📄 Citation

If this repository or the tutorial is useful in your work, please cite:

```bibtex
@misc{asah2026svdlte,
  title  = {SVD-LTE: A Unified Global Explanation Method for
            Latent-Space Top-Eigenvector Analysis of Foundation Models},
  author = {Asah, Victor Kolapo and Yu, Hongchuan},
  year   = {2026},
  note   = {Department of Computing and Informatics, Bournemouth University},
  howpublished = {\url{https://github.com/asahwells/Awesome-XAI-Foundation-Models}}
}
```

---

## 🙏 Acknowledgements

SVD-LTE was developed in collaboration with **Prof Hongchuan Yu**, Department of Computing and Informatics, Bournemouth University. The structure of this repository follows the conventions of the [Awesome](https://awesome.re) list ecosystem.
