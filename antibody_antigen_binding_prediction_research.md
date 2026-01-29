# Antibody-Antigen and VHH-Antigen Binding Prediction: State-of-the-Art Research

**Research Compilation Date: January 2026**

---

## Executive Summary

Antibody-antigen binding prediction represents one of the most challenging domains in computational structural biology. Unlike general protein-protein interactions, antibody-antigen complexes present unique difficulties due to the high variability of complementarity-determining regions (CDRs), particularly the CDR-H3 loop. This document provides a focused review of the latest methods (2024-2025) for predicting antibody-antigen and VHH (nanobody)-antigen interactions, along with comprehensive coverage of scoring functions and filtering methods for ranking predicted complexes.

---

## Table of Contents

1. [The Challenge of Antibody-Antigen Prediction](#1-the-challenge-of-antibody-antigen-prediction)
2. [Structure Prediction Methods](#2-structure-prediction-methods)
3. [VHH/Nanobody-Specific Prediction](#3-vhhnanobody-specific-prediction)
4. [Scoring Functions for Complex Quality](#4-scoring-functions-for-complex-quality)
5. [Filtering and Ranking Methods](#5-filtering-and-ranking-methods)
6. [Consensus and Ensemble Approaches](#6-consensus-and-ensemble-approaches)
7. [Benchmarks and Evaluation Metrics](#7-benchmarks-and-evaluation-metrics)
8. [Practical Recommendations](#8-practical-recommendations)
9. [Tools and Resources](#9-tools-and-resources)
10. [References](#10-references)

---

## 1. The Challenge of Antibody-Antigen Prediction

### Why Antibody-Antigen is Harder Than General PPI

| Factor | Impact |
|--------|--------|
| **CDR Loop Variability** | CDR-H3 loops are highly variable and flexible |
| **Limited Training Data** | Fewer high-quality Ab-Ag structures vs. general PPI |
| **Conformational Changes** | Induced fit upon binding common |
| **Epitope Diversity** | Antibodies can bind diverse epitope types |
| **Paratope Complexity** | 6 CDR loops (heavy + light chain) must be modeled |

### Current Performance Reality

As of 2025, the best methods achieve:

| Method | Antibody Success Rate | Nanobody Success Rate | Notes |
|--------|----------------------|----------------------|-------|
| **AlphaFold3 (1 seed)** | 10-13% | 13.3% | High-accuracy (DockQ > 0.8) |
| **AlphaFold3 (1000 seeds)** | ~60% | - | Extensive sampling required |
| **Boltz-2** | Improved over Boltz-1 | ~43% (3/7 targets) | Structure + affinity prediction |
| **AlphaRED** | 43% | - | AF2-M + Rosetta replica exchange |
| **HelixFold-Multimer** | 52.7% | - | Fine-tuned for Ab-Ag |
| **Boltz-1** | 4.08% | 5% | Open source (MIT) |
| **Chai-1** | 0% | 3.33% | Open source (Apache) |
| **AF2-Multimer v2.3** | 7.6% | - | Baseline comparison |

**Key Insight**: Even the best current methods fail on the majority of antibody-antigen complexes, highlighting the continued difficulty of this prediction task.

---

## 2. Structure Prediction Methods

### 2.1 AlphaFold3

**Architecture:**
- Diffusion-based structure module (replacing Evoformer-based approach)
- Joint modeling of proteins, nucleic acids, small molecules
- Unified treatment of all biomolecular interactions

**Antibody-Antigen Performance:**
- 60% success rate with 1,000 random seeds
- 10-13% with single seed
- Improved CDR-H3 loop prediction (median RMSD: 2.15 Å for antibodies, 2.06 Å for nanobodies)

**Limitations:**
- High computational cost for multi-seed sampling
- Overconfident predictions in some cases
- Performance varies significantly by target

**Access:** AlphaFold Server (academic), open-sourced November 2024

### 2.2 HelixFold-Multimer (December 2024)

**Key Innovation:** Fine-tuned specifically for antigen-antibody systems

**Performance:**
- **52.7% success rate** (vs. AF2 7.6%, RoseTTAFold 4.6%)
- Significantly outperforms baseline methods on Ab-Ag

**Why It Works:**
- Domain-specific training data
- Optimized for CDR loop modeling
- Better handling of Ab-Ag interface physics

### 2.3 AlphaRED

**Approach:** Hybrid method combining:
1. AlphaFold2-Multimer predicted complexes
2. Confidence measures from AF2-M
3. Rosetta-based replica exchange docking

**Performance:** 43% success rate (highest before AF3)

**Advantage:** Leverages both AI prediction and physics-based refinement

### 2.4 Template-Based Docking Tools

**Performance on Epitope Mapping:**
- Template-based tools: up to 35%
- AlphaFold2/Boltz-1: 28%
- AlphaFold3: 47%

**Performance on Antibody Design Scenario:**
- AlphaFold3: 46%
- Boltz-1: 21%
- AlphaFold2: 13%

### 2.5 Boltz-2 (June 2025)

**Major Advancement:** First open-source model to jointly predict structure AND binding affinity.

**Key Capabilities:**
- Structure prediction (building on Boltz-1 architecture)
- Binding affinity prediction approaching FEP accuracy
- ~1000x faster than physics-based Free Energy Perturbation (FEP)
- Cost reduction: ~$100/prediction (FEP) → cents (Boltz-2)
- Prediction time: 6-12 hours (FEP) → ~20 seconds (Boltz-2)

**Antibody-Antigen Performance:**
- Shows improvement over Boltz-1 on Ab-Ag complexes
- Still lags behind AlphaFold3 on antibody benchmarks
- On nanobody-antigen: correctly predicted 3/7 targets (~43%)
- Yields predominantly high and medium-quality models (few incorrect)
- DockQ ≈ 0.91 on training-set-like structures; ≈ 0.70 on novel complexes

**Benchmark Results:**
- CASP16 affinity challenge: Outperformed all top-ranking participants (out-of-the-box, no fine-tuning)
- FEP+ benchmark: 0.6 correlation with experimental results (matching FEP simulations)

**Controllability Features:**
- Experimental method conditioning
- Distance constraints
- Multi-chain template integration

**Why Boltz-2 Matters for Antibody Work:**
1. **Affinity prediction**: Can rank antibody candidates by predicted binding strength
2. **Rapid screening**: Enables evaluation of thousands of candidates
3. **Open source**: MIT license for academic and commercial use
4. **Fine-tunable**: Can be adapted for specific protein-protein affinity tasks (see arXiv:2512.06592)

**Limitations:**
- Still underperforms AlphaFold3 on pure structure prediction for Ab-Ag
- Performance gap on unseen antigens
- Affinity predictions may need calibration for specific systems

**Access:** github.com/jwohlwend/boltz (MIT License)

### 2.6 AI-Augmented Physics-Based Docking

Recent work (Bioinformatics 2025) combines:
- AI-based initial structure prediction
- Physics-based docking refinement
- Energy minimization protocols

This hybrid approach addresses limitations of pure ML methods.

---

## 3. VHH/Nanobody-Specific Prediction

### Why Nanobodies Are Easier to Model

| Factor | Antibody | Nanobody (VHH) |
|--------|----------|----------------|
| **CDR Loops** | 6 (3 heavy + 3 light) | 3 (heavy only) |
| **Size** | ~150 kDa (IgG) | ~15 kDa |
| **Search Space** | Larger | Smaller |
| **Flexibility** | More complex | Simpler |

### Current Performance on Nanobody-Antigen

| Method | High-Accuracy Rate | Notes |
|--------|-------------------|-------|
| **Boltz-2** | ~43% (3/7 targets) | Significant improvement |
| AlphaFold3 | 13.3% | Single seed |
| Boltz-1 | 5% | - |
| Chai-1 | 3.33% | - |

### Specialized Nanobody Tools

#### NanoBodyBuilder2 (ImmuneBuilder)
- Deep learning architecture tailored for nanobodies
- Excels at CDR3 loop prediction
- **100x faster than AlphaFold2**
- Part of the ImmuneBuilder toolkit

#### NanoNet
- Rapid end-to-end nanobody modeling
- Focuses on VHH region prediction
- Suitable for high-throughput applications

#### NanoBERTa-ASP
- Predicts nanobody-antigen interaction sites
- BERT-based sequence model
- Optimizes binding conformation prediction

#### NbX
- Predicts binding sites and structural interactions
- Improves precision in interface prediction

### De Novo VHH Design with RFdiffusion

**Recent Breakthrough (Nature 2025):**
- Fine-tuned RFdiffusion network for antibody design
- Combined with yeast display screening
- Enables de novo generation of:
  - VHH binders
  - Single-chain variable fragments (scFvs)
  - Full antibodies
- Atomic-level precision for user-specified epitopes
- Experimentally validated VHH binders to four disease-relevant epitopes

---

## 4. Scoring Functions for Complex Quality

### 4.1 Ground Truth Metric: DockQ

**Definition:** Continuous quality score combining:
- Ligand RMSD (LRMSD)
- Interface RMSD (iRMSD)
- Fraction of native contacts (fnat)

**Score Range:** 0.0 - 1.0

**Quality Classification (CAPRI Standard):**

| Category | DockQ Range |
|----------|-------------|
| **Incorrect** | < 0.23 |
| **Acceptable** | 0.23 - 0.49 |
| **Medium** | 0.49 - 0.80 |
| **High** | > 0.80 |

### 4.2 AlphaFold Confidence Scores

#### ipTM (Interface Predicted Template Modeling)

**Definition:** Measures predicted quality of inter-chain interfaces

**Calculation:** Derived from predicted aligned error (PAE) matrix, considering only residue pairs from different chains

**Interpretation (AlphaFold Server Guidelines):**

| ipTM Score | Interpretation |
|------------|----------------|
| > 0.8 | High-quality prediction |
| 0.6 - 0.8 | "Gray zone" - may be correct or incorrect |
| < 0.6 | Likely failed prediction |

**AlphaFold Ranking Formula:**
```
ranking_confidence = 0.2 × pTM + 0.8 × ipTM
```

#### Performance Analysis (2025 Study)

Comparison across ColabFold-Template (CF-T), ColabFold-Fast (CF-F), and AlphaFold3 (AF3):

| Metric | Performance | Recommendation |
|--------|-------------|----------------|
| **ipTM** | Highest PCC with DockQ, highest AUC | **Most reliable** |
| **Model Confidence** | High accuracy across all datasets | **Recommended** |
| **pDockQ2** | Lowest AUC consistently | Not recommended for primary filtering |

**Important Finding:** AUC and accuracy values for AF3 were lower than ColabFold, showing comparatively lower correlation between confidence and quality for AF3.

### 4.3 pDockQ and pDockQ2

#### pDockQ (Original)
- Predicts DockQ score with average error of 0.1
- AUC of 0.95 for separating acceptable vs. incorrect models
- Calculated from:
  - Number of interfacial contacts
  - Average quality of interacting residues
  - Fitted to sigmoid function

#### pDockQ2
- Developed for multimeric protein complexes
- Considers PAE between all chains
- Better correlated with DockQ for multi-chain systems
- **However:** Recent benchmarks show lower AUC than ipTM

### 4.4 Physics-Based Scoring Functions

#### FoldX

**Capabilities:**
- Rapid estimation of binding free energy
- Mutation effect prediction (ΔΔG)
- Protein complex interaction energy

**Performance:**
- Stabilizing mutation predictions: ~29% experimental success
- Destabilizing mutation predictions: ~69% correct
- Useful for rapid screening, less reliable for absolute values

#### Rosetta Energy Functions

**Score Functions:**
- ref2015 (default)
- RosettaLigand (for ligand docking)
- Interface energy terms

**Performance:**
- Pearson correlations: 0.5-0.6 for scoring
- Spearman correlation: 0.63 for ranking (RosettaLigand)

**Limitation:** Reliance on static structures; simplified unfolded state representation

### 4.5 Deep Learning Scoring Functions

#### EuDockScore Family (2024)

**Variants:**
| Model | Application |
|-------|-------------|
| **EuDockScore** | General protein-protein docking |
| **EuDockScore-Ab** | Antibody-antigen specific |
| **EuDockScore-AFM** | Tuned for AlphaFold-Multimer outputs |
| **EuDockScore-AFSample** | For AFSample complex predictions |

**Architecture:**
- Euclidean graph neural networks
- Operates on raw 3D coordinates
- Supplemented with DistilProtBert NLP embeddings
- E(3)-equivariant operations

**Performance:**
- State-of-the-art on CAPRI score_set
- Ensembling with DeepRank-GNN-esm and iScore improves discriminative power

**Code:** https://gitlab.com/mcfeemat/eudockscore

#### DeepRank-GNN-esm (2024)

**Features:**
- Graph neural network architecture
- ESM-2 language model embeddings
- Can replace computationally expensive PSSM features

**Advantage:** Maintains or improves performance while reducing computational cost

#### GDockScore

- Graph-based protein-protein docking scoring
- Designed for ranking docked conformations

---

## 5. Filtering and Ranking Methods

### 5.1 Confidence-Based Filtering

#### Recommended Thresholds

**For AlphaFold-based predictions:**

| Filter Level | ipTM Threshold | Expected Outcome |
|--------------|----------------|------------------|
| **Stringent** | > 0.8 | High-quality only, may miss valid predictions |
| **Moderate** | > 0.7 | Balance precision/recall |
| **Permissive** | > 0.6 | Include gray zone, more false positives |

**For pDockQ:**
- Acceptable/Incorrect boundary: ~0.23 (corresponds to DockQ threshold)

### 5.2 Interface-Based Filtering

#### Contact-Based Methods

**Inter-residue Contact Conservation:**
- Calculate contact maps across ensemble
- Filter models with low contact conservation
- Used in Iter-CONSRANK algorithm

**Interface Area Filtering:**
- Typical Ab-Ag interface: 700-900 Å²
- Filter predictions with unrealistic interface sizes

### 5.3 Energy-Based Filtering

**FoldX Interface Energy:**
- Calculate ΔG of binding
- Filter highly unfavorable predictions
- Combine with ML predictions for better accuracy

**Rosetta Interface Score:**
- ddG calculations
- Per-residue energy decomposition
- Identify energetically unfavorable contacts

### 5.4 Structural Quality Filtering

**Clash Detection:**
- Identify steric clashes
- Filter models with severe atomic overlaps

**Secondary Structure Preservation:**
- Verify CDR loops maintain expected structure
- Filter models with distorted framework regions

---

## 6. Consensus and Ensemble Approaches

### 6.1 Iter-CONSRANK Algorithm (2025)

**Approach:**
- Iterative consensus-based scoring
- Builds on established CONSRANK algorithm
- Incorporates iterative filtering

**Method:**
1. Generate ensemble of docking models
2. Calculate inter-residue contact conservation
3. Progressively discard lower-ranked models
4. Refine ensemble through iterations

**Performance:**
- Tested on 3K-BM5up dataset (~1.6 × 10⁵ models)
- Tested on 30K-BM5 dataset (~6.4 × 10⁶ models)
- Tested on CAPRI Score_set (~2.0 × 10⁴ models)
- Significant improvement in correct solution fraction across difficulty categories

### 6.2 Exponential Consensus Ranking (ECR)

**Problem Addressed:**
Traditional consensus methods fail when one docking program performs poorly due to:
- Training-set dependencies
- Scoring-function parameterization

**Solution:**
```
ECR_score = Σ exp(-rank_i / τ)
```

**Advantages:**
- Based on rank (not score)
- Independent of score units, scales, offsets
- Robust to single-program failures
- Theoretical basis for any consensus strategy

### 6.3 Multi-Program Consensus

**MILCDock Approach:**
- Combines predictions from 5 docking tools
- Machine learning to predict binding probability
- Improves ranking of active over inactive ligands

**10-Program Consensus Study:**
Programs tested: ADFR, DOCK, Gemdock, Ledock, PLANTS, PSOVina, QuickVina2, Smina, Autodock Vina, VinaXB

**Finding:** Consensus scoring provides improved docking fidelity with small number of docking combinations

### 6.4 Ensemble Scoring for Antibody-Antigen

**Recommended Ensemble:**

| Component | Weight | Rationale |
|-----------|--------|-----------|
| EuDockScore-Ab | High | Ab-Ag specific |
| ipTM | High | Best single predictor |
| DeepRank-GNN-esm | Medium | Complementary features |
| FoldX ΔΔG | Low | Physics-based validation |

**Implementation:**
```
ensemble_score = w1 × EuDockScore-Ab + w2 × ipTM + w3 × DeepRank + w4 × (-FoldX_ddG)
```

Weights should be optimized on held-out validation set.

---

## 7. Benchmarks and Evaluation Metrics

### 7.1 Key Benchmark Datasets

#### CAPRI Score_set
- Gold-standard for protein-protein docking scoring
- ~20,000 models from CAPRI experiments
- Multiple difficulty categories
- Widely used for scoring function evaluation

#### SAbDab (Structural Antibody Database)
- Updated July 2024
- Comprehensive antibody structure repository
- Includes Ab-Ag complexes
- Annotations for CDR regions, chain pairings

#### Docking Benchmark 5.5
- Standard for protein-protein docking
- Unbound and bound structures
- Difficulty classifications

### 7.2 Evaluation Metrics

#### Classification Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **AUC-ROC** | Area under ROC curve | Overall discriminative ability |
| **AUC-PR** | Area under precision-recall | Imbalanced datasets |
| **Accuracy** | Correct predictions / total | Simple performance measure |
| **F1 Score** | Harmonic mean of precision/recall | Balance precision and recall |

#### Ranking Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Spearman ρ** | Rank correlation | Ranking quality |
| **Pearson r** | Linear correlation | Score correlation with ground truth |
| **Top-N Success Rate** | Correct in top N predictions | Practical utility |
| **Hit Rate** | Acceptable models in top N | Docking success |

#### Structure Quality Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **DockQ** | Combined quality score | >0.23 acceptable |
| **LRMSD** | Ligand RMSD | <10 Å acceptable |
| **iRMSD** | Interface RMSD | <4 Å acceptable |
| **fnat** | Fraction native contacts | >0.1 acceptable |

### 7.3 Recent CAPRI Results (Rounds 47-55)

**8th CAPRI Evaluation Period:**
- Transition to AI-driven tools (AlphaFold era)
- 11 targets, 21 interfaces evaluated
- High difficulty level
- Most targets in "difficult" category

**Round 56 (MHC/Antibody):**
- 4 servers produced medium-quality models
- CLUSPRO, HADDOCK, LZERD, MDOCKPP successful
- Kozakov: high-quality model in top-5
- HADDOCK: medium-quality in top-1

**Overall Assessment:**
> "Highly accurate models were obtained only for ~40% of targets in CASP15-CAPRI, leading assessors to conclude that accurate prediction of protein complexes remained challenging."

---

## 8. Practical Recommendations

### 8.1 Prediction Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Ab + Ag Sequences                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            STEP 1: Structure Prediction                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ AlphaFold3  │  │ HelixFold-  │  │   Boltz-2   │         │
│  │ (if avail)  │  │  Multimer   │  │  (struct+   │         │
│  │             │  │             │  │  affinity)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         Generate multiple seeds (≥100 recommended)          │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            STEP 2: Initial Filtering                         │
│  • ipTM > 0.6 (permissive) or > 0.7 (moderate)             │
│  • Remove models with severe clashes                        │
│  • Check interface area reasonableness                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            STEP 3: Scoring                                   │
│  • EuDockScore-Ab (antibody-specific)                       │
│  • DeepRank-GNN-esm                                         │
│  • FoldX interface energy                                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            STEP 4: Consensus Ranking                         │
│  • Exponential consensus ranking                            │
│  • Or ensemble score with learned weights                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            STEP 5: Final Selection                           │
│  • Top-10 candidates for experimental validation            │
│  • Cluster similar predictions                              │
│  • Diversity in selected set                                │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Method Selection Guide

| Scenario | Recommended Approach |
|----------|---------------------|
| **Academic research (computational access)** | AlphaFold3 with 1000 seeds |
| **Structure + Affinity needed** | **Boltz-2** (joint prediction) |
| **Limited compute** | HelixFold-Multimer or AlphaRED |
| **Nanobody targets** | **Boltz-2** or NanoBodyBuilder2 + AF3 |
| **High-throughput screening** | **Boltz-2** (fast affinity ranking) |
| **Candidate ranking by affinity** | **Boltz-2** affinity predictions |
| **De novo design** | RFdiffusion (fine-tuned) |
| **Mutation effects** | Structure + FoldX/DDMut-PPI or fine-tuned Boltz-2 |

### 8.3 Confidence Calibration

**Problem:** AlphaFold confidence scores may be poorly calibrated for Ab-Ag

**Solution:**
1. Build calibration set from known Ab-Ag structures
2. Map ipTM → empirical success probability
3. Apply calibration to new predictions

### 8.4 When to Trust Predictions

**Higher Confidence:**
- ipTM > 0.8
- Consistent across multiple seeds
- Multiple methods agree
- Similar to known structures

**Lower Confidence:**
- ipTM < 0.7
- High variance across seeds
- Methods disagree
- Novel epitope type

---

## 9. Tools and Resources

### 9.1 Structure Prediction Servers

| Tool | URL | License | Best For |
|------|-----|---------|----------|
| **AlphaFold Server** | alphafoldserver.com | Academic | General Ab-Ag |
| **Boltz-2** | github.com/jwohlwend/boltz | MIT | **Structure + Affinity** |
| **HelixFold** | GitHub | Open | Ab-Ag specific |
| **Boltz-1** | github.com/jwohlwend/boltz | MIT | Open-source structure |
| **Chai-1** | github.com/chaidiscovery/chai-lab | Apache | Commercial use |

### 9.2 Antibody-Specific Tools

| Tool | Purpose | Speed |
|------|---------|-------|
| **IgFold** | Ab structure prediction | Fast |
| **NanoBodyBuilder2** | Nanobody modeling | 100x faster than AF2 |
| **ImmuneBuilder** | Ab/Nb structure suite | Fast |
| **ABodyBuilder2** | Antibody modeling | Fast |

### 9.3 Scoring Tools

| Tool | Type | Code |
|------|------|------|
| **EuDockScore** | DL scoring | gitlab.com/mcfeemat/eudockscore |
| **DeepRank-GNN-esm** | GNN + PLM | GitHub |
| **FoldX** | Physics-based | foldxsuite.crg.eu |
| **HADDOCK** | Docking + scoring | wenmr.science.uu.nl |

### 9.4 Databases

| Database | Content | URL |
|----------|---------|-----|
| **SAbDab** | Ab structures | opig.stats.ox.ac.uk/webapps/sabdab |
| **CAPRI** | Docking benchmarks | ebi.ac.uk/pdbe/complex-pred/capri |
| **PDB** | All structures | rcsb.org |
| **IMGT** | Immunogenetics | imgt.org |

---

## 10. References

### Structure Prediction Methods

1. [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)

2. [On fine-tuning Boltz-2 for protein-protein affinity prediction (arXiv 2025)](https://arxiv.org/abs/2512.06592)

3. [What does AlphaFold3 learn about antibody and nanobody docking, and what remains unsolved? (mAbs 2025)](https://www.tandfonline.com/doi/full/10.1080/19420862.2025.2545601)

4. [AlphaFold and Docking Approaches for Antibody-Antigen and Other Targets: Insights From CAPRI Rounds 47-55 (Proteins 2025)](https://onlinelibrary.wiley.com/doi/10.1002/prot.26801)

5. [AI-augmented physics-based docking for antibody-antigen complex prediction (Bioinformatics 2025)](https://academic.oup.com/bioinformatics/article/41/4/btaf129/8093612)

6. [Atomically accurate de novo design of antibodies with RFdiffusion (Nature 2025)](https://www.nature.com/articles/s41586-025-09721-5)

7. [Unveiling the new chapter in nanobody engineering: advances in traditional construction and AI-driven optimization (J Nanobiotechnology 2025)](https://link.springer.com/article/10.1186/s12951-025-03169-5)

8. [Evaluating Deep Learning Based Structure Prediction Methods on Antibody-Antigen Complexes (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.07.11.662141v1.full)

### Scoring Functions

6. [A comprehensive survey of scoring functions for protein docking models (BMC Bioinformatics 2025)](https://link.springer.com/article/10.1186/s12859-024-05991-4)

7. [Assessing scoring metrics for AlphaFold2 and AlphaFold3 protein complex predictions (Protein Science 2025)](https://onlinelibrary.wiley.com/doi/10.1002/pro.70327)

8. [EuDockScore: Euclidean graph neural networks for scoring protein-protein interfaces (Bioinformatics 2024)](https://academic.oup.com/bioinformatics/article/40/11/btae636/7833366)

9. [DeepRank-GNN-esm: a graph neural network for scoring protein-protein models using protein language model (Bioinformatics Advances 2024)](https://pubmed.ncbi.nlm.nih.gov/38213822/)

### Consensus and Ranking Methods

10. [Increasing the fraction of correct solutions in ensembles of protein-protein docking models by an iterative consensus algorithm (Protein Science 2025)](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.70314)

11. [Exponential consensus ranking improves the outcome in docking and receptor ensemble docking (Scientific Reports 2019)](https://www.nature.com/articles/s41598-019-41594-3)

12. [A generalizable deep learning framework for structure-based protein-ligand affinity ranking (PNAS 2025)](https://www.pnas.org/doi/10.1073/pnas.2508998122)

### Benchmarks and Evaluation

13. [CAPRI-Q: The CAPRI resource evaluating the quality of predicted structures of protein complexes (Proteins 2024)](https://pubmed.ncbi.nlm.nih.gov/39237205/)

14. [Evaluation of AlphaFold-Multimer prediction on multi-chain protein complexes (Bioinformatics 2023)](https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714)

15. [Improved prediction of protein-protein interactions using AlphaFold2 (Nature Communications 2022)](https://www.nature.com/articles/s41467-022-28865-w)

### VHH/Nanobody Specific

16. [Applications and challenges in designing VHH-based bispecific antibodies: leveraging machine learning solutions (mAbs 2024)](https://www.tandfonline.com/doi/full/10.1080/19420862.2024.2341443)

17. [NanoNet: Rapid and accurate end-to-end nanobody modeling by deep learning (Frontiers in Immunology 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9411858/)

### Energy-Based Methods

18. [Estimating Absolute Protein-Protein Binding Free Energies by a Super Learner Model (JCIM 2024)](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01641)

19. [Assessment of Solvated Interaction Energy Function for Ranking Antibody-Antigen Binding Affinities (JCIM 2016)](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.6b00043)

---

## Appendix: Quick Reference Cards

### A. DockQ Quality Thresholds

```
DockQ Score Interpretation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
0.00 ─────── Incorrect ───────── 0.23
0.23 ─────── Acceptable ──────── 0.49
0.49 ─────── Medium ──────────── 0.80
0.80 ─────── High Quality ────── 1.00
```

### B. ipTM Interpretation

```
ipTM Score Interpretation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
0.0 ──── Likely Failed ──── 0.6
0.6 ──── Gray Zone ──────── 0.8
0.8 ──── High Quality ───── 1.0
```

### C. Method Selection Matrix

```
                        │ Speed │ Accuracy │ Open Source │ Ab-Ag Specific │ Affinity │
────────────────────────┼───────┼──────────┼─────────────┼────────────────┼──────────│
AlphaFold3 (1000 seeds) │  Slow │   Best   │   Academic  │       No       │    No    │
Boltz-2                 │  Fast │   Good   │     MIT     │   Improved     │   Yes    │
HelixFold-Multimer      │  Med  │   Good   │     Yes     │      Yes       │    No    │
AlphaRED                │  Med  │   Good   │     Yes     │       No       │    No    │
Boltz-1                 │  Fast │   Fair   │     MIT     │       No       │    No    │
NanoBodyBuilder2        │ VFast │   Good   │     Yes     │   Nanobody     │    No    │
```

### D. Boltz-2 vs FEP Comparison

```
                    │    Boltz-2    │      FEP      │
────────────────────┼───────────────┼───────────────│
Time per prediction │   ~20 sec     │   6-12 hours  │
Cost per prediction │   ~$0.01      │   ~$100       │
Correlation w/ exp  │     0.6       │     0.6       │
Open source         │     Yes       │    Varies     │
```

---

*Document compiled from comprehensive literature review on antibody-antigen binding prediction and protein complex scoring methods, January 2026.*
