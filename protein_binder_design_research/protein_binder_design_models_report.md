# Protein Binder Design Models: Comprehensive Research Report

**Research Compilation Date: January 2026**

**Scope:** De novo design and optimization of protein binders with focus on VHH/antibody formats

---

## Executive Summary

This report provides a comprehensive analysis of state-of-the-art protein binder design models as of January 2026. We profile 12+ major models across generative (diffusion/flow), hallucination, and multi-objective optimization paradigms. The analysis covers architecture, encoding methods, datasets, benchmarks, post-training capabilities, and selection/ranking strategies.

**Key Finding:** The field has matured rapidly, with experimental success rates improving from ~1% (early RFdiffusion) to 14-66% (BoltzGen, BindCraft) for de novo binder design. However, no single model dominates all tasks—optimal pipelines combine multiple approaches.

---

## Table of Contents

1. [Model Profiles](#part-1-model-profiles)
2. [Technical Comparison](#part-2-technical-comparison)
3. [Benchmarks and Evaluation](#part-3-benchmarks-and-evaluation)
4. [Synthesis and Recommendations](#part-4-synthesis-and-recommendations)
5. [References](#references)

---

# Part 1: Model Profiles

## 1.1 RFAntibody / RFdiffusion

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Baker Lab (University of Washington) |
| **Publication** | Nature 2025 (Vol 649, pp 183-193) |
| **License** | MIT (inference); Training code licensed to Xaira |
| **Repository** | github.com/RosettaCommons/RFantibody |

### Architecture
- **Type:** SE(3)-equivariant denoising diffusion probabilistic model
- **Base:** Fine-tuned RFdiffusion specialized for antibody loop generation
- **Encoding:** Backbone-only (Cα, N, C, O atoms as frames)
- **Conditioning:** Target structure, epitope specification

### Pipeline
```
Target Structure → RFdiffusion (backbone) → ProteinMPNN (sequence) → RF2 (validation) → Experimental
```

### Key Capabilities
- De novo VHH, scFv, and full antibody generation
- User-specified epitope targeting with atomic precision
- Multi-motif scaffolding

### Performance
| Metric | Value |
|--------|-------|
| **Experimental validation** | 4/4 disease-relevant epitopes |
| **Cryo-EM confirmation** | 4/5 designed VHHs matched intended pose |
| **Success rate** | Variable (1-10%+ depending on target) |
| **Designs required** | Hundreds to thousands |

### Datasets Used
- PDB antibody-antigen complexes
- Synthetic structures from antibody-specific prediction networks
- Fine-tuned on bound-state complex structures

### Post-Training Capability
- **Training code:** Not publicly available (licensed to Xaira)
- **Fine-tuning:** Demonstrated (RFantibody is fine-tuned RFdiffusion)
- **Custom data integration:** Requires partnership/license

### Strengths
- Atomic-level epitope precision
- Proven experimental success
- Full pipeline provided (design → sequence → validation)

### Limitations
- Requires many designs (~1000+) to find binders
- Training code not public
- Backbone-only generation (side chains via ProteinMPNN)

---

## 1.2 BoltzGen

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | MIT Jameel Clinic / Recursion |
| **Publication** | November 2025 |
| **License** | MIT |
| **Repository** | github.com/HannesStark/boltzgen |

### Architecture
- **Type:** All-atom diffusion model
- **Encoding:** Novel geometric encoding for residue identity via "virtual atoms"
- **Representation:** Full 3D coordinates of every atom
- **Conditioning:** Design specification language with multiple constraints

### Key Innovation
BoltzGen uses a design specification language allowing:
- Covalent bond constraints
- Structure groups
- Binding site specification
- Secondary structure constraints
- Design masks

### Pipeline
```
Target + Constraints → BoltzGen (all-atom) → Direct output (no separate sequence design)
```

### Performance
| Target Type | Success Rate | Best Kd |
|-------------|-------------|---------|
| **Nanobodies (novel targets)** | 66% (6/9) | 6.1 nM |
| **Protein binders** | 80% (4/5) | 0.81 nM (picomolar) |
| **Overall** | Nanomolar binders on 66% of novel targets |

### Datasets Used
- Builds on Boltz-1/Boltz-2 training data
- Structure prediction + design unified

### Post-Training Capability
- **Open source:** Full code under MIT license
- **Fine-tuning:** Possible (architecture supports it)
- **Custom constraints:** Highly flexible via specification language

### Strengths
- All-atom generation (no separate sequence design step)
- Flexible constraint system
- High success rates on diverse targets
- Universal: proteins, nucleic acids, small molecules

### Limitations
- Newer model, less community validation
- Compute intensive (all-atom diffusion)

---

## 1.3 BindCraft

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | ETH Zurich / EPF Lausanne |
| **Publication** | Nature 2025 (Vol 646, pp 483-492) |
| **License** | MIT |
| **Repository** | github.com/martinpacesa/BindCraft |

### Architecture
- **Type:** Hallucination-based (not generative)
- **Core:** AlphaFold2 Multimer backpropagation
- **Approach:** Gradient-based sequence optimization

### Key Innovation
Unlike diffusion models, BindCraft:
1. Starts from random binder sequence + target
2. AF2 predicts complex and calculates error gradient
3. Gradient updates binder sequence iteratively
4. Designs backbone, side chains, and interfaces simultaneously
5. Uses MPNNsol for solubility optimization outside interface

### Pipeline
```
Target → AF2 Hallucination Loop → MPNNsol (non-interface) → AF2 Monomer (validation) → Filter
```

### Performance
| Metric | Value |
|--------|-------|
| **Experimental success** | 10-100% across targets |
| **One-shot design** | ≤10 designs for nM binders |
| **Beta-barrel target** | 6/11 designs validated |
| **Expression rate** | High |

### Datasets Used
- No training (uses pretrained AF2 weights)
- Target structure only required

### Post-Training Capability
- **Fine-tuning:** Not applicable (uses frozen AF2)
- **Customization:** Loss functions can be modified
- **Custom losses:** Fewer interface contacts, physicochemical properties

### Strengths
- Highest reported success rates (up to 100%)
- "One-shot" design capability
- Induced-fit interfaces (target re-predicted each iteration)
- No training required (uses AF2 weights)
- Low design count needed

### Limitations
- Requires 32+ GB GPU memory
- Computationally expensive per design
- Cannot leverage custom training data

---

## 1.4 Mosaic

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Escalante Bio |
| **Publication** | 2024-2025 |
| **License** | Open source |
| **Repository** | github.com/escalante-bio/mosaic |

### Architecture
- **Type:** Multi-objective optimization framework
- **Core:** JAX-based differentiable pipeline
- **Structure Predictors:** Boltz1, Boltz2, AF2, Protenix variants

### Key Innovation
Composite objective protein design:
- Integrates 12+ different ML models into single loss function
- Optimizes multiple properties simultaneously
- Native TPU/XLA integration for scaling

### Objective Examples
```python
loss = binding_affinity + solubility + thermostability + expressibility
```

### Pipeline
```
Target + Objectives → Mosaic Optimization → 1K-50K candidates → Ranking → ~10 designs → Wetlab
```

### Performance
- Generates 1K-50K potential designs per run
- Filters to ~10 designs for experimental testing
- Requires manual tuning (learning rates, etc.)

### Post-Training Capability
- **Custom objectives:** Fully supported
- **Own models:** Can integrate custom-trained models
- **Lab data:** Can train scoring functions on proprietary data

### Strengths
- Ultimate flexibility for multi-objective design
- Integrates any JAX-compatible model
- Scales to TPU clusters
- Full control over optimization

### Limitations
- Requires substantial "hand-holding"
- Not a turnkey solution like BindCraft
- Expertise needed for custom objective tuning

---

## 1.5 PPiFlow

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Academic |
| **Publication** | 2025-2026 |
| **License** | Open source |
| **Repository** | github.com/Mingchenchen/PPIFlow |

### Architecture
- **Type:** Flow matching model
- **Core:** Pairformer architecture
- **Representation:** Backbone rigid-body transformations as continuous flows
- **Generation:** Backbone-only

### Key Innovation
- Explicit pairwise geometric and chemical interactions
- Partial flow for local redesign of existing structures
- End-to-end VHH design pipeline

### Pipeline
```
Target → PPiFlow (backbone) → Sequence Design → Flow-based Affinity Maturation
```

### Capabilities
- PPI binders
- Nanobodies (VHH)
- Antibodies
- Motif scaffolding
- Unconditional monomer generation

### Post-Training Capability
- Code available for training
- Can potentially fine-tune on custom data

### Limitations
- Backbone-only (side chains post-generation)
- Limited to protein-protein (no small molecules/nucleic acids)

---

## 1.6 FlowDesign

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Academic |
| **Publication** | Cell Systems 2025 |
| **License** | Open source |

### Architecture
- **Type:** Flow matching for sequence-structure co-design
- **Key Features:** Flexible prior selection, direct discrete distribution matching

### Key Innovation
- Outperforms baselines in amino acid recovery (AAR), RMSD, Rosetta energy
- Successfully designed anti-HIV antibodies with improved binding and neutralization

### Experimental Validation
- Designed antibodies targeting HIV-1 receptor CD4
- Improved binding affinity vs. ibalizumab
- Improved neutralizing potency across HIV mutants
- Validated by BLI and pseudovirus neutralization

---

## 1.7 EvoDiff

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Microsoft Research |
| **Publication** | 2023-2024 |
| **License** | Open source |

### Architecture
- **Type:** Sequence-space diffusion
- **Scale:** Evolutionary-scale training data
- **Approach:** Controllable generation in sequence space

### Key Innovation
- Generates proteins inaccessible to structure-based models
- Handles intrinsically disordered regions
- Sequence-based formulation (no structure required)

### Experimental Validation
- Intrinsically disordered mitochondrial targeting signals
- Metal-binding proteins
- Protein binders

### Strengths
- Can design disordered regions
- Doesn't require target structure
- Complementary to structure-based methods

---

## 1.8 Genie 2

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | AQ Laboratory |
| **Publication** | arXiv May 2024, ICLR 2025 |
| **License** | Open source |
| **Repository** | github.com/aqlaboratory/genie2 |

### Architecture
- **Type:** SE(3)-equivariant DDPM
- **Representation:** Joint modeling of Cα translations + SO(3) residue frames
- **Scale:** Trained on FoldSeek-clustered AFDB

### Performance
| Metric | Genie 2 | RFDiffusion |
|--------|---------|-------------|
| **Designability** | 0.96 | 0.63 |
| **Diversity** | Higher | Lower |
| **Novelty** | Higher | Lower |

### Key Innovation
- Multi-motif scaffolding with unspecified inter-motif geometry
- State-of-the-art on unconditional and conditional generation
- Massive data augmentation from AlphaFold DB

---

## 1.9 Chroma

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Generate Biomedicines |
| **Publication** | Nature 2023 |
| **License** | Open source |
| **Repository** | github.com/generatebio/chroma |

### Architecture
- **Type:** Diffusion with sub-quadratic scaling
- **Representation:** All-atom, joint sequence + structure
- **Conditioning:** Bayesian inference under constraints

### Key Innovation
- Programmable: symmetries, substructure, shape, semantics, natural language
- Polymer-ensemble-respecting diffusion
- Long-range reasoning with efficient architecture

### Experimental Validation
- 310 proteins characterized
- High expression, proper folding, favorable biophysics
- Crystal structures: ~1.0 Å backbone RMSD

---

## 1.10 AbDiffuser

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Academic |
| **Publication** | NeurIPS 2024 |

### Architecture
- **Type:** Equivariant, physics-informed diffusion
- **Network:** APMixer (novel equivariant architecture)
- **Representation:** Full-atom with side chains
- **Numbering:** AHo system for variable-length handling

### Performance
- Structure RMSD: 0.4962 Å
- Expression: 100% (16/16)
- Binding: 57.1% tight binders

### Key Innovation
- Memory-efficient side chain generation
- Frame-averaging for canonical pose
- Trained on synthetic data from antibody structure prediction

---

## 1.11 IgDiff

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Academic |
| **Publication** | Journal of Computational Biology 2024 |

### Architecture
- **Type:** SE(3) diffusion (FrameDiff-based)
- **Scope:** Paired heavy + light chain backbone
- **Fine-tuned:** For antibody variable regions

### Performance
- scRMSD < 2 Å for best predictions
- 88% of generated antibodies: <2 Å RMSD in all CDR loops
- CDRH3 novelty: 1.39 Å ± 0.56 Å from training set
- Expression: High yield for all tested designs

---

## 1.12 DiffAbXL

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Academic |
| **Publication** | 2024 |

### Architecture
- **Type:** Scaled DiffAb with improved architecture
- **Training:** Large synthetic dataset
- **Focus:** CDR region sequence + structure

### Key Innovation
- Significant improvement from scaling to larger datasets
- Demonstrates importance of data diversity/volume

---

## 1.13 mBER (Million-scale Binder Experimental Ranking)

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Academic |
| **Publication** | bioRxiv 2025 |

### Key Innovation
- Open-source system for antibody-format binders
- Million-scale experimental screening capability
- Designed 1M+ VHH binders against 436 targets
- Screened against 145 targets → 100M+ binding interactions

---

## 1.14 Latent-X2

### Overview
| Attribute | Details |
|-----------|---------|
| **Developer** | Latent Labs |
| **Publication** | Technical Report 2025 |

### Performance
| Metric | Value |
|--------|-------|
| **Target success** | 50% (9/18) |
| **Designs per target** | 4-24 |
| **Formats** | VHH and scFv |
| **Immunogenicity** | Confirmed low in human donor panels |

### Key Innovation
- First de novo antibodies with confirmed low immunogenicity
- Zero-shot design with drug-like properties

---

# Part 2: Technical Comparison

## 2.1 Architecture Paradigms

| Paradigm | Models | Mechanism | Strengths | Weaknesses |
|----------|--------|-----------|-----------|------------|
| **Diffusion** | RFdiffusion, BoltzGen, Genie2, Chroma | Denoise from noise distribution | High quality, controllable | Compute intensive |
| **Flow Matching** | PPiFlow, FlowDesign, EvoDiff | Learn optimal transport | Efficient, flexible priors | Newer paradigm |
| **Hallucination** | BindCraft | Gradient through predictor | High success rate, no training | Can't leverage custom data |
| **Multi-objective** | Mosaic | Composite loss optimization | Ultimate flexibility | Requires expertise |

## 2.2 Structure Encoding Methods

| Method | Models | Description | Pros | Cons |
|--------|--------|-------------|------|------|
| **Backbone frames** | RFdiffusion, Genie2, IgDiff | Cα + orientation as SE(3) frames | Efficient, proven | Needs separate sequence design |
| **All-atom** | BoltzGen, Chroma, AbDiffuser | Every atom coordinate | Complete representation | Computationally expensive |
| **Virtual atoms** | BoltzGen | Geometric encoding of residue type | Novel, unified | Complex implementation |
| **Sequence-only** | EvoDiff | Amino acid tokens | Handles disorder | No explicit structure |

## 2.3 Equivariance Strategies

| Strategy | Models | Computational Cost |
|----------|--------|-------------------|
| **SE(3)-equivariant GNN** | Most diffusion models | High (days-weeks training) |
| **Invariant Point Attention** | AF2-based methods | Moderate |
| **Non-equivariant** | ProteinAE, recent trends | Low (1-2 orders faster) |

**Trend:** Recent work (AlphaFold3, Proteina, ProteinAE) moving toward non-equivariant architectures for efficiency while maintaining quality.

## 2.4 Conditioning Capabilities

| Model | Epitope | Constraints | Multi-motif | Natural Language |
|-------|---------|-------------|-------------|------------------|
| RFAntibody | ✓ | Limited | ✓ | ✗ |
| BoltzGen | ✓ | Full DSL | ✓ | ✗ |
| BindCraft | ✗ (auto) | Via losses | ✗ | ✗ |
| Mosaic | ✓ | Any model | ✓ | ✗ |
| Chroma | ✓ | ✓ | ✓ | ✓ |
| Genie 2 | ✓ | ✓ | ✓ | ✗ |

## 2.5 Post-Training / Fine-Tuning Capability

| Model | Training Code | Fine-tune Possible | Custom Data Integration |
|-------|---------------|-------------------|------------------------|
| **RFAntibody** | Licensed (Xaira) | Yes (demonstrated) | Requires license |
| **BoltzGen** | MIT license | Yes | Full flexibility |
| **BindCraft** | N/A (uses AF2) | No | Via loss modification |
| **Mosaic** | Open source | Yes | Native support |
| **PPiFlow** | Open source | Yes | Possible |
| **Genie 2** | Open source | Yes | Full support |
| **Chroma** | Open source | Yes | Possible |
| **EvoDiff** | Open source | Yes | Sequence data |

---

# Part 3: Benchmarks and Evaluation

## 3.1 Key Benchmarks

### AbBiBench (2025)
- **Scope:** 184,500+ experimental measurements
- **Targets:** 14 antibodies, 9 antigens (flu, HER2, VEGF, SARS-CoV-2, etc.)
- **Innovation:** Considers Ab-Ag complex as functional unit (not antibody alone)

### Adaptyv Protein Design Competition
- **Results:** 95% expression, 14% binding success (53/378)
- **Best designs:** Matched/exceeded Cetuximab (clinical therapeutic)
- **Improvement:** 5x better than 3 months prior

### SAbDab
- **Content:** <10,000 antigen-antibody complexes
- **Limitation:** Restricted data limits model generalization

## 3.2 Success Rate Comparison

| Model | Success Rate | Designs Tested | Notes |
|-------|-------------|----------------|-------|
| **BoltzGen** | 66% (novel targets) | 15/target | nM affinity |
| **BindCraft** | 10-100% | ≤10 | Varies by target |
| **Latent-X2** | 50% | 4-24/target | VHH + scFv |
| **AbDiffuser** | 57% | 16 | Tight binders |
| **RFAntibody** | 1-10%+ | 100-1000s | Target dependent |
| **Adaptyv competition** | 14% | 400 | EGFR target |

## 3.3 Selection and Ranking Methods

### Structure Prediction-Based
| Metric | Source | Performance |
|--------|--------|-------------|
| **ipSAE** (AF3) | Aligned errors | 1.4× better than ipAE |
| **ipTM** | Interface pTM | Standard but lower correlation with AF3 |
| **pDockQ** | Interface contacts | Good for AF2 |
| **DockQ** | Ground truth metric | Gold standard |

### Energy-Based
| Method | Use Case |
|--------|----------|
| FoldX | Rapid ΔΔG estimation |
| Rosetta | Interface scoring |
| Boltz-2 | Joint structure + affinity |

### Meta-Analysis Finding (3,766 binders)
> "Interface-focused metrics, most notably the AF3-derived ipSAE, outperform commonly used scores such as ipAE and ipTM."

---

# Part 4: Synthesis and Recommendations

## 4.1 Comparative Analysis

### For De Novo VHH/Antibody Design

| Criterion | Best Choice | Rationale |
|-----------|-------------|-----------|
| **Highest success rate** | BindCraft | 10-100% with few designs |
| **Lowest design count** | BindCraft | ≤10 designs typically |
| **Most flexible constraints** | BoltzGen | DSL for arbitrary constraints |
| **Epitope precision** | RFAntibody | Atomic-level targeting |
| **All formats (VHH/scFv/IgG)** | RFAntibody, BoltzGen | Full antibody support |
| **Non-antibody binders** | BoltzGen, RFdiffusion | Universal capability |

### For Affinity Maturation

| Criterion | Best Choice | Rationale |
|-----------|-------------|-----------|
| **Flow-based optimization** | PPiFlow | Partial flow for local redesign |
| **Gradient-based** | BindCraft | Continuous optimization |
| **Multi-objective** | Mosaic | Custom fitness functions |
| **With affinity prediction** | Boltz-2 + fine-tuning | Joint prediction |

### For Custom Data Integration

| Data Type | Best Approach |
|-----------|---------------|
| **Binding assay (Kd)** | Fine-tune ESM → predictor head |
| **DMS data** | Train FlowDesign or PPiFlow |
| **Structure data** | Mosaic with custom losses |
| **Large-scale screening** | mBER-style pipeline |

## 4.2 Recommended Pipeline Architecture

Based on your requirements (all formats, all objectives, full control, custom data):

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED HYBRID PIPELINE                            │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: CANDIDATE GENERATION (Parallel approaches)                    │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ BoltzGen    │  │ BindCraft   │  │ RFAntibody  │  │ PPiFlow     │    │
│  │ (universal, │  │ (highest    │  │ (epitope    │  │ (VHH        │    │
│  │ constrained)│  │ success)    │  │ precision)  │  │ maturation) │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│           │              │               │               │              │
│           └──────────────┴───────────────┴───────────────┘              │
│                                  │                                       │
│                                  ▼                                       │
│                         CANDIDATE POOL                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: MULTI-OBJECTIVE FILTERING (Mosaic framework)                  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Objective Functions:                                            │    │
│  │  - Binding affinity (Boltz-2 / custom fine-tuned)               │    │
│  │  - Developability (solubility, aggregation, stability)          │    │
│  │  - Specificity (off-target scoring)                              │    │
│  │  - Epitope compliance (structural alignment)                     │    │
│  │  - YOUR PROPRIETARY MODELS (trained on lab data)                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: STRUCTURE VALIDATION                                          │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │ AF3         │  │ Boltz-2     │  │ Chai-1      │                     │
│  │ (ipSAE)     │  │ (affinity)  │  │ (diversity) │                     │
│  └─────────────┘  └─────────────┘  └─────────────┘                     │
│                                                                          │
│  Consensus scoring: Ensemble predictions                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: RANKING & SELECTION                                           │
│                                                                          │
│  - ipSAE (best single metric)                                           │
│  - Boltz-2 affinity predictions                                         │
│  - Your fine-tuned ranking model                                        │
│  - Diversity clustering (ensure variety)                                │
│                                                                          │
│  Output: 10-50 candidates for experimental testing                      │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: EXPERIMENTAL VALIDATION & FEEDBACK                            │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    ACTIVE LEARNING LOOP                           │   │
│  │                                                                   │   │
│  │  Test in Lab → Collect Results → Update Models → Generate More   │   │
│  │       ▲                                                │          │   │
│  │       └────────────────────────────────────────────────┘          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4.3 Model Selection by Use Case

### Use Case 1: Quick De Novo VHH Design
**Recommended:** BindCraft
- Fastest path to experimental candidates
- Highest success rate per design
- No custom training needed

### Use Case 2: Precise Epitope Targeting
**Recommended:** RFAntibody → BoltzGen validation
- RFAntibody for atomic-level epitope specification
- BoltzGen for constraint satisfaction verification

### Use Case 3: Multi-Property Optimization
**Recommended:** Mosaic framework
- Integrate affinity + developability + specificity
- Train custom models on your proprietary data
- Full control over trade-offs

### Use Case 4: Large-Scale Library Generation
**Recommended:** BoltzGen + mBER-style pipeline
- BoltzGen for diverse generation
- Leverage constraint language for variety
- mBER approach for million-scale screening

### Use Case 5: Affinity Maturation
**Recommended:** PPiFlow + FlowDesign
- Flow-based partial redesign
- Maintains existing beneficial features
- Continuous optimization in structure space

### Use Case 6: Novel Formats (non-antibody)
**Recommended:** BoltzGen or Chroma
- Universal binder capability
- Protein, nucleic acid, small molecule targets

## 4.4 Integrating Your Experimental Data

### Approach 1: Fine-Tune Ranking Models
```python
# Train on your binding assay data
ESM-2 embeddings → Your Kd data → Ranking predictor
```

### Approach 2: Custom Mosaic Objectives
```python
# Integrate your model into Mosaic
loss = standard_objectives + your_proprietary_model(sequence)
```

### Approach 3: Active Learning Loop
1. Generate with BoltzGen/BindCraft
2. Test experimentally
3. Train/update ranking model
4. Use ranking model to prioritize next batch
5. Iterate

### Approach 4: Fine-Tune Generative Models
- BoltzGen, Genie 2, PPiFlow: Full training code available
- Can fine-tune on your specific target class
- Requires GPU cluster (days-weeks)

## 4.5 Recommended Mixture of Approaches

Based on your requirements, here's the optimal combination:

| Component | Primary Tool | Backup/Complement |
|-----------|-------------|-------------------|
| **De novo backbone** | BoltzGen (flexibility) | RFAntibody (precision) |
| **High success rate** | BindCraft | - |
| **Sequence design** | Integrated (BoltzGen) | ProteinMPNN |
| **Multi-objective** | Mosaic | Custom scripts |
| **Affinity prediction** | Boltz-2 | Fine-tuned ESM |
| **Structure validation** | AF3 + Boltz-2 ensemble | Chai-1 |
| **Ranking** | ipSAE + Boltz-2 affinity | Your fine-tuned model |
| **Lab integration** | Active learning loop | - |

### Implementation Priority

1. **Week 1-2:** Set up BindCraft + Mosaic
2. **Week 3-4:** Add BoltzGen for constraint-driven design
3. **Month 2:** Integrate Boltz-2 for affinity prediction
4. **Month 3:** Deploy active learning loop with lab data
5. **Ongoing:** Fine-tune ranking models as data accumulates

---

# References

## Primary Model Papers

1. [Atomically accurate de novo design of antibodies with RFdiffusion (Nature 2025)](https://www.nature.com/articles/s41586-025-09721-5)

2. [BoltzGen: Toward Universal Binder Design (2025)](https://jclinic.mit.edu/boltzgen/)

3. [One-shot design of functional protein binders with BindCraft (Nature 2025)](https://www.nature.com/articles/s41586-025-09429-6)

4. [Mosaic: Composite-objective protein design (GitHub)](https://github.com/escalante-bio/mosaic)

5. [PPIFlow: High-Affinity Protein Binder Design via Flow Matching (bioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.01.19.700484v1)

6. [FlowDesign: Improved design of antibody CDRs (Cell Systems 2025)](https://www.cell.com/cell-systems/abstract/S2405-4712(25)00103-6)

7. [EvoDiff: Protein generation with evolutionary diffusion (Microsoft Research)](https://www.microsoft.com/en-us/research/publication/protein-generation-with-evolutionary-diffusion-sequence-is-all-you-need/)

8. [Genie 2: Designing and Scaffolding Proteins (arXiv 2024)](https://arxiv.org/abs/2405.15489)

9. [Illuminating protein space with Chroma (Nature 2023)](https://www.nature.com/articles/s41586-023-06728-8)

10. [AbDiffuser: Full-atom generation of in-vitro functioning antibodies (NeurIPS 2024)](https://pubmed.ncbi.nlm.nih.gov/39349764/)

11. [IgDiff: De Novo Antibody Design with SE(3) Diffusion (J Comp Biol 2024)](https://www.liebertpub.com/doi/pdf/10.1089/cmb.2024.0768)

## Benchmarks and Evaluation

12. [AbBiBench: Benchmark for Antibody Binding Affinity (arXiv 2025)](https://arxiv.org/abs/2506.04235)

13. [Predicting Experimental Success: Meta-Analysis of 3,766 Binders (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.08.14.670059v1)

14. [Protein Design Competition Results (Adaptyv Bio)](https://www.adaptyvbio.com/blog/po104/)

15. [mBER: Controllable de novo antibody design with million-scale screening (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.09.26.678877v1)

## Technical Reviews

16. [AI-driven antibody design with generative diffusion models (Acta Pharmacol Sin 2024)](https://www.nature.com/articles/s41401-024-01380-y)

17. [Flow matching meets biology: a survey (npj AI 2025)](https://www.nature.com/articles/s44387-025-00066-y)

18. [Diffusion models in protein structure and docking (WIREs 2024)](https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1711)

19. [Structure-based protein and small molecule generation with EGNN and diffusion (CSBJ 2024)](https://www.sciencedirect.com/science/article/pii/S2001037024002228)

20. [Protein-SE(3): Benchmarking SE(3)-based Generative Models (arXiv 2025)](https://arxiv.org/html/2507.20243v1)

---

*Report compiled from comprehensive literature review, January 2026*
