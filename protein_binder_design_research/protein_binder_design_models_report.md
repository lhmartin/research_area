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
5. [Designing and Training Your Own Model](#part-5-designing-and-training-your-own-model)
6. [References](#references)

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

# Part 5: Designing and Training Your Own Model

## 5.1 Decision Framework: Build vs. Fine-Tune vs. Use

Before investing in custom model development, consider:

| Approach | When to Use | Effort | Data Needed |
|----------|-------------|--------|-------------|
| **Use existing models** | Standard targets, proof-of-concept | Low | None |
| **Fine-tune existing** | Specific target class, proprietary data | Medium | 100-10K examples |
| **Train from scratch** | Novel architecture, unique requirements | High | 10K-1M+ examples |
| **Hybrid pipeline** | Combine strengths of multiple approaches | Medium-High | Varies |

**Recommendation:** Start with existing models, fine-tune as data accumulates, only build custom components where existing tools fall short.

---

## 5.2 Architecture Design Choices

### 5.2.1 Generative Paradigm Selection

| Paradigm | Architecture | Pros | Cons | Best For |
|----------|-------------|------|------|----------|
| **Diffusion (DDPM)** | Iterative denoising | High quality, stable training | Slow sampling | Structure generation |
| **Flow Matching** | Continuous normalizing flows | Efficient, flexible priors | Newer, less mature | Sequence-structure co-design |
| **Autoregressive** | Sequential generation | Fast, well-understood | Order dependency | Sequence generation |
| **VAE/CVAE** | Latent space optimization | Fast sampling, interpretable | Mode collapse risk | Optimization/maturation |
| **Hallucination** | Gradient through predictor | No training needed | Can't learn from data | Quick prototyping |

### 5.2.2 Backbone Architecture Options

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE DECISION TREE                            │
└─────────────────────────────────────────────────────────────────────────┘

Do you need 3D structure generation?
├── YES: Choose structure representation
│   ├── Backbone only (Cα or frames)?
│   │   └── Use: SE(3) diffusion (Genie2, RFdiffusion style)
│   │       - Faster training, proven approach
│   │       - Requires separate sequence design (ProteinMPNN)
│   │
│   └── All-atom?
│       └── Use: All-atom diffusion (BoltzGen, Chroma style)
│           - Complete representation, no post-processing
│           - More compute intensive
│
└── NO: Sequence-only
    └── Use: Language model fine-tuning or EvoDiff
        - Can handle disordered regions
        - No explicit structure modeling
```

### 5.2.3 Equivariance Strategy

| Strategy | Implementation | Compute Cost | Quality |
|----------|---------------|--------------|---------|
| **SE(3)-equivariant GNN** | E3NN, EGNN | High (2-24 GPU-weeks) | Excellent |
| **Invariant Point Attention** | AlphaFold-style | Medium | Excellent |
| **Frame averaging** | Average over rotations | Medium | Good |
| **Non-equivariant** | Standard transformer | Low (1-2 GPU-days) | Good (with enough data) |

**Recent Trend:** AlphaFold3, Proteina, and ProteinAE use non-equivariant architectures with comparable quality at 10-100× lower compute.

### 5.2.4 Recommended Architecture Components

```python
# Modern protein binder design architecture (2025 style)

class BinderDesignModel(nn.Module):
    def __init__(self):
        # 1. Encoder: Process target structure
        self.target_encoder = StructureEncoder(
            node_features=128,
            edge_features=64,
            num_layers=6,
            attention_heads=8
        )

        # 2. Conditioning: Incorporate constraints
        self.constraint_encoder = ConstraintEmbedding(
            epitope_dim=64,
            property_dim=32
        )

        # 3. Generator: Diffusion/flow backbone
        self.denoiser = DiffusionNetwork(
            hidden_dim=256,
            num_layers=12,
            time_embedding_dim=64
        )

        # 4. Sequence head (if backbone-only)
        self.sequence_head = ProteinMPNN_style(
            hidden_dim=128,
            num_layers=3
        )
```

---

## 5.3 Structure Representation and Encoding

### 5.3.1 Protein Structure Representations

| Representation | Description | Used By | Dimensionality |
|----------------|-------------|---------|----------------|
| **Cα coordinates** | Alpha carbon positions | Early models | N × 3 |
| **Backbone frames** | Cα + N-Cα-C orientation | RFdiffusion, Genie2 | N × (3 + 9) |
| **Full backbone** | N, Cα, C, O coordinates | ProteinAE | N × 12 |
| **All-atom** | Every atom coordinate | BoltzGen, Chroma | N × ~14 × 3 |
| **Virtual atoms** | Geometric residue encoding | BoltzGen | Novel approach |

### 5.3.2 SE(3) Frame Representation

```python
# Standard frame representation (RFdiffusion/Genie2 style)
class RigidFrame:
    """
    Represents protein residue as rigid body transformation
    - Translation: Cα position (3D vector)
    - Rotation: Local coordinate frame (3×3 rotation matrix)
    """
    def __init__(self, translation, rotation):
        self.t = translation  # Shape: (N, 3)
        self.R = rotation     # Shape: (N, 3, 3)

    def compose(self, other):
        """Compose two rigid transformations"""
        new_R = self.R @ other.R
        new_t = self.t + self.R @ other.t
        return RigidFrame(new_t, new_R)

    def to_coords(self, ideal_coords):
        """Convert frame to atom coordinates"""
        # ideal_coords: canonical N, Cα, C positions
        return self.R @ ideal_coords + self.t
```

### 5.3.3 Node and Edge Features

**Node Features (per residue):**
```python
node_features = {
    'amino_acid': one_hot(20),           # Residue type
    'position': positional_encoding(N),   # Sequence position
    'secondary_structure': one_hot(8),    # DSSP classification
    'phi_psi': sincos_encoding(2),        # Backbone dihedrals
    'chi_angles': sincos_encoding(4),     # Side chain dihedrals
    'sasa': scalar(1),                    # Solvent accessibility
    'bfactor': scalar(1),                 # Flexibility proxy
    'plm_embedding': esm2_embedding(1280) # Pre-trained PLM
}
```

**Edge Features (per residue pair):**
```python
edge_features = {
    'distance': rbf_encoding(16),         # Cα-Cα distance
    'direction': unit_vector(3),          # Relative direction
    'sequence_offset': one_hot(32),       # |i - j| encoding
    'contact_type': one_hot(4),           # Backbone/sidechain contact
    'orientation': quaternion(4)          # Relative frame orientation
}
```

### 5.3.4 Distance and Angle Encodings

```python
# Radial Basis Function encoding for distances
def rbf_encoding(distances, D_min=0.0, D_max=20.0, num_rbf=16):
    D_mu = torch.linspace(D_min, D_max, num_rbf)
    D_sigma = (D_max - D_min) / num_rbf
    return torch.exp(-((distances.unsqueeze(-1) - D_mu) ** 2) / (2 * D_sigma ** 2))

# Sinusoidal encoding for angles
def sincos_encoding(angles):
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

# Fourier positional encoding
def positional_encoding(positions, num_freqs=16):
    freqs = 2 ** torch.linspace(0, num_freqs-1, num_freqs)
    angles = positions.unsqueeze(-1) * freqs * np.pi
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
```

---

## 5.4 Training Data Preparation

### 5.4.1 Data Sources

| Source | Content | Size | Access |
|--------|---------|------|--------|
| **PDB** | Experimental structures | ~220K structures | Public |
| **AlphaFold DB** | Predicted structures | 200M+ proteins | Public |
| **SAbDab** | Antibody structures | ~10K Ab-Ag complexes | Public |
| **Proprietary** | Your experimental data | Varies | Internal |

### 5.4.2 Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────┘

1. DOWNLOAD & FILTER
   ├── PDB: Resolution < 3.0 Å, no missing residues
   ├── Remove redundancy (sequence clustering at 30-50%)
   └── Filter by quality metrics (Rfree, clashscore)

2. STRUCTURE PROCESSING
   ├── Parse PDB/mmCIF files
   ├── Extract backbone coordinates
   ├── Compute frames (translation + rotation)
   ├── Calculate secondary structure (DSSP)
   └── Compute contact maps

3. FEATURE EXTRACTION
   ├── ESM-2 embeddings for each sequence
   ├── Geometric features (distances, angles)
   ├── Surface features (SASA, electrostatics)
   └── Interface annotations (for complexes)

4. DATASET SPLITTING
   ├── Cluster by sequence similarity
   ├── Split clusters (not sequences) into train/val/test
   └── Ensure no leakage between splits

5. DATA AUGMENTATION
   ├── Random rotations/translations (if not equivariant)
   ├── Cropping to fixed length
   └── Coordinate noise (optional)
```

### 5.4.3 Handling Antibody-Specific Data

```python
# Antibody data processing example
class AntibodyDataset:
    def __init__(self, sabdab_path):
        self.data = []

    def process_complex(self, pdb_file):
        # 1. Parse structure
        structure = parse_pdb(pdb_file)

        # 2. Identify chains
        heavy_chain = identify_heavy_chain(structure)  # By sequence patterns
        light_chain = identify_light_chain(structure)
        antigen = identify_antigen(structure)

        # 3. Number residues (IMGT/AHo scheme)
        numbered_heavy = anarci_number(heavy_chain, scheme='imgt')
        numbered_light = anarci_number(light_chain, scheme='imgt')

        # 4. Extract CDR loops
        cdrs = {
            'H1': extract_region(numbered_heavy, 27, 38),
            'H2': extract_region(numbered_heavy, 56, 65),
            'H3': extract_region(numbered_heavy, 105, 117),
            'L1': extract_region(numbered_light, 27, 38),
            'L2': extract_region(numbered_light, 56, 65),
            'L3': extract_region(numbered_light, 105, 117),
        }

        # 5. Identify interface residues
        interface = compute_interface(antibody, antigen, cutoff=8.0)

        return {
            'heavy': heavy_chain,
            'light': light_chain,
            'antigen': antigen,
            'cdrs': cdrs,
            'interface': interface
        }
```

### 5.4.4 Synthetic Data Augmentation

```python
# Generate synthetic training data using structure prediction
def generate_synthetic_complexes(sequences, predictor='boltz'):
    synthetic_data = []

    for ab_seq, ag_seq in sequences:
        # Predict complex structure
        if predictor == 'boltz':
            complex_structure = boltz_predict(ab_seq, ag_seq)
        elif predictor == 'af2m':
            complex_structure = af2m_predict(ab_seq, ag_seq)

        # Filter by confidence
        if complex_structure.confidence > 0.7:
            synthetic_data.append(complex_structure)

    return synthetic_data
```

---

## 5.5 Loss Functions and Training Objectives

### 5.5.1 Diffusion Training Loss

```python
# Standard diffusion loss (score matching)
def diffusion_loss(model, x_0, noise_schedule):
    # Sample timestep
    t = torch.randint(0, T, (batch_size,))

    # Sample noise
    epsilon = torch.randn_like(x_0)

    # Create noisy sample
    alpha_t = noise_schedule.alpha(t)
    x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon

    # Predict noise
    epsilon_pred = model(x_t, t)

    # L2 loss
    loss = F.mse_loss(epsilon_pred, epsilon)

    return loss
```

### 5.5.2 Flow Matching Loss

```python
# Conditional flow matching loss
def flow_matching_loss(model, x_0, x_1):
    # x_0: noise (prior)
    # x_1: data (target structure)

    # Sample time
    t = torch.rand(batch_size, 1)

    # Interpolate
    x_t = (1 - t) * x_0 + t * x_1

    # Target velocity
    v_target = x_1 - x_0

    # Predicted velocity
    v_pred = model(x_t, t)

    # L2 loss on velocity
    loss = F.mse_loss(v_pred, v_target)

    return loss
```

### 5.5.3 Structure-Specific Losses

```python
class StructureLoss:
    def __init__(self):
        self.weights = {
            'coordinate': 1.0,
            'distance': 0.5,
            'angle': 0.3,
            'clash': 0.1,
            'secondary_structure': 0.2
        }

    def coordinate_loss(self, pred, target):
        """L2 loss on atom coordinates (after alignment)"""
        aligned_pred = kabsch_align(pred, target)
        return F.mse_loss(aligned_pred, target)

    def distance_loss(self, pred, target):
        """Loss on pairwise distances (SE(3) invariant)"""
        pred_dist = torch.cdist(pred, pred)
        target_dist = torch.cdist(target, target)
        return F.mse_loss(pred_dist, target_dist)

    def dihedral_loss(self, pred, target):
        """Loss on backbone dihedrals"""
        pred_angles = compute_dihedrals(pred)
        target_angles = compute_dihedrals(target)
        # Angular loss (handles periodicity)
        return 1 - torch.cos(pred_angles - target_angles).mean()

    def clash_loss(self, coords, min_dist=2.0):
        """Penalize steric clashes"""
        distances = torch.cdist(coords, coords)
        clashes = F.relu(min_dist - distances)
        # Mask diagonal
        mask = ~torch.eye(len(coords), dtype=bool)
        return clashes[mask].sum()

    def compute(self, pred, target):
        total = 0
        for name, weight in self.weights.items():
            loss_fn = getattr(self, f'{name}_loss')
            total += weight * loss_fn(pred, target)
        return total
```

### 5.5.4 Binding-Specific Losses

```python
class BindingLoss:
    def interface_quality(self, binder_coords, target_coords):
        """Reward good interface contacts"""
        distances = torch.cdist(binder_coords, target_coords)
        contacts = (distances < 8.0).float()

        # Reward number of contacts
        contact_score = contacts.sum()

        # Penalize too close (clashes)
        clash_score = (distances < 2.5).float().sum()

        return -contact_score + 10 * clash_score

    def shape_complementarity(self, binder_surface, target_surface):
        """Measure shape complementarity at interface"""
        # Compute surface normals
        # Check alignment of normals across interface
        pass

    def hydrophobic_matching(self, binder, target):
        """Ensure hydrophobic residues face hydrophobic patches"""
        pass
```

### 5.5.5 Multi-Objective Training

```python
class MultiObjectiveLoss:
    def __init__(self, objectives):
        self.objectives = objectives

    def compute(self, model_output, targets):
        losses = {}

        # Structure quality
        losses['structure'] = structure_loss(model_output, targets)

        # Binding affinity proxy
        losses['binding'] = binding_score_loss(model_output)

        # Developability
        losses['solubility'] = solubility_loss(model_output.sequence)
        losses['aggregation'] = aggregation_propensity(model_output.sequence)

        # Weighted combination
        total = sum(self.objectives[k] * v for k, v in losses.items())

        return total, losses
```

---

## 5.6 Training Infrastructure and Hyperparameters

### 5.6.1 Compute Requirements

| Model Type | GPU Memory | Training Time | Hardware |
|------------|------------|---------------|----------|
| **Backbone diffusion** | 24-40 GB | 3-7 days | 4-8× A100 |
| **All-atom diffusion** | 40-80 GB | 1-3 weeks | 8× A100/H100 |
| **PLM fine-tuning (LoRA)** | 16-24 GB | 1-3 days | 1-2× A100 |
| **Ranking model** | 8-16 GB | Hours-1 day | 1× A100 |

### 5.6.2 Recommended Hyperparameters

```yaml
# Training configuration (diffusion model)
training:
  batch_size: 32-64
  learning_rate: 1e-4
  lr_schedule: cosine_with_warmup
  warmup_steps: 5000
  total_steps: 500000-1000000
  gradient_clip: 1.0
  ema_decay: 0.999

# Diffusion parameters
diffusion:
  num_timesteps: 1000
  noise_schedule: cosine  # or linear
  beta_start: 0.0001
  beta_end: 0.02

# Architecture
model:
  hidden_dim: 256-512
  num_layers: 12-24
  attention_heads: 8-16
  dropout: 0.0-0.1

# Data
data:
  max_length: 256-512
  crop_size: 128-256
  augmentation: true
```

### 5.6.3 Training Loop Template

```python
def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass
        loss, metrics = compute_loss(model, batch)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update
        optimizer.step()
        scheduler.step()

        # EMA update
        ema_update(model, ema_model, decay=0.999)

        total_loss += loss.item()

        # Log metrics
        if step % log_interval == 0:
            wandb.log(metrics)

    return total_loss / len(dataloader)
```

### 5.6.4 Distributed Training Setup

```python
# Multi-GPU training with PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_distributed():
    local_rank = setup_distributed()

    model = BinderModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Use DistributedSampler for data
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler)

    # Training loop...
```

---

## 5.7 Evaluation and Validation

### 5.7.1 In-Silico Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Designability** | ESMFold RMSD < 2Å | > 90% |
| **scRMSD** | Self-consistency RMSD | < 2Å |
| **pLDDT** | Predicted confidence | > 70 |
| **ipTM/ipSAE** | Interface quality | > 0.7/0.8 |
| **Novelty** | Distance to training set | > 2Å |
| **Diversity** | Pairwise RMSD in generated set | High variance |

### 5.7.2 Validation Pipeline

```python
class DesignValidator:
    def __init__(self):
        self.predictors = {
            'esmfold': ESMFold(),
            'af2': AlphaFold2(),
            'boltz': Boltz()
        }

    def validate_design(self, designed_structure, designed_sequence):
        results = {}

        # 1. Self-consistency (does sequence fold to designed structure?)
        for name, predictor in self.predictors.items():
            predicted = predictor.predict(designed_sequence)
            rmsd = compute_rmsd(predicted, designed_structure)
            results[f'{name}_rmsd'] = rmsd

        # 2. Confidence scores
        results['plddt'] = predicted.plddt.mean()

        # 3. Binding prediction (if complex)
        complex_pred = self.predictors['boltz'].predict_complex(
            designed_sequence, target_sequence
        )
        results['iptm'] = complex_pred.iptm
        results['binding_affinity'] = complex_pred.affinity

        # 4. Developability
        results['solubility'] = predict_solubility(designed_sequence)
        results['aggregation'] = predict_aggregation(designed_sequence)

        return results
```

### 5.7.3 Experimental Validation Priorities

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 EXPERIMENTAL VALIDATION FUNNEL                          │
└─────────────────────────────────────────────────────────────────────────┘

TIER 1: In-silico filtering (free)
├── Structure prediction confidence > threshold
├── Self-consistency check passes
└── No severe predicted clashes
    ↓ ~10-20% pass rate

TIER 2: Expression testing ($, 1-2 weeks)
├── Express in E. coli/mammalian cells
├── Check solubility, yield
└── Basic QC (SDS-PAGE, SEC)
    ↓ ~50-80% expression rate

TIER 3: Binding assays ($$, 1-2 weeks)
├── ELISA/flow cytometry (qualitative)
├── SPR/BLI (quantitative Kd)
└── Competition assays
    ↓ ~10-50% binding rate

TIER 4: Detailed characterization ($$$, weeks-months)
├── Structural validation (X-ray, cryo-EM)
├── Functional assays
├── Stability testing
└── Immunogenicity prediction
```

---

## 5.8 Fine-Tuning Existing Models

### 5.8.1 When to Fine-Tune vs. Train from Scratch

| Scenario | Approach | Rationale |
|----------|----------|-----------|
| < 100 examples | Use pre-trained + ranking | Insufficient for fine-tuning |
| 100-1000 examples | LoRA fine-tuning | Enough for adaptation |
| 1000-10000 examples | Full fine-tuning | Significant domain shift |
| > 10000 examples | Consider training from scratch | May outperform fine-tuning |

### 5.8.2 LoRA Fine-Tuning Protocol

```python
from peft import LoraConfig, get_peft_model

# 1. Load pre-trained model
base_model = load_pretrained('boltzgen')  # or ESM-2, etc.

# 2. Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank (4-16 typical)
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj",  # Attention projections
        "dense"              # Optional: FFN layers
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # or appropriate task type
)

# 3. Create PEFT model
model = get_peft_model(base_model, lora_config)
print(f"Trainable params: {model.num_parameters(only_trainable=True):,}")
# Typically ~1-5% of original parameters

# 4. Fine-tune on your data
trainer = Trainer(
    model=model,
    train_dataset=your_dataset,
    learning_rate=1e-4,  # Can be higher than full fine-tuning
    num_epochs=10-50
)
trainer.train()

# 5. Merge weights for inference (optional)
merged_model = model.merge_and_unload()
```

### 5.8.3 Fine-Tuning Different Components

```python
# Strategy: Freeze backbone, train task head
class FineTunedBinderModel:
    def __init__(self, pretrained_backbone):
        # Freeze backbone
        self.backbone = pretrained_backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable task heads
        self.binding_head = nn.Sequential(
            nn.Linear(backbone.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Binding affinity prediction
        )

        self.design_head = nn.Sequential(
            nn.Linear(backbone.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 20)  # Amino acid probabilities
        )

    def forward(self, x):
        # Get frozen embeddings
        with torch.no_grad():
            embeddings = self.backbone(x)

        # Train task heads
        binding = self.binding_head(embeddings)
        design = self.design_head(embeddings)

        return binding, design
```

---

## 5.9 Building a Complete Pipeline

### 5.9.1 End-to-End Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE BINDER DESIGN PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  INPUTS                                                                  │
│  ├── Target structure (PDB/predicted)                                   │
│  ├── Design constraints (epitope, properties)                           │
│  └── Optional: Starting binder sequence                                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 1: BACKBONE GENERATION                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │ Diffusion   │  │ Flow Match  │  │ Hallucinate │                     │
│  │ (BoltzGen)  │  │ (PPiFlow)   │  │ (BindCraft) │                     │
│  └─────────────┘  └─────────────┘  └─────────────┘                     │
│                         │                                                │
│                         ▼                                                │
│                 N backbone candidates                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 2: SEQUENCE DESIGN (if backbone-only generation)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │ ProteinMPNN │  │ ESM-IF     │  │ LigandMPNN  │                     │
│  └─────────────┘  └─────────────┘  └─────────────┘                     │
│                         │                                                │
│                         ▼                                                │
│               M sequences per backbone                                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 3: STRUCTURE VALIDATION                                         │
│  ├── ESMFold: Self-consistency check                                   │
│  ├── AF2/Boltz: Complex prediction                                     │
│  └── Filter: RMSD < 2Å, confidence > 0.7                               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 4: MULTI-PROPERTY SCORING                                       │
│  ├── Binding affinity (Boltz-2)                                        │
│  ├── Developability (in-house/public models)                           │
│  ├── Specificity (off-target prediction)                               │
│  └── Diversity (cluster representatives)                               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 5: RANKING & SELECTION                                          │
│  ├── Ensemble scoring (combine multiple metrics)                       │
│  ├── Pareto front selection (multi-objective)                          │
│  └── Diversity selection (maximize coverage)                           │
│                         │                                                │
│                         ▼                                                │
│               Top K candidates for testing                              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODULE 6: EXPERIMENTAL FEEDBACK                                         │
│  ├── Express and test candidates                                       │
│  ├── Collect binding data                                              │
│  └── Update models (active learning)                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.9.2 Pipeline Implementation

```python
class BinderDesignPipeline:
    def __init__(self, config):
        # Initialize modules
        self.backbone_generator = BackboneGenerator(config.generator)
        self.sequence_designer = SequenceDesigner(config.seq_design)
        self.validator = StructureValidator(config.validation)
        self.scorer = MultiPropertyScorer(config.scoring)
        self.ranker = CandidateRanker(config.ranking)

    def design(self, target, constraints, num_designs=1000):
        # Stage 1: Generate backbones
        backbones = self.backbone_generator.generate(
            target=target,
            constraints=constraints,
            num_samples=num_designs
        )

        # Stage 2: Design sequences
        candidates = []
        for backbone in backbones:
            sequences = self.sequence_designer.design(
                backbone=backbone,
                target=target,
                num_sequences=8
            )
            for seq in sequences:
                candidates.append({
                    'backbone': backbone,
                    'sequence': seq
                })

        # Stage 3: Validate structures
        validated = []
        for candidate in candidates:
            result = self.validator.validate(candidate)
            if result['passes']:
                candidate['validation'] = result
                validated.append(candidate)

        # Stage 4: Score on multiple properties
        for candidate in validated:
            scores = self.scorer.score(candidate)
            candidate['scores'] = scores

        # Stage 5: Rank and select
        ranked = self.ranker.rank(validated)
        top_candidates = ranked[:config.num_output]

        return top_candidates

    def update_from_experiments(self, results):
        """Update models based on experimental results"""
        self.scorer.update(results)
        self.ranker.update(results)
```

---

## 5.10 Common Pitfalls and Solutions

### 5.10.1 Training Issues

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Mode collapse** | All designs similar | Increase diversity loss, temperature |
| **Poor convergence** | Loss plateaus high | Reduce LR, check data quality |
| **Overfitting** | Train/val gap | More data, dropout, early stopping |
| **Unstable training** | Loss spikes | Gradient clipping, reduce LR |
| **OOM errors** | CUDA out of memory | Reduce batch size, gradient checkpointing |

### 5.10.2 Design Quality Issues

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Poor designability** | ESMFold RMSD high | More structure-aware training |
| **Clashing atoms** | Short interatomic distances | Add clash loss |
| **Unrealistic loops** | Distorted geometry | Better loop modeling, more data |
| **Low binding** | Experimental fails | Interface-focused losses |

### 5.10.3 Data Issues

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Data leakage** | Inflated test metrics | Cluster-based splits |
| **Imbalanced data** | Biased predictions | Resampling, class weights |
| **Noisy labels** | Inconsistent training | Data cleaning, robust losses |
| **Limited data** | Poor generalization | Transfer learning, data augmentation |

---

## 5.11 Recommended Development Roadmap

### Phase 1: Foundation (Weeks 1-4)
```
Week 1-2: Setup & Data
├── Set up compute infrastructure
├── Download and process training data (PDB, SAbDab)
├── Implement data loading pipeline
└── Baseline: Run existing models (BindCraft, BoltzGen)

Week 3-4: Simple Model
├── Implement basic diffusion model
├── Train on subset of data
├── Validate self-consistency
└── Compare to existing baselines
```

### Phase 2: Core Development (Months 2-3)
```
Month 2: Full Training
├── Scale to full dataset
├── Implement all loss functions
├── Add conditioning mechanisms
├── Hyperparameter tuning

Month 3: Refinement
├── Add multi-objective optimization
├── Implement sequence design module
├── Build validation pipeline
└── Initial experimental testing
```

### Phase 3: Production (Months 4-6)
```
Month 4-5: Integration
├── Build end-to-end pipeline
├── Integrate with lab workflows
├── Active learning implementation
└── Scale experimental testing

Month 6: Optimization
├── Fine-tune on experimental data
├── Iterate based on results
├── Document and deploy
└── Continuous improvement
```

### Key Milestones

| Milestone | Target | Success Criterion |
|-----------|--------|-------------------|
| **M1: Self-consistency** | Week 4 | >80% designs fold correctly |
| **M2: Binding prediction** | Week 8 | Correlation with experimental Kd |
| **M3: First binders** | Week 12 | >10% experimental binding rate |
| **M4: Optimized pipeline** | Week 20 | >30% binding rate |
| **M5: Production** | Week 24 | Routine design campaigns |

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
