# State-of-the-Art Research in Protein-Protein Binding Prediction

**Research Compilation Date: January 2026**

---

## Executive Summary

Protein-protein interaction (PPI) prediction has undergone a transformative evolution from 2024-2025, driven by advances in deep learning architectures, protein language models, and diffusion-based generative approaches. This document synthesizes the latest state-of-the-art research across multiple dimensions of PPI prediction, including structure prediction, binding affinity estimation, interaction site identification, and mutation effect prediction.

---

## Table of Contents

1. [Deep Learning Architectures](#1-deep-learning-architectures)
2. [Structure Prediction Models](#2-structure-prediction-models)
3. [Protein Language Models](#3-protein-language-models)
4. [Graph Neural Networks](#4-graph-neural-networks)
5. [Diffusion-Based Methods](#5-diffusion-based-methods)
6. [Binding Affinity Prediction](#6-binding-affinity-prediction)
7. [Antibody-Antigen Interactions](#7-antibody-antigen-interactions)
8. [Interface & Hotspot Prediction](#8-interface--hotspot-prediction)
9. [Mutation Effect Prediction](#9-mutation-effect-prediction)
10. [Key Datasets & Benchmarks](#10-key-datasets--benchmarks)
11. [Tools & Software](#11-tools--software)
12. [Current Challenges & Future Directions](#12-current-challenges--future-directions)
13. [References & Sources](#13-references--sources)

---

## 1. Deep Learning Architectures

### Overview of Current Approaches

Deep learning has become the dominant paradigm for PPI prediction, with several key architectural families:

| Architecture | Strengths | Key Methods |
|-------------|-----------|-------------|
| **Transformers** | Global context, attention mechanisms | ESM-2, ESM3, PLM-interact |
| **Graph Neural Networks** | Natural protein structure representation | MGPPI, GNNGL-PPI, DeepRank-GNN |
| **Convolutional Networks** | Local pattern recognition | DeepCompoundNet, KSGPPI |
| **Diffusion Models** | Generative structure prediction | AlphaFold3, DiffDock, Boltz |

### Transformer-Based Models

Transformers have emerged as the dominant architecture due to their ability to capture long-range dependencies through self-attention mechanisms:

- **GACT-PPIS (2024)**: Integrates enhanced graph attention with deep transformer networks
- **EnsemPPIS**: Uses transformer decoders to extract pairwise residue interactions
- **TransformerPPIS**: Extracts residue interaction information and global protein features

**Key insight**: Transformers overcome GNN limitations by directly modeling global dependencies regardless of spatial or sequential distance, capturing both local and long-range interactions without graph topology constraints.

### Multi-Modal Integration

Modern approaches increasingly combine multiple data modalities:

- Sequence features (amino acid composition, physicochemical properties)
- Structural features (3D coordinates, contact maps)
- Evolutionary features (MSA, conservation scores)
- Network features (PPI network topology)

**EGRET** exemplifies this trend, combining evolutionary features with graph topological features to bridge classical handcrafted approaches with modern transformer-based multimodal frameworks.

---

## 2. Structure Prediction Models

### AlphaFold 3 (2024)

AlphaFold 3 represents the current state-of-the-art for biomolecular structure prediction:

**Key Innovations:**
- Diffusion-based architecture (replacing template-based methods)
- Unified prediction of proteins, nucleic acids, small molecules, and complexes
- Atomic-level and token-level representations with local/global attention

**Performance:**
- 64.9% success rate on protein-ligand interactions (FoldBench)
- 60% success rate for antibody docking (with 1,000 seeds)
- Relatively good Pearson correlation (0.86) on SKEMPI 2.0 complexes

**Limitations:**
- Overconfident predictions in flexible/unconventional systems
- 65% failure rate for single-seed antibody/nanobody docking
- 8.6% increase in RMSE compared to PDB structures for BFE prediction

### Boltz Family

**Boltz-1** (MIT CSAIL/Jameel Clinic):
- First fully open-source model approaching AlphaFold3 accuracy
- Released under MIT license with full training code

**Boltz-2** (2025):
- Joint modeling of complex structures AND binding affinities
- First deep learning model approaching physics-based FEP accuracy
- 1000x faster than FEP methods
- Enables practical in silico screening for drug discovery

### Chai-1

- Multi-modal foundation model for molecular structure prediction
- Architecture follows AlphaFold 3 with key additions:
  - Residue-level PLM embeddings for single-sequence prediction
  - Trainable constraint features (pocket, contact, docking)
- Available under Apache license for commercial drug discovery

### RoseTTAFold Family

**RoseTTAFold (Original):**
- Three-track network (1D sequence, 2D interactions, 3D structure)
- Information flows back and forth between tracks

**RoseTTAFold All-Atom (2024):**
- Extended to proteins, nucleic acids, small molecules, metals, modifications
- Trained on protein-small molecule, protein-metal, and modified protein complexes

**RoseTTAFoldNA:**
- Specialized for protein-DNA and protein-RNA complexes

**RF2-PPI:**
- Outperforms AlphaFold-multimer and AF3 in distinguishing true PPIs from random pairs
- Performance correlates with interface size

### Comparative Analysis (2024-2025)

| Model | Protein-Ligand | Antibody-Antigen | Open Source |
|-------|---------------|------------------|-------------|
| AlphaFold 3 | 64.9% | 60% (1000 seeds) | Academic (Nov 2024) |
| Boltz-1 | ~55% | ~4% | MIT License |
| Chai-1 | Comparable | ~0% | Apache License |
| HelixFold-Multimer | - | 52.7% | - |

### Specialized Tools

**AF3Complex (2025):**
- Enhanced AlphaFold 3 for complex prediction
- Outperforms AF3 on protein-peptide and antibody-antigen cases

**SiteAF3 (2025):**
- Site-specific modeling tool enhancing AF3
- Excels at small molecule binding, peptide, and nucleic acid interactions
- Particularly powerful for orphan proteins and allosteric sites

**CombFold:**
- Combinatorial assembly algorithm with AlphaFold2
- Designed for large protein assemblies

---

## 3. Protein Language Models

### ESM Family (Meta/EvolutionaryScale)

**ESM-2:**
- Widely adopted for sequence encoding
- Pretrained on millions of protein sequences
- Provides embeddings capturing biochemical and evolutionary features

**ESM3 (January 2025, Science):**
- Multi-track transformer for sequence, structure, and function
- 98 billion parameters at largest scale
- Trained on 2.78 billion proteins, 771 billion tokens
- 1.07×10²⁴ FLOPs training compute

**ESM C (December 2024):**
- Best representation learning model from EvolutionaryScale
- Non-diminishing returns up to 6 billion parameters
- Suggests continued scaling will yield improvements

### PLM-interact (2025)

A breakthrough approach extending ESM-2 for PPI prediction:

**Key Innovations:**
1. Extended sequence lengths for paired masked-language training
2. "Next sentence" prediction fine-tuning for protein pairs
3. Joint encoding of protein pairs (analogous to NLP tasks)

**Performance:**
- State-of-the-art cross-species PPI prediction
- Successfully generalizes: human → mouse, fly, worm, yeast, E. coli
- Applicable to virus-host PPI prediction
- Fine-tuning method for mutation effect detection

### Integration with Structural Methods

**DeepRank-GNN-esm (2024):**
- Integrates ESM-2 embeddings into graph neural network framework
- Language model embeddings can replace computationally expensive PSSM features
- Maintains or slightly improves performance vs. traditional approaches

**KSGPPI (2024):**
- Hybrid method integrating sequences and networks
- Uses ESM-2 for global pattern extraction
- CKSAAP + 2D CNN for local information

**DGCPPISP:**
- Two-stage transfer learning with dynamic GCN
- ESM-2 encoding + four sequence features

### Key Trend

> "It has become an irreversible trend for protein language models to gradually replace traditional sequence coding methods. In the current research context, the adoption of protein language models has become an inevitable choice for protein function prediction models to remain competitive."

---

## 4. Graph Neural Networks

### Why Graphs for PPI?

Graphs naturally represent:
- Residues as nodes
- Interactions (bonds, contacts) as edges
- Both intra-protein and inter-protein relationships

### Recent GNN Methods (2024-2025)

**MGPPI (2024) - Multiscale GNN:**
- Addresses insufficient structural information extraction
- Improved interpretability for PPI prediction
- Applications in cancer diagnosis and drug development

**GNNGL-PPI (2024):**
- Multi-category PPI prediction
- Combines global graphs and local subgraphs
- Uses Graph Isomorphism Network (GIN) for global features

**ProtGram-DirectGCN (2025):**
- Models protein structure as hierarchy of n-gram graphs
- Edge weights from residue transition probabilities
- Custom directed GCN with path-specific transformations
- Learnable gating mechanism

**MVGNN-PPIS (2025):**
- Multi-view graph neural network
- Based on AlphaFold3-predicted structures
- Incorporates transfer learning

**LGS-PPIS (2025):**
- Local-Global structural information aggregation
- Published in Proteins: Structure, Function, and Bioinformatics

**ZHMolGraph (2025):**
- For RNA-protein interactions
- Combines GNNs with unsupervised LLMs
- 79.8% AUROC, 82.0% AUPRC for novel RNA/proteins

### Geometric GNNs

**GIGN (Geometric Interaction GNN):**
- Incorporates 3D structures and physical interactions
- Heterogeneous interaction layer for covalent/noncovalent bonds
- Invariant to translations and rotations

---

## 5. Diffusion-Based Methods

### DiffDock Family

**DiffDock (Original):**
- Frames docking as generative modeling over non-Euclidean manifold
- Maps to translational, rotational, and torsional degrees of freedom
- 38% top-1 success rate (RMSD<2Å) on PDBBind
- Outperforms traditional docking (23%) and prior DL methods (20%)

**DiffDock-L (February 2024):**
- Significant improvement in performance and generalization

**DiffDock-PP:**
- Learns to translate and rotate unbound proteins to bound conformations
- 8% sampling success rate on Docking Benchmark 5.5

**DiffDock-site:**
- Integrates point site for pocket identification
- Outperforms DiffDock on several metrics

**DiffPepDock (2025):**
- SE(3)-equivariant diffusion for protein-peptide docking
- Efficient binder screening

### DFMDock (2024)

**Denoising Force Matching Dock:**
- Unifies sampling and ranking in single framework
- Two output heads: forces and energies
- 44% sampling success rate (vs. 8% for DiffDock-PP)
- 16% Top-1 ranking success rate (vs. 0% for DiffDock-PP)

### RFdiffusion

- Represents residues as rigid frames
- Generates backbones from ideal-gas-like prior
- Fine-tuned on PPI data for binder design
- **RFdiffusion2 (2025)**: Atom-level enzyme active site scaffolding

### Key Advantages of Diffusion Models

1. **Generative capability**: Can sample multiple plausible conformations
2. **Uncertainty quantification**: Natural confidence estimates
3. **Flexibility**: Handle diverse molecular interactions
4. **De novo design**: Enable generation of novel binders

---

## 6. Binding Affinity Prediction

### The Challenge

Binding affinity prediction remains more difficult than structure prediction:
- Most structure prediction models (AlphaFold) not designed for affinity
- Limited training data compared to structure data
- Need to capture subtle energetic differences

### Recent Advances

**Boltz-2 Breakthrough:**
- First DL model approaching physics-based FEP accuracy
- Joint structure + affinity prediction
- 1000x faster than traditional FEP

**Current Approaches:**

| Approach | Description | Key Methods |
|----------|-------------|-------------|
| Structure-based | 3D conformational features | GIGN, DeepRank |
| Sequence-based | Raw sequence embeddings | ESM-based models |
| Hybrid | Combined structural + sequence | GES_PPI, DDMut-PPI |
| Physics-enhanced | Incorporate energy terms | FoldX integration |

### PPB-Affinity Dataset (2024)

Addresses critical gap in open-source data:
- Largest publicly available PPB affinity dataset
- Crystal structures of complexes (wild-type and mutant)
- PPB affinity values
- Receptor/ligand chain annotations
- Sources: SKEMPI v2.0, SAbDab, PDBbind v2020, Affinity Benchmark v5.5, ATLAS

### Current Challenges

1. **Limited data quality and diversity**: Biased toward well-studied proteins
2. **Generalizability**: Models fail on novel proteins/chemical series
3. **Dynamic representations**: Most treat interactions as static snapshots
4. **Rare targets**: Insufficient data for resistance mutations

### Emerging Directions

- Integration of molecular dynamics simulations
- Diffusion-based frameworks for ensemble modeling
- Transfer learning from large-scale pretraining

---

## 7. Antibody-Antigen Interactions

### Special Challenges

Antibody-antigen prediction is notably more difficult than general PPI:
- Antibodies often lack accurate structural data
- CDR loops highly variable
- Lower prediction confidence for inferred structures

### AlphaFold Performance Evolution

| Version | Success Rate | Notes |
|---------|-------------|-------|
| AF2 (early) | ~20% | Limited antibody-antigen modeling |
| AF2-Multimer (improved) | >30% | Latest version |
| AF2-Multimer (high sampling) | ~50% | Increased sampling |
| AF3 (1 seed) | 10-13% | High-accuracy rate |
| AF3 (1000 seeds) | 60% | Extensive sampling required |

### Specialized Models

**HelixFold-Multimer (December 2024):**
- Fine-tuned specifically for antigen-antibody systems
- 52.7% success rate (vs. AF2 7.6%, RoseTTAFold 4.6%)
- Significantly outperforms baseline methods

**IgFold:**
- Fast, accurate antibody structure prediction
- Trained on massive natural antibody datasets

### Remaining Limitations

- **Boltz-1**: 4.08% high-accuracy for antibodies
- **Chai-1**: 0% high-accuracy for antibodies
- AF3's 65% failure rate demonstrates need for further improvement

### Sequence-Based Alternatives

When structure unavailable (common in antibody engineering):
- Structure-based methods show limited robustness
- Sequence-only approaches more practical
- Deep learning on B cell sequencing data shows promise

---

## 8. Interface & Hotspot Prediction

### Definitions

- **Interface residue**: Participates in protein-protein contact
- **Hotspot residue**: ΔΔG ≥ 2.0 kcal/mol upon alanine mutation
- **Interface area**: Typically 1200-2000 Å²
- **Hotspot prevalence**: <5% of interface residues

### PPI-hotspotID (2024-2025)

**Method:**
- Ensemble of classifiers
- Only 4 features: conservation, amino acid type, SASA, ΔGgas

**Benchmark Dataset:**
- 158 nonredundant proteins
- 414 known hot spots, 504 non-hot spots

**Performance:**
- Combined with AF-Multimer interface prediction:
  - Sensitivity: 0.70
  - F1: 0.72
- Better than either method alone

### Key Databases

| Database | Description |
|----------|-------------|
| ASEdb | Alanine Scanning Energetics Database |
| BID | Binding Interface Database |
| PINT | Protein-protein Interaction Thermodynamic |
| SKEMPI | Structural Database of Kinetics and Energetics |
| PPI-HotspotDB | 4,039 hotspots in 1,893 proteins |

### Transfer Learning Approaches

**DGCPPISP (2024):**
- Two-stage transfer learning with dynamic GCN
- ESM-2 encoding with additional sequence features
- Protein-peptide binding data helpful for PPI prediction

---

## 9. Mutation Effect Prediction (ΔΔG)

### Importance

- Understanding genetic variant effects on interactions
- Disease mechanism elucidation
- Biomarker identification
- Targeted therapy development

### Recent Methods

**3D-ΔΔG (2025):**
- Dual-channel model using protein 3D structures
- Handles multipoint mutations (unlike many prior methods)
- Addresses limitation of sequence-only or single-mutation methods

**GES_PPI (2025):**
- Combines structural and evolutionary insights
- Gated GNN for local structural information
- Graph Transformer for refinement
- ESM-derived features

**DDMut-PPI (2024):**
- Web server for mutation effect prediction
- Handles single and multiple point mutations
- Key features: FoldX ΔΔG, Δauthority score (network graph)

**PPAC (2025):**
- Uses Protein Large Language Models
- Employs ESM2, ESM C, and ProtT5
- Sequence-based representations for wild-type and mutant

### PLM-interact for Mutations (2025)

- Fine-tuning method for detecting mutation effects
- Leverages joint protein pair encoding
- State-of-the-art performance

### Benchmark: SKEMPI 2.0

- 7,085 samples of affinity changes upon mutations
- 8,338 single mutation entries
- 317 protein-protein complexes
- Gold standard for ΔΔG prediction evaluation

---

## 10. Key Datasets & Benchmarks

### Structure Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| **PDB** | Protein Data Bank | >200,000 structures |
| **AlphaFold DB** | Predicted structures (Sept 2024 update) | >200M proteins |
| **SAbDab** | Structural Antibody Database | Updated July 2024 |

### Interaction Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| **SKEMPI 2.0** | Mutation effects on binding | 8,338 mutations, 345 structures |
| **PPB-Affinity** | PPI binding affinities | Largest available |
| **Docking Benchmark 5.5** | Protein docking evaluation | Standard benchmark |
| **PDBbind v2020** | Binding affinity data | Protein-ligand/PPI |

### Hotspot Datasets

| Dataset | Description |
|---------|-------------|
| **ASEdb** | Alanine scanning energetics |
| **PPI-HotspotDB** | 4,039 hotspots in 1,893 proteins |

### Benchmark Considerations

**Critical Issue - Data Leakage:**

A 2025 study revealed that many reported >90% accuracies are inflated due to:
- Sequence similarity between train/test
- Protein node degree shortcuts
- Improper data splitting

**Actual Performance:**
- When properly evaluated: **0.61-0.65 accuracy**
- Sequence-only methods perform worse than those with functional/expression features

---

## 11. Tools & Software

### Structure Prediction Servers

| Tool | Access | License |
|------|--------|---------|
| AlphaFold Server | Web | Academic (Nov 2024) |
| Boltz | GitHub | MIT |
| Chai-1 | GitHub/Web | Apache |
| HelixFold | GitHub | - |

### Unified Platforms

**ABCFold (2025):**
- Run AlphaFold 3, Boltz-1, Chai-1 with single input
- Converts AF3 JSON to work with all methods

**OmniFold:**
- Unified ensemble framework
- Supports AF3, Chai-1, Boltz-2 in parallel

### PPI Prediction Tools

| Tool | Type | Key Features |
|------|------|-------------|
| PEPPI | Web | Multi-method consensus |
| PLM-interact | Code | Cross-species, mutation effects |
| DDMut-PPI | Web | Mutation effect prediction |
| DeepRank-GNN-esm | Code | GNN with ESM embeddings |
| MGPPI | Code | Multiscale GNN, interpretable |

### Docking Tools

| Tool | Method | Application |
|------|--------|-------------|
| DiffDock | Diffusion | Small molecule docking |
| DiffDock-PP | Diffusion | Protein-protein docking |
| DFMDock | Diffusion | Unified sampling/ranking |
| HDOCK | Traditional + ML | General docking |

---

## 12. Current Challenges & Future Directions

### Major Challenges

1. **Generalizability Gap**
   - Models fail on novel proteins/chemical series
   - Bias toward well-studied targets

2. **Data Limitations**
   - Scarcity of binding affinity data
   - Quality issues in existing datasets
   - Bias toward certain protein families

3. **Antibody-Antigen Modeling**
   - 65% failure rate with single-seed AF3
   - Chai-1 and Boltz-1 show near-zero success

4. **Dynamic Interactions**
   - Most methods treat PPIs as static
   - Transient interactions poorly captured

5. **Computational Cost**
   - Structure prediction requires significant resources
   - PSSM calculation expensive (though replaceable with PLMs)

6. **Realistic Evaluation**
   - Data leakage inflates reported performance
   - Need for gold-standard benchmarks

### Promising Directions

1. **Unified Structure + Affinity Models**
   - Boltz-2 demonstrates feasibility
   - Joint optimization improves both tasks

2. **Molecular Dynamics Integration**
   - Ensemble representations
   - Diffusion-based dynamics modeling

3. **Multimodal Foundation Models**
   - ESM3's joint sequence/structure/function
   - Continued scaling yields improvements

4. **Transfer Learning**
   - Cross-species generalization (PLM-interact)
   - Leverage abundant sequence data

5. **Interpretability**
   - Attention mechanisms reveal key residues
   - High-attention sites correlate with function

6. **De Novo Design**
   - RFdiffusion for binder design
   - Generative approaches for therapeutic development

---

## 13. References & Sources

### Primary Reviews

- [Recent advances in deep learning for protein-protein interaction: a review (BioData Mining 2025)](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-025-00457-6)
- [Advances in protein-protein interaction prediction: a deep learning perspective (Frontiers 2025)](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1710937/full)
- [Structure-Based Approaches for PPI Prediction Using ML and DL (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11763140/)

### Structure Prediction

- [AlphaFold 3 (Nature 2024, Isomorphic Labs)](https://www.isomorphiclabs.com/articles/alphafold-3-predicts-the-structure-and-interactions-of-all-of-lifes-molecules)
- [Comprehensive benchmarking of AlphaFold3 (Briefings in Bioinformatics 2025)](https://academic.oup.com/bib/article/26/6/bbaf616/8351050)
- [Chai-1 GitHub](https://github.com/chaidiscovery/chai-lab)
- [Boltz GitHub](https://github.com/jwohlwend/boltz)
- [ABCFold (Bioinformatics Advances 2025)](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf153/8176613)

### Protein Language Models

- [PLM-interact (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-64512-w)
- [ESM GitHub (EvolutionaryScale)](https://github.com/evolutionaryscale/esm)
- [ESM3 Release Blog](https://www.evolutionaryscale.ai/blog/esm3-release)
- [DeepRank-GNN-esm (Bioinformatics Advances 2024)](https://academic.oup.com/bioinformaticsadvances/article/4/1/vbad191/7511844)

### Graph Neural Networks

- [MGPPI (Frontiers in Genetics 2024)](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2024.1440448/full)
- [GNNGL-PPI (BMC Genomics 2024)](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-024-10299-x)
- [GNN for PPI Survey (arXiv 2024)](https://arxiv.org/abs/2404.10450)

### Diffusion Models

- [Diffusion models in protein structure and docking (WIREs 2024)](https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1711)
- [DiffDock (arXiv)](https://arxiv.org/abs/2210.01776)
- [DiffDock-PP (arXiv)](https://arxiv.org/abs/2304.03889)
- [DFMDock (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11463455/)

### Binding Affinity

- [PPB-Affinity Dataset (Scientific Data 2024)](https://www.nature.com/articles/s41597-024-03997-4)
- [Recent advances in ML predictions of protein-ligand binding affinities (Current Opinion 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0959440X25002118)
- [Binding Affinity Prediction Review (arXiv 2024)](https://arxiv.org/html/2410.00709v2)

### Antibody-Antigen

- [Improved deep learning prediction of antigen-antibody interactions (PNAS 2024)](https://www.pnas.org/doi/10.1073/pnas.2410529121)
- [HelixFold-Multimer (arXiv 2024)](https://arxiv.org/html/2412.09826v1)
- [AlphaFold antibody-antigen evaluation (Protein Science 2024)](https://onlinelibrary.wiley.com/doi/abs/10.1002/pro.4865)
- [What does AF3 learn about antibody docking (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12360200/)

### Interface & Hotspots

- [PPI-hotspotID (eLife 2024)](https://elifesciences.org/articles/96643)
- [Nanobody hotspot prediction (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12268110/)

### Mutation Effects

- [3D-ΔΔG (PubMed 2025)](https://pubmed.ncbi.nlm.nih.gov/40375059/)
- [GES_PPI (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11927297/)
- [DDMut-PPI (NAR 2024)](https://academic.oup.com/nar/article/52/W1/W207/7680621)
- [Decoding mutation effects with ML (PubMed 2025)](https://pubmed.ncbi.nlm.nih.gov/40013003/)

### Benchmarking

- [Deep learning for PPI plateau at 0.65 accuracy (Bioinformatics 2025)](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i590/8199378)
- [AlphaFold 3 evaluation on SKEMPI (JCIM 2024)](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00976)
- [Baker Lab RF2-PPI study (Science 2025)](https://www.bakerlab.org/wp-content/uploads/2025/11/Zhang-Science-Predicting-protein-protein-interactions-in-the-human-proteome-3.pdf)

---

## Appendix: Quick Reference Tables

### Model Performance Summary (2024-2025)

| Task | Best Method | Performance | Notes |
|------|-------------|-------------|-------|
| Protein-Ligand Structure | AlphaFold 3 | 64.9% | FoldBench |
| Antibody-Antigen | HelixFold-Multimer | 52.7% | Specialized fine-tuning |
| PPI Binary Classification | PLM-interact | SOTA | Cross-species |
| Mutation ΔΔG | DDMut-PPI/3D-ΔΔG | SOTA | Multi-point capable |
| Protein Docking | DFMDock | 44% | Unified sampling/ranking |
| Binding Affinity | Boltz-2 | ~FEP accuracy | 1000x faster than FEP |

### Key Architecture Choices

| Data Available | Recommended Approach |
|---------------|---------------------|
| Sequence only | ESM-2/ESM3 + PLM-interact |
| Sequence + Structure | GNN + PLM embeddings |
| Known complex structure | AlphaFold3 / Boltz-2 |
| De novo design | RFdiffusion |
| Mutation screening | DDMut-PPI / 3D-ΔΔG |
| Antibody design | HelixFold-Multimer |

---

*Document compiled from comprehensive web research on the state-of-the-art in protein-protein binding prediction as of January 2026.*
