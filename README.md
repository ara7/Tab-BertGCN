# Tab-BertGCN
Tab-BertGCN
# Tab-BertGCN: Multimodal Graph Learning for Postoperative Delirium Prediction

This repository contains the official PyTorch implementation of the **Tab-BertGCN** architecture, as described in our manuscript: *"Graph-Based Multimodal Learning with Transformers for Postoperative Delirium Prediction."*

This model extends the BertGCN framework by integrating structured tabular patient data (demographics, comorbidities, and illness severity) with unstructured clinical notes using a heterogeneous graph structure.

## Acknowledgements and Code Lineage
This project builds upon and significantly extends two excellent foundational repositories. We thank the original authors for open-sourcing their work:
1. **[TextGCN](https://github.com/yao8839836/text_gcn):** Utilized for the initial construction of the heterogeneous Document-Word (TF-IDF) and Word-Word (PPMI) corpus graph.
2. **[BertGCN](https://github.com/ZeroRin/BertGCN/tree/main):** Utilized as the base architecture for integrating transformer-derived text embeddings into the Graph Convolutional Network.

## Proposed Tab-BertGCN
While the original BertGCN relies solely on text, our `Tab-BertGCN` introduces a multimodal fusion layer. We modified the training pipeline to simultaneously ingest clinical notes and structured clinical variables. 

Key modifications include:
* **The `.tab` Data Pipeline:** We introduced a parallel data loader that reads structured patient features from custom `.tab` files.
* **Dimensionality Alignment:** A trainable linear projection layer that maps the $p$-dimensional tabular data (e.g., 39 features) into the exact same dense continuous space as the 384-dimensional Bioformer text embeddings.
* **Multimodal Node Fusion:** Aggregating the text and tabular representations prior to graph propagation.

⚠️ Repository Status: The source code for this project is currently undergoing final formatting and documentation. The complete suite of scripts (including model.py and train.py) and the synthetic dataset will be pushed to this public repository during the peer-review process.

## Data Availability & Privacy Notice
**Due to strict patient privacy regulations, the original Electronic Health Record (EHR) dataset from the Indiana Network for Patient Care (INPC) cannot be shared.** However, to facilitate reproducibility and allow researchers to test the `Tab-BertGCN` architecture, we have provided a **synthetic dummy dataset** in the `/data` folder. This dummy data perfectly mimics the structural format of our real data without containing any Protected Health Information (PHI).

## Repository Structure
* `/data/`: Contains the synthetic corpus, labels, and the `.tab` files for structured features.
* `/build_graph_fused.py`: Script (adapted from TextGCN) to calculate TF-IDF and PPMI and build the adjacency matrix.
* `/model.py`: Contains the `Tab-BertGCN` PyTorch class, including the tabular linear projection layer.
* `/train.py`: The main training loop handling the multimodal data fusion and GCN propagation.

## Data Format Example
To run this model on your own data, the text corpus and a corresponding `.tab` file is needed. 

**Example of `tabular/patient_data.tab`:**
(A comma-separated or tab-separated list of binary/continuous clinical features)
```text
Age, Fluid_Electrolyte_Disorder, Cardiac_Arrhythmia, Charlson Comorbidity Index, Emergency..
69, 1, 0, 3, 1

**## Usage (Coming Soon)**

Build the Graph: 
python build_graph_fused.py --dataset synthetic_data

2. **Train Tab-BertGCN:**

python train.py --dataset synthetic_data --epochs 50 --batch_size 64
