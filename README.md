# Dasamoola Phytochemical Clustering & Disease Target Mapping

An unsupervised machine learning and knowledge graph pipeline that groups plant-derived phytochemicals by structural similarity and maps them to therapeutic disease targets.

## 📊 Project Overview

| Feature | Description |
| :--- | :--- |
| **What the Project Is** | An unsupervised machine learning pipeline designed to analyze the Dasamoola polyherbal formulation by mapping structural clusters of plant-derived molecules to their therapeutic disease targets[cite: 27]. |
| **What It Does** | • Converts 490 molecular structures into 2048-bit digital vectors using Morgan circular fingerprints[cite: 29].<br>• Applies K-Means clustering and t-SNE dimensionality reduction to systematically group structurally similar phytochemicals[cite: 27, 30].<br>• Generates interactive heat maps to evaluate the complex, many-to-many associations between molecular clusters and disease categories[cite: 29, 32]. |
| **Data Source** | Real-world phytochemical data consisting of 490 molecules from 10 medicinal plants, sourced from the IMPPAT and Dr. Duke's databases[cite: 29]. The targets are mapped against 87 diseases standardized by the WHO ICD-11 classification framework[cite: 29, 33]. |
| **Results & Insights** | Successfully grouped 490 phytoconstituents into 49 distinct, chemically coherent clusters[cite: 32].<br><br>💡 **Key Insight:** Demonstrated that traditional plant medicines do not act as isolated molecules; instead, they function as synergistic molecular clusters that converge on shared biological targets to treat specific disease pathways[cite: 27, 38]. |

## 📚 Publications

**This repository contains the code and datasets used in the following peer-reviewed publication:**

* **Knowledge graph integration of clustered medicinal plants, molecules, diseases, and targets** 
  *Authors: UK Shajil, Jaleel UCA, S. Sathish, Sandesh EPA, A. Sujith, Baiju G. Nair*[cite: 27]
  *Journal: Computational Biology and Chemistry (Elsevier), Volume 122, 2026, 108895*[cite: 27]
  *[Link to Paper](https://www.sciencedirect.com/science/article/pii/S1476927126000204)*


## 🚀 Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/sathish-dream-ai/plant-molecules-plant-diseases-clustering-and-heat-map.git](https://github.com/sathish-dream-ai/plant-molecules-plant-diseases-clustering-and-heat-map.git)
