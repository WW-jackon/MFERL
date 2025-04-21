# MFERL
MFERL is a model designed to predict associations between circRNAs and diseases based on **multi-feature enhanced representation learning**.

## License

Copyright (C) 2025 [Your Name] (your_email@example.com)

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

---

## Overview

**MFERL** (Multi-Feature Enhanced Representation Learning) integrates various biological features and similarity networks to effectively learn embeddings for predicting potential circRNA–disease associations.

---

## Environment Requirement

To run this project, make sure you have the following Python environment:

```bash
torch == 1.7.1+cu110
numpy == 1.19.5
matplotlib == 3.5.1
dgl-cu110 == 0.5.3
scikit-learn == 0.24.2
```

---

## Model Structure

- **`MFERL_model.py`**: The core model implementing multi-feature enhanced representation learning.
- **`attention_layer.py`**: Feature-level attention module.
- **`fivefold_CV.py`**: Implements 5-fold cross-validation for performance evaluation.
- **`case_study.py`**: Predicts candidate circRNAs for given diseases.

---

## Datasets

The `datasets/` folder contains the benchmark datasets used in the experiments. These include:
- circRNA and miRNA features
- Similarity matrices
- Interaction matrices

Note: Some large dataset files are managed with Git LFS.

---

## Running the Model

### 5-Fold Cross-Validation
```bash
python fivefold_CV.py
```

### Case Study Prediction
```bash
python case_study.py
```

---



## Contact

For questions or collaboration, feel free to contact:
- **Your Name**: your_email@example.com

