# MFERL
Leveraging Explainable Multi-scale Features for Fine-Grained circRNA-miRNA Interaction Prediction.

## License

Copyright (C) 2025 Li Peng (plpeng@hnu.edu.cn), Wang Wang (wang_master717@163.com)

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

---

## abstract

Background: Circular RNAs (circRNAs) and microRNAs (miRNAs) interactions have essential implications in various biological processes and diseases. Computational science approaches have emerged as powerful tools for studying and predicting these intricate molecular interactions, garnering considerable attention. Current methods face two significant limitations: the lack of precise interpretable models and insufficient representation of homogeneous and heterogeneous molecules. 

Results: We propose a novel method, MFERL, that addresses both limitations through multi-scale representation learning and an explainable fine-grained model for predicting circRNA-miRNA interactions (CMI). MFERL learns multi-scale representations by aggregating homogeneous node features and interacting with heterogeneous node features, as well as through novel dual-convolution attention mechanisms and contrastive learning to enhance features. 

Conclusions: We utilize a manifold-based method to examine model performance in detail, revealing that MFERL exhibits robust generalization, robustness, and interpretability. Extensive experiments show that MFERL outperforms state-of-the-art models and offers a promising direction for understanding CMI intrinsic mechanisms.

---

## Environment Requirement

To run this project, make sure you have the following Python environment:

```bash
numpy==1.24.4
scipy==1.8.1
torch-cluster==1.6.1+pt20cu118
torch-geometric==2.5.3
torch-scatter==2.1.1+pt20cu118
torch-sparse==0.6.17+pt20cu118
torch-spline-conv==1.2.2+pt20cu118
```

---

## Model Structure

- **`datapro.py`**: This script handles the reading, preprocessing, and transformation of data into a format suitable for model training and evaluation.
- **`new_model.py`**: the core model proposed in the paper.
- **`train.py`**: completion of a 5-fold cross-validation experiment.
- **`clac_metric.py`**: Computes various performance metrics and evaluation indicators to assess the model's accuracy and effectiveness.

---

## Datasets

The `datasets/` folder contains the benchmark datasets used in the experiments. 

