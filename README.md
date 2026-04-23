# DiffGCC: Diffusion-Enhanced Global-Local Graph Contrastive Clustering

## 🛠️ Methodology
![Framework](https://github.com/user-attachments/assets/7045bf0b-01f3-4f7d-83fe-9f92d7012c88)

## ⚙️ Dependencies
- PyTorch, dgl (only as the source of datasets), scikit-learn, numpy

## 📊 Datasets
We use four benchmark datasets provided by DGL:
| Dataset | Type | #Nodes | #Edges | #Features | #Classes |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Cora | Citation | 2,708 | 10,556 | 1,433 | 7 |
| CiteSeer | Citation | 3,327 | 9,228 | 3,703 | 6 |
| Photo | Co-purchase | 7,650 | 238,163 | 745 | 8 |
| Computer | Co-purchase | 13,752 | 491,722 | 767 | 10 |

## 🚀 Usage
```bash
python main.py --dataset Cora
