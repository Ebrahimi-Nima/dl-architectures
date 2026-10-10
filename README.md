# Deep Learning Architectures: 10 Famous Models, Implemented and Documented

A hands-on series where I implement, train and analyze 10 landmark deep learning architectures in PyTorch. Each project lives in its own folder with a README (architecture explanation, results, analysis), clean code, and a public Kaggle notebook.

## Projects

| # | Model | Task / Dataset | Status | Kaggle | Result |
|---|-------|----------------|--------|--------|--------|
| 01 | [LeNet-5](01-lenet5-mnist) | Digit classification / MNIST | Done | [Notebook](PASTE_KAGGLE_LINK_1) | 99.02% test acc |
| 02 | [VGG-style CNN (ablation study)](02-vgg-cifar10) | Image classification / CIFAR-10 | Done | [Notebook](PASTE_KAGGLE_LINK_2) | 88.93% test acc (baseline: 78.79%) |
| 03 | [ResNet: plain vs residual](03-resnet-cifar10) | Image classification / CIFAR-10 | Done | [Notebook](PASTE_KAGGLE_LINK_3) | 90.72% test acc (ResNet-18); ResNet-56 85.67% vs Plain-56 63.35% |
| 04 | [EfficientNet transfer learning](04-efficientnet-plant-disease) | Plant disease classification | Done | [Notebook](https://www.kaggle.com/code/ebrahiminima/day-4-efficientnet-transfer-learning-pytorch-p) | 99.85% test acc (98.18% with only 5% of the data) |
| 05 | U-Net | Segmentation | Planned | | |
| 06 | LSTM / GRU | Sentiment analysis / IMDB | Planned | | |
| 07 | Seq2Seq + Attention | Machine translation | Planned | | |
| 08 | Transformer / BERT | Text classification | Planned | | |
| 09 | Autoencoder / VAE | Image reconstruction / generation | Planned | | |
| 10 | DCGAN | Image generation | Planned | | |

## Stack
Python, PyTorch, torchvision, scikit-learn, matplotlib, Kaggle (GPU).

## Structure of each project
```
NN-model-name/
├── README.md       # problem, architecture, training setup, results, analysis
├── *.ipynb         # Kaggle-ready notebook
├── *.py            # script version
├── requirements.txt
└── images/
```