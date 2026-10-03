# Deep Learning Architectures: 10 Famous Models, Implemented and Documented

A hands-on series where I implement, train and analyze 10 landmark deep learning architectures in PyTorch. Each project lives in its own folder with a README (architecture explanation, results, analysis), clean code, and a public Kaggle notebook.

## Projects

| # | Model | Task / Dataset | Status | Kaggle | Result |
|---|-------|----------------|--------|--------|--------|
| 01 | [LeNet-5](01-lenet5-mnist) | Digit classification / MNIST | Done | _link_ | 99.02% test acc |
| 02 | AlexNet / VGG | Image classification / CIFAR-10 | Planned | | |
| 03 | ResNet-18 | Image classification / CIFAR-10/100 | Planned | | |
| 04 | EfficientNet (transfer learning) | Plant disease / Kaggle | Planned | | |
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
