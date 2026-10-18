# Deep Learning Architectures: 10 Famous Models, Implemented and Documented

A hands-on series where I implement, train and analyze 10 landmark deep learning architectures in PyTorch. Each project lives in its own folder with a README (architecture explanation, results, analysis), clean code, and a public Kaggle notebook.

## Projects

| # | Model | Task / Dataset | Status | Kaggle | Result |
|---|-------|----------------|--------|--------|--------|
| 01 | [LeNet-5](01-lenet5-mnist) | Digit classification / MNIST | Done | [Notebook](PASTE_KAGGLE_LINK_1) | 99.02% test acc |
| 02 | [VGG-style CNN (ablation study)](02-vgg-cifar10) | Image classification / CIFAR-10 | Done | [Notebook](PASTE_KAGGLE_LINK_2) | 88.93% test acc (baseline: 78.79%) |
| 03 | [ResNet: plain vs residual](03-resnet-cifar10) | Image classification / CIFAR-10 | Done | [Notebook](PASTE_KAGGLE_LINK_3) | 90.72% test acc (ResNet-18); ResNet-56 85.67% vs Plain-56 63.35% |
| 04 | [EfficientNet transfer learning](04-efficientnet-plant-disease) | Plant disease classification | Done | [Notebook](https://www.kaggle.com/code/ebrahiminima/day-4-efficientnet-transfer-learning-pytorch-p) | 99.85% test acc (98.18% with only 5% of the data) |
| 05 | [U-Net: brain MRI segmentation](05-unet-brain-mri) | Medical image segmentation / LGG MRI | Done | [Notebook](https://www.kaggle.com/code/ebrahiminima/day-5-u-net-from-scratch-pytorch-brain-mri-seg) | Test Dice 0.754 (patient-level split); a random slice split inflates it to 0.836 |
| 06 | [RNN, LSTM & GRU: sentiment analysis](06-lstm-gru-imdb) | Sentiment analysis / IMDB | Done | [Notebook](https://www.kaggle.com/code/ebrahiminima/day-6-rnn-lstm-gru-pytorch-imdb-sentiment) | 87.84% test acc (GRU); the vanilla RNN's gradient vanishes by ~10 orders of magnitude over 250 tokens |
| 07 | [Seq2Seq with attention](07-seq2seq-attention-multi30k) | Machine translation / Multi30k EN-DE | Done | [Notebook](https://www.kaggle.com/code/ebrahiminima/day-7-seq2seq-with-attention-pytorch-multi30k) | BLEU 32.8 with attention vs 19.1 without; general (Luong) attention collapses to near-one-hot weights |
| 08 | [Transformer & BERT: text classification](08-transformer-bert-imdb) | Sentiment classification / IMDB | Done | [Notebook](https://www.kaggle.com/code/ebrahiminima/day-8-transformer-bert-pytorch-imdb-classifi) | 92.29% test acc (BERT-base) vs 87.96% for a Transformer from scratch (GRU on Day 6: 87.84%) |
| 09 | [Autoencoder & VAE](09-autoencoder-vae-fashion-mnist) | Image reconstruction & generation / Fashion-MNIST | Done | [Notebook](https://www.kaggle.com/code/ebrahiminima/day-9-autoencoder-vae-pytorch-fashion-mnist) | VAE rate-distortion study; an autoencoder with a fitted Gaussian generates as well as the VAE (FD-CNN 65.8 vs 68.4) |
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