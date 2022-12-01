# DASCL: Dictionary-Assisted Supervised Contrastive Learning

Last Updated: November 29, 2022

Authors: Patrick Y. Wu, Richard Bonneau, Joshua A. Tucker, and Jonathan Nagler 

This repository includes a PyTorch implementation of the dictionary-assisted supervised contrastive learning framework, described [here](https://arxiv.org/abs/2210.15172). We will be updating this repository in the lead up to EMNLP 2022; please check back as we get closer to the conference. 

## Usage

To use the functions in this package, install the package first. To do this, navigate to the directory you would like to install the files in. Then,

```
git clone https://github.com/SMAPPNYU/DASCL
cd DASCL
pip install -r requirements.txt
python setup.py develop
```

We have implemented DASCL for BERT and RoBERTa. The functions are used in a very similar manner to `BertForSequenceClassification` and `RobertaForSequenceClassification`. 