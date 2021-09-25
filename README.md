# Fine-Grained Entity Typing for Domain Independent Entity Linking

[![Paper](https://img.shields.io/badge/Paper-%09arXiv%3A1909.05780-red?style=flat-square)](https://arxiv.org/abs/1909.05780)
[![Conference](https://img.shields.io/badge/AIII-2020-blueviolet?style=flat-square)](https://ojs.aaai.org//index.php/AAAI/article/view/6234)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg?style=flat-square)](https://www.python.org/downloads/release/python-3711/)
[![Requires.io](https://img.shields.io/requires/github/yasumasaonoe/ET4EL?style=flat-square)](./requirements.txt)

## Description

This project is based on the [Pytorch-Lightning template](https://github.com/PyTorchLightning/deep-learning-project-template).

Included a LightningModule and a DataModule with automatic data downloads.

## How to run

First, install dependencies

```bash
# clone project   
git clone https://github.com/yasumasaonoe/ET4EL

# install project   
cd et4el-lightning
pip install -e . # This will setup the project as a package so it the imports work
pip install -r requirements.txt
 ```

 Next, navigate to any file and run it.

 ```bash
# module folder
cd et4el

# run module 
python cli.py
```

## Data

We train our entity typing model on data derived from March 2019 English Wikipedia dump. This data can be downloaded from [here](https://drive.google.com/file/d/1m9CPaehSjlsFA6Na-bYZ2GWt_kzyfJTo/view?usp=sharing) (12GB).

### Entity Linking Data for Evaluation

* CoNLL-YAGO: This data is not publicly available. You can find more information [here](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/).
* WikilinksNED Unseen-Mentions: This data is created by splitting the WikilinksNED training set (Eshel et al. 2017) into train, development, and test sets by unique mentions (15.5k for train, 1k for dev, and 1k for test). There are no common mentions between the train, dev, and test sets. The dataset can be downloaded from [here](https://drive.google.com/a/utexas.edu/file/d/1jANXLqsDwZvRBxhgDRryOQgClqyzciII/view?usp=drive_web). Note that the training set is used for baselines only.

## Questions

Contact us at `yasumasa@cs.utexas.edu` if you have any questions!

## Acknowledgements

Code for entity typing model is based on Eunsol Choi's pytorch implementation:
> GitHub: <https://github.com/uwnlp/open_type>  
> Paper : <https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf>

## Citation

```bibtex
@article{onoe2020finegrained,
  title={Fine-Grained Entity Typing for Domain Independent Entity Linking},
  author={Yasumasa Onoe and Greg Durrett},
  journal={AIII},
  year={2020},
  eprint={1909.05780},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
