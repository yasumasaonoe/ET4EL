# ET4EL

> [**Fine-Grained Entity Typing for Domain Independent Entity Linking**](https://arxiv.org/pdf/1909.05780.pdf)<br/>
> Yasumasa Onoe and Greg Durrett<br/>
> AAAI 2020


## Prerequisites

* The code is developed with `python 3.7` and `pytorch 1.0.0` or newer versions (we've tested our code on `pytorch 1.4.0`).  


## Training Entity Typing Models

* Download our training data (`entity_typing_data`) from [here](https://drive.google.com/file/d/1m9CPaehSjlsFA6Na-bYZ2GWt_kzyfJTo/view?usp=sharing) and put under `data`.
  * `entity_typing_data/train/et_conll_60k` uses the type set `data/onotology/conll_categories.txt`.
* Check `entity_typing/constant.py` to make sure paths are correct.
* Run the training function in `entity_typing/main.py`. Please see example commands in `entity_typing/scripts`.

## Evaluating Models on Entity Linking

* Put entity linking evaluation data in the appropriate folder.
* Run the evaluation function in `entity_typing/main.py`. Please see example commands in `entity_typing/scripts`.


## Data

### Training Data 

* We train our entity typing model on data derived from March 2019 English Wikipedia dump. This data can be downloaded from [here](https://drive.google.com/file/d/1m9CPaehSjlsFA6Na-bYZ2GWt_kzyfJTo/view?usp=sharing).

### Entity Linking Data for Evaluation

#### CoNLL-YAGO

* This data is not publicly available. You can find more information [here](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/).


#### WikilinksNED Unseen-Mentions   

* This data is created by splitting the WikilinksNED training set (Eshel et al. 2017) into train, development, and test sets by unique mentions (15.5k for train, 1k for dev, and 1k for test). There are no common mentions between the train, dev, and test sets. The dataset can be downloaded from [here](https://drive.google.com/a/utexas.edu/file/d/1jANXLqsDwZvRBxhgDRryOQgClqyzciII/view?usp=drive_web). Note that the training set is used for baselines only.

## Questions
Contact us at `yasumasa@cs.utexas.edu` if you have any questions!


## Acknowledgements
Code for entity typing model is based on Eunsol Choi's pytorch implementation.

GitHub: https://github.com/uwnlp/open_type<br/>
Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf

