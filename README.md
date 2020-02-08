# ET4EL

> [**Fine-Grained Entity Typing for Domain Independent Entity Linking**](https://arxiv.org/pdf/1909.05780.pdf)<br/>
> Yasumasa Onoe and Greg Durrett<br/>
> AAAI 2020


## Prerequisites

The code is developed with `python 3.7` and `pytorch 1.0.0`. 


## Training Entity Typing Models

Coming soon...

## Evaluating Models on Entity Linking

Coming soon...


## Data

### WikilinksNED Unseen-Mentions   

This data is created by splitting the WikilinksNED training set (Eshel et al. 2017) into train, development, and test sets by unique mentions (15.5k for train, 1k for dev, and 1k for test). There are no common mentions between the train, dev, and test sets. The dataset can be downloaded from [here](https://drive.google.com/a/utexas.edu/file/d/1jANXLqsDwZvRBxhgDRryOQgClqyzciII/view?usp=drive_web).


## Questions
Contact us at `yasumasa@cs.utexas.edu` if you have any questions!


## Acknowledgements
Code for entity typing model is based on Eunsol Choi's pytorch implementation.

GitHub: https://github.com/uwnlp/open_type<br/>
Paper : https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf

