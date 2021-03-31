# A General Framework to Evaluate Robust Aggregation Algorithms
### [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defensesfor Federated Learning](https://www.ndss-symposium.org/wp-content/uploads/2021-498-paper.pdf)
by [Virat Shejwalkar](https://people.cs.umass.edu/~vshejwalkar/) and [Amir Houmansadr](https://people.cs.umass.edu/~amir/index.php) published at [ISOC Network and Distributed Systems Security Symposium, (NDSS) 2021](https://www.ndss-symposium.org/)

## Motivation

## Result Highlights

## Understanding the code and using the notebooks
We have given the code in the form of notebooks which are self-explanatory in that the description of each cell is given in the respective notebooks. 
To run the code, please clone/download the repo and simply start running the notebooks in usual manner.
Various evaluation dimensions are below
* Datasets included are CIFAR10 (covers iid and cross-silo FL cases) and FEMNIST (covers non-iid and cross-device FL cases).
* We have given codes for five state-of-the-art aggregation algorithms, which give theoretical convergence guarantees: [Krum](https://dl.acm.org/doi/abs/10.5555/3294771.3294783), [Multi-krum](https://dl.acm.org/doi/abs/10.5555/3294771.3294783), [Bulyan](https://arxiv.org/pdf/1802.07927), [Trimmed-mean](http://proceedings.mlr.press/v80/yin18a/yin18a.pdf), [Median](http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)
* Baseline model poisoning attacks [Fang](https://www.usenix.org/system/files/sec20-fang.pdf) and [LIE](https://papers.nips.cc/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf).
* Our state-of-the-art model poisoning attacks, Aggregation-tailored attacks and Aggregation-agnsotic attacks, for the above mentioned aggregation algorithms. For any other aggregation algorithms, the code allows for simple plug-and-attack framework.


## Requirements

