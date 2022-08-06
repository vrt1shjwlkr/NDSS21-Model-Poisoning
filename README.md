# A General Framework to Evaluate Robustness of Aggregation Algorithms in Federated Learning

## Motivation
This repository provides a code to investigate the security of a given federated learning algorithm, and especially the security provided by _robust aggregation algorithms_ that are used to aggregate the client updates in federated learning. The code can be used to reproduce results and conclusions from two of our works: [[1] Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defensesfor Federated Learning](https://www.ndss-symposium.org/wp-content/uploads/2021-498-paper.pdf) and [[2] Back to the drawing board: A critical evaluation of poisoning attacks on production federated learning](https://arxiv.org/pdf/2108.10241.pdf).

Our attacks are untargeted poisoning attacks, i.e., they aim to reduce the accuracy of the global model on all of the test inputs. For a detailed review of different types of attacks please refer to our thorough review paper [2](https://arxiv.org/pdf/2108.10241.pdf).

## Result Highlights


## Understanding the code and using the notebooks
We have given the code in the form of notebooks which are self-explanatory, because the description of each cell is given in the respective notebooks. 
To run the code and reproduce the results in the paper, please clone/download the repo and simply run the notebooks, preferrably on a GPU.
Various evaluation dimensions are below
* Datasets included are CIFAR10 (covers iid and cross-silo FL cases) and FEMNIST (covers non-iid and cross-device FL cases).
* We have given codes for five state-of-the-art aggregation algorithms, which give theoretical convergence guarantees: [Krum](https://dl.acm.org/doi/abs/10.5555/3294771.3294783), [Multi-krum](https://dl.acm.org/doi/abs/10.5555/3294771.3294783), [Bulyan](https://arxiv.org/pdf/1802.07927), [Trimmed-mean](http://proceedings.mlr.press/v80/yin18a/yin18a.pdf), [Median](http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)
* Baseline model poisoning attacks [Fang](https://www.usenix.org/system/files/sec20-fang.pdf) and [LIE](https://papers.nips.cc/paper/2019/file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf).
* Our state-of-the-art model poisoning attacks, Aggregation-tailored attacks and Aggregation-agnsotic attacks, for the above mentioned aggregation algorithms. For any other aggregation algorithms, the code allows for simple plug-and-attack framework.

## Citation

```
@inproceedings{shejwalkar2021manipulating,
  title={Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning},
  author={Shejwalkar, Virat and Houmansadr, Amir},
  booktitle={NDSS},
  year={2021}
}

@inproceedings{shejwalkar2022back,
  title={Back to the drawing board: A critical evaluation of poisoning attacks on production federated learning},
  author={Shejwalkar, Virat and Houmansadr, Amir and Kairouz, Peter and Ramage, Daniel},
  booktitle={2022 IEEE Symposium on Security and Privacy (SP)},
  pages={1354--1371},
  year={2022},
  organization={IEEE}
}
```
