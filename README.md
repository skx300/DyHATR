# DyHATR

This is the source code for ECML-PKDD 2020 paper ["Modeling Dynamic Heterogeneous Network forLink Prediction using Hierarchical Attentionwith Temporal RNN"](https://arxiv.org/abs/2004.01024).


All readers are welcome to star/fork this repository and use it to reproduce our experiments or train your own data. Please kindly cite our paper:
```
@inproceedings{Xue2020DyHATR,
  title     = {Modeling Dynamic Heterogeneous Network forLink Prediction using Hierarchical Attentionwith Temporal RNN},
  author    = {Xue, Hansheng and Yang, Luwei and Jiang, Wen and Wei, Yi and Hu, Yi and Lin, Yu},
  booktitle = {Proceedings of the 2020 European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year      = {2020},
}
```

## Requirement
```
Python 3.6
networkx == 1.11
numpy == 1.18
sklearn == 0.22
tensorflow == 1.14
```

## Dataset
In this resposity, we provide EComm dataset as an example, you can also download all the other datasets from the SNAP platform ([Twitter](http://snap.stanford.edu/data/higgs-twitter.html), and [Math-Overflow](http://snap.stanford.edu/data/sx-mathoverflow.html)). Besides, you can also use your own Dynamic Heterogeneous Networks dateset, as long as it fits the following template.
```
Node_one	Node_two	Edge_type	Timestamp
  n1		  n2		  e1			1
  n1		  n2		  e2			2
  n1		  n3		  e2			2
  .
  .
```

## Example Usage
To reproduce the experiments on EComm dataset, simply run:
```
python3 src/main.py
```


### Acknowledgement
The original version of this code base was originally forked from [GraphSAGE](https://github.com/williamleif/GraphSAGE), and [GAT](https://github.com/PetarV-/GAT), and we owe many thanks to these authors for making their code available. If you have some questions about the code or paper, you are welcome to open an issue or send us an email. We will respond to that as soon as possible.


