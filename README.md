# Parallelism-detection
OpenSource Code for **Multi-graph Learning for Parallelism Discovery in Sequential Programs**

This project is the implementation of our work on the parallelism discovery with multi-graph learning. 
Our framework leverages ASTNN-based graph neural network model and DGCNN-based graph neural network model. About the original implementation
 of ASTNN and DGCNN, you can find the code here:  
 * **ASTNN**: (https://github.com/zhangj111/astnn)  
 * **DGCNN**: ( https://github.com/muhanzhang/DGCNN)  
 * **DGCNN-tensorflow**: (https://github.com/hitlic/DGCNN-tensorflow)
   
 
## Requirement  
Python3  
pytorch
pycparser
Tensorflow  
Pluto==0.11.4(http://pluto-compiler.sourceforge.net/)  
Rose (http://rosecompiler.org/)  
Clang/LLVM  
 
## Generating dataset  
We provide a dataset generator. This step assumes that you want generating your own dataset. You
 may add your source code manually to `utils/data/source_code` , then run:  
`python3 data_gen.py`  

**Note** that you need replace the `polycc` and `autoPar` with your own.  

## Prediction  with neural network model  
Train and test:  
`python3 model_train.py`

## Please cite as
```bibtex
@article{SHEN2021515,
author = {Yuanyuan Shen and Manman Peng and Shiling Wang and Qiang Wu},
title = {Towards parallelism detection of sequential programs with graph neural network},
journal = {Future Generation Computer Systems},
volume = {125},
pages = {515-525},
year = {2021},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2021.07.001},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X21002557}
}
@article{SHEN,
author = {Yuanyuan Shen and Manman Peng and Qiang Wu and Guoqi Xie},
title = {Multi-graph Learning for Parallelism Discovery in Sequential Programs}
}
```
