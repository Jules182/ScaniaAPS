# Predictive Maintenance using Automated Machine Learning
### Predicting Air Pressure System (APS) failures in Scania trucks using a TPOT AutoML pipeline

This classification task is based on the Industrial Challenge 2016 at The 15th International Symposium on Intelligent Data Analysis (IDA). It was published in the [UCI Machine Learning Library](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks) and is available on [Kaggle](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks).

Results of the original challenge:

| Top 3 contestants    |      Score      |  Type 1 faults | Type 2 faults |
|:----------|:-------------|:------|:------|
| [Camila F. Costa and Mario A. Nascimento](https://doi.org/10.1007/978-3-319-46349-0_33) | 9920  | 542 | 9 |
| [Christopher Gondek, Daniel Hafner and Oliver R. Sampson](https://doi.org/10.1007/978-3-319-46349-0_36) | 10900 | 490 | 12 |
| Sumeet Garnaik, Sushovan Das, Rama Syamala Sreepada and Bidyut Kr. Patra | 11480 | 398 | 15 |

## Automated machine learning (AutoML)
[TPOT](https://epistasislab.github.io/tpot/) is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

## Usage
This repository contains 
* a module for preprocessing the data set [preprocessing.py](preprocessing.py).
* a script for [executing the TPOT process](tpot_script.py)
* a notebook for evaluating the results of the TPOT process
  * [Python script](scania_analysis.py)
  * [Jupyter notebook with results](scania_analysis.ipynb)
  
## Results
|      Score      |  Type 1 faults | Type 2 faults |
|:-------------|:------|:------|
| 11960  | 646 | 11 |
