# CSCI 548 Data Integration Toolkit - Relation Extraction
CSCI 548 Spring 2019 Group Project. Data Integration Toolkit Library(ditk) - **Relation Extraction Group**

## Team Members

| Name                 | Email                 | Module 1              | Module 2               |
|:--------------------:|:---------------------:|:---------------------:|:----------------------:|
| Haozhe Gu(Ash)       | haozhegu@usc.edu      | soft_label_RE         | local_global_CRNN_bio  |

## Benchmark Dataset
### NYT10 Dataset

NYT10 is a distantly supervised dataset originally released by the paper "Sebastian Riedel, Limin Yao, and Andrew McCallum. Modeling relations and their mentions without labeled text.". [[paper]](http://www.riedelcastro.org//publications/papers/riedel10modeling.pdf) [[download]](http://iesl.cs.umass.edu/riedel/ecml/)
## Evaluation Metrics
    1. Precision
    2. Recall 
    3. F1 Score

## Example Usage (Locally for now)
Go to **package** folder and install the package
```
pip install --user .
```
After that, you can test it using example method in both Python2 and Python3

Python2
  ```
  from ditk.relationExtraction.example_python2 import Example as model1

  m1=model1()
  m1.read_dataset()
  m1.train()
  evaluate()
  ```

Python3
  ```
  from ditk.relationExtraction.example_python3 import Example as model2

  m2=model2()
  m2.read_dataset()
  m2.train()
  evaluate()
  ```
