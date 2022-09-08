# Neural Network implementation for breast cancer classification
Machine learning project for internal evaluation of syllabus [COMP 484], [Kathmandu University](https://ku.edu.np)

---

## Overview
Medical conditions, if diagnosed on time, can be treated. Breast Cancer is one of the fatal medical conditions which is the world’s most prevalent cancer, according to the World Health Organisation. There are around 4 main types of breast cancers, each requiring different types of treatment. The purpose of classification is to identify the type of breast cancer and select the best treatment. Treatment algorithms rely on breast cancer classification to define specific subgroups that are each treated according to the best evidence available.

So, we have chosen the project **“Neural Network implementation for Breast Cancer Classification”** for the mini project of COMP 484 (Machine Learning). The project's core idea is to classify the different types of breast cancer considering patients’ different attributes using neural networks. The neural network, built with Tensorflow and Keras in this project, will consume a dataset available in Kaggle to build a model that can correctly classify the type of breast cancer after weighing certain medical details.


---

## Dataset(s)
- [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) [Kaggle]
- [Seer Breast Cancer Data](https://ieee-dataport.org/open-access/seer-breast-cancer-data) [IEEEDataPort]
  
---

## API Server

To start the local API server:

```sh
uvicorn main:app
```

The terminal command above will start a local API server which will be accessible at [http://localhost:8000](http://localhost:8000).

Go to *[/docs](http://localhost:8000/docs)* to access interactive API client.

---

## Project Members
- [Oshan Shrestha](https://oshanshrestha.com.np)
- [Melina Shakya]()
- [Diwas Shrestha](https://github.com/Ge7alt)
