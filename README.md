# Food Demand Prediction Project

## Introduction
In the realm of meal delivery, delivery centers often face the challenge of unpredictable demand, impacting raw material procurement and staffing. The solution? Accurate Demand Forecasting. Our project aims to provide a solution by predicting the demand for different meal types in upcoming weeks based on various parameters.

## Developers
- [Vansh Raja](https://github.com/vansh-raja)
- [Pranav Tyagi](https://github.com/PranavTyagi-3)
- [Ayush Bhatt](https://github.com/AyushB21)
- [Kunal Paliwal](https://github.com/kunalpaliwal13)
- [Arnav Gupta](https://github.com/arrnavgg)
- [Rishabh Sharma](https://github.com/rishabh301)
- [Shreyansh Negi](https://github.com/Shreyanshnegi13)

## Tools & Libraries Used
- PowerBI & Tableau
- Apache Kafka
- Spark & PySpark
- Scikit-Learn (SkLearn)
- HTML, CSS, & JavaScript
- DVC (Data Version Control)
- SQLite
- MongoDB
- MLFlow
- Flask
- Pandas
- NumPy
- PyMongo
- Pickle

## Data Pipeline
### Extract
Data is retrieved from the Kafka messaging queue in batches using Spark for initial pre-processing. Spark saves the data in multiple files, which are then merged for further operations.
### Transform and Train
The merged file is sent through the DVC pipeline for additional preprocessing. Preprocessed data is saved into a CSV file for training the model, which is tracked using MLflow.
### Load
The preprocessed file is then stored either in SQLite or MongoDB for further access and analysis.

## MLOPS Pipeline
### Database Implementation
We employ structured (SQLite) and unstructured (MongoDB) databases to efficiently handle different data types. SQLite manages structured data, while MongoDB is used for unstructured data.
### Messaging Queue System
Apache Kafka, integrated with PySpark, ensures seamless communication between pipeline components, facilitating real-time data integration and transmission.
### Data Visualization
Insights from demand forecasting are visually represented using PowerBI and Tableau, offering informative and visually appealing analyses.
### Model Building
Machine learning models are trained with various parameters to predict future order quantities based on fetched data.
### Model Tracking
MLFlow tracks, stores, and deploys machine learning models, managing versions, parameters, and dependencies effectively.
### Deployment
MLFlow serves trained models, providing access to different versions via Flask endpoints for data input and prediction output.

## Screenshots
![ss1](https://github.com/Vansh-Raja/Gryffindor-Internship/assets/64516886/5ee4671f-d688-417e-bc94-5f1b39fa5b4c)
![ss2](https://github.com/Vansh-Raja/Gryffindor-Internship/assets/64516886/3e161586-940c-47a6-9b3c-0abf54b82937)
![ss3](https://github.com/Vansh-Raja/Gryffindor-Internship/assets/64516886/352021e6-4949-4500-b6c2-47841a19fc1e)
![ss4](https://github.com/Vansh-Raja/Gryffindor-Internship/assets/64516886/e122c30a-83b4-4655-bdb3-2cd58ba3e76b)
