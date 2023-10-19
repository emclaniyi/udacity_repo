# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The **Predict Customer Churn** project seeks to predict customer churn in the context of a business. Customer churn, also known as customer attrition, is the phenomena in which a company's clients cease doing business with it. Predicting customer churn allows firms to identify customers who are likely to leave and take proactive actions to keep them.
We use machine learning approaches in this project to build a prediction model that can detect probable churners based on past customer data

## Files and data description
1. 'churn_library_solution.py': Python module comprising data preprocessing, feature engineering, model training, and evaluation methods and utilities.

2. 'data/': this folder contains dataset needed to train and test the prediction model. A CSV file with historical customer data.

3. 'models/': this folder contains all machine learning models. These models may be loaded and used to predict outcomes.

4. 'images/': contains all images generated during exploratory data analysis (EDA) and model assessment. Histograms, bar graphs, and ROC curves are examples of such images.

5. 'logs/': containing log files. Log files record information regarding code execution, such as INFO and ERROR messages. These logs are important for troubleshooting and tracking project progress. 

## Running Files
Follow these steps to execute the project and predict customer churn:

1. Ensure that Python and the required libraries in the requirements.txt file are installed on your PC. 'pip' is commonly used to install needed libraries. You might also create a virtual environment to handle dependencies.

2. Put dataset in the 'data/' folder. The dataset should be in CSV format and include historical customer data, including features and a churn target variable.

#### to run the test scripts
```bash
# Example command to run the test script
pytest churn_script_logging_and_tests.py
```

```bash
# Example command to run the script
ipython churn_library.py
```




