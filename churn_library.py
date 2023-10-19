"""
Module to perform all model training steps, including visualization and hyperparameter tuning.
Author: Emerald
Date: September 2023
"""

import os
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use the Tkinter backend
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import shap

# Set environment variable
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    Returns a DataFrame for the CSV found at pth.

    Args:
        pth (str): Path to the CSV file.

    Returns:
        df (pd.DataFrame): Pandas DataFrame.
    '''
    df = pd.read_csv(pth)
    # Create a Churn column if it doesn't exist
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    '''
    Perform Exploratory Data Analysis (EDA) on df and save figures to the "images" folder.

    Args:
        df (pd.DataFrame): Pandas DataFrame to be analyzed.

    Returns:
        None
    '''
    # Create the "images" folder if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")

    # Create the "eda" subfolder within the "images" folder if it doesn't exist
    eda_folder = os.path.join("images", "eda")
    if not os.path.exists(eda_folder):
        os.makedirs(eda_folder)

    # Plot and save figures
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title('Churn Histogram')
    plt.savefig(os.path.join(eda_folder, "churn_histogram.png"))

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title('Customer Age Histogram')
    plt.savefig(os.path.join(eda_folder, "customer_age_histogram.png"))

    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts(normalize=True).plot(kind='bar')
    plt.title('Marital Status Bar Plot')
    plt.savefig(os.path.join(eda_folder, "marital_status_bar_plot.png"))
 

    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'], hist=True, kde=True)
    plt.title('Total Transaction Count Distribution')
    plt.savefig(os.path.join(eda_folder, "total_transaction_count_distribution.png"))
 
    quant_columns = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio']
    plt.figure(figsize=(20, 10))
    sns.heatmap(df[quant_columns].corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(eda_folder, "correlation_heatmap.png"))
    


def encoder_helper(df, category_lst, response):
    '''
    Helper function to turn each categorical column into a new column with
    the proportion of churn for each category.

    Args:
        df (pd.DataFrame): Pandas DataFrame.
        category_lst (list): List of columns that contain categorical features.
        response (str): String of response name.

    Returns:
        df (pd.DataFrame): Pandas DataFrame with new columns for encoded categorical variables.
    '''
    for category in category_lst:
        category_lst = []
        category_groups = df.groupby(category).mean()[response]

        for val in df[category]:
            category_lst.append(category_groups.loc[val])

        df[f'{category}_{response}'] = category_lst
  

    return df

def perform_feature_engineering(df, response):
    '''
    Perform feature engineering and split the data into training and testing sets.

    Args:
        df (pd.DataFrame): Pandas DataFrame.
        response (str): String of response name.

    Returns:
        X_train (pd.DataFrame): Features for training.
        X_test (pd.DataFrame): Features for testing.
        y_train (pd.Series): Target variable for training.
        y_test (pd.Series): Target variable for testing.
    '''

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']
    # Assuming that 'response' is the target variable
    target = df[response]

    # Drop the response variable from the features
    features = df[keep_cols]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    '''
    Produces classification reports for training and testing results and stores reports as images in
    the "images" folder.

    Args:
        y_train (array-like): Training response values.
        y_test (array-like): Test response values.
        y_train_preds_lr (array-like): Training predictions from logistic regression.
        y_train_preds_rf (array-like): Training predictions from random forest.
        y_test_preds_lr (array-like): Test predictions from logistic regression.
        y_test_preds_rf (array-like): Test predictions from random forest.

    Returns:
        None
    '''
    # Create the "images" folder if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")

    # Create the "eda" subfolder within the "images" folder if it doesn't exist
    result_folder = os.path.join("images", "results")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    plt.rc('figure', figsize=(5, 5))

    # Random Forest Reports and Images
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig(os.path.join(result_folder, "random_forest_classification_report.png"))
    plt.close()

    # Logistic Regression Reports and Images
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig(os.path.join(result_folder, "logistic_regression_classification_report.png"))
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    Creates and stores the feature importances plot and SHAP model explainability plot.

    Args:
        model: Model object containing feature_importances_ attribute.
        x_data: Pandas DataFrame of X values.
        output_pth: Path to store the figures.

    Returns:
        None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot for feature importances
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save the feature importances figure
    plt.savefig(os.path.join(output_pth, "rf_Features_importance.png"))
    plt.close()




def train_models(x_train, x_test, y_train, y_test):
    '''
    Train, store model results: images + scores, and store models.

    Args:
        x_train (pd.DataFrame): Features for training.
        x_test (pd.DataFrame): Features for testing.
        y_train (pd.Series): Target variable for training.
        y_test (pd.Series): Target variable for testing.

    Returns:
        None
    '''

    # Grid search for Random Forest
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    # Train Logistic Regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)

    # Predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Print scores
    print('Random Forest Results')
    print('Test Results')
    print(classification_report(y_test, y_test_preds_rf))
    print('Train Results')
    print(classification_report(y_train, y_train_preds_rf))

    print('Logistic Regression Results')
    print('Test Results')
    print(classification_report(y_test, y_test_preds_lr))
    print('Train Results')
    print(classification_report(y_train, y_train_preds_lr))

    # Plot ROC curves
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_preds_rf)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_preds_lr)

    plt.figure(figsize=(15, 8))
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig("images/results/roc_curves.png")
    plt.close()

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
