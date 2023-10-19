"""
Module to perform test all functions and log info and errors
Author: Emerald
Date: September 2023
"""

import os
import logging
import churn_library as cls

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

if not os.path.exists("logs"):
    os.makedirs("logs")


# Configure logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    Test data import.
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("ERROR: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("SUCCESS: DataFrame shape is valid")
    except AssertionError as err:
        logging.error(
            "ERROR: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    Test perform eda function.
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing perform_eda: Data imported successfully")
    except Exception as err:
        logging.error("Testing perform_eda: Error occurred during data import")
        raise err

    # Perform EDA on the same data
    try:
        cls.perform_eda(df)
        logging.info("Testing perform_eda: EDA completed successfully")
    except Exception as err:
        logging.error("Testing perform_eda: Error occurred during EDA")
        raise err

    # Add assertions for checking if image files exist
    try:
        assert os.path.isfile("images/eda/churn_histogram.png")
        assert os.path.isfile("images/eda/customer_age_histogram.png")
        assert os.path.isfile("images/eda/marital_status_bar_plot.png")
        assert os.path.isfile("images/eda/total_transaction_count_distribution.png")
        assert os.path.isfile("images/eda/correlation_heatmap.png")
        logging.info("Testing perform_eda: Image files generated successfully")
    except AssertionError as err:
        logging.error("Testing perform_eda: Image files not found")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    Test encoder helper.
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        df_encoded = cls.encoder_helper(df, category_lst, 'Churn')
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: Error occurred during encoding")
        raise err

    # Add assertions for checking if encoded columns are added
    try:
        assert all(
            col in df_encoded.columns for col in [
                'Gender_Churn',
                'Education_Level_Churn',
                'Marital_Status_Churn',
                'Income_Category_Churn',
                'Card_Category_Churn'])
        logging.info(
            "Testing encoder_helper: Encoded columns added successfully")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Encoded columns not found")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    Test perform_feature_engineering.
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df, 'Churn')
        logging.info("SUCCESS: Testing perform_feature_engineering.")
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: Error occurred during feature engineering")
        raise err

    # Add assertions for checking if data splits are not empty
    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info(
            "Testing perform_feature_engineering: Data splits are valid")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Data splits are empty")
        raise err


def test_train_models(train_models):
    '''
    Test train_models.
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df, 'Churn')
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing train_models: Error occurred during model training")
        raise err

    # Add assertions for checking if model files and ROC curve image exist
    try:
        assert os.path.isfile("./models/rfc_model.pkl")
        assert os.path.isfile("./models/logistic_model.pkl")
        assert os.path.isfile("images/roc_curves.png")
        logging.info(
            "Testing train_models: Model files and ROC curve image created successfully")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Model files or ROC curve image not found")
        raise err


if __name__ == "__main__":
    # Run test functions
    test_import(cls.import_data)
    test_eda()
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
