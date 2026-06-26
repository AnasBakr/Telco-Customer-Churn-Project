
## Telco Customer Churn Prediction

### Overview

This project focuses on building a machine learning model to predict customer churn in a telecommunications dataset. Customer churn is a critical problem for many businesses, and accurately predicting which customers are likely to leave can enable companies to implement proactive retention strategies. This notebook covers the entire machine learning pipeline, from data loading and preprocessing to model training, evaluation, and preparation for deployment.

### Dataset

The dataset used in this project is the "Telco Customer Churn" dataset from Kaggle, which includes information about customers' demographics, services they've subscribed to, contract details, and whether they churned or not.

### Project Structure

The notebook is organized into the following main sections:

1.  **Data Loading and Initial Inspection**: Loading the dataset and performing initial checks on its structure, data types, and basic statistics.
2.  **Data Cleaning and Feature Engineering**: Handling missing values, converting data types, dropping irrelevant features, and encoding categorical variables into a numerical format suitable for machine learning models.
    *   `TotalCharges` column converted to numeric.
    *   `customerID` dropped.
    *   `Churn` target variable converted to 0/1.
    *   Binary categorical features (e.g., `gender`, `Partner`) mapped to 0/1.
    *   Multi-category nominal features (e.g., `InternetService`, `PaymentMethod`) one-hot encoded.
3.  **Exploratory Data Analysis (EDA)**: Visualizing data distributions, correlations, and relationships between various features and the target variable (`Churn`) to gain insights into customer behavior.
    *   Churn distribution analysis.
    *   Numerical feature analysis (`tenure`, `MonthlyCharges`, `TotalCharges`) vs. Churn.
    *   Categorical feature analysis (e.g., `gender`, `Internet Service`, `Contract`, `Payment Method`) vs. Churn.
    *   Multivariate analysis to identify critical customer segments.
4.  **Data Preprocessing for Model Training**: Preparing the data further for machine learning models.
    *   **Feature Scaling**: Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) scaled using `StandardScaler`.
    *   **Class Imbalance Handling**: Synthetic Minority Over-sampling Technique (SMOTE) applied to balance the imbalanced churn classes.
    *   **Train-Test Split**: Data split into training and testing sets with stratification.
5.  **Machine Learning Model Training and Evaluation**: Training and evaluating multiple classification models to predict churn.
    *   **Models Trained**: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, XGBoost Classifier, LightGBM Classifier.
    *   **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix for each model.
6.  **Model Deployment Preparation**: Saving the best-performing model and scaler for future deployment as a predictive service.
7.  **Interactive Prediction Interface (Gradio)**: Building a user-friendly web interface using Gradio to allow interactive testing of the churn prediction model.
8.  **Cloud Deployment Guide: Google Cloud Run**: Providing instructions and necessary files (`requirements.txt`, `app.py`, `Dockerfile`) for deploying the model as a Flask API on Google Cloud Run.

### Key Findings

*   **Churn Rate**: Approximately 26.6% of customers in the dataset churn.
*   **Significant Predictors**: Customers with shorter `tenure`, higher `MonthlyCharges`, `Fiber Optic` internet service, and using `Electronic Check` payment methods are more likely to churn.
*   **Retention Factors**: Customers on `Two-year contracts` show significantly lower churn rates.
*   **Gender Impact**: `gender` does not appear to be a significant predictor of churn.

### Best Performing Model

Among the evaluated models, the **Random Forest Classifier** achieved the best performance:

*   **Accuracy**: 0.8408
*   **F1-Score**: 0.8446

This model provides a robust balance between correctly identifying churning customers (high recall) and minimizing false positives (good precision).

### Deployment

The project includes all necessary files and instructions to deploy the trained Random Forest model as a RESTful API using Flask, containerized with Docker, and hosted on Google Cloud Run. An interactive Gradio interface is also provided for local testing and demonstration.

### How to Use (Local Development)

1.  Clone this repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the Jupyter/Colab notebook to train the model and generate `random_forest_model.joblib` and `scaler.joblib`.
4.  Optionally, run the Gradio interface directly from the notebook for interactive predictions.
5.  For local API testing, save `app.py`, `random_forest_model.joblib`, and `scaler.joblib` in the same directory and run `python app.py`.

### Future Enhancements

*   Explore advanced feature engineering techniques.
*   Hyperparameter tuning for all models.
*   Implement more sophisticated deployment strategies (e.g., CI/CD pipelines).
*   Integrate model monitoring and retraining mechanisms.

Feel free to explore the notebook, provide feedback, or contribute to this project!
