# Telcom_Churn_Prediction

<br/> :office: **Background:**

Any business wants to maximize the number of customers. To achieve this goal, it is important not only to try to attract new ones, but also to retain existing ones. Retaining a client will cost the company less than attracting a new one. In addition, a new client may be weakly interested in business services and it will be difficult to work with him, while old clients already have the necessary data on interaction with the service.

Accordingly, predicting the churn, we can react in time and try to keep the client who wants to leave. Based on the data about the services that the client uses, we can make him a special offer, trying to change his decision to leave the operator. This will make the task of retention easier to implement than the task of attracting new users, about which we do not know anything yet.

You are provided with a dataset from a telecommunications company. The data contains information about almost six thousand users, their demographic characteristics, the services they use, the duration of using the operator's services, the method of payment, and the amount of payment.

<br/> :thought_balloon: **Task:** 

In this project we predicted which customers are likely to churn based on certain attributes.



<br/> :page_with_curl: **Feature Descriptions:**

  - customerID - customer id

  - gender - client gender (male / female)

  - SeniorCitizen - is the client retired (1, 0)

  - Partner - is the client married (Yes, No)

  - tenure - how many months a person has been a client of the company

  - PhoneService - is the telephone service connected (Yes, No)

  -  MultipleLines - are multiple phone lines connected (Yes, No, No phone service)

  - InternetService - client’s Internet service provider (DSL, Fiber optic, No)

  - OnlineSecurity - is the online security service connected (Yes, No, No internet service)

  - OnlineBackup - is the online backup service activated (Yes, No, No internet service)

  - DeviceProtection - does the client have equipment insurance (Yes, No, No internet service)

  - TechSupport - is the technical support service connected (Yes, No, No internet service)

  - StreamingTV - is the streaming TV service connected (Yes, No, No internet service)

  - StreamingMovies - is the streaming cinema service activated (Yes, No, No internet service)

  - Contract - type of customer contract (Month-to-month, One year, Two year)

  - PaperlessBilling - whether the client uses paperless billing (Yes, No)

  - PaymentMethod - payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))

  - MonthlyCharges - current monthly payment

  - TotalCharges - the total amount that the client paid for the services for the entire time

  - Churn - whether there was a churn (Yes or No)

<br/> :abacus: **Methodology:**
  
  - Data Understanding: 
    - 5986 rows and 22 columns 
    - Remove two columns "Unnamed: 0" and "customerID"
    - Feature distribution: Histogramm of all numerical features
    - Correlation among the numerical features: Strong correlation (0.83) between "TotalCharges" and "tenure"
    - Check for imbalance between the two classes in "Curn": 4300 times "0" and 1587 times "1"

  - Data Cleaning: 
    - Data leakage: no data leakage (see feature description)
    - Outliers: no outliers (see Boxplots)
    - Duplicates: no duplicates (based on the column "customerID")
    - Missing values:10 missing values in "TotalCharges" (replaced " " by "nan")
    - Changing Data Types: "TotalCharges" to numeric; "SenioCitizen" to categorical (No/ Yes instead of 0/1); "Churn" to numeric (0/1 instead of No/ Yes) 
  
  - Data preparation and preprocessing (sklearn):
    - Split numerical and categorical features
    - Missing Values: Imputation of numerical data (KNNImputer)
    - Data Imbalance: Upsamling of categorical data (SMOTENC)
    - Define X and Y ("Churn")
    - Split the data into train and test data: train_test_split (sklearn.model_selection)
    - Pipeline: we created pipelines for both the categorical (incl. OneHotEncoder) and the numeric data (incl. PowerTransformer and StandardScaler) as well as a preprocessor (ColumnTransformer) to combine these two pipelines
    - Why PowerTransformer: to make numeric data more Gaussian-line
    
  - Superviced Machine Learning Algorithm:
    - Supervised Machine Learning: The training data/ labels feeding to the algorithm includes the desired solutions (Churn)
    - Classification Algorithm: uses input training data to predict the likelihood that subsequent data will fall into one of the predetermined categories.
    - Binary Classifier: Distinguishes between the two classes "Yes" and "No"
    
  - Superviced Machine Learning - Model & Hyperparameter Tuning and Evaluation(pycaret and sklearn):
    - Metric: F1 (combination of recall and precision)
    - Model Selection (pycaret): CatBoost, RandomForest, LightGradientBoostMachine
    - CatBoost: F1-score 0.8372
    - LightGradientBoostingMachine: F1-score 0.8324
    - RandomForest: F1-score 0.8309
    - Tuned CatBoost: 0.8380 --> Best Model
    - Tuned LightGradientBoostingMachine: 0.8318
    - Tuned RandomForest: 0.8257
    - Best parameters of tuned CatBoost: depth 9, l2_leaf_reg 7, random_strength 0.8, n_estimators 210, eta 0.15
    - prediction:
    - Confusion matrix: 

  - Feature Interpretation (SHAP)

<br/> :chart_with_upwards_trend: **Results:**

<br/> :speech_balloon: **Recommendations:**
