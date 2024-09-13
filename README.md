# Breast Cancer Detection Using Machine Learning
This project is focused on building a machine learning model to detect whether a breast tumor is malignant (cancerous) or benign (non-cancerous). Various machine learning algorithms have been tested, including Logistic Regression, SVM, Random Forest, KNN, Naive Bayes, and XGBoost, with hyperparameter tuning applied to improve performance.

# Project Overview
# Objective: 
To predict whether a breast tumor is malignant or benign based on input features from a dataset.
# Dataset: 
The Breast Cancer Wisconsin dataset, provided by scikit-learn, was used for model training and testing.
# Algorithms: 
Multiple machine learning algorithms were applied and compared for performance.
# Key Features
# Data Preprocessing: 
Standardization of the dataset to ensure optimal model performance.
# Model Evaluation: 
Performance was measured using key metrics like Accuracy, Precision, Recall, and F1-Score.
# Hyperparameter Tuning: 
GridSearchCV was used to optimize hyperparameters, particularly for Logistic Regression.
# Best Model: 
Based on the evaluation, XGBoost showed the best performance, but Logistic Regression was further tuned for improvement.
# Steps in the Project
# Data Loading: 
Loaded the Breast Cancer Wisconsin dataset from sklearn.datasets.
# Data Preprocessing: 
Standardization using StandardScaler to normalize the data.
# Model Training: 
Trained six machine learning models:
Logistic Regression
Support Vector Machine (SVM)
Random Forest
K-Nearest Neighbors (KNN)
Naive Bayes
XGBoost
# Hyperparameter Tuning: 
Fine-tuned Logistic Regression using GridSearchCV to optimize C, penalty, solver, and max_iter.
# Model Evaluation: 
Measured accuracy, precision, recall, and F1-score for each model. The best model was chosen based on these metrics.
# Model Export: 
Saved the best model using Pickle for future use.
# Technologies Used
Python 3.9+
Pandas: For data manipulation and analysis.
Scikit-Learn: For machine learning models and evaluation.
XGBoost: For gradient boosting.
Matplotlib/Seaborn: For data visualization.
Pickle: For saving and loading the trained model.

# Model	Accuracy	
Logistic Reg.	 0.973684   
SVM	0.956140   0.971429  
Random Forest	0.964912   
KNN	0.947368  
Naive Bayes	0.964912   
XGBoost	0.956140  
Logistic Reg and SVM showed the highest performance, with Logistic Reg selected as the final model for deployment.

# Predict the result
prediction = model.predict(sample_data)
print("Prediction:", "Malignant" if prediction[0] == 1 else "Benign")
Future Enhancements
Add more advanced models like Deep Learning for further improvement.
Deploy the model as a web application using Flask or FastAPI.
Contributing
Feel free to open issues or create pull requests if you have any suggestions or improvements for the project.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
