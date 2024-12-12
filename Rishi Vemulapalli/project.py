import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("Customer-Churn-Records.csv")

# Clean the data: Drop irrelevant columns
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Prepare features and labels
X = df.drop(columns=['Exited'])
y = df['Exited']

# Preprocessing function for creating features
def Customer_Sentiment(row):
    if row['Complain'] == 0:
        if row['SatisfactionScore'] > 3:
            return 'Positive_Sentiment'
        else:
            return 'Negative_Sentiment'
    elif row['Complain'] == 1:
        if row['SatisfactionScore'] >= 3:
            return 'Inaccurate'
        else:
            return 'Negative_Sentiment'
    return 'unknown'

# Apply the sentiment function
X['Customer_Sentiment'] = X.apply(Customer_Sentiment, axis=1)

# Drop the original columns that are no longer needed
X.drop(columns=['Complain', 'SatisfactionScore'], inplace=True)

# List of categorical and continuous columns
categorical_columns = ['Geography', 'Gender', 'Card Type', 'HasCrCard', 'IsActiveMember', 'Customer_Sentiment']
continuous_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'PointsEarned']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),
        ('num', StandardScaler(), continuous_columns)
    ])

# Split data into train/validation/test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# Define the models
models = {
    'GaussianNB': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='binary:logistic',
        random_state=42
    )
}

# Train and evaluate models
output_file = open("model_evaluation_results.txt", "w")
output_file.write("Model Evaluation Results\n")
output_file.write("========================\n\n")

for model_name, model in models.items():
    output_file.write(f"Best Model Name: {model_name}\n")
    
    if model_name == 'KNN':
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'euclidean']
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        output_file.write(f"Best Parameters for {model_name}: {grid_search.best_params_}\n\n")
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    # Get predictions
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    # Validation metrics
    output_file.write(f"{model_name} Validation Accuracy: {accuracy_score(y_val, y_val_pred):.2f}\n\n")
    output_file.write("              precision    recall  f1-score   support\n\n")
    
    # Class 0
    output_file.write(f"           0       {precision_score(y_val, y_val_pred, pos_label=0):.2f}      {recall_score(y_val, y_val_pred, pos_label=0):.2f}      {f1_score(y_val, y_val_pred, pos_label=0):.2f}      {sum(y_val == 0)}\n")
    
    # Class 1
    output_file.write(f"           1       {precision_score(y_val, y_val_pred, pos_label=1):.2f}      {recall_score(y_val, y_val_pred, pos_label=1):.2f}      {f1_score(y_val, y_val_pred, pos_label=1):.2f}      {sum(y_val == 1)}\n\n")
    
    # Overall metrics
    output_file.write(f"    accuracy                           {accuracy_score(y_val, y_val_pred):.2f}      {len(y_val)}\n")
    output_file.write(f"   macro avg       {precision_score(y_val, y_val_pred, average='macro'):.2f}      {recall_score(y_val, y_val_pred, average='macro'):.2f}      {f1_score(y_val, y_val_pred, average='macro'):.2f}      {len(y_val)}\n")
    output_file.write(f"weighted avg       {precision_score(y_val, y_val_pred, average='weighted'):.2f}      {recall_score(y_val, y_val_pred, average='weighted'):.2f}      {f1_score(y_val, y_val_pred, average='weighted'):.2f}      {len(y_val)}\n\n")

    # Test metrics
    output_file.write(f"{model_name} Test Accuracy: {accuracy_score(y_test, y_test_pred):.2f}\n\n")
    output_file.write("              precision    recall  f1-score   support\n\n")
    
    # Class 0
    output_file.write(f"           0       {precision_score(y_test, y_test_pred, pos_label=0):.2f}      {recall_score(y_test, y_test_pred, pos_label=0):.2f}      {f1_score(y_test, y_test_pred, pos_label=0):.2f}      {sum(y_test == 0)}\n")
    
    # Class 1
    output_file.write(f"           1       {precision_score(y_test, y_test_pred, pos_label=1):.2f}      {recall_score(y_test, y_test_pred, pos_label=1):.2f}      {f1_score(y_test, y_test_pred, pos_label=1):.2f}      {sum(y_test == 1)}\n\n")
    
    # Overall metrics
    output_file.write(f"    accuracy                           {accuracy_score(y_test, y_test_pred):.2f}      {len(y_test)}\n")
    output_file.write(f"   macro avg       {precision_score(y_test, y_test_pred, average='macro'):.2f}      {recall_score(y_test, y_test_pred, average='macro'):.2f}      {f1_score(y_test, y_test_pred, average='macro'):.2f}      {len(y_test)}\n")
    output_file.write(f"weighted avg       {precision_score(y_test, y_test_pred, average='weighted'):.2f}      {recall_score(y_test, y_test_pred, average='weighted'):.2f}      {f1_score(y_test, y_test_pred, average='weighted'):.2f}      {len(y_test)}\n\n")
    
    output_file.write("="*50 + "\n\n")

output_file.close()
print("Model evaluation completed. Results saved to 'model_evaluation_results.txt'.")


