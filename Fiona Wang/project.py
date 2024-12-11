import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import io

seed = 0
string_builder = io.StringIO()

def get_seed():
    global seed
    seed = seed + 1
    return seed

df = pd.read_csv("Customer-Churn-Records.csv")

print(df['Exited'].value_counts())

array=df.shape
print(array)
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

X = df.drop(columns=['Exited'])
y = df['Exited']

def Customer_Sentiment(row):
    if row['Complain'] == 0:
        if row['SatisfactionScore'] > 3:
            return 'Positive_Sentiment'  # High satisfaction, no complaint
        elif row['SatisfactionScore'] <= 3:
            return 'Negative_Sentiment'  # Low satisfaction, no complaint
    elif row['Complain'] == 1:
        if row['SatisfactionScore'] >= 3:
            return 'Inaccurate'  # High satisfaction, with a complaint [inaccurate]
        elif row['SatisfactionScore'] < 3:
            return 'Negative_Sentiment' # Low satisfaction
        return 'unknown'
    
X['Customer_Sentiment'] = X.apply(Customer_Sentiment, axis=1)
X.drop(columns=['Complain', 'SatisfactionScore'])

categorical_columns = ['Geography', 'Gender', 'Card Type', 'HasCrCard', 'IsActiveMember', 'Customer_Sentiment']
continuous_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'PointsEarned']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),  # One-hot encode categorical columns
        ('num', StandardScaler(), continuous_columns)   # Scale continuous columns
    ])

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=seed)

models = {
    'MLP': MLPClassifier(max_iter=1000, random_state=seed),
    'SVM': SVC(random_state=seed),
    'LogisticRegression': LogisticRegression(random_state=get_seed(),max_iter=5000)
}

param_grids = {
    'MLP': {
        'classifier__hidden_layer_sizes': [(8,), (8, 4), (16, 8)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__solver': ['adam', 'sgd'],
        'classifier__alpha': [0.001, 0.01, 0.1],
        'classifier__learning_rate_init': [0.001, 0.01],  
        'classifier__max_iter': [3000, 4000],  
        'classifier__tol': [1e-4],  
        'classifier__early_stopping': [True],  
        'classifier__validation_fraction': [0.1],  
        'classifier__n_iter_no_change': [10]  
    },
    'SVM': {
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__C': [0.1, 1, 10]
    },
    'LogisticRegression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['liblinear', 'saga']
    }
}

best_models = {}
best_model_name = None
best_model_val_accuracy = -float('inf')

output_file = open("model_evaluation_results.txt", "w") 
output_file.write("Model Evaluation Results\n") 
output_file.write("========================\n")

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    
    grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_models[model_name] = grid_search.best_estimator_
    string_builder.write(f"Best Parameters for {model_name}: {grid_search.best_params_}\n")
    
    # Evaluate on validation set
    y_val_pred = best_models[model_name].predict(X_val)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    string_builder.write(f"{model_name} Validation Accuracy: {accuracy_val:.2f}\n")
    string_builder.write(classification_report(y_val, y_val_pred))

    # Evaluate on test dataset
    y_test_pred = best_models[model_name].predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    string_builder.write(f"{model_name} Test Accuracy: {accuracy_test:.2f}\n")
    string_builder.write(classification_report(y_test, y_test_pred))
    
    if accuracy_val > best_model_val_accuracy:
        best_model_val_accuracy = accuracy_val
        best_model_name = model_name
        
output_file.write(f"\nBest Model Name: {best_model_name}\n") 
output_file.write(string_builder.getvalue())
string_builder.close()
output_file.close()

print(f" Best Model name: {best_model_name} ")
