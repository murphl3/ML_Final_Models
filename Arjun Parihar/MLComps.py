import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

#Load and preprocess the dataset
data = pd.read_csv("Customer-Churn-Records.csv")

#Drop unnecessary columns
columns_to_drop = ["RowNumber", "CustomerId", "Surname"]
data = data.drop(columns=columns_to_drop)

#Encode categorical variables
label_encoders = {}
for column in ["Geography", "Gender", "Card Type"]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

#Separate features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

#Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Define models and hyperparameter grids
models = {
    "Logistic Regression": LogisticRegression(),
    "MLP": MLPClassifier(max_iter=1000, random_state=11),
    "KNN": KNeighborsClassifier()
}

param_grids = {
    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }
}

#Perform hyperparameter tuning and evaluate models
best_models = {}
results = []

for model_name, model in models.items():
    print(f"Tuning hyperparameters for {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    #Make predictions
    y_pred = best_model.predict(X_test)
    
    #Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Model": model_name,
        "Best Params": grid_search.best_params_,
        "Accuracy": accuracy,
        "Precision": precision,
        "F1 Score": f1
    })

#Display results
results_df = pd.DataFrame(results)
print(results_df)

#Identify the best model
best_model_name = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]["Model"]
print(f"\nThe best model is: {best_model_name}")