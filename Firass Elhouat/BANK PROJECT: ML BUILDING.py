import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------- Load the dataset
ChurnData = pd.read_csv(
    r'/Users/firasxcx/Documents/Loyola /Fall Semster/Machine learning /Project_ML/Customer-Churn-Records.csv')
import pandas as pd


# deriving a new variable based on Complain in conditional to Satisfaction Score
# Define the functions
def Customer_Sentiment(row):
    if row['Complain'] == 0:
        if row['Satisfaction Score'] > 3:
            return 'Positive_Sentiment'  # High satisfaction, no complaint
        else:
            return 'Negative_Sentiment'  # Low satisfaction, no complaint
    elif row['Complain'] == 1:
        if row['Satisfaction Score'] >= 3:
            return 'Inaccurate'  # High satisfaction, with a complaint [inaccurate]
        else:
            return 'Negative_Sentiment'  # Low satisfaction


def More_then_One_Pro(row):
    if row['NumOfProducts'] > 1:
        return 'More_Then_One'
    else:
        return 'One_Product'


# Apply the functions to the DataFrame
ChurnData['Customer_Sentiment'] = ChurnData.apply(Customer_Sentiment, axis=1)
ChurnData['More_then_One_Pro'] = ChurnData.apply(More_then_One_Pro, axis=1)

sns.countplot(x='Customer_Sentiment', data=ChurnData, palette="seismic", hue='Exited')
plt.show()
sns.countplot(x='More_then_One_Pro', data=ChurnData, palette="seismic", hue='Exited')
plt.show()

# Get counts of each sentiment category
sentiment_counts_by_exited = ChurnData.groupby('Exited')['Customer_Sentiment'].value_counts().unstack(fill_value=0)

# Convert the result into a DataFrame
sentiment_counts_df = sentiment_counts_by_exited.reset_index()

# Print the DataFrame
# Display the full DataFrame without truncation
pd.set_option('display.max_rows', None)  # To display all rows
pd.set_option('display.max_columns', None)  # To display all columns
print(sentiment_counts_df)

# Data Preprocessing: Drop unnecessary columns
ChurnData_filtered = ChurnData.drop(columns=['Surname', 'RowNumber', 'Complain', 'Satisfaction Score', 'NumOfProducts','CustomerId'])

# One-Hot Encoding for categorical variables
ChurnData_encoded = pd.get_dummies(ChurnData_filtered, drop_first=True)

# Correlation Analysis
correlation_matrix = ChurnData_encoded.corr()

# Correlation Heatmap
mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))  # Create a mask for the upper triangle

plt.figure(figsize=(9, 6))  # Keep the original figure size
ax = sns.heatmap(correlation_matrix, annot=True, cmap='seismic',
                 linewidths=0.1, annot_kws={"size": 5}, fmt=".2f")
ax.tick_params(axis='x', labelsize=5)
ax.tick_params(axis='y', labelsize=6)
plt.xticks(rotation=25, ha='right')  # Rotate labels 45 degrees for better readability
plt.title('Correlation Heatmap')
plt.show()

# Extracting Only X features
X = ChurnData_encoded.drop('Exited', axis=1)  # Features
y = ChurnData_encoded['Exited']  # Target

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Define the models and their hyperparameters
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Neural Network': MLPClassifier()
}
# Defining Parameter grid
param_grids = {
    'Logistic Regression': {'C': [2 ** i for i in range(-5, 5)],
                            'solver': ['liblinear', 'newton-cg', 'lbfgs',
                                       'saga', 'sag', 'newton-cholesky'],
                            'max_iter': [1000]},
    'SVM': {'C': [0.01, 0.1, 1, 10],
            'kernel': ['rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]},
    'KNN': {'n_neighbors': [3, 5, 7, 15, 20],
            'weights': ['uniform', 'distance'],
            'p': [1, 2, 3]},
    'Neural Network': {
        'hidden_layer_sizes': [(5, 5, 5), (10, 10),
                               (20,20), (10, 10, 10),
                               (50, 50)],
        'activation': ['relu', 'tanh'],
        'max_iter': [5000, 7000]
    }
}
file = open("Model_Evaluation_New_Variable.txt", "w")

# Running GridSearchCV for each model
best_models = {}
for model_name in models:
    print(f"Running GridSearchCV for {model_name}...")
    file.write(f"Running GridSearchCV for {model_name}...")
    print("=" * 50)
    file.write("=" * 50)

    # Set up GridSearchCV with 10-fold cross-validation
    grid_search = GridSearchCV(estimator=models[model_name],
                               param_grid=param_grids[model_name],
                               cv=10,
                               n_jobs=-1,  # Use all processors
                               verbose=1)  # To see progress

    # Fit GridSearchCV to training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_models[model_name] = grid_search.best_estimator_

    # Print the best parameters
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    file.write(f"Best Parameters for {model_name}: {grid_search.best_params_}\n")
    print("=" * 50)
    file.write("=" * 50)

    # Evaluate on test set
    y_pred = best_models[model_name].predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy for {model_name}: {accuracy:.4f}")
    file.write(f"Test accuracy for {model_name}: {accuracy:.4f}")
    print("=" * 50)
    file.write("=" * 50)

    # Print Classification Report
    file.write(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
    print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
    print("=" * 50)
    file.write("=" * 50)

    # Optionally, print all the best models
    print("\nBest models from GridSearchCV:")
    for model_name, model in best_models.items():
        print(f"{model_name}: {model}")
file.write("\nBest models from GridSearchCV:")
file.write(f"{model_name}: {model}")
print("=" * 50)
file.write("=" * 50)

file.close()
