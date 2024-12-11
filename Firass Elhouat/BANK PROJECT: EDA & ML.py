import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
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

#######################################
# Data Introduction & Exploratory Analysis
#######################################
# -------- Load the dataset
ChurnData = pd.read_csv(r'/Users/firasxcx/Documents/Loyola /Fall Semster/Machine learning /Project_ML/Customer-Churn-Records.csv')

# Print top portion
print(ChurnData.head(5))

# Structure of the data
print(ChurnData.info())

# Descriptive statistics
print(ChurnData.describe())

# Data Preprocessing: Drop unnecessary columns
ChurnData_filtered = ChurnData.drop(columns=['Surname', 'RowNumber'])

# One-Hot Encoding for categorical variables
ChurnData_encoded = pd.get_dummies(ChurnData_filtered, drop_first=1)

# Correlation Analysis
correlation_matrix = ChurnData_encoded.corr()

# Correlation heatmap interactive plot
fig = px.imshow(
    correlation_matrix,
    text_auto=".2f",
    color_continuous_scale='balance',
    title='Correlation Heatmap',
    labels=dict(color='Correlation'),)
fig.update_layout(width=800, height=800)
# Display the figure interactively
fig.show()

# Define color palette
palette = {0: 'blue', 1: 'red'}
# Create subplots
fig = sp.make_subplots(
    rows=2,
    cols=3,
    subplot_titles=[
        'Exited Distribution',
        'Customer Exited by Geography',
        'Customer Exited by Gender',
        'Customer Exited by Number of Products',
        'Customer Exited by Has Credit Card',
        'Customer Exited by Has Complained'
    ]
)

# Define the data groups and their locations
data_groups = [
    ('Exited', ChurnData_filtered['Exited'].value_counts(), 1, 1),
    ('Geography', ChurnData_filtered.groupby('Geography')['Exited'].value_counts().unstack(), 1, 2),
    ('Gender', ChurnData_filtered.groupby('Gender')['Exited'].value_counts().unstack(), 1, 3),
    ('NumOfProducts', ChurnData_filtered.groupby('NumOfProducts')['Exited'].value_counts().unstack(), 2, 1),
    ('HasCrCard', ChurnData_filtered.groupby('HasCrCard')['Exited'].value_counts().unstack(), 2, 2),
    ('Complain', ChurnData_filtered.groupby('Complain')['Exited'].value_counts().unstack(), 2, 3)
]

# Add traces dynamically
for group_name, data, row, col in data_groups:
    if group_name == 'Exited':
        # For Exited distribution, plot a single bar chart
        fig.add_trace(go.Bar(
            x=data.index.astype(str),
            y=data.values,
            marker_color=[palette[0], palette[1]],
            name=f'{group_name} Distribution'
        ), row=row, col=col)
    else:
        # For grouped data, plot Not Exited first, then Exited
        for exited_status in [0, 1]:  # Non-Exited (0) first, Exited (1) on top
            fig.add_trace(go.Bar(
                x=data.index.astype(str),
                y=data[exited_status],
                marker_color=palette[exited_status],
                name=f'{group_name} (Exited={exited_status})',
                opacity=1 if exited_status == 1 else 0.8  # Highlight Exited
            ), row=row, col=col)

# Update layout
fig.update_layout(height=650, title_text="Customer Exited Analysis",
                  barmode='group',showlegend=False)


fig.show()


# Continuous variables distribution plots
# Create subplots
fig = sp.make_subplots(rows=2, cols=2,
                       subplot_titles=[
                           'Age Distribution by Exited',
                           'Credit Score Distribution by Exited',
                           'Balance Distribution by Exited',
                           'Non-Zero Balance Distribution by Exited'])

# Filtering Balance to exclude when Balance = 0
ChurnData_filtered_no_zero_balance = ChurnData_filtered[ChurnData_filtered['Balance'] > 0]

# Defining columns to plot
columns = ['Age', 'CreditScore', 'Balance', 'Balance']
datasets = [ChurnData_filtered, ChurnData_filtered, ChurnData_filtered, ChurnData_filtered_no_zero_balance]

# Define a color palette for Exited categories
palette = {0: 'blue', 1: 'red'}

# Defining interactive plot
for i, (column, data) in enumerate(zip(columns, datasets)):
    for exited_status in [0, 1]:
        filtered_data = data[data['Exited'] == exited_status]
        fig.add_trace(go.Histogram(
            x=filtered_data[column],
            nbinsx=50,
            name=f"{column} (Exited={exited_status})",
            marker=dict(color=palette[exited_status]),
            opacity=0.7),
            row=(i // 2) + 1, col=(i % 2) + 1)

# Update layout
fig.update_layout(height=600, title_text="Customer Churn Numerical Analysis",
                  barmode='overlay',
                  showlegend=False,
                  legend=dict(title='Exited', orientation='h',
                              yanchor='top',
                              y=-0.2,
                              xanchor='right',
                              x=0.5)) # legend turned off
fig.show()




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


# Data Preprocessing: Drop unnecessary columns
ChurnData_filtered = ChurnData.drop(columns=['Surname', 'RowNumber', 'Complain',
                                             'Satisfaction Score',
                                             'NumOfProducts','CustomerId'])

# One-Hot Encoding for categorical variables
ChurnData_encoded = pd.get_dummies(ChurnData_filtered, drop_first=True)

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
