import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd


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
