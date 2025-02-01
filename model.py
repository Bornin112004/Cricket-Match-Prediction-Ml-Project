import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load Data
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

# Data Cleaning and Transformation
def clean_team_names(df):
    df['team1'] = df['team1'].replace({'Delhi Daredevils': 'Delhi Capitals', 'Deccan Chargers': 'Sunrisers Hyderabad'})
    df['team2'] = df['team2'].replace({'Delhi Daredevils': 'Delhi Capitals', 'Deccan Chargers': 'Sunrisers Hyderabad'})
    return df

match = clean_team_names(match)
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']
match = match[(match['team1'].isin(teams)) & (match['team2'].isin(teams)) & (match['dl_applied'] == 0)]

# Feature Engineering
def create_features(match, delivery):
    total_score_df = delivery.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
    total_score_df = total_score_df[total_score_df['inning'] == 1]
    match = match.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')
    match = match[['match_id', 'city', 'winner', 'total_runs']]
    delivery = match.merge(delivery, on='match_id')
    delivery = delivery[delivery['inning'] == 2]
    return delivery

delivery_df = create_features(match, delivery)

def calculate_match_metrics(df):
    df['current_score'] = df.groupby('match_id')['total_runs_y'].cumsum()
    df['runs_left'] = df['total_runs_x'] - df['current_score']
    df['balls_left'] = 126 - (df['over'] * 6 + df['ball'])
    df['player_dismissed'] = df['player_dismissed'].fillna(0).apply(lambda x: 1 if x != 0 else 0)
    df['wickets'] = 10 - df.groupby('match_id')['player_dismissed'].cumsum()
    df['crr'] = (df['current_score'] * 6) / (120 - df['balls_left'])
    df['rrr'] = (df['runs_left'] * 6) / df['balls_left']
    df['result'] = df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)
    return df

delivery_df = calculate_match_metrics(delivery_df)
final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr', 'result']].dropna()
final_df = final_df[final_df['balls_left'] != 0]

# Splitting Data
X = final_df.iloc[:, :-1]
y = final_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Model Training
def train_model(X_train, y_train, model):
    trf = ColumnTransformer([
        ('encoder', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')

    pipe = Pipeline(steps=[('preprocessor', trf), ('classifier', model)])
    pipe.fit(X_train, y_train)
    return pipe

logistic_model = train_model(X_train, y_train, LogisticRegression(solver='liblinear'))
random_forest_model = train_model(X_train, y_train, RandomForestClassifier(random_state=42))

# Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

evaluation_logistic = evaluate_model(logistic_model, X_test, y_test)
evaluation_random_forest = evaluate_model(random_forest_model, X_test, y_test)

# Displaying Comparison Table
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Logistic Regression': [evaluation_logistic['Accuracy'], evaluation_logistic['Precision'], evaluation_logistic['Recall'], evaluation_logistic['F1 Score']],
    'Random Forest Classifier': [evaluation_random_forest['Accuracy'], evaluation_random_forest['Precision'], evaluation_random_forest['Recall'], evaluation_random_forest['F1 Score']]
})

print(comparison_df.to_string(index=False))

# Save Model
pickle.dump(random_forest_model, open('ipl_win_predictor.pkl', 'wb'))
