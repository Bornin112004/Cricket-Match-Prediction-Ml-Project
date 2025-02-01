from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open('ipl_win_predictor.pkl', 'rb'))

# Define teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', teams=teams, cities=cities)

@app.route('/predict', methods=['POST'])
def predict():
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    city = request.form['city']
    runs_left = int(request.form['runs_left'])
    balls_left = int(request.form['balls_left'])
    wickets = int(request.form['wickets'])
    total_runs = int(request.form['total_runs'])
    
    # Calculate Current Run Rate (CRR) and Required Run Rate (RRR)
    crr = (total_runs - runs_left) * 6 / (120 - balls_left)
    rrr = runs_left * 6 / balls_left if balls_left > 0 else 0
    
    input_data = pd.DataFrame([[batting_team, bowling_team, city, runs_left, balls_left, wickets, total_runs, crr, rrr]],
                              columns=['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr'])
    
    win_probability = model.predict_proba(input_data)[0][1] * 100  # Probability of batting team winning
    lose_probability = 100 - win_probability  # Probability of bowling team winning
    
    return render_template('index.html', teams=teams, cities=cities, batting_team=batting_team, bowling_team=bowling_team, 
                           prediction_text=f'{batting_team} Win Probability: {win_probability:.2f}% | {bowling_team} Win Probability: {lose_probability:.2f}%')

if __name__ == '__main__':
    app.run(debug=True)
