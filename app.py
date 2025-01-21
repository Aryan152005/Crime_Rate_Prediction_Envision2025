from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load the trained model
with open('Model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Simulate the label encoding for cities
cities = ['Ahmedabad', 'Bengaluru', 'Chennai', 'Delhi', 'Mumbai']
city_encoder = {city: idx for idx, city in enumerate(cities)}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs
        year = int(request.form['year'])
        city = request.form['city']
        population = float(request.form['population'])
        murder = int(request.form['murder'])

        # Encode city
        city_encoded = city_encoder.get(city, -1)
        if city_encoded == -1:
            return render_template('error.html', message="Invalid city input")

        # Create input array
        input_data = np.array([[year, city_encoded, population, murder]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Create synthetic dataset for visualization
        synthetic_data = pd.DataFrame({
            'Year': [2018, 2016, 2017, 2019, 2017],
            'City': ['Bengaluru', 'Delhi', 'Mumbai', 'Delhi', 'Delhi'],
            'Population': [86.31, 91.18, 53.66, 72.24, 52.37],
            'Murder': [136, 176, 97, 112, 115],
            'Predicted Crime Rate': [2.03, 2.14, 1.70, 1.64, 1.85]
        })

        synthetic_data['City'] = synthetic_data['City'].map(city_encoder)

        # Add new prediction to the dataset
        synthetic_data = pd.concat([synthetic_data, pd.DataFrame([{
            'Year': year, 'City': city_encoded, 'Population': population,
            'Murder': murder, 'Predicted Crime Rate': prediction
        }])], ignore_index=True)

        # Generate and save plots
        generate_visualizations(synthetic_data)

        return render_template('result.html', prediction=round(prediction, 2))
    except Exception as e:
        return render_template('error.html', message=str(e))


def generate_visualizations(data):
    """Generate and save visualizations based on the predicted data."""
    # Decode city names
    reverse_city_encoder = {v: k for k, v in city_encoder.items()}
    data['City'] = data['City'].map(reverse_city_encoder)

    # Bar plot: Average Predicted Crime Rate by City
    avg_crime_by_city = data.groupby('City')['Predicted Crime Rate'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='City', y='Predicted Crime Rate', data=avg_crime_by_city)
    plt.title('Average Predicted Crime Rate by City')
    plt.xticks(rotation=45)
    plt.ylabel('Predicted Crime Rate')
    plt.xlabel('City')
    plt.tight_layout()
    plt.savefig('static/avg_crime_by_city.png')

    # Line plot: Yearly Trend of Predicted Crime Rate
    avg_crime_by_year = data.groupby('Year')['Predicted Crime Rate'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='Predicted Crime Rate', data=avg_crime_by_year, marker='o')
    plt.title('Yearly Trend of Predicted Crime Rate')
    plt.ylabel('Predicted Crime Rate')
    plt.xlabel('Year')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/yearly_trend.png')


if __name__ == '__main__':
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
