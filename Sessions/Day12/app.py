from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__, template_folder='templates')

# Load the model and scaler
try:
    model = joblib.load(r'Sessions\Day12\insurance_prediction_model.joblib')
    scaler = joblib.load(r'Sessions\Day12\insurance_scaler.joblib')
except Exception as e:
    print(f"Error loading model or scaler: {e}")

def prepare_input_data(form_data):
    """Prepare input data for prediction"""
    data = {
        'age': float(form_data['age']),
        'sex': 1 if form_data['sex'] == 'male' else 0,
        'bmi': float(form_data['bmi']),
        'children': int(form_data['children']),
        'smoker': 1 if form_data['smoker'] == 'yes' else 0
    }
    
    # Add region columns (one-hot encoding)
    selected_region = form_data['region']
    regions = ['northeast', 'northwest', 'southeast', 'southwest']
    for region in regions:
        data[f'region_{region}'] = 1 if region == selected_region else 0
    
    # Create interaction terms
    data['age_smoker'] = data['age'] * data['smoker']
    data['bmi_smoker'] = data['bmi'] * data['smoker']
    data['age_bmi'] = data['age'] * data['bmi']
    data['age_squared'] = data['age'] ** 2
    data['bmi_squared'] = data['bmi'] ** 2
    
    return pd.DataFrame([data])

def calculate_model_metrics():
    """Calculate model performance metrics"""
    try:
        df = pd.read_csv('insurance.csv')
        
        # Prepare features
        df['smoker'] = (df['smoker'] == 'yes').astype(int)
        df['sex'] = (df['sex'] == 'male').astype(int)
        
        # Create interaction terms
        df['age_smoker'] = df['age'] * df['smoker']
        df['bmi_smoker'] = df['bmi'] * df['smoker']
        df['age_bmi'] = df['age'] * df['bmi']
        df['age_squared'] = df['age'] ** 2
        df['bmi_squared'] = df['bmi'] ** 2
        
        # One-hot encode region
        region_dummies = pd.get_dummies(df['region'], prefix='region')
        df = pd.concat([df, region_dummies], axis=1)
        df.drop('region', axis=1, inplace=True)
        
        X = df.drop('charges', axis=1)
        y = df['charges']
        
        # Scale features and make predictions
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        return {
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

@app.route('/')
def home():
    return render_template('base.html', content='home')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    error = None
    
    if request.method == 'POST':
        try:
            # Prepare input data
            input_df = prepare_input_data(request.form)
            
            # Scale the features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_result = f"${prediction:,.2f}"
            
        except Exception as e:
            error = f"Error making prediction: {str(e)}"
    
    return render_template('base.html', 
                         content='predict',
                         prediction=prediction_result,
                         error=error)

@app.route('/model-info')
def model_info():
    metrics = calculate_model_metrics()
    
    if metrics is None:
        return render_template('base.html', 
                             content='model_info',
                             error="Error calculating model metrics")
    
    return render_template('base.html', 
                         content='model_info',
                         metrics=metrics)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('base.html', 
                         content='home',
                         error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('base.html',
                         content='home',
                         error="Internal server error"), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 