from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load files
model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))
scaler = pickle.load(open('feature_scaler.pkl', 'rb'))

# Crop name mapping (must match training encoding)
crop_dict_reverse = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'papaya',
    6: 'orange', 7: 'apple', 8: 'muskmelon', 9: 'watermelon',
    10: 'grapes', 11: 'mango', 12: 'banana', 13: 'pomegranate',
    14: 'lentil', 15: 'blackgram', 16: 'mungbean', 17: 'mothbeans',
    18: 'pigeonpeas', 19: 'kidneybeans', 20: 'chickpea', 21: 'coffee',
    22: 'coconut'
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Returns multiple suitable crops ranked by probability.
    """
    try:
        # Get input features
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Validate inputs
        if not (0 <= ph <= 14):
            return jsonify({'error': 'pH must be between 0 and 14'}), 400

        # Create feature array (order must match training)
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale features using the trained scaler (NOT fit_transform)
        features_scaled = scaler.transform(features)

        # Get probability predictions for all crops
        probabilities = model.predict_proba(features_scaled)[0]
        class_indices = model.classes_

        # Create list of (crop_name, probability) sorted by likelihood
        crop_predictions = []
        for i, class_idx in enumerate(class_indices):
            crop_predictions.append({
                'crop': crop_dict_reverse.get(class_idx, 'Unknown'),
                'probability': float(probabilities[i]),
                'percentage': round(probabilities[i] * 100, 2)
            })

        # Sort by probability descending
        crop_predictions.sort(key=lambda x: x['probability'], reverse=True)

        # Filter crops above threshold (adjust as needed)
        threshold = 0.05  # 5% - show any crop with at least 5% probability
        suitable_crops = [c for c in crop_predictions if c['probability'] >= threshold]

        # If no crops meet threshold, show top 5 anyway
        if not suitable_crops:
            suitable_crops = crop_predictions[:5]

        # Return results
        return render_template('index.html',
                             top_crop=suitable_crops[0],
                             all_crops=suitable_crops,
                             threshold_used=threshold)

    except ValueError as e:
        return render_template('index.html', error=f'Invalid input: {str(e)}')
    except Exception as e:
        return render_template('index.html', error=f'Prediction error: {str(e)}')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint that returns JSON (useful for frontend frameworks).
    """
    try:
        data = request.get_json()

        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        if not (0 <= ph <= 14):
            return jsonify({'error': 'pH must be between 0 and 14'}), 400

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)
        probabilities = model.predict_proba(features_scaled)[0]
        class_indices = model.classes_

        crop_predictions = []
        for i, class_idx in enumerate(class_indices):
            crop_predictions.append({
                'crop': crop_dict_reverse.get(class_idx, 'Unknown'),
                'probability': float(probabilities[i]),
                'percentage': round(probabilities[i] * 100, 2)
            })

        crop_predictions.sort(key=lambda x: x['probability'], reverse=True)

        threshold = 0.05
        suitable_crops = [c for c in crop_predictions if c['probability'] >= threshold]

        if not suitable_crops:
            suitable_crops = crop_predictions[:5]

        return jsonify({
            'success': True,
            'top_crop': suitable_crops[0],
            'all_recommendations': suitable_crops
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
