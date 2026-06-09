from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)                           

# Load model and encoder
model = joblib.load('career_model.joblib')
le = joblib.load('label_encoder.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload received'}), 400

    expected_keys = [
        'math_score','english_score','biology_score','chemistry_score','physics_score',
        'economics_score','tech_interest','health_interest','business_interest','arts_interest',
        'social_help_interest','analytical_thinking','problem_solving','creativity','leadership',
        'teamwork','communication','decision_making','stem_score','health_score','business_score'
    ]

    try:
        features_list = []
        for k in expected_keys:
            if k not in data:
                raise KeyError(k)
            features_list.append(float(data[k]))

        features = np.array([features_list])
    except KeyError as e:
        return jsonify({'error': 'Missing field', 'field': str(e)}), 400
    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Invalid field type', 'details': str(e)}), 400

    try:
        prediction = model.predict(features)
    except Exception as e:
        return jsonify({'error': 'Model prediction error', 'details': str(e)}), 500

    # Try to inverse-transform prediction using label encoder; fall back to raw prediction
    try:
        career = le.inverse_transform(prediction)[0]
    except Exception:
        try:
            career = prediction[0]
        except Exception:
            career = str(prediction)

    return jsonify({'recommended_career': career})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)