
#!/usr/bin/env python3
"""
LabVio ML API - HuggingFace Spaces Deployment
Flask API for unified health risk predictions
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import pytz

app = Flask(__name__)
CORS(app)

# Global model storage
models = {
    'scaler': None,
    'gradient_boosting': None,
    'logistic_regression': None,
    'features': None
}

# Risk level mapping
RISK_LEVELS = {
    0: 'low',
    1: 'moderate',
    2: 'high'
}

# Lab type mapping
LAB_TYPE_MAP = {
    'cbc': 0,
    'urinalysis': 1,
    'urine': 1,
    'lipid': 2,
    'lipid profile': 2
}

def load_models():
    """Load unified trained models"""
    try:
        scaler_path = 'saved_models/scaler.pkl'
        gb_path = 'saved_models/gradient_boosting.pkl'
        lr_path = 'saved_models/logistic_regression.pkl'
        features_path = 'saved_models/features.pkl'
        
        if not all(os.path.exists(p) for p in [scaler_path, gb_path, lr_path, features_path]):
            raise FileNotFoundError("Model files not found.")
        
        models['scaler'] = joblib.load(scaler_path)
        models['gradient_boosting'] = joblib.load(gb_path)
        models['logistic_regression'] = joblib.load(lr_path)
        models['features'] = joblib.load(features_path)
        
        print("✅ Models loaded successfully!")
        print(f"   Features: {len(models['features'])} features")
        
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def parse_value(value, default=0.0):
    """Parse value from string or number"""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            if value.lower() in ['positive', 'trace']:
                return 1.0
            if value.lower() in ['negative']:
                return 0.0
            return default
    return default

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = all(v is not None for v in models.values())
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'service': 'LabVio ML API',
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict health risk from lab values"""
    try:
        ph_tz = pytz.timezone('Asia/Manila')
        ph_time = datetime.now(ph_tz).strftime('%Y-%m-%d %I:%M:%S %p')
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        print(f"\n🔬 Prediction Request | {ph_time}")
        print(f"   Data: {list(data.keys())}")
        
        # Get lab type from request
        lab_type = data.get('lab_type', 'cbc').lower().strip()
        
        # Map lab type to ID
        for key, value in LAB_TYPE_MAP.items():
            if key in lab_type:
                lab_type_id = value
                break
        else:
            lab_type_id = 0  # Default to CBC
        
        # Check if models are loaded
        if not all(v is not None for v in models.values()):
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Default values
        cbc_defaults = {'wbc': 7.5, 'rbc': 4.7, 'hemoglobin': 14.0, 'platelets': 250}
        lipid_defaults = {'cholesterol': 180, 'hdl': 55, 'ldl': 100, 'triglycerides': 140, 'vldl': 28}
        urine_defaults = {'ph': 6.5, 'specific_gravity': 1.015, 'protein': 0.0, 'ketones': 0.0, 'blood': 0.0, 'nitrites': 0.0, 'leukocyte_esterase': 0.0}
        common_defaults = {'glucose': 95, 'a1c': 5.4}
        
        # Build feature dict
        features_dict = {'lab_type': lab_type_id}
        
        for key in cbc_defaults.keys():
            features_dict[key] = parse_value(data.get(key), cbc_defaults[key])
        
        for key in lipid_defaults.keys():
            features_dict[key] = parse_value(data.get(key), lipid_defaults[key])
        
        features_dict['glucose'] = parse_value(data.get('glucose'), common_defaults['glucose'])
        features_dict['a1c'] = parse_value(data.get('a1c'), common_defaults['a1c'])
        
        for key in urine_defaults.keys():
            features_dict[key] = parse_value(data.get(key), urine_defaults[key])
        
        # Create DataFrame
        X = pd.DataFrame([features_dict], columns=models['features'])
        
        # Scale features
        X_scaled = models['scaler'].transform(X)
        
        # Predict
        risk_class = models['gradient_boosting'].predict(X_scaled)[0]
        risk_probabilities = models['gradient_boosting'].predict_proba(X_scaled)[0]
        
        risk_level = RISK_LEVELS[risk_class]
        confidence = int(risk_probabilities[risk_class] * 100)
        
        risk_score = int(
            risk_probabilities[0] * 15 +
            risk_probabilities[1] * 50 +
            risk_probabilities[2] * 85
        )
        
        result = {
            'riskLevel': risk_level,
            'riskScore': risk_score,
            'confidence': confidence,
            'model': 'gradient_boosting_unified',
            'probabilities': {
                'low': float(risk_probabilities[0]),
                'moderate': float(risk_probabilities[1]),
                'high': float(risk_probabilities[2])
            }
        }
        
        print(f"   ✅ Prediction: {risk_level} (Score: {risk_score}, Confidence: {confidence}%)")
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error: {e}")
        print(error_details)
        return jsonify({'error': str(e), 'details': error_details}), 500

if __name__ == '__main__':
    print("\n🚀 Starting LabVio ML API (HuggingFace Spaces)...\n")
    
    if load_models():
        print("🌐 Starting server on 0.0.0.0:7860")
        app.run(host='0.0.0.0', port=7860, debug=False)
    else:
        print("❌ Failed to load models.")
        exit(1)
