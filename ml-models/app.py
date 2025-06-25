from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from flask_cors import CORS
from marshmallow import Schema, fields, ValidationError

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model = joblib.load('model.pkl')  # Load pre-trained model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a schema for input validation
class InputSchema(Schema):
    userId = fields.Str(required=True)
    keystrokeData = fields.List(fields.Dict(keys=fields.Str(), values=fields.Int()), required=True)
    mouseData = fields.List(fields.Dict(keys=fields.Str(), values=fields.Int()), required=True)

def extract_features(data):
    keystrokes = data.get('keystrokeData', [])
    mouse_data = data.get('mouseData', [])

    # Calculate average key press duration
    durations = [k['downTime'] for k in keystrokes]
    avg_key_time = np.mean(np.diff(durations)) if len(durations) > 1 else 0

    # Calculate average mouse speed
    speeds = []
    for i in range(1, len(mouse_data)):
        dx = mouse_data[i]['x'] - mouse_data[i-1]['x']
        dy = mouse_data[i]['y'] - mouse_data[i-1]['y']
        dt = (mouse_data[i]['time'] - mouse_data[i-1]['time']) / 1000.0  # Convert to seconds
        if dt > 0:
            speed = ((dx ** 2 + dy ** 2) ** 0.5) / dt
            speeds.append(speed)

    avg_mouse_speed = np.mean(speeds) if speeds else 0
    return [avg_key_time, avg_mouse_speed]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate input data
        data = request.json
        logger.info("Received data: %s", data)
        validated_data = InputSchema().load(data)

        # Extract features and make prediction
        features = extract_features(validated_data)
        prediction = model.predict([features])[0]
        logger.info("Prediction made: %s", prediction)

        return jsonify({"fraud": bool(prediction)})

    except ValidationError as err:
        logger.error("Validation error: %s", err.messages)
        return jsonify({"error": "Invalid input data", "details": err.messages}), 400
    except Exception as e:
        logger.exception("An error occurred during prediction")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
