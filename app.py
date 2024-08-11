from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the models
models = {
    'linear_regression': joblib.load('Linear Regression.joblib'),
    'ridge_regression': joblib.load('Lasso Regression.joblib'),
    'lasso_regression': joblib.load('Lasso Regression.joblib'),
    'decision_tree': joblib.load('Decision Tree.joblib'),
    # 'random_forest': joblib.load('random_forest_regressor.joblib')
}

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    model = models[model_name]
    data = request.json
    X_new = np.array(data['features']).reshape(1, -1)
    
    prediction = model.predict(X_new)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)


# curl -X POST -H "Content-Type: application/json" -d '{"features": [138374, 3521, 70003, 28175, 8710, 9343, 793, 759, 465, 261, 197, 112, 71, 70, 36, 11, 118, 906, 50, 26, 28, 8, 40, 8]}' http://127.0.0.1:5000/predict/random_forest

# {
#   "error": "Model not found"
# }
# (base) dhyey@Dhyeys-MacBook-Air MLOPS OEE %  curl -X POST -H "Content-Type: application/json" -d '{"features": [138374, 3(base) dhyey@Dhyeys-MacBook-Air MLOPS OEE %  curl -X POST -H "Content-Type: application/json" -d '{"features": [138374, 3521, 70003, 28175, 8710, 9343, 793, 759, 465, 261, 197, 112, 71, 70, 36, 11, 118, 906, 50, 26, 28, 8, 40, 8]}' http://127.0.0.1:5000/predict/ridge_regression                 
# {
#   "prediction": [
#     98.805375
#   ]
# }
# (base) dhyey@Dhyeys-MacBook-Air MLOPS OEE %  curl -X POST -H "Content-Type: application/json" -d '{"features": [138374, 3521, 70003, 28175, 8710, 9343, 793, 759, 465, 261, 197, 112, 71, 70, 36, 11, 118, 906, 50, 26, 28, 8, 40, 8]}' http://127.0.0.1:5000/predict/lasso_regression
# {
#   "prediction": [
#     98.805375
#   ]
# }
# (base) dhyey@Dhyeys-MacBook-Air MLOPS OEE %  curl -X POST -H "Content-Type: application/json" -d '{"features": [138374, 3521, 70003, 28175, 8710, 9343, 793, 759, 465, 261, 197, 112, 71, 70, 36, 11, 118, 906, 50, 26, 28, 8, 40, 8]}' http://127.0.0.1:5000/predict/decision_tree  
# {
#   "prediction": [
#     107.0
#   ]
# }
