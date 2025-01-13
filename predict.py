import pickle
import json
from flask import Flask, request, jsonify
import numpy as np

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('stickers')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data_array = np.array([list(data.values())])
    X = data_array.reshape(1, -1)
    num_sold = model.predict(X)
    
    result = {'sticker sales': float(num_sold)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
