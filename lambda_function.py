import pickle
import json
import numpy as np

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

def predict(data):
    data_array = np.array([list(data.values())])
    X = data_array.reshape(1, -1)
    num_sold = model.predict(X)
    
    result = {'sticker sales': float(num_sold)}
    return result


def lambda_handler(event, context):
    X = event['data']
    result = predict(X)
    return result
