import requests
import json
import sys

url_local = f'http://localhost:9696/predict'

data = {
    "country":0.0,
    "store":0.0,
    "product":1.0,
    "date_year":2010.0,
    "date_month":1.0,
    "date_day":1.0,
    "date_day_of_week":4.0,
    "date_year_sin":0.000657569,
    "date_year_cos":0.9999997616,
    "date_month_sin":0.5,
    "date_month_cos":0.8660253882
}

response = json.dumps(requests.post(url_local, json=data).json())

print(response)
