import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

X = {
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

data = {'data': X}

response = requests.post(url, json=data).json()

print(response)
