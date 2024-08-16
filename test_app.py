import pandas as pd
import requests 

df = pd.read_csv('./training/nearest-earth-objects(1910-2024).csv')
test_data = df.loc[:10]
test_data = test_data.drop(columns='is_hazardous')
#print(test_data)
json_data = test_data.to_json(orient='records')

url = 'http://localhost:9696/predict'
response = requests.post(url, json=json_data)
print(response.json())