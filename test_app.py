import time
import pandas as pd
import requests

df = pd.read_csv('./training/nearest-earth-objects(1910-2024).csv')
for i in range(1, 10):
    test_data = df.loc[1000*i:1000*(i+1)]
    #print(test_data)
    json_data = test_data.to_json(orient='records')
    url = 'http://localhost:9696/predict'
    response = requests.post(url, json=json_data)
    print(response.json())
    time.sleep(10)