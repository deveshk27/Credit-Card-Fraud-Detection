import requests
import json

url = "http://127.0.0.1:8000/predict"

payload = {
  "Time": 10.0,
  "features": {f"V{i}": 0.0 for i in range(1, 29)},
  "Amount": 99.99
}

print("Sending transaction to API...")
response = requests.post(url, json=payload)

print(f"Status Code: {response.status_code}")
print(f"API Response: {json.dumps(response.json(), indent=2)}")