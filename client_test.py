import requests
import json

url = "https://credit-card-fraud-detection-gv3r.onrender.com/predict"

payload = {
  "Time": 10.0,
  "features": {f"V{i}": 0.0 for i in range(1, 29)},
  "Amount": 99.99
}

print("Sending transaction to API...")
response = requests.post(url, json=payload)

print(f"Status Code: {response.status_code}")
print(f"API Response: {json.dumps(response.json(), indent=2)}")