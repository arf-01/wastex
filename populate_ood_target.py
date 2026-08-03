import requests
import os
import time

backend_url = "http://localhost:8000"
token = "8e9140f20bf1895edbf2860d40be913fc5120bed"
bin_id = "pi_001"
url = f"{backend_url}/api/pi/inference/"
list_url = f"{backend_url}/api/ood/"

headers = {
    "Authorization": f"Token {token}",
}

def get_ood_count():
    try:
        # We fetch the list of OOD images from the API
        response = requests.get(list_url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # DRF pagination typically returns results in 'results' or as a list, and total count in 'count'
            if isinstance(data, dict):
                return data.get("count", len(data.get("results", [])))
            elif isinstance(data, list):
                return len(data)
        else:
            print(f"Failed to fetch OOD list: {response.text}")
    except Exception as e:
        print(f"Error checking OOD count: {e}")
    return 0

print("Checking current OOD image count on Edge server...")
current_count = get_ood_count()
print(f"Current OOD count: {current_count}")

target_count = 3
attempts = 0
max_attempts = 15

while current_count < target_count and attempts < max_attempts:
    attempts += 1
    print(f"\nOOD count ({current_count}) is less than target ({target_count}). Attempting download/upload {attempts}...")
    
    img_response = requests.get(f"https://picsum.photos/400/300?random={time.time_ns()}", timeout=15)
    if img_response.status_code != 200:
        print("Failed to download random image")
        continue
        
    temp_filename = f"temp_ood_fill.jpg"
    with open(temp_filename, "wb") as f:
        f.write(img_response.content)
        
    try:
        with open(temp_filename, "rb") as f:
            files = {"image": f}
            data = {"source": bin_id}
            response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            
        if response.status_code == 200:
            result = response.json()
            print(f"Success: Predicted Class: {result.get('predicted_class')}, Is OOD: {result.get('ood')}, Saved: {result.get('saved_to_db')}")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass
                
    time.sleep(1)
    current_count = get_ood_count()
    print(f"New OOD count: {current_count}")

print(f"\nDone. Final OOD count: {current_count}")
