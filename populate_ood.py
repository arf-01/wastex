import requests
import os
import time

backend_url = "http://localhost:8000"
token = "8e9140f20bf1895edbf2860d40be913fc5120bed"
bin_id = "pi_001"
url = f"{backend_url}/api/pi/inference/"

headers = {
    "Authorization": f"Token {token}",
}

print("Starting to populate Edge OOD dashboard with 3 images...")

for i in range(1, 4):
    print(f"\nDownloading image {i}/3 from picsum.photos...")
    img_response = requests.get(f"https://picsum.photos/400/300?random={i+10}", timeout=15)
    if img_response.status_code != 200:
        print(f"Failed to download image {i}")
        continue
        
    temp_filename = f"temp_ood_{i}.jpg"
    with open(temp_filename, "wb") as f:
        f.write(img_response.content)
        
    print(f"Uploading {temp_filename} to {url}...")
    try:
        with open(temp_filename, "rb") as f:
            files = {"image": f}
            data = {"source": bin_id}
            response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            
        print(f"Response status code: {response.status_code}")
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
            
print("\nDone populating dashboard.")
