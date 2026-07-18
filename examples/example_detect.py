"""Example script for calling the YOLOv8 detection API.""" 
 
import requests 
 
def detect_objects(image_path): 
    with open(image_path, "rb") as f: 
        response = requests.post("http://localhost:8000/detect", files={"file": f}) 
    return response.json() 
 
if __name__ == "__main__": 
    result = detect_objects("sample.jpg") 
    print(result) 
