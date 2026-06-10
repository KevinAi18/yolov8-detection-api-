"""
Batch Detection Utility
Allows processing multiple images in a single API call
"""
import requests
import os
from pathlib import Path

def batch_detect(image_paths: list, api_host: str = "http://0.0.0.0:8001/", confidence: float = 0.5):
    """
    Process multiple images and return combined detection results
    
    Args:
        image_paths: List of image file paths
        api_host: API server URL
        confidence: Minimum confidence threshold (0-1)
    
    Returns:
        dict: Detection results for all images
    """
    results = {}
    for image_path in image_paths:
        if not os.path.exists(image_path):
            results[image_path] = {"error": "File not found"}
            continue
        
        files = {"file": open(image_path, "rb")}
        params = {"confidence": confidence}
        response = requests.post(
            f"{api_host}img_object_detection_to_json",
            files=files,
            params=params
        )
        
        if response.status_code == 200:
            results[image_path] = response.json()
        else:
            results[image_path] = {"error": f"Detection failed: {response.status_code}"}
    
    return results

if __name__ == "__main__":
    # Example usage
    images = ["test1.jpg", "test2.jpg"]
    detections = batch_detect(images, confidence=0.6)
    for img, result in detections.items():
        print(f"{img}: {result}")
