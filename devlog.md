 
## 2026-06-15 
### YOLOv8 Object Detection API Notes 
- YOLOv8 supports detection, segmentation and classification tasks 
- FastAPI used to wrap YOLOv8 model as REST API endpoint 
- Input image sent as base64 or multipart form to API 
- Response returns bounding boxes, class labels and confidence 
 
## 2026-06-17 
### YOLOv8 Model Optimization Notes 
- TensorRT export reduces YOLOv8 inference time by 3x on GPU 
- ONNX export allows model to run without PyTorch dependency 
- Half precision FP16 inference speeds up detection on GPU 
- Batch inference processes multiple images in single forward pass 
