 
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
 
## 2026-06-20 
### YOLOv8 Custom Dataset Training Notes 
- Custom dataset needs images and labels in YOLO format 
- Labels stored as txt files with class id and bounding box 
- Roboflow used to annotate and export dataset in YOLO format 
- data.yaml file defines train val test paths and class names 
 
## 2026-06-22 
### YOLOv8 Inference Optimization Notes 
- Confidence threshold filters out low quality detections 
- NMS removes duplicate bounding boxes for same object 
- IOU threshold controls how aggressively NMS removes boxes 
- Tracking with BoT-SORT maintains object ID across video frames 
 
## 2026-06-24 
### YOLOv8 Segmentation Mode Notes 
- YOLOv8 segmentation outputs pixel level masks for each object 
- Mask head added on top of detection head in model architecture 
- Instance segmentation distinguishes between same class objects 
- Segmentation useful for medical imaging and autonomous driving 
 
## 2026-06-27 
### YOLOv8 Edge Deployment Notes 
- YOLOv8 nano and small variants suited for edge device deployment 
- OpenVINO export optimizes model for Intel CPU based edge devices 
- CoreML export used for deploying detection model on iOS devices 
- Quantization to INT8 reduces model size for embedded hardware 
 
## 2026-06-29 
### YOLOv8 Pose Estimation Notes 
- YOLOv8 pose mode detects human keypoints like joints and limbs 
- 17 keypoints predicted per detected person in standard COCO format 
- Useful for fitness tracking and fall detection applications 
- Keypoint confidence scores help filter unreliable pose predictions 
 
## 2026-07-01 
### YOLOv8 Multi Camera Pipeline Notes 
- Multi camera setup processes several video streams in parallel 
- Threading used to handle concurrent camera feeds efficiently 
- Results aggregated across cameras for unified detection dashboard 
- RTSP protocol used to connect to IP camera streams directly 
 
## 2026-07-04 
### YOLOv8 Export Format Comparison Notes 
- TorchScript export keeps compatibility within PyTorch ecosystem 
- TFLite export needed for Android mobile app deployment 
- Comparing inference speed across export formats on same hardware 
- Choosing format depends on target deployment environment 
