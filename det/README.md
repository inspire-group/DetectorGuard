## Base Detector Output
- This is directory holds the prediction outputs of Base Detector.

- You can download a sample of prediction files (the ones used in the paper) from [here](https://drive.google.com/drive/folders/1aezBxFOuGa-EmLMdI5TCXeAFjk2n12Y-?usp=sharing).

- You can also generate your own prediction files following the same format of the sample prediction files. Check out popular object detector implementations such as [detectron2](https://github.com/facebookresearch/detectron2) and [ultralytics-yolo](https://github.com/ultralytics/yolov5)
  - We use [detectron2](https://github.com/facebookresearch/detectron2) to build Faster R-CNN and [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4) for YOLO in our paper.

- The predictions are organized by classes (e.g., `det_cls_yolo`) and images (e.g., `det_img_yolo`).
  - the bounding box is in PASCAL VOC format (i.e., `x_min y_min x_max y_max`)
  - `det_cls_{detector}` 
    - the file name has the format of `{class_name}.txt` 
    - each line has the format of `img_id prediction_confidence x_min y_min x_max y_max`
  - `det_img_{detector}` 
    - the file name has the format of `{img_id}.txt` 
    - each line has the format of `class_label prediction_confidence x_min y_min x_max y_max`


