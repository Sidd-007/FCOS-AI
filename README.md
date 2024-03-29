# FCOS: Fully Convolutional One-Stage Object Detection

* Fully Convolutional One-Stage Object Detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation. 
* Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3, and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor box free, as well as proposal free. 
* By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxes such as calculating overlapping during training. More importantly, we also avoid all hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. 
* With the only post-processing non-maximum suppression (NMS), FCOS with ResNeXt-64x4d-101 achieves 44.7% in AP with single-model and single-scale testing, surpassing previous one-stage detectors with the advantage of being much simpler.


## Input
![image_1](https://user-images.githubusercontent.com/84730469/216777768-5a708ae0-a5f6-409c-aa72-207ab93167de.jpg)



## Output
![image_1_05](https://user-images.githubusercontent.com/84730469/216777847-69f6a05a-5e4f-4e93-81b0-5835d7124af1.jpg)
