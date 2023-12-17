# Segment-Anything-Model-Applied-to-Alovas-platform

* Automatically generated prompt which is used to guide the ["Segment Anything Model"](https://github.com/facebookresearch/segment-anything) to segment the target area.

* Structure of the "Segment Anything Model" applied to [Alovas platform](https://www.alovas.com/).
![](./readme_img/structure.jpg)
## Dependencies
* Pytorch-21.08-py3:latest
* Download the meta segment_anything weight:
    ```shell
        mkdir weights
        cd weights
        wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ```
* Install others related library and dependencies:
   ```shell
       cd segment-anything; pip install -e .
       pip install opencv-python pycocotools matplotlib onnxruntime onnx
       sudo apt-get update
       sudo apt-get install ffmpeg libsm6 libxext6  -y
       sudo apt-get install -y libvips
   ```
## Codes
 
