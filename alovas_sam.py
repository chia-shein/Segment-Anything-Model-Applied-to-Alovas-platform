import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
#import os
#import pyvips
import json
import torchvision.transforms as T
#from typing import Any, Dict, List
from utils import getprompt,load_image,generate_json,contour_to_points,cal_area
from segment_anything import sam_model_registry, SamPredictor

class Alovas_SAM():
    def __init__(self):
        super().__init__()
    def init_parameters(self, image_path, save_base):
        self.image_path=image_path
        self.save_base=save_base
        #self.preprocess_result_path=preprocess_result_path
        #self.result_file_name_list=result_file_name_list
        self.sam_checkpoint='./weights/sam_vit_h_4b8939.pth'
        self.model_type='vit_h'
        self.device = "cuda"
        self.level=3

    def execute(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        image = load_image(self.image_path, self.level)
        predictor.set_image(image)
        input_boxes = torch.tensor(getprompt(self.image_path), device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        #print(masks)
        transform = T.ToPILImage()
        num=0
        for i in masks:
            num+=1
            pic=i.to(torch.float16)
            img = transform(pic)
            img1=img.copy()
            #img.show()

            img=np.array(img1)
            contours, area = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            #print(contour)
            '''
            M = cv2.moments(contour)
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            xs = [v[0][0] - x for v in contour]
            ys = [-(v[0][1] - y) for v in contour]
            plt.plot(xs, ys)
            plt.show()
            '''
            contour = contours[0]
            polygons=generate_json(num,contour_to_points(contour),cal_area(contours))
            json_object = json.dumps(polygons, indent=4)
            if num==1:
                with open(f"{self.save_base}/{self.image_path[9:-4]}.json", "w") as f:
                    f.write(json_object)
            else:
                index=generate_json(num,contour_to_points(contour),cal_area(contours))
                data=json.load(open(f"{self.save_base}/{self.image_path[9:-4]}.json",encoding='utf-8'))
                data['annotation'].append(index)
                #print(data)
                with open(f"{self.save_base}/{self.image_path[9:-4]}.json",'w') as outfile:
                    json.dump(data,outfile)
            print(str(num)+": json create done!")


# For testing
AlgorithmName_runner = Alovas_SAM()
image_path = './images/CMU-2.svs'
save_base='./Annotation/'
AlgorithmName_runner.init_parameters(image_path, save_base)
AlgorithmName_runner.execute()




