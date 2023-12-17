from segment_anything import SamPredictor, sam_model_registry
from utils import load_image, write_masks_to_folder, generate_json, get_points
from contour_prompt import getprompt
import torch
import json
import os
import numpy as np

class SegmentAnything_main():
    def __init__(self):
        super().__init__()
        # initialize parameter
        
    def init_parameters(self, image_path, preprocessed_result_path, result_file_name_list):
        #init image path(string), preprocessed result path(string), result file name list(string array)
        self.image_path=image_path
        #self.preprocess_result_path=preprocess_result_path
        self.result_file_name_list=result_file_name_list
        self.input_boxes=torch.tensor(getprompt(image_path),device=predictor.device)
        self.weights_path='./weights/sam_vit_h_4b8939.pth'
        self.model_type='vit_h'
        #self.point=getprompt(image_path)
        self.level=3
        self.convert_to_rle=True
        # return algorithm progress (0-100)

    def execute(self):
        # algorithm executing here
        print("Loading model...")
        sam = sam_model_registry[self.model_type](checkpoint=self.weights_path)
        _ = sam.to(device='cuda')
        output_mode = "coco_rle" if self.convert_to_rle else "binary_mask"
        mask_predictor = SamPredictor(sam)

        if not os.path.isdir(self.image_path):
            targets = [self.image_path]
        else:
            targets = [
                f for f in os.listdir(self.image_path) if not os.path.isdir(os.path.join(self.image_path, f))
            ]
            targets = [os.path.join(self.image_path, f) for f in targets]

        os.makedirs(self.result_file_name_list, exist_ok=True)

        for t in targets:
            print(f"Processing '{t}'...")
            image = load_image(t, level=self.level)
            print(getprompt(image))
            
            #input_point, input_label = get_points(self.point, self.level)
            input_boxes = torch.tensor(getprompt(image_path), device=predictor.device)
            mask_predictor.set_image(image)
            masks, scores, logits = mask_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )
            base = os.path.basename(t)
            base = os.path.splitext(base)[0]
            save_base = os.path.join(self.result_file_name_list, base)
            if output_mode == "binary_mask":
                os.makedirs(save_base, exist_ok=False)
                write_masks_to_folder(masks, save_base)
            else:
                polygons = generate_json(masks)
                json_object = json.dumps(polygons, indent=4)
                with open(f"{save_base}.json", "w") as f:
                    f.write(json_object)

        print("Done!")

# For testing
AlgorithmName_runner = SegmentAnything_main()
image_path = '../images/1026261.svs'
preprocessed_result_path = './out/result.json'
result_file_name_list = './out/output_file1.json'
AlgorithmName_runner.init_parameters(image_path, preprocessed_result_path, result_file_name_list)
AlgorithmName_runner.execute()
