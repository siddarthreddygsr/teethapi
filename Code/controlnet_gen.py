import matplotlib.pyplot as plt
from Stage1_ToothSegm import Stage1
from Stage2_Mask2Mask import Stage2_Mask2Mask
import yaml
from np2b64 import convert_to_base64
import numpy as np
from PIL import Image as im 

def cn_image_gen(img_path):
    with open("./Config.yaml", 'r') as f:
        GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['C2C2T_v2_facecolor_lightcolor']
    stage1_data = Stage1(img_path, mode=GeneratorConfig['mode'], state=GeneratorConfig['stage1'], if_visual=False)
    stage2_data = Stage2_Mask2Mask(stage1_data, mode=GeneratorConfig['mode'], state=GeneratorConfig['stage2'], if_visual=False)
    stage2_data.update(stage1_data)
    controlnet_image = im.fromarray(stage2_data["crop_teeth_align"])
    controlnet_np =  np.array(controlnet_image)
    return convert_to_base64(controlnet_np)

# print(cn_image_gen("../Data/case1.jpg"))