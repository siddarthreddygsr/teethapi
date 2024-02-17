import base64
import numpy as np

def convert_to_base64(img_np, out_type='uint8'):
    img_as_base64 = base64.b64encode(img_np.astype(out_type).squeeze()).decode('utf-8')
    return img_as_base64