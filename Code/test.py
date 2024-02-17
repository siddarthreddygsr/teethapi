import numpy as np
from PIL import Image as im
from np2b64 import convert_to_base64
from controlnet_gen import cn_image_gen
from maskgen import mask_generator
from fooocus import RunpodAPI

image_path = "../Data/case1.jpg"
image = im.open(image_path)
image_np = np.array(image)
base64_image = convert_to_base64(image_np)
controlnet_img = cn_image_gen(image_path)
mask_img = mask_generator(image_path)

api_key = "PT9H5DJMNWJ1PQKJW98PTT99COHABVR5MLEG4WY2"
runpod_api = RunpodAPI(api_key)

input_data = {
    "api_name": "v2/generation/image-prompt",
    "input_image": base64_image,
    "cn_image": controlnet_img,
    "input_mask": mask_img,
    "require_base64": True,
}

job_response = runpod_api.run_job(input_data)
print(job_response)

if "job_id" in job_response:
    job_id = job_response["job_id"]
    status_response = runpod_api.get_job_status(job_id)
    print(status_response)