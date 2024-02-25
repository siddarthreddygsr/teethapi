import time
import requests
import json
import base64


start = time.time()
host = "https://3y7wkrmlouh1fu-8888.proxy.runpod.net"

def inpaint_outpaint(params: dict) -> dict:
    """
    example for inpaint outpaint v2
    """
    response = requests.post(url=f"{host}/v2/generation/image-prompt",
                        data=json.dumps(params),
                        headers={"Content-Type": "application/json"})
    return response.json()


source = open("case1 (1).png", "rb").read()
mask = open("mask (20).png", "rb").read()
cn = open("image - 2024-02-14T012901.261.png", "rb").read()
result = inpaint_outpaint(params={
                            "prompt": "perfect teeth, shiny teeth,veneer teeth, super white teeth, perfect shape of teeth",
                            "negative_prompt" : "imperfect teeth",
                            "style_selections":["Fooocus V2,Fooocus Enhance,Fooocus Sharp, Fooocus Negative"],
                            "input_image": base64.b64encode(source).decode('utf-8'),
                            "input_mask": base64.b64encode(mask).decode('utf-8'),
                            "image_prompts": [
                              {
                            "cn_img": base64.b64encode(cn).decode('utf-8'),
                            "cn_stop": 0.6,
                            "cn_weight": 1,
                             "cn_type": "PyraCanny"
                              }],
                            "require_base64":True,
                            "async_process": False})
# print(json.dumps(result, indent=4, ensure_ascii=False))
base = result[0]['base64']
# Decode the base64 string
image_data = base64.b64decode(base)
end = time.time()
# Write the decoded data to a file
with open("image2.png", "wb") as f:
    f.write(image_data)
final = end - start
print(final)    