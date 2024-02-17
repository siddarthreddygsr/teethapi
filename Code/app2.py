from flask import Flask, request, jsonify, send_file
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image as im
from Stage1_ToothSegm import Stage1
from Stage2_Mask2Mask import Stage2_Mask2Mask
from Stage3_Mask2Teeth import Stage3_Mask2Teeth
from Restore.Restore import Restore
import yaml
from werkzeug.utils import secure_filename
import time
import numpy as np
from np2b64 import convert_to_base64
from controlnet_gen import cn_image_gen
from maskgen import mask_generator
from fooocus import RunpodAPI
import base64
from io import BytesIO

app = Flask(__name__)

api_key = "PT9H5DJMNWJ1PQKJW98PTT99COHABVR5MLEG4WY2"
runpod_api = RunpodAPI(api_key)
output_directory = "processed_files"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

@app.route('/2Dteethalignment', methods=['POST'])
def twoDteethalign():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{filename.split('.')[0]}-{int(time.time())}.{file_extension}"
        temp_path = os.path.join(output_directory, unique_filename)
        file.save(temp_path)

        result_path = twoDteethalign_logic(temp_path)

        return send_file(result_path, as_attachment=True)

def twoDteethalign_logic(img_path):
    with open("./Config.yaml", 'r') as f:
        GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['C2C2T_v2_facecolor_lightcolor']

    img_name = os.path.basename(img_path).split('.')[0]
    stage1_data = Stage1(img_path, mode=GeneratorConfig['mode'], state=GeneratorConfig['stage1'], if_visual=False)
    stage2_data = Stage2_Mask2Mask(stage1_data, mode=GeneratorConfig['mode'], state=GeneratorConfig['stage2'], if_visual=False)
    stage2_data.update(stage1_data)
    stage3_data = Stage3_Mask2Teeth(stage2_data, mode=GeneratorConfig['mode'], state=GeneratorConfig['stage3'], if_visual=False)
    stage3_data.update(stage2_data)
    pred = Restore(stage3_data['crop_mouth_align'], stage3_data)   # restore to original size
    pred_face = pred['pred_ori_face']
   
    result_directory = os.path.join(output_directory, 'prediction')
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    result_filename = f"{img_name}-{int(time.time())}.png"
    result_path = os.path.join(result_directory, result_filename)
    cv2.imwrite(result_path, pred_face)

    return result_path

@app.route('/focus_teethalignment_submitter', methods=['POST'])
def focus_teethalign():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{filename.split('.')[0]}-{int(time.time())}.{file_extension}"
        temp_path = os.path.join(output_directory, unique_filename)
        file.save(temp_path)

        result_response = focus_teethalign_logic(temp_path)

        return result_response

def focus_teethalign_logic(image_path):
    image = im.open(image_path)
    # image_np = np.array(image)
    # base64_image = convert_to_base64(image_np)
    # controlnet_img = cn_image_gen(image_path)
    # mask_img = mask_generator(image_path)
    input_data = {
                "input": {
                    "api_name": "inpaint-outpaint",
                    "inpaint_additional_prompt": "veener teeth",
                    "input_image": "http://0x0.st/HnXJ.png",
                    "input_mask": "https://i.imgur.com/tC6DV8I.png",
                    "cn_img1": "https://i.imgur.com/RKi2BAY.png",
                    "cn_weight1": "1",
                    "cn_stop1": "0.5",
                    "cn_type1": "ImagePrompt",
                    "require_base64": "true"
                }
            }
    job_response = runpod_api.run_job(input_data)

    return job_response

@app.route('/check_job_status', methods=['POST'])
def check_status():
    # try:
    job_id = request.get_json()["jobid"]
    response = runpod_api.get_job_status(job_id)
    # return response
    img =  response['output'][0]['base64']
    image_data = base64.b64decode(img)
    image = im.open(BytesIO(image_data))
    image.save('decoded_image.png')
    print("image saved")
    return jsonify({"status": response["status"]})
    # except Exception as e:
        # return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True,port=8000,host= '0.0.0.0')
