from flask import Flask, request, jsonify, send_file
import os
import cv2
import matplotlib.pyplot as plt
from Stage1_ToothSegm import Stage1
from Stage2_Mask2Mask import Stage2_Mask2Mask
from Stage3_Mask2Teeth import Stage3_Mask2Teeth
from Restore.Restore import Restore
import yaml
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

# Define the directory to store files
output_directory = "processed_files"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser may submit an empty part without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Generate a unique filename using epoch time
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{filename.split('.')[0]}-{int(time.time())}.{file_extension}"

        # Save the file to the output directory
        temp_path = os.path.join(output_directory, unique_filename)
        file.save(temp_path)

        # Process the image
        result_path = process_image_logic(temp_path)

        # Return the result file
        return send_file(result_path, as_attachment=True)

def process_image_logic(img_path):
    with open("./Config.yaml", 'r') as f:
        GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['C2C2T_v2_facecolor_lightcolor']

    ### main program
    img_name = os.path.basename(img_path).split('.')[0]
    stage1_data = Stage1(img_path, mode=GeneratorConfig['mode'], state=GeneratorConfig['stage1'], if_visual=False)
    stage2_data = Stage2_Mask2Mask(stage1_data, mode=GeneratorConfig['mode'], state=GeneratorConfig['stage2'], if_visual=False)
    stage2_data.update(stage1_data)
    stage3_data = Stage3_Mask2Teeth(stage2_data, mode=GeneratorConfig['mode'], state=GeneratorConfig['stage3'], if_visual=False)
    stage3_data.update(stage2_data)
    pred = Restore(stage3_data['crop_mouth_align'], stage3_data)   # restore to original size
    pred_face = pred['pred_ori_face']

    # Save the visual results    
    result_directory = os.path.join(output_directory, 'prediction')
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    result_filename = f"{img_name}-{int(time.time())}.png"
    result_path = os.path.join(result_directory, result_filename)
    cv2.imwrite(result_path, pred_face)

    return result_path

if __name__ == '__main__':
    app.run(debug=True)
