import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import random
import string
import time
import os
import boto3

def load_aws_credentials(file_path='secrets.json'):
    with open(file_path, 'r') as file:
        credentials = json.load(file)
    return credentials.get('aws_key'), credentials.get('aws_secret')

def convert_to_url(img_np, out_type='uint8'):
    random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    image = Image.fromarray(img_np)
    unique_filename = f"image_{int(time.time())}_{random_chars}.png"
    image_file_path = os.path.join("processed_files", unique_filename)
    image.save(image_file_path)

    # Load AWS credentials from the secrets file
    aws_access_key_id, aws_secret_access_key = load_aws_credentials()

    region_name = 'ap-south-1'
    bucket_name = 'teethe'
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
    s3.upload_file(image_file_path, bucket_name, unique_filename)
    os.remove(image_file_path)
    url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{unique_filename}"
    return url

# image_path = '../Data/case1.jpg'

# # Open the image using PIL
# image = Image.open(image_path)

# # Convert the image to a NumPy array
# img_np = np.array(image)
# print(convert_to_url(img_np))
