import numpy as np
import tensorflow as tf
from PIL import Image
from np2b64 import convert_to_base64

def mask_generator(image_path):
    interpreter = tf.lite.Interpreter(model_path="model_files/teeth_segment.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    im = Image.open(image_path)
    res_im = im.resize((416, 416))

    np_res_im = np.array(res_im)
    np_res_im = (np_res_im/255).astype('float32')

    if len(np_res_im.shape) == 3:
        np_res_im = np.transpose(np_res_im, (2, 0, 1))
        np_res_im = np.expand_dims(np_res_im, 0)
    input_data = np_res_im
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    # output_data = interpreter.get_tensor(output_details[0]['index'])

    output_details = interpreter.get_output_details()
    output_tensor_index = output_details[0]['index']
    segmentation_result = interpreter.tensor(output_tensor_index)()

    threshold = 0.5
    binary_result = (segmentation_result > threshold).astype(int)

    binary_mask = binary_result

    binary_mask = binary_mask.squeeze()  # Remove singleton dimensions if any
    black_white_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    black_white_mask[binary_mask > 0] = 255  # Set foreground pixels to white (255)

    # Convert the numpy array to a PIL Image
    data = Image.fromarray(black_white_mask)
    image_np = np.array(data)
    return convert_to_base64(image_np)

mask_generator("../Data/case1.jpg")
