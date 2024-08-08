import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request
#from PIL import Image

interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = ['dragon', 'dino']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x /= 255
    return x

def predict(url):
    img = download_image(url)
    target_size = (150, 150)
    img_prep = prepare_image(img, target_size)
    x = np.array(img_prep, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return dict(zip(classes, preds[0]))

def lambda_handler(event, context):
    url = event['url']
    preds = predict(url)
    return {
        'statusCode': 200,
        'body': str(preds)
    }