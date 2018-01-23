'''
Created on 2018/01/21

@author: jiman
'''
import argparse
from trainer import model
from keras.models import load_model
import cv2
import subprocess
import numpy as np

def prepare_image(img_file):
    print(img_file)
    img = cv2.imread(img_file, 1)
    img = cv2.resize(img, (60, 60))
    img = img / 255.0
    return img

def run(model_file, img_file):
    cmd = 'gsutil cp %s /tmp' % model_file[0]
    subprocess.check_call(cmd.split())
    filename = model_file[0].split('/')[-1]
    print(filename)
    ordinal_model = load_model('/tmp/%s' % filename)
    #ordinal_model = model.compile_model(
    #                ordinal_model, learning_rate=0.001)
    img = prepare_image(img_file)
    predicted = ordinal_model.predict(np.array([img]))
    print(np.sum(predicted[0] > 0.5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file',
                        required=True,
                        type=str,
                        help='Trained model file local or GCS', nargs='+')
    parser.add_argument('--img-file',
                        required=True,
                        type=str,
                        help='Image for predict')
    parse_args, unknown = parser.parse_known_args()

    run(**parse_args.__dict__)
    