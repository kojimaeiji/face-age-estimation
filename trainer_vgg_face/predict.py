'''
Created on 2018/01/21

@author: jiman
'''
import argparse
from trainer_vgg_face import model
from keras.models import load_model
import cv2
import subprocess
import numpy as np

def prepare_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)
    img -= mean 
    return img

def predict_core(face_model, img_np_mat):
    img = prepare_image(img_np_mat)
    predicted = face_model.predict(np.array([img]))
    pre_num = np.array([predicted[0][i]*i for i in range(101)])        
    return np.sum(pre_num)

def run(model_file, img_file):
    if model_file.startswith('gs://'):
        cmd = 'gsutil cp %s /tmp' % model_file[0]
        subprocess.check_call(cmd.split())
        filename = model_file.split('/')[-1]
        filename = '/tmp/%s' % filename
    else:
        filename = model_file
    face_model = load_model(filename, compile=False)
    model.compile_model(face_model, learning_rate=0.001)

    img = cv2.imread(img_file, 1)
    img = cv2.resize(img, (224, 224))

    #ordinal_model = model.compile_model(
    #                ordinal_model, learning_rate=0.001)
    return predict_core(face_model, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file',
                        required=True,
                        type=str,
                        help='Trained model file local or GCS')
    parser.add_argument('--img-file',
                        required=True,
                        type=str,
                        help='Image for predict')
    parse_args, unknown = parser.parse_known_args()

    result = run(**parse_args.__dict__)
    print(result)
    