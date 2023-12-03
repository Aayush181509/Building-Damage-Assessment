from sys import argv
import json
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.preprocessing.image as image
import pandas as pd
import fnmatch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)


def classify_patches(model_path, patch_path, **kwargs):
    if len(tf.config.experimental.list_physical_devices('GPU')) >= 1:
        if kwargs.get("verbose", 0) >= 1:
            print("Working with GPU")
        with tf.device("/GPU:0"):
            model = load_model(model_path)
            if kwargs.get("verbose", 0) >= 1:
                print('Setting model on device - ok')

    else:
        if kwargs.get("verbose", 0) >= 1:
            print("Working with CPU")
        with tf.device("/device:CPU:0"):
            model = load_model(model_path)
            if kwargs.get("verbose", 0) >= 1:
                print('Setting model on device - ok')

    df_pred = pd.DataFrame(data = {'id_row': [], 
                                   'CONDITION': []})

    #Load images and predict on each
    for img_name in tqdm(fnmatch.filter(os.listdir(patch_path), '*.jpg')):
        image_path  = os.path.join(patch_path, img_name)
        input_image = image.load_img(image_path, target_size=(512, 512))
        input_image = image.img_to_array(input_image)
        input_image = np.array([input_image])/255
        preds       = model.predict(input_image)
        preds_value = np.argmax(preds,axis=1)[0]

        if kwargs.get("damage_precision", 0) >= 1:
            if preds_value==2: 
                preds_value = 200
            preds_value = preds[0][1] - preds[0][0]
        
        else:
            if preds_value==2:
                preds_value = np.argmax(preds[0][0:2])
        
        df_pred_row = pd.DataFrame({'id_row': [str(img_name.replace('.jpg',''))],
                                    'CONDITION': [preds_value]})

        df_pred = pd.concat([df_pred, df_pred_row], ignore_index=True)
    
    df_pred.id_row = df_pred.id_row.astype(int)
    df_pred.to_csv(os.path.join(patch_path, "pred_df.csv"))

def main():
    with open(argv[1]) as config_json:
        config = json.load(config_json)
    
    classify_patches(model_path = config.get("model_path"),
                     patch_path = config.get("patch_path"),
                     damage_precision = config.get("damage_precision"),
                     verbose = 1)

if __name__ == '__main__':
    main()
