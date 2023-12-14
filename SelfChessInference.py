import cv2
import typing
import numpy as np
import matplotlib.pyplot as plt

import os
from urllib.request import urlopen
import tarfile
from io import BytesIO
from zipfile import ZipFile

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = ['#', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'B', 'K', 'N', 'O', 'P', 'Q', 'R', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x']

    def predict(self, image: np.ndarray):
        # plt.title("Original Image")
        # plt.imshow(image)
        # plt.show()
        try:
            # print(self.input_shape)
            # print(image.shape)
            image = cv2.resize(image, self.input_shape[:2][::-1])
        except:
            print("ERROR")
            return
        # height, width, channels = image.shape
        # print(f"resized image {height, width, channels}")
        # plt.title("Resized Image")
        # plt.imshow(image)
        # plt.show()

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        # print(f"shape of image_pred {image_pred.shape}")

        preds = self.model.run(None, {self.input_name: image_pred})[0]
        
        text = ctc_decoder(preds, self.vocab)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    model = ImageToWordModel(model_path="C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/chessChild/model.onnx")

    df = pd.read_csv("C:/Users/ericz/OneDrive/Desktop/IAM Model/validationSet.csv").values.tolist()
    #df[0] is the file path and df[1] is the label
    accum_cer = []
    accum_wer = [] 
    
    # image_path = "C:/Users/ericz/Handwriting-Transformers/HCS Data/HCS Dataset December 2021/extracted move boxes/004_0_12_black.png"
    # image_path = "C:/Users/ericz/OneDrive/Desktop/IAM Model/chessTest/Bxa3#Line.png"
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path)
        prediction_text = model.predict(image)
        label = label.strip()
        
        try:
            cer = get_cer(prediction_text, label)
            print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

            wer = get_wer(prediction_text, label)
            print(f"WER: {wer}")

            accum_cer.append(cer)
            accum_wer.append(wer)

        except: 
            print()
        
    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
