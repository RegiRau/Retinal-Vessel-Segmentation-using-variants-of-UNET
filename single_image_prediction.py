import os
import cv2
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import PIL
np.random.seed(0)

# CLAHE
def clahe_equalized(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized

#loading model architectures
from model import attentionunet
from tensorflow.keras.optimizers import Adam
from evaluation_metrics import IoU_coef,IoU_loss

SIZE_X = 1632
SIZE_Y = 1216
IMG_HEIGHT = SIZE_Y
IMG_WIDTH = SIZE_X

IMG_CHANNELS = 1
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = attentionunet(input_shape)
model.compile(optimizer = Adam(learning_rate = 1e-3), loss= IoU_loss, metrics= ['accuracy', IoU_coef])
model.load_weights('Veins_Trained_models/Veins_Attention_Unet_12images_10epochs_01.hdf5') #loading weights


path1 = "04_Vein_Dataset/images"
path2 = '04_Vein_Dataset/labels'

testimages = sorted(os.listdir(path1))
testmasks =  sorted(os.listdir(path2))

test = cv2.imread(path1 + '/' + testimages[14], 0)
label = cv2.imread(path2 + '/' + testmasks[14], 0)

test = clahe_equalized(test) #applying CLAHE
test_res = cv2.resize(test, dsize=(SIZE_X, SIZE_Y), interpolation=cv2.INTER_CUBIC)
test_res = np.array(test, dtype="float32")

prediction = (model.predict(test_res)[0,:,:,0] > 0.5).astype(np.uint8) #predict on single patch
prediction = np.array(prediction)



cv2.imwrite('prediction.png', prediction)
cv2.imwrite('label.png', label)
