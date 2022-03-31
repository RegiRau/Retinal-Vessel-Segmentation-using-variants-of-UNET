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

SIZE_X = 1632
SIZE_Y = 1216

#loading model architectures
from model import unetmodel, residualunet, attentionunet, residual_attentionunet
from tensorflow.keras.optimizers import Adam
from evaluation_metrics import IoU_coef,IoU_loss

IMG_HEIGHT = SIZE_Y
IMG_WIDTH = SIZE_X
IMG_CHANNELS = 1

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = attentionunet(input_shape) #/residualunet(input_shape)/unetmodel(input_shape)/attention_residualunet(input_shape)
model.compile(optimizer = Adam(learning_rate = 1e-3), loss= IoU_loss, metrics= ['accuracy', IoU_coef])
model.load_weights('Retina_Trained models/retina_attentionUnet_150epochs.hdf5') #loading weights
#model.load_weights('/content/drive/MyDrive/training/retina_Unet_150epochs.hdf5') #loading weights


# path1 = '/content/drive/MyDrive/training/images'    #test dataset images directory path
# path2 = '/content/drive/MyDrive/training/masks'     #test dataset mask directory path

path1 = "../04_Vein_Dataset/images"
path2 = "../04_Vein_Dataset/labels"

from sklearn.metrics import jaccard_score,confusion_matrix

testimg = []
ground_truth = []
prediction = []
global_IoU = []
global_accuracy = []

testimages = sorted(os.listdir(path1))
testmasks =  sorted(os.listdir(path2))

for idx, image_name in enumerate(testimages):
    test_img = cv2.imread(path1 + '/' + image_name, 0)
    test = clahe_equalized(test_img) #applying CLAHE
    res_test = cv2.resize(test, dsize=(SIZE_X, SIZE_Y), interpolation=cv2.INTER_CUBIC)
    res_test = np.array(res_test, dtype="float32")
    testimg.append(res_test)

    test_img = (res_test.astype('float32')) / 255.
    test_img_norm = np.expand_dims(np.array(test_img), axis=-1)
    test_img_input = np.expand_dims(test_img_norm, 0)
    test_img_prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(
        np.uint8)  # predict on single patch

    prediction.append(test_img_prediction)


    groundtruth = cv2.imread(path2 + '/' + testmasks[idx], 0)
    groundtruth = cv2.resize(groundtruth, dsize=(SIZE_X, SIZE_Y), interpolation=cv2.INTER_CUBIC)
    groundtruth[groundtruth < 200] = 0
    groundtruth[groundtruth >= 200] = 255
    groundtruth = np.array(groundtruth, dtype="uint8")
    ground_truth.append(groundtruth)

    y_true = groundtruth # 0 - 255
    y_pred = test_img_prediction  # 1 and 0
    labels = [0, 1]
    IoU = []  #Intersection over Union -> Schwellenwert, um zu ermitteln, ob ein vorhergesagtes Ergebnis ein
            #True Positive oder ein False Positive ist

    for label in labels:
      jaccard = jaccard_score(y_pred.flatten(),y_true.flatten(), pos_label=label, average='weighted')
      IoU.append(jaccard)
    IoU = np.mean(IoU) #jacard/IoU of single image
    global_IoU.append(IoU)

    cm=[]
    accuracy = []
    cm = confusion_matrix(y_true.flatten(),y_pred.flatten(), labels=[0, 1])
    accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]) #accuracy of single image
    #cm[0,0]: true negatives, c[1,1]: true positives, c[1,0]: false negatives, c[0,1]: false positives
    global_accuracy.append(accuracy)


avg_acc =  np.mean(global_accuracy)
mean_IoU = np.mean(global_IoU)

print('Average accuracy is',avg_acc)
print('mean IoU is',mean_IoU)


#checking segmentation results
import random
test_img_number = random.randint(0, len(testimg))
plt.figure(figsize=(20, 18))
plt.subplot(231)
plt.title('Test Image', fontsize = 25)
plt.xticks([])
plt.yticks([])
plt.imshow(testimg[test_img_number])
plt.subplot(232)
plt.title('Ground Truth', fontsize = 25)
plt.xticks([])
plt.yticks([])
plt.imshow(ground_truth[test_img_number],cmap='gray')
plt.subplot(233)
plt.title('Prediction', fontsize = 25)
plt.xticks([])
plt.yticks([])
plt.imshow(prediction[test_img_number],cmap='gray')

plt.show()



#prediction on single image
from datetime import datetime 
reconstructed_image = []

testimages = sorted(os.listdir(path1))
testmasks =  sorted(os.listdir(path2))

test = cv2.imread(path1 + '/' + testimages[14], 0)
label = cv2.imread(path2 + '/' + testmasks[14], 0)

predicted_patches = []
start = datetime.now()   

test = clahe_equalized(test) #applying CLAHE
res_test = cv2.resize(test, dsize=(SIZE_X, SIZE_Y), interpolation=cv2.INTER_CUBIC)
res_test = np.array(res_test)

test_img = (res_test.astype('float32')) / 255.
test_img_norm = np.expand_dims(np.array(test_img), axis=-1)
test_img_input = np.expand_dims(test_img_norm, 0)
test_img_prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(
    np.uint8)  # predict on single patch
stop = datetime.now()
print('Execution time: ',(stop-start)) #computation time

test_shape = np.shape(test)

res_prediction = cv2.resize(test, dsize=(test_shape), interpolation=cv2.INTER_CUBIC)


plt.subplot(121)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.imshow(test)
plt.subplot(122)
plt.title('Prediction')
plt.xticks([])
plt.yticks([])
plt.imshow(reconstructed_image,cmap='gray')

plt.show()
