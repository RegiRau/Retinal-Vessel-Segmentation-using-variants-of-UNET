import os
import cv2
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
np.random.seed(0)


#CLAHE
def clahe_equalized(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized

# path1 = '../healthy'              #training images directory path
# path2 = '../healthy_manualsegm'   #training images directory path

path1 = "../04_Vein_Dataset/images"
path2 = "../04_Vein_Dataset/labels"

image_dataset = []
mask_dataset = []

with_patches = False

if with_patches:

    patch_size = 512

    images = sorted(os.listdir(path1))
    for i, image_name in enumerate(images):
       #if image_name.endswith(".jpg"):
       #image = skimage.io.imread(path1+"/"+image_name)  #Read image
       image = cv2.imread(path1 + '/' + image_name, 0)
       #image = image[:,:,1] #selecting green channel
       image = clahe_equalized(image) #applying CLAHE
       SIZE_X = (image.shape[1]//patch_size)*patch_size #getting size multiple of patch size
       SIZE_Y = (image.shape[0]//patch_size)*patch_size #getting size multiple of patch size
       image = Image.fromarray(image)
       image = image.resize((SIZE_X, SIZE_Y)) #resize image
       image = np.array(image)
       patches_img = patchify(image, (patch_size, patch_size), step=patch_size)  #create patches(patch_sizexpatch_sizex1)

       for i in range(patches_img.shape[0]):
           for j in range(patches_img.shape[1]):
               single_patch_img = patches_img[i,j,:,:]
               single_patch_img = (single_patch_img.astype('float32')) / 255.
               image_dataset.append(single_patch_img)

    masks = sorted(os.listdir(path2))
    for i, mask_name in enumerate(masks):
        # if mask_name.endswith(".jpg"):
        #     mask = skimage.io.imread(path2+"/"+mask_name)   #Read masks
        mask = cv2.imread(path2 + '/' + mask_name, 0)
        #mask = skimage.io.imread(path2 + '/' + mask_name, plugin='pil')
        SIZE_X = (mask.shape[1]//patch_size)*patch_size #getting size multiple of patch size
        SIZE_Y = (mask.shape[0]//patch_size)*patch_size #getting size multiple of patch size
        mask = Image.fromarray(mask)
        mask = mask.resize((SIZE_X, SIZE_Y))  #resize image
        mask = np.array(mask)
        patches_mask = patchify(mask, (patch_size, patch_size), step=patch_size)  #create patches(patch_sizexpatch_sizex1)

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i,j,:,:]
                single_patch_mask = (single_patch_mask.astype('float32'))/255.
                mask_dataset.append(single_patch_mask)

    IMG_HEIGHT = patch_size
    IMG_WIDTH = patch_size

else:
    # ohne patches:
    SIZE_X = 1632
    SIZE_Y = 1216
    images = sorted(os.listdir(path1))
    for i, image_name in enumerate(images[:12]):
       image = cv2.imread(path1 + '/' + image_name, 0)
       image = clahe_equalized(image) #applying CLAHE
       image = cv2.resize(image, dsize=(SIZE_X, SIZE_Y), interpolation=cv2.INTER_CUBIC)
       image = np.array(image, dtype="float32")
       image_dataset.append(image)

    masks = sorted(os.listdir(path2))
    for i, mask_name in enumerate(masks[:12]):
        mask = cv2.imread(path2 + '/' + mask_name, 0)
        mask = cv2.resize(mask, dsize=(SIZE_X, SIZE_Y), interpolation=cv2.INTER_CUBIC)
        mask[mask < 200] = 0
        mask[mask >= 200] = 1
        mask = np.array(mask, dtype="float32")
        mask_dataset.append(mask)

    IMG_HEIGHT = SIZE_Y
    IMG_WIDTH = SIZE_X


image_dataset = np.array(image_dataset)
mask_dataset =  np.array(mask_dataset)
image_dataset = np.expand_dims(image_dataset,axis=-1)
mask_dataset =  np.expand_dims(mask_dataset,axis=-1)

print('built dataset')

#importing models
from model import unetmodel, residualunet, attentionunet, residual_attentionunet
from tensorflow.keras.optimizers import Adam
from evaluation_metrics import IoU_coef,IoU_loss


IMG_CHANNELS = 1
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# model = unetmodel(input_shape)
# model.compile(optimizer = Adam(lr = 1e-3), loss= IoU_loss, metrics= ['accuracy', IoU_coef])
#model = residualunet(input_shape)
#model.compile(optimizer = Adam(lr = 1e-3), loss= IoU_loss, metrics= ['accuracy', IoU_coef])
model = attentionunet(input_shape)
model.compile(optimizer = Adam(learning_rate = 1e-3), loss= IoU_loss, metrics= ['accuracy', IoU_coef])
# model = residual_attentionunet(input_shape)
# model.compile(optimizer = Adam(lr = 1e-3), loss= IoU_loss, metrics= ['accuracy', IoU_coef])


#splitting data into 70-30 ratio to validate training performance
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.3, random_state=0)

np.save('train_test_split/x_train.npy', x_train)
np.save('train_test_split/y_train.npy', y_train)
np.save('train_test_split/x_test.npy', x_test)
np.save('train_test_split/y_test.npy', y_test)

#train model
history = model.fit(x_train, y_train, 
                    verbose=1,
                    batch_size = 2,
                    validation_data=(x_test, y_test ), 
                    shuffle=False,
                    epochs=10)

# #training-validation loss curve
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure(figsize=(7,5))
# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'y', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# #training-validation accuracy curve
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.figure(figsize=(7,5))
# plt.plot(epochs, acc, 'r', label='Training Accuracy')
# plt.plot(epochs, val_acc, 'y', label='Validation Accuracy')
# plt.title('Training and validation accuracies')
# plt.xlabel('Epochs')
# plt.ylabel('IoU')
# plt.legend()
# plt.show()

# #training-validation IoU curve
# iou_coef = history.history['IoU_coef']
# val_iou_coef = history.history['val_IoU_coef']
# plt.figure(figsize=(7,5))
# plt.plot(epochs, iou_coef, 'r', label='Training IoU')
# plt.plot(epochs, val_iou_coef, 'y', label='Validation IoU')
# plt.title('Training and validation IoU coefficients')
# plt.xlabel('Epochs')
# plt.ylabel('IoU')
# plt.legend()
# plt.show()

#save model
model.save('Veins_Trained_models/Veins_Attention_Unet_150epochs.hdf5')
