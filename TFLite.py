import tensorflow as tf

patch_size = 512

#loading model architectures
from model import unetmodel, residualunet, attentionunet, residual_attentionunet
from tensorflow.keras.optimizers import Adam
from evaluation_metrics import IoU_coef,IoU_loss
import os
import cv2
import numpy as np
from patchify import patchify, unpatchify
import skimage

# CLAHE
def clahe_equalized(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized

IMG_HEIGHT = patch_size
IMG_WIDTH = patch_size
IMG_CHANNELS = 1

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = attentionunet(input_shape) #/residualunet(input_shape)/unetmodel(input_shape)/attention_residualunet(input_shape)
model.compile(optimizer = Adam(learning_rate = 1e-3), loss= IoU_loss, metrics= ['accuracy', IoU_coef])
model.load_weights('M:\Regine Rausch/05 Data/05 Segmentation Network\Retinal-Vessel-Segmentation-using-variants-of-UNET'
                   '\Trained models/retina_attentionUnet_150epochs.hdf5') #loading weights

# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
open("attention_Unet_lite.tflite", "wb").write(tflite_model)

# Convert the model to the TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
quant_tflite_model = converter.convert()

# Save the model to disk
open("attention_Unet_quant_lite.tflite", "wb").write(quant_tflite_model)

#load test image
path1 = 'M:\Regine Rausch/05 Data/05 Segmentation Network/healthy'              #test images directory path
path2 = 'M:\Regine Rausch/05 Data/05 Segmentation Network/healthy_manualsegm'   #label images directory path

testimg = []
ground_truth = []
prediction = []
global_IoU = []
global_accuracy = []

testimages = sorted(os.listdir(path1))
testmasks =  sorted(os.listdir(path2))

idx = 0
image_name = testimages[0]
#for idx, image_name in enumerate(testimages):
if image_name.endswith(".jpg"):
    predicted_patches = []
    test_img = skimage.io.imread(path1 + "/" + image_name)

    test = test_img[:, :, 1]  # selecting green channel
    test = clahe_equalized(test)  # applying CLAHE
    SIZE_X = (test_img.shape[1] // patch_size) * patch_size  # getting size multiple of patch size
    SIZE_Y = (test_img.shape[0] // patch_size) * patch_size  # getting size multiple of patch size
    test = cv2.resize(test, (SIZE_X, SIZE_Y))
    testimg.append(test)
    test = np.array(test)

    patches = patchify(test, (patch_size, patch_size), step=patch_size)  # create patches(patch_sizexpatch_sizex1)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch_norm = (single_patch.astype('float32')) / 255.
            single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
            single_patch_input = np.expand_dims(single_patch_norm, 0)

            # TF model
            single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(
                np.uint8)  # predict on single patch

            # Load the TFLite model and allocate tensors.
            #https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
            interpreter = tf.lite.Interpreter(model_path="attention_Unet_lite.tflite")
            interpreter.allocate_tensors()
            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            # Test the model on random input data.
            input_shape = input_details[0]['shape']
            #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            input_data = single_patch_input
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output = (output_data[0, :, :, 0] > 0.5).astype(np.uint8)

            predicted_patches.append(single_patch_prediction)
    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches,
                                            (patches.shape[0], patches.shape[1], patch_size, patch_size))
    reconstructed_image = unpatchify(predicted_patches_reshaped, test.shape)  # join patches to form whole img
    prediction.append(reconstructed_image)




