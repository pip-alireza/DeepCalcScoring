import os
from tensorflow import keras
from matplotlib import pyplot as plt
import segmentation_models as sm
import cv2
import numpy as np
import glob
from natsort import natsorted
import tensorflow as tf
from keras.callbacks import CSVLogger
from keras import backend as K
import skimage
from skimage import exposure
from utils import*


path = "data/all data/JAB-006/data_np"
folder_to_save = "data/all data/JAB-006/ton2_pred"
model_path = 'ton_cfj_400_2.h5'
beg_s = 63  # index starts from 0, make sure to reduce it by 1
end_s = 395
CT_gap_ratio = 1 # this is drived from ImgeJ
bifurcation_index = 110



X_pixel = 512
Y_pixel = 512
n_channel = 3
train_img = []
train_msk = []
iou_list = []
F1_list = []
calc_score=[]
pred_msk_ck = []


for im in natsorted(glob.glob(path + '/*img.png')): # sorted func is not correctly sorting but natsorted does
    image = cv2.imread(im, cv2.IMREAD_COLOR)
    train_img.append(image)
train_images = np.array(train_img)


for msk in natsorted(glob.glob(path + '/*mask.png')):
    mask = cv2.imread(msk, cv2.IMREAD_COLOR)
    # mask = cv2.resize(mask, (X_pixel, Y_pixel))
    mask[mask > 0] = 255 # this turns the image to B&W
    mask = mask/255 # this is necessary for iou since we want the value be btw 0 and 1
    mask = mask[:, :, 0] # for mask it is 3 dimension image (512,512,3) but all 3 are same
    train_msk.append(mask)

train_masks = np.array(train_msk)
print(train_masks.shape)
train_masks=np.expand_dims(train_masks,axis=3)


# Loading the model
model = keras.models.load_model(model_path, custom_objects={'TransformerBlock':TransformerBlock}, compile=False)


test_res = open(f"{folder_to_save}/calc_score.txt", 'w')
count = 0
for idx in range(train_images.shape[0]):

    # predicting and plotting  the segmentation
    pred = model.predict(np.expand_dims(train_images[idx, :,:,:], axis=0))  #shape_required (1,X,X,1)
    pred_mk = np.squeeze(pred)     #shape (X,X)
    # pred_msk = cv2.resize(pred_mk, (512, 512), interpolation=cv2.INTER_CUBIC) # increasing the area of interest by 10% 512 +51
    pred_msk = pred_mk > 0.5  # this creates a True, False elements in the matrix
    img_o = train_images[idx, :, :, 0]

    #In this part we check if the model has failed
    pred_msk_ck.append(pred_msk)

    img_o[pred_msk == False] = 0   # extracting vascular system


    # counting calcification in every image by setting a threshold 145 over average intensity
    calc_img = np.zeros((X_pixel, Y_pixel, n_channel)).astype(np.uint8)
    calc_img = img_o + calc_img
    calc_img[calc_img<=145] =0

    # plt.imsave(f"{folder_to_save}/calc/{idx}_calc.png", calc_img.astype(np.uint8))
    calc = np.count_nonzero(calc_img)/3  #since we have 3 channels, it counts the number 3 times. That's why we divide by 3
    calc_score.append(calc)
    test_res.write(
        f"Mean calcium score at {idx} "  "%s %s \n" % ((":  ", calc)))


   # calculating accuracy in every slice and plotting it
    mask_org = train_masks.astype(np.uint8)
    mask_org = mask_org[idx, :, :, 0]  # deleting the Red and Green element of the mask
    pred_msk_ol = pred_msk * 1  # reconstructing a

    intersect = np.zeros((X_pixel, Y_pixel, n_channel)).astype(np.float32)
    true_missed = np.zeros((X_pixel, Y_pixel, n_channel)).astype(np.float32)
    false_seg = np.zeros((X_pixel, Y_pixel, n_channel)).astype(np.float32)
    segmented = np.zeros((X_pixel, Y_pixel, n_channel)).astype(np.uint8)
    
    # we use float32 so in the below subtraction it captures negative values.
    # then we transfer it back to uint8 for subtraction from the original img and visualization

    intersect[:, :, 1] = mask_org + pred_msk_ol  # intersection pixels will be 2. Intersection will have green color
    intersect[intersect != 2] = 0  # 0
    intersect[intersect == 2] = 255
    intersect_pixel = np.count_nonzero(intersect[:, :, 1] > 2)

    true_missed[:, :, 2] = mask_org - pred_msk_ol  # true pixels that are missed are 1. True missed will have blue channel
    true_missed[true_missed != 1] = 0  # 0
    true_missed[true_missed == 1] = 255
    true_m_pixel = np.count_nonzero(true_missed[:, :, 2] > 2)

    false_seg[:, :, 0] = pred_msk_ol - mask_org  # falsely segmented pixels are 1. False segmented is Red
    false_seg[false_seg != 1] = 0  # 0
    false_seg[false_seg == 1] = 255
    false_s_pixel = np.count_nonzero(false_seg[:, :, 0] > 2)

    intersect = intersect.astype(np.uint8)
    true_missed = true_missed.astype(np.uint8)
    false_seg = false_seg.astype(np.uint8)

    segmented += intersect + true_missed + false_seg  # adding the mask to the original image
    # img_o += intersect + true_missed + false_seg  # adding the mask to the original image
    iou_list.append(intersect_pixel / (intersect_pixel + true_m_pixel + false_s_pixel))
    F1_list.append(2*intersect_pixel / (2*intersect_pixel + true_m_pixel + false_s_pixel))

    test_res.write(
        f"test image {idx}: " "%s %s %s %s %s \n" % (
        ("intersect_pixel ", intersect_pixel), ("true_m_pixel: ", true_m_pixel), ("false_s_pixel: ", false_s_pixel),
        ("IOU_score: ", (intersect_pixel / (intersect_pixel + true_m_pixel + false_s_pixel))), ("cal score: ", calc_score[idx])))


    # TO SAVE SEGMENTATION ON THE ORIGINAL IMAGE
    vessel_seg = np.zeros((X_pixel, Y_pixel, n_channel)).astype(np.float32)
    vessel_seg[:,:,0] = pred_msk_ol # assign red to predicted mask
    vessel_seg[vessel_seg == 1] = 255
    vessel_seg = vessel_seg.astype(np.uint8) 
    train_images[idx, :,:,:][np.repeat(pred_msk_ol[:, :, np.newaxis], 3, axis=2)>0] = 0 # zeros the areas where predicted is 1
    vessel_seg = train_images[idx, :,:,:] + vessel_seg

    plt.imsave(f"{folder_to_save}/{idx}vs.png", vessel_seg)

    train_images[idx, :,:,:][segmented>0] = 0
    overlaid = train_images[idx, :,:,:] + segmented

    # plt.imsave(f"{folder_to_save}/overlayed/{idx}ovld.png", overlaid)
    # plt.imsave(f"{folder_to_save}/all overlayed/{idx}_segmented.png", segmented)
    print(idx)

iou_mean = np.mean(iou_list)
iou_mean_aorta = np.mean(iou_list[:108]) # accuracy for aorta
F1_mean = np.mean(F1_list)
F1_mean_aorta = np.mean(F1_list[:bifurcation_index])
calc_ss= sum(calc_score[beg_s:end_s]) * CT_gap_ratio


print("Mean iou score", iou_mean)
test_res.write(
    f"Mean iou score at {idx} "  "%s %s \n" % ((":  ", iou_mean)))
test_res.write(
    f"mean iou score for only aorta"  "%s %s \n" % ((":  ", iou_mean_aorta)))
test_res.write(
    f"mean F1 score"  "%s %s \n" % ((":  ", F1_mean)))
test_res.write(
     f"Calcium score between slice {beg_s} and {end_s}"  "%s %s \n" % ((":  ", calc_ss)))
test_res.write(
    f"mean F1 score for only aorta"  "%s %s \n" % ((":  ", F1_mean_aorta)))
test_res.close()


print("calc score", calc_score)

