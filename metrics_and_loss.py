import tensorflow as tf
from tensorflow.keras import losses
import numpy as np
def recall(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.round(y_true_f)
    y_pred_f = tf.round(y_pred_f)
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    re=intersection/(tf.reduce_sum(y_true_f)+1)
    return re

def precision(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.round(y_true_f)
    y_pred_f = tf.round(y_pred_f)
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    pr=intersection/(tf.reduce_sum(y_pred_f)+1)
    return pr

def specificity(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.round(y_true_f)
    y_pred_f = tf.round(y_pred_f)
    intersection = tf.reduce_sum(tf.multiply((1-y_true_f), (1-y_pred_f)))
    sp=intersection/(tf.reduce_sum((1-y_pred_f))+1)
    return sp

def mean_iou(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.round(y_true_f)
    y_pred_f = tf.round(y_pred_f)
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    miou=intersection/(tf.reduce_sum(y_pred_f)+tf.reduce_sum(y_true_f)-intersection+1)
    return miou

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(tf.multiply(y_true_f,y_pred_f))
    score = (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_coeff1(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.round(y_true_f)
    y_pred_f = tf.round(y_pred_f)
    intersection = tf.reduce_sum(tf.multiply(y_true_f,y_pred_f))
    score = (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score 

def dice_loss(y_true,y_pred):
    loss = 1-dice_coeff(y_true, y_pred)
    return loss
    
def bce_dice_loss(y_true,y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def tversky(y_true, y_pred,alpha=0.7):
    smooth=1
    y_true_pos = tf.reshape(y_true,[-1])
    y_pred_pos = tf.reshape(y_pred,[-1])
    true_pos = tf.reduce_sum(tf.multiply(y_true_pos, y_pred_pos))
    false_neg = tf.reduce_sum(tf.multiply(y_true_pos, (1-y_pred_pos)))
    false_pos = tf.reduce_sum(tf.multiply((1-y_true_pos),y_pred_pos))
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
    
def focal_tversky(y_true,y_pred):
    gamma = 0.75
    pt_1 = tversky(y_true, y_pred,alpha=0.7)
    return tf.math.pow((1-pt_1), gamma)

def BCE_tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)+losses.binary_crossentropy(y_true, y_pred)

def bce_focal_tversky(y_true,y_pred):
    return focal_tversky(y_true,y_pred)+losses.binary_crossentropy(y_true, y_pred)
