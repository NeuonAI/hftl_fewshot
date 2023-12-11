# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:25:14 2023

@author: user
"""
import sys

sys.path.append("path/to/tf_slim/models/research/slim")

from preprocessing import inception_preprocessing_1_12_no_crop
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import cv2
from nets.inception_resnet_v2 import inception_resnet_v2
from nets import inception_utils
from PIL import Image
from six.moves import cPickle
import datetime



train_herbarium_txt = "path/to/dictionary_method1.txt"
checkpoint_model = "path/to/trained/osm_model.ckpt"
pkl_file = "path/to/save/dictionary.pkl"



# ----- Network hyperparameters ----- #
batch = 100 # 10 images x 10 corner crops
n_crop = 10
n_image_per_batch = 10
input_size = (299,299,3)
numclasses = 997 



# ----- Initiate tensors ----- #
is_training = tf.placeholder(tf.bool)    
x = tf.placeholder(tf.float32,(batch,) + input_size)

def read_txt(txt_file):
    with open(txt_file, 'r') as t1:
        lines = [x.strip() for x in t1.readlines()]
    
    paths = [x.split(" ")[0] for x in lines]

    fam_lbls = [x.split(" ")[1] for x in lines]
    gen_lbls = [x.split(" ")[2] for x in lines]
    spe_lbls = [x.split(" ")[3] for x in lines]
    
    return paths, fam_lbls, gen_lbls, spe_lbls


def datetimestr():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")


def crop_images(filepath):
    img = []
    
    try:
    
        im = cv2.imread(filepath)
    
        if im is None:
           im = cv2.cvtColor(np.asarray(Image.open(filepath).convert('RGB')),cv2.COLOR_RGB2BGR)
        im = cv2.resize(im,(input_size[0:2]))
    
        if np.ndim(im) == 2:
            im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)          
        else:
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
               
        # Flip image
        flip_img = cv2.flip(im, 1)
        
        im1 = im[0:260,0:260,:]
        im2 = im[0:260,-260:,:]
        im3 = im[-260:,0:260,:]
        im4 = im[-260:,-260:,:]
        im5 = im[19:279,19:279,:]
        
        imtemp = [cv2.resize(ims,(input_size[0:2])) for ims in (im1,im2,im3,im4,im5)]
                
        [img.append(ims) for ims in imtemp]


        flip_im1 = flip_img[0:260,0:260,:]
        flip_im2 = flip_img[0:260,-260:,:]
        flip_im3 = flip_img[-260:,0:260,:]
        flip_im4 = flip_img[-260:,-260:,:]
        flip_im5 = flip_img[19:279,19:279,:]
        
        flip_imtemp = [cv2.resize(imf,(input_size[0:2])) for imf in (flip_im1,flip_im2,flip_im3,flip_im4,flip_im5)]
                
        [img.append(imf) for imf in flip_imtemp]  
        
        

    except:
        print("Exception: Image not read...")
   
    
    return img



# ----- Image preprocessing methods ----- #
train_preproc = lambda xi: inception_preprocessing_1_12_no_crop.preprocess_image(
        xi,input_size[0],input_size[1],is_training=True)

test_preproc = lambda xi: inception_preprocessing_1_12_no_crop.preprocess_image(
        xi,input_size[0],input_size[1], is_training=False)  

def data_in_train():
    return tf.map_fn(fn = train_preproc,elems = x,dtype=np.float32)      

def data_in_test():
    return tf.map_fn(fn = test_preproc,elems = x,dtype=np.float32)

data_in = tf.cond(
        is_training,
        true_fn = data_in_train,
        false_fn = data_in_test
        )


# ----- Construct network ----- #

with slim.arg_scope(inception_utils.inception_arg_scope()):
    logits,endpoints = inception_resnet_v2(data_in,
                                    num_classes=numclasses,
                                    is_training=is_training,
                                    create_aux_logits=False)
    
with tf.variable_scope('mixed_embedding'):      
      
    feat_500 = slim.fully_connected(
                    inputs=endpoints['PreLogitsFlatten'] ,
                    num_outputs=500,
                    activation_fn=None,
                    normalizer_fn=None,
                    trainable=True,
                    scope='feat_500'                            
            )
    logits_500 = slim.fully_connected(feat_500,997,activation_fn=None,
                                scope='Species_500')


variables_to_restore = slim.get_variables_to_restore()
restorer = tf.train.Saver(variables_to_restore)


# ----- Get dictionary data ----- #
train_anchor_paths, train_anchor_fam_lbls, train_anchor_gen_lbls, train_anchor_spe_lbls = read_txt(train_herbarium_txt)
dictionary = {}
for spe, path in zip(train_anchor_spe_lbls, train_anchor_paths):
    if spe not in dictionary:
        dictionary[spe] = []
    dictionary[spe].append(path)


# ----- Start session ----- #
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
print("Start session")
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, checkpoint_model)


    mean_emb_dict = {}

    counter = 0
    for key in dictionary.keys():
        
        print("Counter:", counter, "Key:", key)
        
        key_paths = dictionary[key]
        print("key paths len:", len(key_paths))
        
        key_all_embs = []
       
        n_key_paths_per_batch = [key_paths[i:i + n_image_per_batch] for i in range(0, len(key_paths), n_image_per_batch)]
       
        for batch_filepaths in n_key_paths_per_batch:
            
            add_padding = False
            
            batch_im_crops = []

            
            if len(batch_filepaths) < n_image_per_batch:
                padding_num = n_image_per_batch - len(batch_filepaths)
                for k in range(padding_num):
                    batch_filepaths.append(batch_filepaths[0])
                    add_padding = True

            
            for i in range(n_image_per_batch): 
                filepath = batch_filepaths[i]

                im_crops = crop_images(filepath)
                im_crops = np.asarray(im_crops,dtype=np.float32)/255.0 
                  
                batch_im_crops.append(im_crops)
            
            
            batch_im_crops_reshaped = np.reshape(batch_im_crops, (batch, 299, 299, 3))
                        
            batch_embeddings = sess.run(
                        feat_500,
                        feed_dict = {
                                    x : batch_im_crops_reshaped,                            
                                    is_training : False
                                }
                    )
            
            batch_embeddings_grouped = [batch_embeddings[i:i + n_crop] for i in range(0, len(batch_embeddings), n_crop)]
            
            if add_padding is True:
                n_to_keep = n_image_per_batch - padding_num
            else:
                n_to_keep = n_image_per_batch
            
            for i in range(n_to_keep):
                current_embs = batch_embeddings_grouped[i]
                
                current_embs_mean = np.mean(current_embs, axis=0)
                key_all_embs.append(current_embs_mean)
                       
       
        # ----- Get mean embs ----- #
        key_mean_embs = np.mean(key_all_embs, axis=0)
        
        if key not in mean_emb_dict:
            mean_emb_dict[key] = key_mean_embs
            
        counter += 1
        
        
        

        
  
with open(pkl_file,'wb') as fid:
    cPickle.dump(mean_emb_dict,fid,protocol=cPickle.HIGHEST_PROTOCOL)
    print(f"[{datetimestr()}] Pkl file created")
    








