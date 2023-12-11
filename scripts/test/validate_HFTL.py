# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:05:45 2023

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
import os
from six.moves import cPickle
from sklearn.metrics.pairwise import cosine_similarity
import datetime





test_field_file = "path/to/lists/test_seen.txt"
image_dir_parent_test = "path/to/dataset"
checkpoint_model = "path/to/trained_hftl_model.ckpt"
pkl_file = "path/to/saved/dictionary.pkl"


# ----- Network hyperparameters ----- #
batch1 = 10
batch2 = 10
input_size = (299,299,3)
numclasses1 = 997
numclasses2 = 997  


# ----- Initiate tensors ----- #
is_training = tf.placeholder(tf.bool)
x1 = tf.placeholder(tf.float32,(batch1,) + input_size)
x2 = tf.placeholder(tf.float32,(batch2,) + input_size)


def read_txt(txt_file):
    with open(txt_file, 'r') as t1:
        lines = [x.strip() for x in t1.readlines()]
    
    paths = [os.path.join(image_dir_parent_test,x.split(" ")[0]) for x in lines]

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




def match_herbarium_dictionary(test_embedding_list, herbarium_emb_list):
    similarity = cosine_similarity(test_embedding_list, herbarium_emb_list)
        
    k_distribution = []
    # 1 - Cosine
    print("Get probability distribution")
    for sim in similarity:
        new_distribution = []
        for d in sim:
            new_similarity = 1 - d
            new_distribution.append(new_similarity)
        k_distribution.append(new_distribution)
        
    k_distribution = np.array(k_distribution)
        
              
    softmax_list = []
    # Inverse weighting
    for d in k_distribution:
        inverse_weighting = (1/np.power(d,5))/np.sum(1/np.power(d,5))
        softmax_list.append(inverse_weighting)
    
    softmax_list = np.array(softmax_list)    
    
    return softmax_list


def get_rank(dict_value):
    prob = dict_value['prob']
    label = dict_value['label']
    
    idx = np.argsort(prob)[::-1]
    
    np.argmax(prob) == label
    
    rank_i = np.squeeze(np.where(idx==label)) + 1
    
    return rank_i


# ----- Image preprocessing methods ----- #
train_preproc = lambda xi: inception_preprocessing_1_12_no_crop.preprocess_image(
        xi,input_size[0],input_size[1],is_training=True)

test_preproc = lambda xi: inception_preprocessing_1_12_no_crop.preprocess_image(
        xi,input_size[0],input_size[1],is_training=False) 

def data_in_train1():
    return tf.map_fn(fn = train_preproc,elems = x1,dtype=np.float32)

def data_in_test1():
    return tf.map_fn(fn = test_preproc,elems = x1,dtype=np.float32)

def data_in_train2():
    return tf.map_fn(fn = train_preproc,elems = x2,dtype=np.float32)

def data_in_test2():
    return tf.map_fn(fn = test_preproc,elems = x2,dtype=np.float32)

data_in1 = tf.cond(
        is_training,
        true_fn = data_in_train1,
        false_fn = data_in_test1
        )

data_in2 = tf.cond(
        is_training,
        true_fn = data_in_train2,
        false_fn = data_in_test2
        )




# ----- Construct network ----- #
# ----- Network 1 construction ----- #
with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits,endpoints = inception_resnet_v2(data_in1,
                                                num_classes=numclasses1,
                                                is_training=is_training,
                                                scope='herbarium',
                                                create_aux_logits=False
                                                )
        
with tf.variable_scope('herbarium_embedding'):
           
    herbarium_fc = slim.fully_connected(
                inputs=endpoints['PreLogitsFlatten'],
                num_outputs=500,
                activation_fn=None,
                normalizer_fn=None,   
                weights_regularizer=slim.l2_regularizer(0.00004),                     
                trainable=True,
                scope='herbarium_embedding'
        )

    herbarium_feat = tf.math.l2_normalize(
            herbarium_fc,
            axis=1 
            )    
            
 

        
# ----- Network 2 construction ----- #
with slim.arg_scope(inception_utils.inception_arg_scope()):
    logits2,endpoints2 = inception_resnet_v2(data_in2,
                                    num_classes=numclasses2,
                                    is_training=is_training,
                                    scope='field',
                                    create_aux_logits=False
                                    )


with tf.variable_scope('field_embedding'):  
    
    field_fc = slim.fully_connected(
                    inputs=endpoints2['PreLogitsFlatten'],
                    num_outputs=500,
                    activation_fn=None,
                    normalizer_fn=None,
                    weights_regularizer=slim.l2_regularizer(0.00004), 
                    trainable=True,
                    scope='field_embedding'
            )

    field_feat = tf.math.l2_normalize(
                                    field_fc,
                                    axis=1 
                                )         
    

     

    
variables_to_restore = slim.get_variables_to_restore()
restorer = tf.train.Saver(variables_to_restore)


# ----- Get test set data ----- #
test_positive_paths, test_positive_fam_lbls, test_positive_gen_lbls, test_positive_spe_lbls = read_txt(test_field_file)


# ----- Get herbarium / acnhor dictionary ----- #
with open(pkl_file,'rb') as fid1:
	herbarium_dictionary = cPickle.load(fid1)

herbarium_dictionary_embs = []
for i in range(numclasses1):
    k = str(i)
    emb = herbarium_dictionary[k]
    herbarium_dictionary_embs.append(emb)
    


herbarium_dictionary_embs = np.asarray(herbarium_dictionary_embs)

prediction_dictionary = {}


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
print("Start process")
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, checkpoint_model)
    
    correct_count = 0
    test_embedding_list = []
    imgs_filepaths = []
    ground_truth_list = []
    
    for i, gt_label, path in zip(range(len(test_positive_paths)), test_positive_spe_lbls, test_positive_paths):
        print(i, path)


        
        gt_index = int(gt_label)
        ground_truth_list.append(gt_index)
        path_positive = test_positive_paths[i]
        ims_positive = crop_images(path_positive)
 
               
        ims_positive = np.asarray(ims_positive,dtype=np.float32)/255.0
        
        sample_embeddings = sess.run(
                    field_feat,
                    feed_dict = {
                                x2:ims_positive,   
                                is_training : False
                            }
                )
        
    
        
        
        for emb in sample_embeddings:   
            test_embedding_list.append(emb.reshape(1,500)) 
        imgs_filepaths.append(path)
        
        
        
    len_test_embedding_list = len(test_embedding_list)     
    test_embedding_all_crop_list = np.asarray(test_embedding_list)
    test_embedding_all_crop_list = np.reshape(test_embedding_all_crop_list, (len_test_embedding_list,500))
        
    # ----- Iterate sample over herbarium mean class ----- #
    print("Comparing sample embedding with herbarium distance...")
    softmax_all_crop_list = match_herbarium_dictionary(test_embedding_all_crop_list, herbarium_dictionary_embs)
    
    
    # ----- Average the 10 crops predictions ----- #
    softmax_all_crop_list_grouped = [softmax_all_crop_list[i:i + 10] for i in range(0, len(softmax_all_crop_list), 10)]
    softmax_all_crop_list_avg = [np.mean(x, axis=0) for x in softmax_all_crop_list_grouped]
    softmax_all_crop_list_avg_np = np.asarray(softmax_all_crop_list_avg)       

    # ----- Get all crops results ----- #
    print("Get top N predictions all crops...")
    for prediction, key, fp in zip(softmax_all_crop_list_avg_np, ground_truth_list, imgs_filepaths):
        if fp not in prediction_dictionary:
            prediction_dictionary[fp] = {'prob' : [], 'label' : []}
        prediction_dictionary[fp]['prob'] = prediction
        prediction_dictionary[fp]['label'] = key              
                

    top1 = 0
    top5 = 0
    top1_species_counter = 0
    top5_species_counter = 0

        
    for filepath in prediction_dictionary.keys():
        groundtruth = prediction_dictionary[filepath]["label"]
        pred_probabilities = prediction_dictionary[filepath]["prob"]
        
        top_5_pred = pred_probabilities.argsort()[-5:][::-1]
        top_5_prob = np.sort(pred_probabilities)[-5:][::-1]
        
        top_1_pred = pred_probabilities.argsort()[-1:][::-1]
        top_1_prob = np.sort(pred_probabilities)[-1:][::-1]
        
        if groundtruth in top_5_pred:
            top5 = 1
        else:
            top5 = 0
        
        if groundtruth == top_1_pred:
            top1 = 1
        else:
            top1 = 0
        
        top1_species_counter += top1
        top5_species_counter += top5
        
        print("Top-1 species:", round(top1_species_counter / len(imgs_filepaths),4), top1_species_counter,"/",len(imgs_filepaths), " ----- Top-5 species:", round(top5_species_counter / len(imgs_filepaths),4), top5_species_counter,"/",len(imgs_filepaths))



    #   Save prediction file
    ranks = np.asarray([get_rank(value) for key,value in prediction_dictionary.items()])
    mrr = np.sum((1/ranks))/len(prediction_dictionary)
    print("MRR score:", round(mrr,4))

                  
                   