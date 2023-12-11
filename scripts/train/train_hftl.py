# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:04:43 2023

@author: user
"""

"""
Network 1 - herbarium network pretrained ckpt plantclef 2021
Network 2 - field network pretrained ckpt plantclef 2021

"""

from network_module_HFTL import inception_resnetv2_module as incept_resv2_net


train_file1 = "lists/hftl_herbarium_train.txt"
test_file1 = "lists/hftl_herbarium_validation.txt"

train_file2 = "lists/hftl_field_train.txt"
test_file2 = "lists/hftl_field_validation.txt"

checkpoint_model1 = "saved_weights/inceptionres_herbarium_run9/migrated_ckpt/herbarium_migrated_run9_best.ckpt"
checkpoint_model2 = "saved_weights/inceptionres_field_new_run1/migrated_ckpt/field_migrated_new_run1_best.ckpt"
checkpoint_save_dir = "dir/to/save/checkpoints"




batch1 = 16
batch2 = 16
input_size = (299,299,3)
numclasses1 = 997
numclasses2 = 997
learning_rate = 0.0001
iterbatch = 4
max_iter = 5000000
val_freq = 50
val_iter = 5


network = incept_resv2_net(
        batch1 = batch1,
        batch2 = batch2,
        iterbatch = iterbatch,
        numclasses1 = numclasses1,
        numclasses2 = numclasses2,
        input_size = input_size,       
        train_file1 = train_file1,
        train_file2 = train_file2,
        test_file1 = test_file1,
        test_file2 = test_file2,        
        checkpoint_model1 = checkpoint_model1,
        checkpoint_model2 = checkpoint_model2,
        save_dir = checkpoint_save_dir,
        learning_rate = learning_rate,
        max_iter = max_iter,
        val_freq = val_freq,
        val_iter = val_iter
        )
