# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:47:15 2023

@author: user
"""

from network_module_OSM import inception_resnetv2_module as incept_resv2_net


train_file = "lists/osm-435_train.txt"
test_file = "lists/osm-435_validation.txt"

checkpoint_model = "slim_models/inception_resnet_v2_2016_08_30/inception_resnet_v2_2016_08_30.ckpt"
checkpoint_save_dir = "dir/to/save/checkpoints"
tensorboard_save_dir = "dir/to/save/tensorboard/output"



batch = 32
input_size = (299,299,3)
numclasses = 997
learning_rate = 0.0001
iterbatch = 4
max_iter = 5000000
val_freq = 50
val_iter = 10


network = incept_resv2_net(
        batch = batch,
        iterbatch = iterbatch,
        numclasses = numclasses,
        input_size = input_size,
        train_file = train_file,
        test_file = test_file,
        checkpoint_model = checkpoint_model,
        save_dir = checkpoint_save_dir,
        tensorboard_dir = tensorboard_save_dir,
        learning_rate = learning_rate,
        max_iter = max_iter,
        val_freq = val_freq,
        val_iter = val_iter
        )