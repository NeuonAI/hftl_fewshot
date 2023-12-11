# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:21:07 2023

@author: user
"""


import os 
import sys
sys.path.append("path/to/tf_slim/models/research/slim")


import tensorflow as tf
from preprocessing import inception_preprocessing
slim = tf.contrib.slim

import numpy as np

from nets.inception_resnet_v2 import inception_resnet_v2
from nets import inception_utils


from database_module_HFTL import database_module
from sklearn.metrics.pairwise import euclidean_distances


class inception_resnetv2_module(object):
    def __init__(self,
                 batch1,
                 batch2,
                 iterbatch,
                 numclasses1,
                 numclasses2,               
                 train_file1,
                 train_file2,
                 test_file1,
                 test_file2,   
                 input_size,
                 checkpoint_model1,
                 checkpoint_model2,
                 learning_rate,
                 save_dir,
                 max_iter,
                 val_freq,
                 val_iter):
        
        self.batch1 = batch1
        self.batch2 = batch2
        self.iterbatch = iterbatch
        self.train_file1 = train_file1
        self.train_file2 = train_file2
        self.test_file1 = test_file1
        self.test_file2 = test_file2 
        self.input_size = input_size
        self.numclasses1 = numclasses1
        self.numclasses2 = numclasses2
        self.checkpoint_model1 = checkpoint_model1
        self.checkpoint_model2 = checkpoint_model2
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.max_iter = max_iter
        self.val_freq = val_freq
        self.val_iter = val_iter
        
               
        print('Initiating database...')
        
        
        
        # =============================================================================
        #   Database module
        # =============================================================================
        self.train_database = database_module(
                database_file1 = self.train_file1,
                database_file2 = self.train_file2,
                batch1 = self.batch1,
                batch2 = self.batch2,
                input_size = self.input_size,
                numclasses1 = self.numclasses1,
                numclasses2 = self.numclasses2)

        self.test_database = database_module(
                database_file1 = self.test_file1,
                database_file2 = self.test_file2,
                batch1 = self.batch1,
                batch2 = self.batch2,
                input_size = self.input_size,
                numclasses1 = self.numclasses1,
                numclasses2 = self.numclasses2)
        
        
        
        
        
        
        
       
        print('Initiating tensors...')
        # =============================================================================
        #   Tensors 
        # =============================================================================
        x1 = tf.placeholder(tf.float32,(self.batch1,) + self.input_size)   # herbarium 
        x2 = tf.placeholder(tf.float32,(self.batch2,) + self.input_size)   # field
        y1 = tf.placeholder(tf.int32,(self.batch1,))
        y2 = tf.placeholder(tf.int32,(self.batch2,))
        self.is_training = tf.placeholder(tf.bool)
        



               
     
        # =============================================================================
        #   Image pre-processing methods   
        # =============================================================================
        train_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=True)
        
        test_preproc = lambda xi: inception_preprocessing.preprocess_image(
                xi,self.input_size[0],self.input_size[1],is_training=False) 
        
        def data_in_train1():
            return tf.map_fn(fn = train_preproc,elems = x1,dtype=np.float32)
        
        def data_in_test1():
            return tf.map_fn(fn = test_preproc,elems = x1,dtype=np.float32)
        
        def data_in_train2():
            return tf.map_fn(fn = train_preproc,elems = x2,dtype=np.float32)
        
        def data_in_test2():
            return tf.map_fn(fn = test_preproc,elems = x2,dtype=np.float32)
        
        data_in1 = tf.cond(
                self.is_training,
                true_fn = data_in_train1,
                false_fn = data_in_test1
                )
        
        data_in2 = tf.cond(
                self.is_training,
                true_fn = data_in_train2,
                false_fn = data_in_test2
                )
        

                
        # =============================================================================
        #   Constuct Network 1
        # =============================================================================
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits,endpoints = inception_resnet_v2(data_in1,
                                                num_classes=self.numclasses1,
                                                is_training=self.is_training,
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
            
 
        
        # =============================================================================
        #   Construct Network 2       
        # =============================================================================
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits2,endpoints2 = inception_resnet_v2(data_in2,
                                            num_classes=self.numclasses2,
                                            is_training=self.is_training,
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
       
        
        feat_concat = tf.concat([herbarium_feat, field_feat], axis=0)
        lbl_concat = tf.concat([y1, y2], axis=0)
 
           
        # =============================================================================
        #   Get all variables
        # =============================================================================
        self.variables_to_restore = tf.trainable_variables()
        
        self.variables_herbarium = [k for k in self.variables_to_restore if k.name.startswith('herbarium')]
        self.variables_field = [k for k in self.variables_to_restore if k.name.startswith('field')]


        # =============================================================================
        #   Get front and end variables to train
        # =============================================================================
        self.var_list_front = self.variables_herbarium[:-4] + self.variables_field[:-4]
        
        self.var_list_end = self.variables_herbarium[-2:] + self.variables_field[-2:]
    
        self.var_list_train = self.var_list_front + self.var_list_end

        
        
        
        
        
        # =============================================================================
        #   Network losses
        # =============================================================================
        with tf.name_scope("loss_calculation"):                 
            
            with tf.name_scope("triplet_loss_batch_semihard"):
                                
                self.triplet_loss = tf.reduce_mean(
                        tf.contrib.losses.metric_learning.triplet_semihard_loss(
                                labels=lbl_concat, embeddings=feat_concat, margin=1.0))
                    
                

            with tf.name_scope("L2_reg_loss"):
                self.regularization_loss = tf.add_n([ tf.nn.l2_loss(v) for v in self.var_list_train if 'biases' not in v.name])  * 0.00004 
                
            with tf.name_scope("total_loss"):
                self.totalloss = self.triplet_loss + self.regularization_loss
                
                        
            
            
            
        # =============================================================================
        #   Update operation
        # =============================================================================
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)    


        
        
        
        # =============================================================================
        #   Load model
        # =============================================================================
        self.vars_ckpt = slim.get_variables_to_restore()

        
        vars_ckpt_herbarium = [k for k in self.vars_ckpt if k.name.startswith('herbarium')]
        
        vars_ckpt_field = [k for k in self.vars_ckpt if k.name.startswith('field')]
              
        
        # ----- Restore model 1 ----- #
        restore_fn1 = slim.assign_from_checkpoint_fn(
            self.checkpoint_model1, vars_ckpt_herbarium[:-4])        
 
        
        # ----- Restore model 2 ----- #
        restore_fn2 = slim.assign_from_checkpoint_fn(
            self.checkpoint_model2, vars_ckpt_field[:-4]) 




        # =============================================================================
        #   Training scope       
        # =============================================================================
        with tf.name_scope("train"):
            loss_accumulator = tf.Variable(0.0, trainable=False)
            
            self.collect_loss = loss_accumulator.assign_add(self.totalloss)
                        
            self.average_loss = tf.cond(self.is_training,
                                        lambda: loss_accumulator / self.iterbatch,
                                        lambda: loss_accumulator / self.val_iter)
            
            self.zero_op_loss = tf.assign(loss_accumulator,0.0)


    
            # ----- Separate vars ----- #
            self.accum_train_front = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_front] 
            self.accum_train_end = [tf.Variable(tf.zeros_like(
                    tv.initialized_value()), trainable=False) for tv in self.var_list_end]                                               
        
            self.zero_ops_train_front = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train_front]
            self.zero_ops_train_end = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_train_end]




            
            # =============================================================================
            #   Set up optimizer / Compute gradients
            # =============================================================================
            with tf.control_dependencies(self.update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate * 0.1)
                # create another optimizer
                optimizer_end_layers = tf.train.AdamOptimizer(self.learning_rate)
                
                
                # compute gradient with another list of var_list
                gradient1 = optimizer.compute_gradients(self.totalloss,self.var_list_front)
                gradient2 = optimizer_end_layers.compute_gradients(self.totalloss,self.var_list_end)

              
                gradient_only_front = [gc[0] for gc in gradient1]
                gradient_only_front,_ = tf.clip_by_global_norm(gradient_only_front,1.25)
                
                gradient_only_back = [gc[0] for gc in gradient2]
                gradient_only_back,_ = tf.clip_by_global_norm(gradient_only_back,1.25)
                

               
                self.accum_train_ops_front = [self.accum_train_front[i].assign_add(gc) for i,gc in enumerate(gradient_only_front)]
            
                self.accum_train_ops_end = [self.accum_train_end[i].assign_add(gc) for i,gc in enumerate(gradient_only_back)]






            # =============================================================================
            #   Apply gradients
            # =============================================================================
            self.train_step_front = optimizer.apply_gradients(
                    [(self.accum_train_front[i], gc[1]) for i, gc in enumerate(gradient1)])
      
            self.train_step_end = optimizer_end_layers.apply_gradients(
                    [(self.accum_train_end[i], gc[1]) for i, gc in enumerate(gradient2)])
            
            



        # =============================================================================
        #   Global variables
        # =============================================================================
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]             
        var_list += bn_moving_vars
        
        
        
        
        
        # =============================================================================
        #   Create saver
        # =============================================================================
        saver = tf.train.Saver(var_list=var_list, max_to_keep=0)

        tf.summary.scalar('loss',self.average_loss) 
        self.merged = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'loss')])
        
        
        
        
        
        # =============================================================================
        #   Tensorboard writer
        # =============================================================================
        tensorboar_dir = os.path.join(self.save_dir,'_tensorboard')
        writer_train = tf.summary.FileWriter(tensorboar_dir+'/train')
        writer_test = tf.summary.FileWriter(tensorboar_dir+'/test')





        print('Commencing training...')      
        # =============================================================================
        #   TF Session     
        # =============================================================================
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            
            sess.run(tf.global_variables_initializer())
            writer_train.add_graph(sess.graph)
            writer_test.add_graph(sess.graph)
            
            restore_fn1(sess)
            restore_fn2(sess)

            initial_loss_val = 0.0
            initial_loss_train = 0.0
            loss_val = 0.0
            loss_train = 0.0            
            
            for i in range(self.max_iter+1):
                try:
                    sess.run(self.zero_ops_train_front)
                    sess.run(self.zero_ops_train_end)
                    sess.run([self.zero_op_loss])                    
                    
                    
                    
                    # =============================================================================
                    #   Validation
                    # =============================================================================
                    if i % self.val_freq == 0:
                        print('Start:%f'%sess.run(loss_accumulator))
                        for j in range(self.val_iter):
                            batch_true_counter = 0
                            batch_filepaths_current = []                            

                            img1, img2, lbl1, lbl2, batch_filepaths  = self.test_database.read_batch()
                            
                            for path in batch_filepaths:
                                batch_filepaths_current.append(path)
                            
                            _, herbarium_embeddings, field_embeddings = sess.run(
                                    [self.collect_loss, herbarium_feat, field_feat],
                                        feed_dict = {x1 : img1,
                                                     x2 : img2,
                                                     y1 : lbl1,
                                                     y2 : lbl2,
                                                     self.is_training : False
                                            
                                            }
                            )
                            
                            
                            
                            embs_h = herbarium_embeddings
                            embs_f = field_embeddings                                
                            embs_f_inverted = field_embeddings[::-1]
                            
                            for emb_h, emb_f, emb_f_inv in zip(embs_h, embs_f, embs_f_inverted):
                                emb_h = np.reshape(emb_h, (1,500))
                                emb_f = np.reshape(emb_f, (1,500))
                                emb_f_inv = np.reshape(emb_f_inv, (1,500))
                                
                                d_positive = euclidean_distances(emb_h, emb_f)
                                d_negative = euclidean_distances(emb_h, emb_f_inv)
                                
                                if d_negative > d_positive:
                                    print("d positive:", d_positive, "d negative:", d_negative, "--- True")
                                    batch_true_counter += 1
                                else:
                                     print("d positive:", d_positive, "d negative:", d_negative, "--- False")
                            
                            if j == (self.val_iter - 1):
                                print("d positive: %f ----- d negative: %f"%(d_positive, d_negative))                                    
                                

                                                       
                            
                            print('[%i]:%f'%(j,sess.run(loss_accumulator)))
                                
                            print("\n")                       
                            print("TOTAL TRUE: %i"%(batch_true_counter))

                            
                            
                        print('End:%f'%sess.run(loss_accumulator))

                        s,self.netLoss = sess.run(                        
                                [self.merged,self.average_loss],
                                    feed_dict = {
                                            self.is_training : False
                                    }                            
                                ) 

                        if (self.netLoss > initial_loss_val) and (i == 0):
                            initial_loss_val = self.netLoss
                        
                        if self.netLoss > loss_val:
                            loss_val = self.netLoss
                        
                        if (self.netLoss > initial_loss_val / 2) and (self.netLoss > loss_val):
                            for path in batch_filepaths_current:
                                print("[Valid] Anomaly Loss: %f Path: %s"%(self.netLoss, path))
                                
                        
                        writer_test.add_summary(s, i)
                        print('[Valid] [Epoch] (H): %i (F): %i Labels: %i [Iter]: %i [Loss]: %f'%(self.test_database.epoch1,self.test_database.epoch2,self.test_database.epoch_labels,i,self.netLoss))
                        print("\n")


                        sess.run([self.zero_op_loss])
                        




                    # =============================================================================
                    #   Train
                    # ============================================================================= 
                    for j in range(self.iterbatch):
                        batch_true_counter = 0 
                        batch_filepaths_current = []                         
                        img1, img2, lbl1, lbl2, batch_filepaths = self.train_database.read_batch()
                        
                        for path in batch_filepaths:
                            batch_filepaths_current.append(path)                        

                        _, _, _, herbarium_embeddings, field_embeddings = sess.run(
                                [self.collect_loss,self.accum_train_ops_front,self.accum_train_ops_end, herbarium_feat, field_feat],
                                        feed_dict = {x1 : img1,
                                                     x2 : img2, 
                                                     y1 : lbl1,
                                                     y2 : lbl2,
                                                     self.is_training : True
                                            
                                            }
                            )
                        
                        embs_h = herbarium_embeddings
                        embs_f = field_embeddings                                
                        embs_f_inverted = field_embeddings[::-1]
                        
                        for emb_h, emb_f, emb_f_inv in zip(embs_h, embs_f, embs_f_inverted):
                            emb_h = np.reshape(emb_h, (1,500))
                            emb_f = np.reshape(emb_f, (1,500))
                            emb_f_inv = np.reshape(emb_f_inv, (1,500))
                            
                            d_positive = euclidean_distances(emb_h, emb_f)
                            d_negative = euclidean_distances(emb_h, emb_f_inv)
                            
                            if d_negative > d_positive:
                                print("d positive:", d_positive, "d negative:", d_negative, "--- True")
                                batch_true_counter += 1
                            else:
                                 print("d positive:", d_positive, "d negative:", d_negative, "--- False")
                        
                        if j == (self.iterbatch - 1):
                            print("d positive: %f ----- d negative: %f"%(d_positive, d_negative))
                        
                        print("TOTAL TRUE: %i"%(batch_true_counter))



                    s,self.netLoss = sess.run(
                            [self.merged,self.average_loss],
                                feed_dict = {
                                        self.is_training : True
                                }                            
                            ) 

                    if (self.netLoss > initial_loss_train) and (i == 0):
                        initial_loss_train = self.netLoss
                    
                    if self.netLoss > loss_train:
                        loss_train = self.netLoss
                    
                    if (self.netLoss > initial_loss_train / 2) and (self.netLoss > loss_train):
                        for path in batch_filepaths:
                            print("[Train] Anomaly Loss: %f Path: %s"%(self.netLoss, path))

                    
                    writer_train.add_summary(s, i)
                    
                    sess.run([self.train_step_front])
                    sess.run([self.train_step_end])
                        
                    print('[Train] [Epoch] (H): %i (F): %i Labels: %i [Iter]: %i [Loss]: %f'%(self.train_database.epoch1,self.train_database.epoch2,self.train_database.epoch_labels,i,self.netLoss))
                    print("\n")


                    
                    if i % 5000 == 0:
                        saver.save(sess, os.path.join(self.save_dir,'%08i.ckpt'%i)) 
                    
                except KeyboardInterrupt:
                    print('Interrupt detected. Ending...')
                    break




                
            # =============================================================================
            #   Save model
            # =============================================================================
            saver.save(sess, os.path.join(self.save_dir,'final.ckpt')) 
            print('Model saved')




           

    
     

    
    
    
    
    
    
    
    
    
