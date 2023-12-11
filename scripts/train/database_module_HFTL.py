# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:53:01 2022

@author: user
"""


import numpy as np
from PIL import Image
import cv2
import random
import copy


class database_module(object):
    def __init__(
                self,
                database_file1, 
                database_file2,
                batch1,
                batch2,
                input_size,
                numclasses1,
                numclasses2
            ):
        
        print("Initialising...")
        self.database_file1 = database_file1
        self.database_file2 = database_file2
        self.batch1 = batch1
        self.batch2 = batch2
        self.input_size = input_size
        self.numclasses1 = numclasses1
        self.numclasses2 = numclasses2
        
        self.load_data_list()

        
    def load_data_list(self):        
        self.database1_dict = {}    #  herbarium
        self.database2_dict = {}    #   field      
        
        
        # ----- Dataset 1 ----- #
        with open(self.database_file1,'r') as fid:
            lines = [x.strip() for x in fid.readlines()]
            
        self.data_paths1 = [x.split(' ')[0] for x in lines]          
        self.data_labels1 = [int(x.split(' ')[3]) for x in lines]

        for key, value in zip(self.data_labels1, self.data_paths1):
            if key not in self.database1_dict:
                self.database1_dict[key] = [] 

            self.database1_dict[key].append(value)
                
            
        # ----- Dataset 2 ----- #
        with open(self.database_file2,'r') as fid2:
            lines2 = [x.strip() for x in fid2.readlines()]
            
        self.data_paths2 = [x.split(' ')[0] for x in lines2]
         
        self.data_labels2 = [int(x.split(' ')[3]) for x in lines2]

        for key, value in zip(self.data_labels2, self.data_paths2):
            if key not in self.database2_dict:
                self.database2_dict[key] = [] 

            self.database2_dict[key].append(value)
      
        


        self.database1_dict_copy = copy.deepcopy(self.database1_dict)
        self.database2_dict_copy = copy.deepcopy(self.database2_dict)

        self.unique_labels1 = list(set(self.data_labels1))
        self.unique_labels2 = list(set(self.data_labels2))
        
        self.unique_labels_both = list(set(self.unique_labels1).intersection(self.unique_labels2))
        self.unique_labels_both_len = len(self.unique_labels_both)
        
        self.labels_idx = np.arange(self.unique_labels_both_len)
        self.labels_cursor = 0
        self.epoch1 = 0 #   herbarium
        self.epoch2 = 0 #   field
        self.epoch_labels = 0
        self.refilled_keys_list1 = []
        self.refilled_keys_list2 = []
        
        self.shuffle_unique_labels_cursor()
        
        
    def shuffle_unique_labels_cursor(self):
        print('shuffling')
        print(self.labels_idx[0:10])            
        np.random.shuffle(self.labels_idx)
        print(self.labels_idx[0:10])
        self.cursor = 0        
          

        
    def read_image(self, filepath):
        try:
            im = cv2.imread(filepath)
        
            if im is None:
               im = cv2.cvtColor(np.asarray(Image.open(filepath).convert('RGB')),cv2.COLOR_RGB2BGR)
            im = cv2.resize(im,(self.input_size[0:2]))
        
            if np.ndim(im) == 2:
                im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
                
            else:
                im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            
        except:
            pass

        return im


  
    
    def get_path(self, current_class, dictionary):        
        try:
            current_path = random.choice(dictionary.get(current_class))
        except:
            current_path = None
            
        return current_path
    
    
    def remove_dictionary_value(self, current_class, current_path, dictionary):
        #   Remove used path from dictionary
        for key, value in dictionary.items():
            if key == current_class:
                if current_path in value:
                    value.remove(current_path)
        return current_class, current_path                    

    
    def refill_dict1_key(self, key):
        self.database1_dict_copy[key] = copy.deepcopy(self.database1_dict[key])
    
    def refill_dict2_key(self, key):
        self.database2_dict_copy[key] = copy.deepcopy(self.database2_dict[key])
        
        
        
    def read_batch(self):        
        total_filepaths = []
        
        img1 = []
        img2 = []
        
        lbl1 = []
        lbl2 = []

        
        while len(total_filepaths) < self.batch1 + self.batch2:          

            # ----- Select random class ----- #
            current_class = self.unique_labels_both[self.labels_idx[self.cursor]]
                      
            # ----- Check dictionary availability ANCHOR (HERBARIUM) and POSITIVE (FIELD) ----- #
            class_dict1_len = len(self.database1_dict_copy[current_class])
            class_dict2_len = len(self.database2_dict_copy[current_class])

            remaining_filepaths = (self.batch1 + self.batch2) - len(total_filepaths)

            
            if (class_dict1_len >= 4) and (class_dict2_len >= 4) and (remaining_filepaths % 4*4 == 0):           
                i_iter = 4

            elif (class_dict1_len >= 2) and (class_dict2_len >= 2) and (remaining_filepaths % 2*4 == 0):
                i_iter = 2

            elif (class_dict1_len >= 1) and (class_dict2_len >= 1) and (remaining_filepaths % 1*4 == 0):  
                i_iter = 1

            else:
                i_iter = 0                
                
                if class_dict1_len < 1:  
                    #   Check dictionary len - ANCHOR (HERBARIUM)
                    self.refill_dict1_key(current_class)
                    self.refilled_keys_list1.append(current_class)                    


                        
                if class_dict2_len < 1:
                    #   Check dictionary len - POSITIVE (FIELD)
                    self.refill_dict2_key(current_class)
                    self.refilled_keys_list2.append(current_class)

            


            for i in range(i_iter):
                #   Get POSITIVE (FIELD)
                if len(lbl2) < self.batch2:
                    current_path2 = self.get_path(current_class, self.database2_dict_copy)
                    im2 = self.read_image(current_path2)
                    
                    if (current_path2 is not None) and (im2 is not None) and (len(total_filepaths) < self.batch1 + self.batch2):
                        current_class, current_path2 = self.remove_dictionary_value(current_class, current_path2, self.database2_dict_copy)
                        img2.append(im2)
                        lbl2.append(current_class)
                        total_filepaths.append(current_path2)

                        
                        
            class_dict2_len = len(self.database2_dict_copy[current_class])
            if class_dict2_len < 1:
                #   Check dictionary len - POSITIVE (FIELD)
                self.refill_dict2_key(current_class)
                self.refilled_keys_list2.append(current_class)


                        

            for i in range(i_iter):
                #   GET ANCHOR (HERBARIUM)
                if len(lbl1) < self.batch1:
                    current_path1 = self.get_path(current_class, self.database1_dict_copy)
                    im1 = self.read_image(current_path1)
                    
                    if (current_path1 is not None) and (im1 is not None) and (len(total_filepaths) < self.batch1 + self.batch2):
                        current_class, current_path1 = self.remove_dictionary_value(current_class, current_path1, self.database1_dict_copy)
                        img1.append(im1)
                        lbl1.append(current_class)
                        total_filepaths.append(current_path1)


            class_dict1_len = len(self.database1_dict_copy[current_class])
            if class_dict1_len < 1:  
                #   Check dictionary len - ANCHOR (HERBARIUM)
                self.refill_dict1_key(current_class)
                self.refilled_keys_list1.append(current_class)                    


                        
            
            self.cursor += 1
            if self.cursor >= self.unique_labels_both_len:
                self.shuffle_unique_labels_cursor()
                self.epoch_labels += 1
            
            check_epoch1 = all(elem in self.refilled_keys_list1 for elem in self.unique_labels1)
            if check_epoch1 is True:
                self.epoch1 +=1
                self.refilled_keys_list1 = []

            check_epoch2 = all(elem in self.refilled_keys_list2 for elem in self.unique_labels2)
            if check_epoch2 is True:
                self.epoch2 +=1
                self.refilled_keys_list2 = []
            
                    
                

            
        img1 = np.asarray(img1,dtype=np.float32)/255.0
        img2 = np.asarray(img2,dtype=np.float32)/255.0
        lbl1 = np.asarray(lbl1, dtype=np.int32)
        lbl2 = np.asarray(lbl2, dtype=np.int32)
            
        return (img1, img2, lbl1, lbl2, total_filepaths)










        
    
    