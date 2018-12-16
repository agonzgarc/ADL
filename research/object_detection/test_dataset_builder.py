


import pdb
import random
import numpy as np

import functools
import json
import os
import tensorflow as tf
import imp

from object_detection import trainer
from object_detection import evaluator_al as evaluator
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.save_subset_imagenetvid_tf_record import save_tf_record
from object_detection.utils import label_map_util
from object_detection.utils import np_box_ops
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops

from pycocotools import mask

from PIL import Image
from object_detection.utils import visualization_utils as vis_utils


tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)
#tf.logging.set_verbosity(tf.logging.WARN)

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/datatmp/Experiments/Javad/tf/data/ILSVRC/',
                    'Directory that contains data.')

FLAGS = flags.FLAGS

data_info = {'data_dir': FLAGS.data_dir,
          'annotations_dir':'Annotations',
          'label_map_path': './data/imagenetvid_label_map.pbtxt',
          'set': 'train_150K_clean'}

data_info['output_path'] = FLAGS.data_dir + 'imagenet_2nd_half_test_split' + '.record'

def get_dataset(data_info):
    """ Gathers information about the dataset given and stores it in a
    structure at the frame level.
    Args:
        data_info: dictionary with information about the dataset
    Returns:
        dataset: structure in form of list, each element corresponds to a
            frame and its a dictionary with multiple keys
        videos: list of videos
    """
    dataset = []
    path_file = os.path.join(data_info['data_dir'],'AL', data_info['set'] + '.txt')
    with open(path_file,'r') as pF:
        idx = 0
        for line in pF:
            # Separate frame path and clean annotation flag
            split_line = line.split(' ')
            # Remove trailing \n
            verified = True if split_line[1][:-1] == '1' else False
            path = split_line[0]
            split_path = path.split('/')
            filename = split_path[-1]
            video = split_path[-3]+'/'+split_path[-2]
            dataset.append({'idx':idx,'filename':filename,'video':video,'verified':verified})
            idx+=1
    videos = set([d['video'] for d in dataset])
    #return dataset, videos

#================================================================================
#   TRAIN/TEST SPLITTING
#================================================================================
    dataset_train=[]
    dataset_test=[]
    train_iter=0
    test_iter=0
    train_75K_clean = open(data_info['data_dir']+'AL/train_75K_clean.txt','w') 
    test_75K_clean = open(data_info['data_dir']+'AL/test_75K_clean.txt','w')     

    for v in videos:             

      frames = [[f['idx'],f['filename'],f['verified']] for f in dataset if f['video'] == v]

      if len(frames)>20:   # if number of frames in video is less than 20 don't split and consider it for train only

        cut=int(len(frames)/2)+1
        frames_train=frames[0:cut]
        frames_test=frames[cut:]

        for f in frames_train:
           dataset_train.append({'idx':train_iter,'filename':f[1],'video':v,'verified':f[2]})
           train_75K_clean.write(data_info['data_dir']+'Data/VID/train/'+v+'/'+f[1]+' '+str(int(f[2])))
           train_75K_clean.write('\n')
           train_iter+=1

        for f in frames_test:
           dataset_test.append({'idx':test_iter,'filename':f[1],'video':v,'verified':f[2]})
           test_75K_clean.write(data_info['data_dir']+'Data/VID/train/'+v+'/'+f[1]+' '+str(int(f[2])))
           test_75K_clean.write('\n')
           test_iter+=1

      else:
        for f in frames:
           dataset_train.append({'idx':train_iter,'filename':f[1],'video':v,'verified':f[2]})
           train_75K_clean.write(data_info['data_dir']+'Data/VID/train/'+v+'/'+f[1]+' '+str(int(f[2])))
           train_75K_clean.write('\n')
           train_iter+=1
          
    videos_train = set([d['video'] for d in dataset_train])  
    videos_test = set([d['video'] for d in dataset_test])  

    train_75K_clean.close()
    test_75K_clean.close()
    return dataset_test, videos_test

#================================================================================

dataset_test,videos_test = get_dataset(data_info)

indices=[f['idx'] for f in dataset_test]   
shoreted_indices=indices[0::5]
#pdb.set_trace()
save_tf_record(data_info,shoreted_indices)

