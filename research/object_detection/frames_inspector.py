import pdb
import random
import numpy as np

import functools
import json
import os
import tensorflow as tf
import imp
import pickle

from object_detection import trainer
from object_detection import selection_funcs as sel
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

False_PN='FP'

data_dir='/datatmp/Experiments/Javad/tf/data/ILSVRC/'
#train_dir='/datatmp/Experiments/Javad/tf/model/'+False_PN+'_gtR1cycle1/'

eval_train_dir='/datatmp/Experiments/Javad/tf/model/R1cycle0/'+False_PN+'_gtR1cycle1eval_train/'
current_cycle_path='/datatmp/Experiments/Javad/tf/model/R1cycle0/'
next_cycle_path='/datatmp/Experiments/Javad/tf/model/'+False_PN+'_gtR1cycle1/'


#eval_train_dir='/datatmp/Experiments/Javad/tf/model/'+False_PN+'_gtR1cycle1/'+False_PN+'_gtR1cycle2eval_train/'
#current_cycle_path='/datatmp/Experiments/Javad/tf/model/'+False_PN+'_gtR1cycle1/'
#next_cycle_path='/datatmp/Experiments/Javad/tf/model/'+False_PN+'_gtR1cycle2/'

current_active_set=[]
next_active_set=[]
with open(current_cycle_path + 'active_set.txt', 'r') as f:
   for line in f:
      current_active_set.append(int(line))
with open(next_cycle_path + 'active_set.txt', 'r') as f:
   for line in f:
      next_active_set.append(int(line))
newly_added_frames=[f for f in next_active_set if f not in current_active_set]



data_info = {'data_dir': FLAGS.data_dir,
          'annotations_dir':'Annotations',
          'label_map_path': './data/imagenetvid_label_map.pbtxt',
          'set': 'train_150K_clean'}

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
    return dataset,videos
dataset,videos = get_dataset(data_info)

def normalize_box(box,w,h):
    """ Input: [ymin, xmin,ymax,xmax]
        Output: normalized by width and height of image
    """
    nbox = box.copy()
    nbox[:,0] = nbox[:,0]/h
    nbox[:,1] = nbox[:,1]/w
    nbox[:,2] = nbox[:,2]/h
    nbox[:,3] = nbox[:,3]/w
    return nbox

def augment_active_set(dataset,videos,active_set,num_neighbors=5):
    """ Augment set of indices in active_set by adding a given number of neighbors
    Arg:
        dataset: structure with information about each frames
        videos: list of video names
        active_set: list of indices of active_set
        num_neighbors: number of neighbors to include
    Returns:
        aug_active_set: augmented list of indices with neighbors
    """
    aug_active_set = []

    # We need to do this per video to keep limits in check
    for v in videos:
        frames_video = [f['idx'] for f in dataset if f['video'] == v]
        max_frame = np.max(frames_video)
        idx_videos_active_set = [idx for idx in frames_video if idx in active_set]
        idx_with_neighbors = [i for idx in idx_videos_active_set for i in range(idx-num_neighbors,idx+num_neighbors+1) if i >= 0 and i
         <= max_frame ]
        aug_active_set.extend(idx_with_neighbors)

    return aug_active_set


# loading detected boxes
if os.path.exists(eval_train_dir + 'detections.dat'):
            with open(eval_train_dir + 'detections.dat','rb') as infile:
            ###### pdb remove latinq
                detections = pickle.load(infile)
            	#detected_boxes = pickle.load(infile,encoding='latin1')


aug_active_set =  augment_active_set(dataset,videos,current_active_set,num_neighbors=5)
unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

save_tf_record(data_info,unlabeled_set)


BOXES = detections['boxes'] 
SCORES= detections['scores']
score_thresh=0.5
j=1

pdb.set_trace()

for f in newly_added_frames:
  
  anno_ind=unlabeled_set.index(f)
  ind=SCORES[anno_ind] > score_thresh # Extracting boxes with score greater than threshold
  boxes=np.array(BOXES[anno_ind])[ind,:]
  v=dataset[f]['video']
  video_dir = os.path.join(data_dir,'Data','VID','train',v)
  curr_im = Image.open(os.path.join(video_dir,dataset[f]['filename']))
  im_w,im_h = curr_im.size
  vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes,im_w,im_h))
  curr_im.save(data_dir+False_PN+'_samples'+'/'+str(j)+'_'+dataset[f]['filename'])
  print(v)
  print(dataset[f]['filename'])
  j+=1
	    
