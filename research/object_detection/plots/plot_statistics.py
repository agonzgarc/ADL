from __future__ import division
import numpy as np
import json
import pdb
import os
import cv2
from PIL import Image
from object_detection.utils import visualization_utils as vis_utils
import matplotlib.pyplot as plt

FPN='FP'

json_dir = '/datatmp/Experiments/Javad/tf/data/ILSVRC/stat_data/'

for c in range(1,7):

  json_file=FPN+'_stat_data_cycle'+str(c)+'.json'
  rel_loc=[]
  with open(json_dir+json_file) as f:
      data = json.load(f)
      vid_length=[d['video_length'] for d in data[FPN+'_info']]
      print(len(vid_length))
      print('Number of videos without '+FPN+' in cycle '+str(c)+' = ',len(data['videos_wo_'+FPN]))
      
      for item in data[FPN+'_info']:
          rel_loc.append(item[FPN+'_loc']/item['video_length'])

#------------plot of where FN/FN samples fall in the video ----------------------

  plt.subplot(3,2,c)
  plt.xlabel('normalized video length')
  plt.ylabel('Number of '+FPN)
  plt.hist(rel_loc,bins=500,color='orange')
  plt.title('cycle '+str(c))

plt.subplots_adjust(hspace=0.9,  wspace = 0.9)
plt.savefig(json_dir+'location_in_videos')

#----------ploting the length of videos -----------------------------------------

"""
plt.xlabel('video index ')
plt.ylabel('length')
plt.hist(vid_length,bins=500 ,color = 'orange')
plt.axis([0, 800, 0, 50])
plt.grid(True)
plt.savefig(json_dir+'video_length') 
"""
