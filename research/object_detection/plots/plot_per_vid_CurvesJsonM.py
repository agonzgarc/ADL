
import numpy as np
import json
import pdb
import os
import cv2
from PIL import Image
from object_detection.utils import visualization_utils as vis_utils
import matplotlib.pyplot as plt

data_dir = '/datatmp/Experiments/Javad/images/'
output_dir = '/datatmp/Experiments/Javad/output/'
save_name = 'coco_model_vs_cycle_model'
video_dir='/datatmp/Experiments/Javad/tf/data/ILSVRC/Data/VID/val/'

video_name='ILSVRC2015_val_00035008'

json_files = [
    '/datatmp/Experiments/Javad/tf/data/ILSVRC/coco_performances/',
    '/datatmp/Experiments/Javad/tf/data/ILSVRC/performances_1cycle/']

labels = ['coco_model','1cycle_model']
colors = ['r','b']
linestyles = ['-','-']

num_curves = 2
num_of_videos= len(os.listdir(json_files[1]))
mAP=np.zeros((num_of_videos,2))

#----------------------------mAP Comparison ---------------------------------------
"""
for i in range(num_curves):
   directory = json_files[i]
   vids=np.array(os.listdir(directory)) 
   for v in range(num_of_videos):
       vid_name=vids[v]
       with open(directory+vid_name) as f:
           data = json.load(f)
           mAP[v,i]=data['performance_on_vid'][0]

diff=mAP[:,1]-mAP[:,0]
print(diff)
print('videos that the model is less performant', vids[diff<-0.1]) 
xaxis=range(1,num_of_videos+1)
plt.plot(xaxis,mAP[:,0])
plt.plot(xaxis,mAP[:,1])
plt.legend(labels)

plt.xlabel('index of video')
plt.ylabel('mAP')
plt.savefig(output_dir[:-7]+save_name)
#plt.show()
"""
#---------------------Visualizing videos ---------------------------------------

def normalize_box(box,w,h):
    # Input: [ymin, xmin,ymax,xmax]
    # Output: normalized by width and height of image
    #pdb.set_trace()
    nbox = box.copy()
    nbox[:,0] = nbox[:,0]/h
    nbox[:,1] = nbox[:,1]/w
    nbox[:,2] = nbox[:,2]/h
    nbox[:,3] = nbox[:,3]/w
    #print(nbox)
    return nbox

for i in range(num_curves):
   with open(json_files[i]+video_name+'.json') as f:
      data = json.load(f)      
      num_frames=len(data['frame_id'])
      for ind in range(num_frames):   

          item=data['frame_id'][ind]           
	  frame_num=item.keys()[0]
          frame_id=item[frame_num]
          curr_im = Image.open(os.path.join(video_dir,video_name,frame_id))
          im_w,im_h = curr_im.size          
          
          gt_boxes=np.asarray(data['gt_boxes'][ind][frame_num])
	  boxes=np.asarray(data['boxes'][ind][frame_num])

          
          if boxes.any():
             #pdb.set_trace()
             #vis_utils.draw_bounding_boxes_on_image(curr_im,boxes)
             N_boxes=len(boxes)
             vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes,im_w,im_h))
	  if gt_boxes.any():
             #print(ind)
             #print(frame_id)
	     #vis_utils.draw_bounding_boxes_on_image(curr_im,gt_boxes,color='green')
             N_boxes=len(gt_boxes)
	     vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes,im_w,im_h),color='green')
          #pdb.set_trace()
          curr_im.save(data_dir+str(i)+'_'+frame_id)

# visualize concatenated images
with open(json_files[i]+video_name+'.json') as f:
   data = json.load(f)
   for ind in range(num_frames):
      item=data['frame_id'][ind]
      frame_num=item.keys()[0]
      frame_id=item[frame_num]
      img1 = cv2.imread(data_dir+str(0)+'_'+frame_id)
      img2 = cv2.imread(data_dir+str(1)+'_'+frame_id)
      vis = np.concatenate((img1, img2), axis=1)
      cv2.imwrite(output_dir+frame_id, vis)

