import pdb
import random
import numpy as np

import functools
import json
import os
import tensorflow as tf
import imp
import time

# Visualization
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from object_detection.utils import visualization_utils as vis_utils
from object_detection.utils import np_box_ops
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops


# Tracking module
import siamfc.siamese as siam
from siamfc.tracker import tracker_full_video
from siamfc.parse_arguments import parse_arguments
from siamfc.region_to_bbox import region_to_bbox



####### Auxiliary functions - consider placing them in a separate file
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

    # Convert to set to remove duplicates
    return list(set(aug_active_set))


def compute_entropy_with_threshold(predictions, threshold, measure='max'):
    """ Given a list of predictions (class scores with background), it computes
    the entropy of each prediction in the list
    Args:
        predictions: list of predictions. Each element corresponds to an image,
            containing a numpy nd array of shape (num windows, num_classes+1)
        threshold: minimum value for acceptable prediction
    Returns:
        entropies: list of the same dimension, each item is the summary entropy
            for the corresponding image
    """
    # Add more summary measures, now we only have max

    def softmax_pred(x):
        e = np.exp(x)
        return e/np.sum(e,axis=1,keepdims=True)

    def entropy(x):
        if len(x)>0:
            return np.sum(-x*np.log(x),axis=1)
        else:
            return -1

    all_sm = [softmax_pred(i)[1:] for i in predictions]
    all_sm_no_background = [i[:,1:] for i in all_sm]

    # Compute maximum score of each prediction
    max_scores = np.amax(all_sm_no_background,axis=2)
    dets_sm = [all_sm[i][max_scores[i]>threshold] for i in range(len(max_scores))]
    if measure == 'max':
        entropies = [np.max(entropy(d_sm)) for d_sm in dets_sm]
    elif measure == 'avg':
        entropies = [np.mean(entropy(d_sm)) for d_sm in dets_sm]

    return entropies


def filter_detections(boxes,scores,labels,thresh_detection = 0.5):
    idx_good_det = scores > thresh_detection
    return boxes[idx_good_det,:],scores[idx_good_det],labels[idx_good_det]

def convert_boxes_wh(box):
    """ Detection boxes come as [ymin,xmin,ymax,xmax]
        We need [x,y,w,h] for tracking
    """
    whbox = np.array([box[1],box[0],box[3]-box[1],box[2]-box[0]])
    return whbox

def convert_boxes_xy(box):
    """ Tracking results come as [x, y, w, h]
        Convert back to the API's [ymin,xmin,ymax,xmax]
    """
    return np.array([box[1],box[0],box[1]+box[3],box[0]+box[2]])


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


##################################################### End of auxiliary functions

def top_score_frames_selector(scores_videos,idx_videos,num_neighbors=5,budget=3200):

    number_of_vids=len(idx_videos)
    vid_length=[len(v) for v in idx_videos]
    max_vid_length=max(vid_length)

    SCORES=np.zeros((number_of_vids,max_vid_length))
    IDX=np.zeros((number_of_vids,max_vid_length),dtype=int)
    CANDIDATES=np.zeros((number_of_vids,max_vid_length),dtype=int)-1
    CANDIDATES_SC=np.zeros((number_of_vids,max_vid_length))-1

    for v in range(number_of_vids):

      idx=idx_videos[v]
      scores=scores_videos[v]
      shuffled_vec=[i for i in range(len(scores))]
      random.shuffle(shuffled_vec)
      SCORES[v,:]=np.pad(scores[shuffled_vec], (0,max_vid_length-len(scores)), 'constant', constant_values=-1)
      IDX[v,:]=np.pad(idx[shuffled_vec], (0,max_vid_length-len(idx)), 'constant', constant_values=-1)
    #print('--------------------------------------------------------------------------------')

    #--------------------------SORTING THE SCORES IN DESCENDING ORDER-----------------------
    sorted_SCORES=np.flip(np.sort(SCORES,axis=1),1)
    AUX=np.flip(np.argsort(SCORES,axis=1),1)  
    sorted_INDICES=np.zeros((number_of_vids,max_vid_length),dtype=int)
    for i in range(number_of_vids):
    	sorted_INDICES[i,:]=IDX[i,AUX[i,:]]
    #print('--------------------------------------------------------------------------------')


    for v in range(0,number_of_vids):
      iter=0
      sorted_indices=sorted_INDICES[v,:]
      sorted_scores=sorted_SCORES[v,:]
      idx_max=sorted_indices[0]
      score_max=sorted_scores[0]
      while(score_max>=0):
        #print('score_max= ',score_max)
        #print('idx_max= ',idx_max)
        #print('video= ',v)      
        #print('iter= ',iter)
        #print('sorted_indices= ',sorted_indices)
        #print('sorted_scores= ',sorted_scores)
        #print('idx_max= ',idx_max)
        #print('score_max= ',score_max)    
        CANDIDATES[v,iter]=idx_max
        CANDIDATES_SC[v,iter]=score_max
        left=max(idx_max-num_neighbors,0)
        right=min(idx_max+num_neighbors,max(sorted_indices))
        frames_to_remove=np.arange(left,right+1,1)
        #print('frames_to_remove= ',frames_to_remove)    
        #print('frames to remove= ', frames_to_remove)
        IND = np.in1d(sorted_indices, frames_to_remove) #intersection
        shrinked_indices=sorted_indices[~IND] # removing frames from indices
        shrinked_scores=sorted_scores[~IND] # removing frames from scores    
        #print('shrinked_indices= ',shrinked_indices)
        #print('shrinked_scores= ',shrinked_scores)
        sorted_scores=shrinked_scores
        sorted_indices=shrinked_indices
        #print('size of sorted indices= ',sorted_indices.size)
        if sorted_indices.size != 0:
           idx_max=sorted_indices[0]
           score_max=sorted_scores[0]
        else:
           #print('************************************************')
           break
        iter=iter+1    
        #print('************************************************')
    
    #----------------SELECTING FRAMES FROM TOP CANDIDATES------------------------
    scores_greater_than_zero=np.sum(np.array(CANDIDATES_SC) > 0)
    scrores_equal_to_zero=np.sum(np.array(CANDIDATES_SC) == 0)
    print('total number of frames containing at least 1 FP or FN= ', scores_greater_than_zero)
    print('total number of frames containing 0 FP or FN= ', scrores_equal_to_zero)


    with open('/data/users/javad/tf/data/ILSVRC/stat_data/frame_statistics'+str(number_of_vids)+'.txt','a') as myfile:
       myfile.write('scores_greater_than_zero= '+str(scores_greater_than_zero)) 
       myfile.write('scrores_equal_to_zero= '+str(scrores_equal_to_zero))
       myfile.write('======================================') 

    b=0    
    sel_idx=np.zeros(budget,dtype=int)-1


    for j in range(0,len(CANDIDATES[0])):    
        for i in range(0,len(CANDIDATES)):
            #print('i= ',i, ' j= ',j)
            if CANDIDATES_SC[i,j]>0:   # first pick the frames with scores > 0
                sel_idx[b]=CANDIDATES[i,j]
                b=b+1
                if b==budget:
                   break
        if b==budget:
           break

    if b < budget :   # if there is budget left pick frames with score 0
      for j in range(0,len(CANDIDATES[0])):    
        for i in range(0,len(CANDIDATES)):
            #print('i= ',i, ' j= ',j)
            if CANDIDATES_SC[i,j]== 0:
                sel_idx[b]=CANDIDATES[i,j]
                b=b+1
                if b==budget:
                   break
        if b==budget:
           break	
  
    indices=sel_idx[sel_idx>=0]
    #print(indices)
    #print('length of selected frames = ',len(indices))
    return indices                



def select_random(dataset,videos,active_set,budget=3200):

    # Random might start with an empty active_set (first cycle)
    if active_set:
        aug_active_set = augment_active_set(dataset,videos,active_set,num_neighbors=3)
    else:
        aug_active_set = active_set

    unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

    scores_videos = []
    idx_videos = []
    #num_frames = []

    for v in videos:
        print(v)
        frames = [f['idx'] for f in dataset if f['video'] == v and f['idx'] in unlabeled_set]
        if len(frames) > 0:
            random.shuffle(frames)
            idx_videos.append(np.asarray(frames))
            scores_videos.append(np.zeros(len(frames)))
            #num_frames.append(len(frames))

    indices=top_score_frames_selector(scores_videos, idx_videos,num_neighbors=5,budget=budget)
    return indices

# Pass unlabeled set as argument instead of recomputing here?
def select_least_confident(dataset,videos,active_set,detections,budget=3200,measure='max'):

        thresh_detection = 0.5

        # We have detections only for the unlabeled dataset, be careful with indexing
        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=3)
        unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        predictions = detections['scores']

        scores_videos = []
        idx_videos = []
        #num_frames = []

        t_start = time.time()

        for v in videos:

            # Select frames in current video
            frames = [f['idx'] for f in dataset if f['video'] == v]

            # Get only those that are not labeled
            frames = [f for f in frames if f in unlabeled_set]

            # If all frames of video are in active set, ignore video
            if len(frames) > 0:
                # Extract corresponding predictions
                det_frames = [predictions[unlabeled_set.index(f)] for f in frames]

                # Compute average frame confidence
                avg_conf = []
                for df in det_frames:
                    sel_dets = df[df > thresh_detection]

                    # Do inverse of least confidence --> selection prioritizes higher scores
                    if len(sel_dets) > 0:
                        if measure == 'avg':
                            acf = 1-sel_dets.mean()
                        elif measure == 'max':
                            acf = 1-sel_dets.max()
                        else:
                            raise ValueError('Summary measure error')
                    else:
                        acf = 0
                    avg_conf.append(acf)

                # Convert to array for easier processing
                avg_conf = np.asarray(avg_conf)

                # Add scores
                scores_videos.append(avg_conf)

                # Frames already contains list of global indices
                idx_videos.append(np.asarray(frames))

                # Save number of frames for padding purposes
                #num_frames.append(len(frames))

        elapsed_time = time.time() - t_start
        print("All videos processed in:{:.2f} seconds".format(elapsed_time))

        # Javad, call your function here
        indices=top_score_frames_selector(scores_videos, idx_videos,num_neighbors=5,budget=budget)

        return indices

def select_entropy(dataset,videos,active_set,detections,budget=3200):

        thresh_detection = 0.5

        # We have detections only for the labeled dataset, be careful with indexing
        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=3)
        unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        predictions = detections['scores_with_background']
        sel_predictions = detections['scores']

        scores_videos = []
        idx_videos = []
        #num_frames = []

        t_start = time.time()

        for v in videos:

            # Select frames in current video
            frames = [f['idx'] for f in dataset if f['video'] == v]

            # Get only those that are not labeled
            frames = [f for f in frames if f in unlabeled_set]

            # If all frames of video are in active set, ignore video
            if len(frames) > 0:
                # Extract corresponding predictions
                pred_frames = [predictions[unlabeled_set.index(f)] for f in frames]

                # Compute and summarize entropy
                ent = np.asarray(compute_entropy_with_threshold(pred_frames,thresh_detection,measure='avg'))

                # Add scores
                scores_videos.append(ent)
                # NO DETECTIONS now returns -1 --> change?

                # Frames already contains list of global indices
                idx_videos.append(np.asarray(frames))

                # Save number of frames for padding purposes
                #num_frames.append(len(frames))

        elapsed_time = time.time() - t_start
        print("All videos processed in:{:.2f} seconds".format(elapsed_time))

        indices=top_score_frames_selector(scores_videos, idx_videos,num_neighbors=5,budget=budget)
        return indices

"""

def select_FPN_PerVideo(dataset,videos,active_set,detections,groundtruth_boxes,cycle):
        #data_dir='/datatmp/Experiments/Javad/tf/data/ILSVRC'
    score_thresh=0.5
    iou_thresh=0.5 	    
        #indices = []
        scores_videos = []
        idx_videos = []
"""

def selectFnPerVideo(dataset,videos,active_set,detections,groundtruth_boxes,cycle, budget=3200):
	
	data_dir='/data/users/javad/tf/data/ILSVRC' #imagenet
	#data_dir='/data/datasets/synthia' #synthia

	score_thresh=0.5
	iou_thresh=0.75 	    
	scores_videos = []
	idx_videos = []

	aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=3)
	unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        # We have detections only for the labeled dataset, be careful with indexing
        #unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

	BOXES = detections['boxes'] 
	SCORES= detections['scores']
	gt_boxes = groundtruth_boxes['boxes']

	stat_data={}
	list_of_FNs=[]
	stat_data['FN_info']=[]

	for v in videos:
		# Select frames in current video
		frames = [f['idx'] for f in dataset if f['video'] == v and f['idx'] in unlabeled_set]
		# If all frames of video are in active set, ignore video
		if len(frames)>0:
			FN = np.zeros((len(frames)))
			j=0
			for f in frames:
				anno_ind=unlabeled_set.index(f)
				# Extracting boxes with score greater than threshold
				ind=SCORES[anno_ind] > score_thresh
				boxes=np.array(BOXES[anno_ind])[ind,:]		                
				# FN selection part                      
				if gt_boxes[anno_ind].any():	           		    
					if boxes.any():
						# Compute IOU between gt and detected bbox 
						iou_mat= np_box_ops.iou(gt_boxes[anno_ind], boxes)
						ind = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
						iter=0
						while(iou_mat[ind]>=iou_thresh):
							iou_mat[ind[0],:]=-1
							iou_mat[:,ind[1]]=-1
							iter=iter+1
							ind = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
						FN[j]=len(iou_mat)-iter
					else:
						FN[j]=len(gt_boxes[anno_ind])		
				j+=1               

			scores_videos.append(FN)
			idx_videos.append(np.asarray(frames))

        ##========================statistics ===============================================
			list_of_FNs.append(FN.tolist())
		stat_data['FN_info'].append({'video':v,'frames':len(frames),'scores':list_of_FNs})
	output_file = data_dir+'/stat_data/FN_stat_data_cycle'+str(cycle)+'.json'
	with open(output_file, 'w') as fn:
		json.dump(stat_data, fn)

		##========================visualization to check FNs================================
		"""
		for f in frames:
			for d in dataset:
				if d['idx']==f and d['video']==v:
					IndInDs=dataset.index(d)
					anno_ind=unlabeled_set.index(f)                                          
					video_dir = os.path.join(data_dir,'Data','VID','train',v) #imagenet
					#video_dir = os.path.join(data_dir,'train',v,'RGB') # synthia						
					curr_im = Image.open(os.path.join(video_dir,dataset[IndInDs]['filename']))
					im_w,im_h = curr_im.size
					vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes[anno_ind],im_w,im_h),color='green')
					ind=SCORES[anno_ind] > score_thresh # Extracting boxes with score greater than threshold
					boxes=np.array(BOXES[anno_ind])[ind,:]
					vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes,im_w,im_h))
					draw = ImageDraw.Draw(curr_im)
					curr_im.show()
					curr_im.save(data_dir+'/FN_samples'+'/FN_'+str(int(FN[frames.index(f)]))+'_'+dataset[IndInDs]['filename'])
		pdb.set_trace()
		"""
		##====================================================================================

	indices=top_score_frames_selector(scores_videos, idx_videos, num_neighbors=5, budget=budget)
	return indices


def selectFpPerVideo(dataset,videos,active_set,detections,groundtruth_boxes,cycle, budget=3200):
	
	#data_dir='/data/datasets/synthia' #synthia
	data_dir='/data/users/javad/tf/data/ILSVRC' #imagenet
	score_thresh=0.5
	iou_thresh=0.75 	    
	scores_videos = []
	idx_videos = []

	aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=3)
	unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        # We have detections only for the labeled dataset, be careful with indexing
        #unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

	BOXES = detections['boxes'] 
	SCORES= detections['scores']
	gt_boxes = groundtruth_boxes['boxes']

	stat_data={}
	list_of_FPs=[]
	stat_data['FP_info']=[]

	for v in videos:
		# Select frames in current video
		frames = [f['idx'] for f in dataset if f['video'] == v and f['idx'] in unlabeled_set]
		# If all frames of video are in active set, ignore video
		if len(frames)>0:
			FP = np.zeros((len(frames)))
			j=0
			for f in frames:
				anno_ind=unlabeled_set.index(f)
				# Extracting boxes with score greater than threshold
				ind=SCORES[anno_ind] > score_thresh
				boxes=np.array(BOXES[anno_ind])[ind,:]
				# FP selection part            
				if boxes.any(): # if the frame has detections with high score
					if gt_boxes[anno_ind].any():
						# Compute IOU between gt and detected bbox 
						iou_mat= np_box_ops.iou(gt_boxes[anno_ind], boxes)
						# Check if there are multiple detections for single groundtruth box
						ind = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
						iter=0
						while(iou_mat[ind]>=iou_thresh):
							iou_mat[ind[0],:]=-1
							iou_mat[:,ind[1]]=-1
							iter=iter+1
							ind = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
						FP[j]=len(iou_mat[0])-iter
					else:
						FP[j]=len(boxes)		                	
				j=j+1
			scores_videos.append(FP)
			idx_videos.append(np.asarray(frames))

        ##========================statistics ===============================================
			list_of_FPs.append(FP.tolist())
		stat_data['FP_info'].append({'video':v,'frames':len(frames),'scores':list_of_FPs})
	output_file = data_dir+'/stat_data/FP_stat_data_cycle'+str(cycle)+'.json'
	with open(output_file, 'w') as fp:
		json.dump(stat_data, fp)
	##========================visualization to check FPs================================
		"""
		for f in frames:
			for d in dataset:
				if d['idx']==f and d['video']==v:
					IndInDs=dataset.index(d)
					anno_ind=unlabeled_set.index(f)
					video_dir = os.path.join(data_dir,'Data','VID','train',v)# imagenet                                          
					#video_dir = os.path.join(data_dir,'train',v,'RGB') # synthia
					curr_im = Image.open(os.path.join(video_dir,dataset[IndInDs]['filename']))
					im_w,im_h = curr_im.size
					vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes[anno_ind],im_w,im_h),color='green')
					ind=SCORES[anno_ind] > score_thresh # Extracting boxes with score greater than threshold
					boxes=np.array(BOXES[anno_ind])[ind,:]
					vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes,im_w,im_h))
					draw = ImageDraw.Draw(curr_im)
					curr_im.show()
					curr_im.save(data_dir+'/FP_samples'+'/FP_'+str(int(FP[frames.index(f)]))+'_'+dataset[IndInDs]['filename'])
		pdb.set_trace()
		"""		
		##==================================================================================
	indices=top_score_frames_selector(scores_videos, idx_videos, num_neighbors=5, budget=budget)
	return indices

def select_FPN_PerVideo(dataset,videos,active_set,detections,groundtruth_boxes,cycle, budget=3200):
	
	#data_dir='/data/datasets/synthia' #synthia
	data_dir='/data/users/javad/tf/data/ILSVRC' #imagenet
	score_thresh=0.5
	iou_thresh=0.75 	    
	scores_videos = []
	idx_videos = []

	aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=3)
	unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        # We have detections only for the labeled dataset, be careful with indexing
        #unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

	BOXES = detections['boxes'] 
	SCORES= detections['scores']
	gt_boxes = groundtruth_boxes['boxes']

	stat_data={}
	list_of_FPNs=[]
	stat_data['FPN_info']=[]

	for v in videos:
		# Select frames in current video
		frames = [f['idx'] for f in dataset if f['video'] == v and f['idx'] in unlabeled_set]
		# If all frames of video are in active set, ignore video
		if len(frames)>0:
			FP = np.zeros((len(frames)))
			FN = np.zeros((len(frames)))
			j=0
			for f in frames:
				anno_ind=unlabeled_set.index(f)
				# Extracting boxes with score greater than threshold
				ind=SCORES[anno_ind] > score_thresh
				boxes=np.array(BOXES[anno_ind])[ind,:]

				# FP selection part            
				if boxes.any(): # if the frame has detections with high score
					if gt_boxes[anno_ind].any():
						# Compute IOU between gt and detected bbox 
						iou_mat= np_box_ops.iou(gt_boxes[anno_ind], boxes)
						# Check if there are multiple detections for single groundtruth box
						ind = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
						iter=0
						while(iou_mat[ind]>=iou_thresh):
							iou_mat[ind[0],:]=-1
							iou_mat[:,ind[1]]=-1
							iter=iter+1
							ind = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
						FP[j]=len(iou_mat[0])-iter
					else:
						FP[j]=len(boxes)
		                
				# FN selection part                      
				if gt_boxes[anno_ind].any():	           		    
					if boxes.any():
						# Compute IOU between gt and detected bbox 
						iou_mat= np_box_ops.iou(gt_boxes[anno_ind], boxes)
						ind = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
						iter=0
						while(iou_mat[ind]>=iou_thresh):
							iou_mat[ind[0],:]=-1
							iou_mat[:,ind[1]]=-1
							iter=iter+1
							ind = np.unravel_index(np.argmax(iou_mat, axis=None), iou_mat.shape)
						FN[j]=len(iou_mat)-iter
					else:
						FN[j]=len(gt_boxes[anno_ind])			
				j+=1               
			FPN=FP+FN
			scores_videos.append(FPN)
			idx_videos.append(np.asarray(frames))

        	##========================statistics ===============================================
			list_sum_of_FPNs.append(FPN.tolist())
		stat_data['FPN_info'].append({'video':v,'frames':len(frames),'scores':list_of_FPNs})
	output_file = data_dir+'/stat_data/FPN_stat_data_cycle'+str(cycle)+'.json'
	with open(output_file, 'w') as fpn:
		json.dump(stat_data, fpn)


		##========================visualization to check FPNs================================
		"""
		for f in frames:
			for d in dataset:
				if d['idx']==f and d['video']==v:
					IndInDs=dataset.index(d)
					anno_ind=unlabeled_set.index(f)                                          
					#video_dir = os.path.join(data_dir,'Data','VID','train',v) #imagenet
					video_dir = os.path.join(data_dir,'train',v,'RGB') # synthia
					curr_im = Image.open(os.path.join(video_dir,dataset[IndInDs]['filename']))
					im_w,im_h = curr_im.size
					vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes[anno_ind],im_w,im_h),color='green')
					ind=SCORES[anno_ind] > score_thresh # Extracting boxes with score greater than threshold
					boxes=np.array(BOXES[anno_ind])[ind,:]
					vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes,im_w,im_h))
					draw = ImageDraw.Draw(curr_im)
					curr_im.show()
					curr_im.save(data_dir+'/FPN_samples'+'/FPN_'+str(int(FPN[frames.index(f)]))+'_'+dataset[IndInDs]['filename'])
		pdb.set_trace()
		"""
		##====================================================================================
	indices=top_score_frames_selector(scores_videos, idx_videos, num_neighbors=5, budget=budget)
	return indices




def select_TCFP(dataset,videos,data_dir,candidate_set,evaluation_set,detections,dataset_name='imagenet',budget=3200):

    # Selector configuration
    threshold_track = 0.7
    num_frames_to_track = 3

    # Tracker configuration
    hp, evaluation, run, env, design = parse_arguments()
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1

    total_frames = len(candidate_set)
    overall_frame_counter = 0

    detected_boxes = detections['boxes']
    detected_scores = detections['scores']
    detected_labels = detections['labels']

    indices = []
    elapsed_time = []


    scores_videos = []
    idx_videos = []
    #num_frames = []

    t_start = time.time()


    # Get only top detections
    for i in range(len(detected_boxes)):
        detected_boxes[i],detected_scores[i],detected_labels[i] = filter_detections(detected_boxes[i],detected_scores[i],detected_labels[i])

    #gt_boxes = groundtruths['boxes']
    for v in videos:

        if dataset_name == 'imagenet':
            video_dir = os.path.join(data_dir,'Data','VID','train',v)
        elif dataset_name == 'synthia':
            video_dir = os.path.join(data_dir,'train',v,'RGB')
        else:
            raise ValueError('Dataset error: select imagenet or synthia')

        # Select frames in current video (even those with wrong GTs)
        frames = [[f['idx'],f['filename'],f['verified']] for f in dataset if f['video'] == v]

        # Get maximium index of frames in video
        idx_all_frames_video = [f[0] for f in frames]
        max_frame = np.max(idx_all_frames_video)

        # Get frames in video that are in the candidate set 
        frames_candidate = [f for f in frames if f[0] in candidate_set and f[2]]


        if len(frames_candidate) > 0:

            frame_counter = 0

            frame_list_video = []
            pos_x_video = []
            pos_y_video = []
            target_w_video = []
            target_h_video = []

            num_good_dets_video = []

            detections_neighbors_video = []

            for fu in frames_candidate:

                idx_frame_video = idx_all_frames_video.index(fu[0])

                frame_counter += 1
                overall_frame_counter += 1

                print("Adding information about frame in video: {}/{}, overall: {}/{}".format(frame_counter,len(frames_candidate),overall_frame_counter, total_frames))

                # Map frame in the evaluation set to obtain detections
                idx_evaluated_frames = evaluation_set.index(fu[0])

                # Get boxes for current frame
                boxes_frame = detected_boxes[idx_evaluated_frames]
                scores_frame = detected_scores[idx_evaluated_frames]
                labels_frame = detected_labels[idx_evaluated_frames]

                #gt_frame = gt_boxes[fu[0]]

                # Visualization of frame's GT and detections
                #curr_im = Image.open(os.path.join(video_dir,frames[idx_frame_video][1]))
                #im_w,im_h = curr_im.size
                ##vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_frame,im_w,im_h))
                #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes_frame[:50,:],im_w,im_h),color='green')
                #curr_im.show()


                num_good_dets = labels_frame.shape[0]

                num_good_dets_video.append(num_good_dets)

                for idx_det in range(num_good_dets):

                    ###### Common part for forward and backward tracking
                    # Convert [y x y x] to [y x w h]
                    curr_box = convert_boxes_wh(boxes_frame[idx_det])
                    pos_x, pos_y, target_w, target_h = region_to_bbox(curr_box)

                    # Append them twice, forward and backward
                    pos_x_video.append(pos_x)
                    pos_y_video.append(pos_y)
                    target_w_video.append(target_w)
                    target_h_video.append(target_h)

                    ###### Forward part
                    detections_neighbors_f = []
                    frame_list_f = [os.path.join(video_dir,frames[idx_frame_video][1])]

                    for t in range(1,num_frames_to_track+1):
                        idx_neighbor = idx_frame_video+t

                        # Check if neighbor still in video
                        if idx_neighbor < len(frames):
                            frame_list_f.append(os.path.join(video_dir,frames[idx_neighbor][1]))
                            # Take only those of the current class
                            detections_neighbors_f.append(detected_boxes[evaluation_set.index(fu[0]+t)][detected_labels[evaluation_set.index(fu[0]+t)] == labels_frame[idx_det]])

                    ###### Backward part
                    detections_neighbors_b = []
                    frame_list_b = [os.path.join(video_dir,frames[idx_frame_video][1])]

                    for t in range(1,num_frames_to_track+1):
                        idx_neighbor = idx_frame_video-t
                        if idx_neighbor >= 0:
                            frame_list_b.append(os.path.join(video_dir,frames[idx_neighbor][1]))
                            # Take only those of the current class
                            detections_neighbors_b.append(detected_boxes[evaluation_set.index(fu[0]-t)][detected_labels[evaluation_set.index(fu[0]-t)] == labels_frame[idx_det]])

                    # Save frames and detections in lists
                    frame_list_video.append([frame_list_f,frame_list_b])
                    detections_neighbors_video.append([detections_neighbors_f,detections_neighbors_b])



            # Track ALL frames and all detections in video with one call
            bboxes_video, elapsed_time_video = tracker_full_video(hp, run, design, frame_list_video, pos_x_video, pos_y_video, target_w_video, target_h_video, final_score_sz, env)

            elapsed_time.append(elapsed_time_video)

            # Computation of TC-FP score
            frame_counter = 0

            tc_scores = np.zeros(len(frames_candidate))

            for fu in frames_candidate:

                num_good_dets = num_good_dets_video[frame_counter]

                tc_sum_frame = np.zeros(num_good_dets)
                tc_neigh_frame = np.zeros(num_good_dets)

                for idx_det in range(num_good_dets):

                    # Return and delete from list first element, going in the same order as before
                    bboxes = bboxes_video.pop(0)
                    detections_neighbors = detections_neighbors_video.pop(0)
                    frame_list = frame_list_video.pop(0)

                    # First position contains forward
                    bboxes_f = bboxes[0]
                    frame_list_f = frame_list[0]
                    detections_neighbors_f = detections_neighbors[0]

                    for t in range(1,len(frame_list_f)):

                        # Visualize track and detections
                        #curr_im = Image.open(frame_list_f[t])
                        #im_w,im_h = curr_im.size
                        #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(convert_boxes_xy(bboxes_f[t]).reshape((1,4)),im_w,im_h))
                        #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(detections_neighbors_f[t-1],im_w,im_h),color='green')
                        #curr_im.show()

                        tc_neigh_frame[idx_det] += 1

                        # Check if tracked detection matches any detection in neighbor frame, if any
                        if len(detections_neighbors_f[t-1]) > 0:
                            ovTr = np_box_ops.iou(convert_boxes_xy(bboxes_f[t]).reshape((1,4)),detections_neighbors_f[t-1])
                            # Increment score if it does
                            if np.max(ovTr) > threshold_track:
                                tc_sum_frame[idx_det] += 1


                    # Second position contains backward
                    bboxes_b = bboxes[1]
                    frame_list_b = frame_list[1]
                    detections_neighbors_b = detections_neighbors[1]


                    for t in range(1,len(frame_list_b)):

                        ## Visualize track and detections
                        #curr_im = Image.open(frame_list_b[t])
                        #im_w,im_h = curr_im.size
                        #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(convert_boxes_xy(bboxes_b[t]).reshape((1,4)),im_w,im_h))
                        #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(detections_neighbors_b[t-1],im_w,im_h),color='green')
                        #curr_im.show()


                        tc_neigh_frame[idx_det] += 1

                        # Check if tracked detection matches any detection in neighbor frame, if any
                        if len(detections_neighbors_b[t-1]) > 0:
                            ovTr = np_box_ops.iou(convert_boxes_xy(bboxes_b[t]).reshape((1,4)),detections_neighbors_b[t-1])
                            # Increment score if it does
                            if np.max(ovTr) > threshold_track:
                                tc_sum_frame[idx_det] += 1


                # Compute and save mean score per frame
                if num_good_dets > 0:
                    # Score is normalized count
                    tc_scores[frame_counter] = 1 - np.mean(tc_sum_frame/tc_neigh_frame)
                else:
                    # Frames with no detections don't have TCFP score (inf so they aren't taken)
                    tc_scores[frame_counter] = -1

                frame_counter += 1


            scores_videos.append(tc_scores)
            idx_videos.append(np.asarray([fc[0] for fc in frames_candidate]))
            #num_frames.append(len(frames_candidate))

            print("Current average elapsed time per video: {:.2f}".format(np.mean(elapsed_time)))

    elapsed_time = time.time() - t_start
    print("All videos processed in: {:.2f} seconds".format(elapsed_time))

    # Call selection function
    indices=top_score_frames_selector(scores_videos, idx_videos,num_neighbors=5,budget=budget)
    return indices


