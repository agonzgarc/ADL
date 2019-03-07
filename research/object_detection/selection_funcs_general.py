import pdb
import random
import numpy as np

import functools
import json
import os
import tensorflow as tf
import imp
import time
import pickle
#import maxflow
from itertools import compress

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


def compute_entropy_with_threshold(predictions, threshold, measure='max',topk=3):
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
    elif measure == 'sum':
        entropies = [np.sum(entropy(d_sm)) for d_sm in dets_sm]
    elif measure == 'topk':
        topk_lists = [entropy(d_sm).tolist() if len(d_sm) > 0 else []  for d_sm in dets_sm]
        entropies = []
        for i in topk_lists:
            if len(i)>0:
                i.sort(reverse=True)
                entropies.append(np.mean(i[min(len(i)-1,topk)]))
            else:
                entropies.append(-1)
    else:
        raise ValueError('Summary measure error')
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


def top_score_frames_selector(scores_videos,idx_videos,data_dir,name,cycle,run,num_neighbors=5,budget=3200,thresh_video=0,disambiguity=None):


    if not os.path.exists(data_dir +'/frames_and_scores'):
         os.mkdir(data_dir +'/frames_and_scores')

    with open(data_dir +'/frames_and_scores/scores_videos_'+name+'cycle_'+str(cycle)+'_run_'+str(run)+'.dat','wb') as outfile_scores:
         pickle.dump(scores_videos,outfile_scores, protocol=pickle.HIGHEST_PROTOCOL)

    if disambiguity:
         with open(data_dir +'/frames_and_scores/disambiguity_'+name+'cycle_'+str(cycle)+'_run_'+str(run)+'.dat','wb') as outfile_scores:
             pickle.dump(disambiguity,outfile_scores, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_dir +'/frames_and_scores/idx_videos_'+name+'cycle_'+str(cycle)+'_run_'+str(run)+'.dat','wb') as outfile_videos_idx:
         pickle.dump(idx_videos,outfile_videos_idx, protocol=pickle.HIGHEST_PROTOCOL)

    """
    #----------------statistics before sorting and selecting candidates------------------------
    scores_greater_than_zero=np.sum(np.array(scores_videos) > 0)
    scores_equal_to_zero=np.sum(np.array(scores_videos) == 0)

    with open(data_dir +'/frames_and_scores/before_selection_frame_statistics_'+name+'run_'+str(run)+'_cycle_'+str(cycle)+'.txt','a') as myfile:
       myfile.write('scores_greater_than_zero= '+str(scores_greater_than_zero)+'\n') 
       myfile.write('scores_equal_to_zero= '+str(scores_equal_to_zero)+'\n')
       if disambiguity:
          myfile.write('scores_disambiguity= '+str(scores_disambiguity)+'\n')
       myfile.write('=================================================='+'\n') 
    """

    number_of_vids=len(idx_videos)
    vid_length=[len(v) for v in idx_videos]
    max_vid_length=max(vid_length)
    SCORES=np.zeros((number_of_vids,max_vid_length))
    IDX=np.zeros((number_of_vids,max_vid_length),dtype=int)
    CANDIDATES=np.zeros((number_of_vids,max_vid_length),dtype=int)-1
    CANDIDATES_SC=np.zeros((number_of_vids,max_vid_length))-1


    if disambiguity:
       DISAMBIGUITY=np.zeros((number_of_vids,max_vid_length))-1 
       DISAMBIGUITY_SC=np.zeros((number_of_vids,max_vid_length))-1


    for v in range(number_of_vids):
        idx=idx_videos[v]
        scores=scores_videos[v]
        shuffled_vec=[i for i in range(len(scores))]
        random.shuffle(shuffled_vec)
        SCORES[v,:]=np.pad(scores[shuffled_vec], (0,max_vid_length-len(scores)), 'constant', constant_values=-1)
        IDX[v,:]=np.pad(idx[shuffled_vec], (0,max_vid_length-len(idx)), 'constant', constant_values=-1)
        if disambiguity:
           disambig=disambiguity[v]
           DISAMBIGUITY[v,:]=np.pad(disambig[shuffled_vec], (0,max_vid_length-len(disambig)), 'constant', constant_values=-1)

    #--------------------------SORTING THE SCORES IN DESCENDING ORDER-----------------------
    sorted_SCORES=np.flip(np.sort(SCORES,axis=1),1)
    AUX=np.flip(np.argsort(SCORES,axis=1),1)  
    sorted_INDICES=np.zeros((number_of_vids,max_vid_length),dtype=int)
    if disambiguity:
        sorted_DISAMBIGUITY=np.zeros((number_of_vids,max_vid_length)) 
    #---------------------------------------------------------------------------------------

    for i in range(number_of_vids):
        sorted_INDICES[i,:]=IDX[i,AUX[i,:]]
        if disambiguity:
           sorted_DISAMBIGUITY[i,:]=DISAMBIGUITY[i,AUX[i,:]]

    for v in range(0,number_of_vids):

      iter=0
      sorted_indices=sorted_INDICES[v,:]
      sorted_scores=sorted_SCORES[v,:]
      if disambiguity:
          sorted_disamb=sorted_DISAMBIGUITY[v,:]
          idx_tie1=np.where(sorted_scores[0]==sorted_scores)
          if len(idx_tie1[0])>1:
             max_tie1=np.max(sorted_disamb[idx_tie1])
             idx_tie2=np.where(sorted_disamb[idx_tie1]==max_tie1)
             if len(idx_tie2[0])>1: 
                idx=np.random.choice(idx_tie2[0]) 
             else:
                idx=idx_tie2[0]     
          else: 
             idx=idx_tie1[0]
          score_max=sorted_scores[idx]
          idx_max=sorted_indices[idx]
      else:
          score_max=sorted_scores[0]
          idx_max=sorted_indices[0]

      while(score_max>=0):

        CANDIDATES[v,iter]=idx_max
        CANDIDATES_SC[v,iter]=score_max

        left=max(idx_max-num_neighbors,0)
        right=min(idx_max+num_neighbors,max(sorted_indices))
        frames_to_remove=np.arange(left,right+1,1)
        print('frames_to_remove= ', frames_to_remove)
        IND = np.in1d(sorted_indices, frames_to_remove) #intersection
        shrinked_indices=sorted_indices[~IND] # removing frames from indices
        shrinked_scores=sorted_scores[~IND] # removing frames from scores
        sorted_indices=shrinked_indices
        sorted_scores=shrinked_scores
        if disambiguity:
           shrinked_disamb=sorted_disamb[~IND]
           sorted_disamb=shrinked_disamb

        if sorted_indices.size != 0:
           if disambiguity:
              idx_tie1=np.where(sorted_scores[0]==sorted_scores)
              if len(idx_tie1[0])>1:
                 max_tie1=np.max(sorted_disamb[idx_tie1])
                 idx_tie2=np.where(sorted_disamb[idx_tie1]==max_tie1)
                 if len(idx_tie2[0])>1: 
                    idx=np.random.choice(idx_tie2[0]) 
                 else:
                    idx=idx_tie2[0]     
              else: 
                 idx=idx_tie1[0]
              score_max=sorted_scores[idx]
              idx_max=sorted_indices[idx]
           else:
              score_max=sorted_scores[0]
              idx_max=sorted_indices[0]
        else:
           break
        iter=iter+1    
    
    #----------------statistics after sorting and selection and removing of neighbors------------------------

    """
    if disambiguity:	
       A=np.array(CANDIDATES_SC) > 0
       DISAMBIGUITY_SC[A]

    scores_greater_than_zero=np.sum(np.array(CANDIDATES_SC) > 0)
    scores_equal_to_zero=np.sum(np.array(CANDIDATES_SC) == 0)

    with open(data_dir +'/frames_and_scores/after_selection_frame_statistics_'+name+'run_'+str(run)+'_cycle_'+str(cycle)+'.txt','a') as myfile:
       myfile.write('scores_greater_than_zero= '+str(scores_greater_than_zero)+'\n') 
       myfile.write('scores_equal_to_zero= '+str(scores_equal_to_zero)+'\n')
       if disambiguity:
          myfile.write('scores_equal_to_zero= '+str(scores_equal_to_zero)+'\n')
       myfile.write('=================================================='+'\n') 
    """

    b=0    
    sel_idx=np.zeros(budget,dtype=int)-1
    for j in range(0,len(CANDIDATES[0])):    
        for i in range(0,len(CANDIDATES)):
            #print('i= ',i, ' j= ',j)
            if CANDIDATES_SC[i,j]>thresh_video:   # first pick the frames with scores > 0
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
            if CANDIDATES_SC[i,j] == thresh_video:
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



def select_random(dataset,videos,active_set,data_dir='', name='emptyName', cycle=1, run=1,budget=3200,neighbors_across=3,neighbors_in=5):

    # Random might start with an empty active_set (first cycle)
    if active_set:
        aug_active_set = augment_active_set(dataset,videos,active_set,num_neighbors=neighbors_across)
    else:
        aug_active_set = active_set

    unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

    scores_videos = []
    idx_videos = []

    for v in videos:
        print(v)
        frames = [f['idx'] for f in dataset if f['video'] == v and f['idx'] in unlabeled_set]
        if len(frames) > 0:
            random.shuffle(frames)
            idx_videos.append(np.asarray(frames))
            scores_videos.append(np.zeros(len(frames)))
            #num_frames.append(len(frames))

    indices=top_score_frames_selector(scores_videos, idx_videos, data_dir=data_dir, name=name, cycle=cycle, run=run, num_neighbors=neighbors_in, budget=budget)
    return indices

# Pass unlabeled set as argument instead of recomputing here?
def select_least_confident(dataset,videos,active_set, detections, data_dir='', name='emptyName', cycle=1, run=1, budget=3200, neighbors_across=3, neighbors_in=5,measure='max',topk=3):

        thresh_detection = 0.5

        # We have detections only for the unlabeled dataset, be careful with indexing
        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=neighbors_across)
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
                        elif measure == 'sum':
                            acf = len(sel_dets)-sel_dets.sum()
                        elif measure == 'topk':
                            sel_dets = list(sel_dets)
                            sel_dets.sort(reverse=True)
                            acf = 1 - np.mean(sel_dets[:min(topk,len(sel_dets))])
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
        indices=top_score_frames_selector(scores_videos, idx_videos, data_dir=data_dir, name=name, cycle=cycle, run=run, num_neighbors=neighbors_in, budget=budget)

        return indices

# Pass unlabeled set as argument instead of recomputing here?
def select_marginal(dataset,videos,active_set, detections, data_dir='', name='emptyName', cycle=1, run=1, budget=3200, neighbors_across=3, neighbors_in=5,measure='max',topk=3):

        thresh_detection = 0.5

        # We have detections only for the unlabeled dataset, be careful with indexing
        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=neighbors_across)
        unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        predictions = detections['scores']
        predictions = detections['scores_with_background']

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
                        elif measure == 'topk':
                            sel_dets = list(sel_dets)
                            sel_dets.sort(reverse=True)
                            acf = 1 - np.mean(sel_dets[:min(topk,len(sel_dets))])
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
        indices=top_score_frames_selector(scores_videos, idx_videos, data_dir=data_dir, name=name, cycle=cycle, run=run, num_neighbors=neighbors_in, budget=budget)

        return indices

def select_entropy(dataset,videos,active_set,detections,data_dir='', name='emptyName', cycle=1, run=1,budget=3200, neighbors_across=3, neighbors_in=5,measure='max',topk=3):

        thresh_detection = 0.5

        # We have detections only for the labeled dataset, be careful with indexing
        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=neighbors_across)
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
                ent = np.asarray(compute_entropy_with_threshold(pred_frames,thresh_detection,measure=measure,topk=topk))

                # Add scores
                scores_videos.append(ent)
                # NO DETECTIONS now returns -1 --> change?

                # Frames already contains list of global indices
                idx_videos.append(np.asarray(frames))

                # Save number of frames for padding purposes
                #num_frames.append(len(frames))

        elapsed_time = time.time() - t_start
        print("All videos processed in:{:.2f} seconds".format(elapsed_time))

        indices=top_score_frames_selector(scores_videos, idx_videos, data_dir=data_dir, name=name, cycle=cycle, run=run, num_neighbors=neighbors_in, budget=budget)
        return indices



def select_FP_FN_FPN_PerVideo(dataset,videos,active_set,detections,groundtruth_boxes,data_dir,name,cycle,run,budget=3200):
	
	score_thresh=0.5
	iou_thresh=0.75 	    
	scores_videos = []
	idx_videos = []
	disambiguity=[]

	aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=3)
	unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        # We have detections only for the labeled dataset, be careful with indexing
        #unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

	BOXES = detections['boxes'] 
	SCORES= detections['scores']
	LABELS = detections['labels']
	gt_boxes = groundtruth_boxes['boxes']
	gt_labels = groundtruth_boxes['labels']
	

	stat_data={}
	list_of_FNs=[]
	list_of_FPs=[]
	stat_data['FN_info']=[]
	stat_data['FP_info']=[]

	for v in videos: 

		#if v=='test5_14segs_weather_0_spawn_0_roadTexture_2_P_None_C_None_B_None_WC_None/24-10-2018_20-30-41': # "cyclist" video in val set
		#if v=='test5_28segs_weather_4_spawn_1_roadTexture_2_P_None_C_None_B_None_WC_None/24-10-2018_22-38-23': # "bicycle" video in val set
		#if v=='test5_14segs_weather_2_spawn_1_roadTexture_1_P_None_C_None_B_None_WC_None/24-10-2018_21-27-06':
		#if v=='test5_15segs_weather_0_spawn_1_roadTexture_0_P_None_C_None_B_None_WC_None/24-10-2018_20-32-51':
		#	print('yes !')		
		#else:
		#	print('no !')
		#	continue


		# Select frames in current video
		frames = [f['idx'] for f in dataset if f['video'] == v and f['idx'] in unlabeled_set]
		# If all frames of video are in active set, ignore video
		if len(frames)>0:
			FN = np.zeros((len(frames)))
			FP = np.zeros((len(frames)))
			TP = np.zeros((len(frames)))
			j=0
			for f in frames:
				anno_ind=unlabeled_set.index(f)
				# Extracting boxes with score greater than threshold
				ind=SCORES[anno_ind] > score_thresh
				boxes=np.array(BOXES[anno_ind])[ind,:]
				labels=LABELS[anno_ind][ind]
				gt_labels_frame_f=np.copy(gt_labels[anno_ind])
				gt_boxes_frame_f=np.copy(gt_boxes[anno_ind])

				if gt_boxes_frame_f.any():	           		    
					if boxes.any():						
						for it in range(0,len(boxes)):							
							box=boxes[[it]] # maximum score box
							iou_vec=np_box_ops.iou(gt_boxes_frame_f, box)
							idx_max=np.argmax(iou_vec)
							if iou_vec[idx_max]>=iou_thresh:
								if labels[it]==gt_labels_frame_f[idx_max]: # box label matches gt label ?
									gt_boxes_frame_f[idx_max,:]=-1
									gt_labels_frame_f[idx_max]=-1
									TP[j]+=1
								else : 
									FP[j]+=1
							else:
								FP[j]+=1

						FN[j]=sum(gt_labels_frame_f>-1)
		
					else:
						FN[j]=len(gt_boxes_frame_f)					
				else:
					FP[j]=len(boxes)					                
				j+=1               

			if 'FP_gt' in name:
				scores_videos.append(FP)
				idx_videos.append(np.asarray(frames))
			elif 'FN_gt' in name: 
				scores_videos.append(FN)
				idx_videos.append(np.asarray(frames))			
			elif 'FPN' in name: 
				scores_videos.append(FP+FN)
				disambiguity.append(FN)
				idx_videos.append(np.asarray(frames))	
		"""
		##========================visualization to check FN/FP ================================		
		for f in frames:	
					# -----------------------------------plotting the groundtruth boxes ------------------------------------
					IndInDs=f
					#pdb.set_trace()
					anno_ind=unlabeled_set.index(f)                                          
					#video_dir = os.path.join(data_dir,'Data','VID','train',v) #imagenet
					video_dir = os.path.join(data_dir,'val',v,'RGB') # synthia						
					curr_im = Image.open(os.path.join(video_dir,dataset[IndInDs]['filename']))
					im_w,im_h = curr_im.size
					for l in range(len(gt_labels[anno_ind])):  # plotting the groundtruth boxes
						if gt_labels[anno_ind][l]==-1:
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes[anno_ind][[l]],im_w,im_h),color='sienna')
						elif gt_labels[anno_ind][l]==0: #car
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes[anno_ind][[l]],im_w,im_h),color='yellow')
						elif gt_labels[anno_ind][l]==1: #pedestrian
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes[anno_ind][[l]],im_w,im_h),color='lightpink')
						elif gt_labels[anno_ind][l]==2: #bicycle
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes[anno_ind][[l]],im_w,im_h),color='springgreen')
						elif gt_labels[anno_ind][l]==3: #cyclist
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes[anno_ind][[l]],im_w,im_h),color='sandybrown')
						elif gt_labels[anno_ind][l]==4: #truck
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes[anno_ind][[l]],im_w,im_h),color='orange')
						else :
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_boxes[anno_ind][[l]],im_w,im_h),color='skyblue')

					# -----------------------------------plotting the detection boxes ------------------------------------

					ind=SCORES[anno_ind] > score_thresh # Extracting boxes with score greater than threshold
					boxes=np.array(BOXES[anno_ind])[ind,:]
					labels=LABELS[anno_ind][ind]
					for l in range(len(labels)):   
						if labels[l]==-1:
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes[[l]],im_w,im_h),color='sienna',thickness=1)
						elif labels[l]==0: #car
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes[[l]],im_w,im_h),color='yellow',thickness=1)
						elif labels[l]==1: #pedestrian
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes[[l]],im_w,im_h),color='lightpink',thickness=1)
						elif labels[l]==2: #bicycle
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes[[l]],im_w,im_h),color='springgreen',thickness=1)
						elif labels[l]==3: #cyclist
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes[[l]],im_w,im_h),color='sandybrown',thickness=1)
						elif labels[l]==4: #truck
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes[[l]],im_w,im_h),color='orange',thickness=1)
						else :
							vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes[[l]],im_w,im_h),color='skyblue',thickness=1)

					draw = ImageDraw.Draw(curr_im)
					curr_im.show()
					curr_im.save(data_dir+'/Samples/'+dataset[IndInDs]['filename'][:-4]+'_FN_'+str(int(FN[frames.index(f)]))+'_FP_'\
					+str(int(FP[frames.index(f)]))+'_TP_'+str(int(TP[frames.index(f)]))+'.png')

		pdb.set_trace()
		"""

	indices=top_score_frames_selector(scores_videos, idx_videos, data_dir=data_dir, name=name, cycle=cycle, run=run, num_neighbors=5, budget=budget, disambiguity=disambiguity)
	return indices



#def select_TCFP(dataset,videos,data_dir,candidate_set,evaluation_set,detections,dataset_name='imagenet',budget=3200):

    ## Selector configuration
    #threshold_track = 0.7
    #num_frames_to_track = 3

    ## Tracker configuration
    #hp, evaluation, run, env, design = parse_arguments()
    #final_score_sz = hp.response_up * (design.score_sz - 1) + 1

    #total_frames = len(candidate_set)
    #overall_frame_counter = 0

    #detected_boxes = detections['boxes']
    #detected_scores = detections['scores']
    #detected_labels = detections['labels']

    #indices = []
    #elapsed_time = []


    #scores_videos = []
    #idx_videos = []
    ##num_frames = []

    #t_start = time.time()


    ## Get only top detections
    #for i in range(len(detected_boxes)):
        #detected_boxes[i],detected_scores[i],detected_labels[i] = filter_detections(detected_boxes[i],detected_scores[i],detected_labels[i])

    ##gt_boxes = groundtruths['boxes']
    #for v in videos:

        #if dataset_name == 'imagenet':
            #video_dir = os.path.join(data_dir,'Data','VID','train',v)
        #elif dataset_name == 'synthia':
            #video_dir = os.path.join(data_dir,'train',v,'RGB')
        #else:
            #raise ValueError('Dataset error: select imagenet or synthia')

        ## Select frames in current video (even those with wrong GTs)
        #frames = [[f['idx'],f['filename'],f['verified']] for f in dataset if f['video'] == v]

        ## Get maximium index of frames in video
        #idx_all_frames_video = [f[0] for f in frames]
        #max_frame = np.max(idx_all_frames_video)

        ## Get frames in video that are in the candidate set 
        #frames_candidate = [f for f in frames if f[0] in candidate_set and f[2]]


        #if len(frames_candidate) > 0:

            #frame_counter = 0

            #frame_list_video = []
            #pos_x_video = []
            #pos_y_video = []
            #target_w_video = []
            #target_h_video = []

            #num_good_dets_video = []

            #detections_neighbors_video = []

            #num_good_dets_video = []

            #for fu in frames_candidate:

                #idx_frame_video = idx_all_frames_video.index(fu[0])

                #frame_counter += 1
                #overall_frame_counter += 1

                #print("Adding information about frame in video: {}/{}, overall: {}/{}".format(frame_counter,len(frames_candidate),overall_frame_counter, total_frames))

                ## Map frame in the evaluation set to obtain detections
                #idx_evaluated_frames = evaluation_set.index(fu[0])

                ## Get boxes for current frame
                #boxes_frame = detected_boxes[idx_evaluated_frames]
                #scores_frame = detected_scores[idx_evaluated_frames]
                #labels_frame = detected_labels[idx_evaluated_frames]

                ##gt_frame = gt_boxes[fu[0]]

                ## Visualization of frame's GT and detections
                ##curr_im = Image.open(os.path.join(video_dir,frames[idx_frame_video][1]))
                ##im_w,im_h = curr_im.size
                ###vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_frame,im_w,im_h))
                ##vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(boxes_frame[:50,:],im_w,im_h),color='green')
                ##curr_im.show()


                #num_good_dets = labels_frame.shape[0]

                #num_good_dets_video.append(num_good_dets)

                #for idx_det in range(num_good_dets):

                    ####### Common part for forward and backward tracking
                    ## Convert [y x y x] to [y x w h]
                    #curr_box = convert_boxes_wh(boxes_frame[idx_det])
                    #pos_x, pos_y, target_w, target_h = region_to_bbox(curr_box)

                    ## Append them twice, forward and backward
                    #pos_x_video.append(pos_x)
                    #pos_y_video.append(pos_y)
                    #target_w_video.append(target_w)
                    #target_h_video.append(target_h)

                    ####### Forward part
                    #detections_neighbors_f = []
                    #frame_list_f = [os.path.join(video_dir,frames[idx_frame_video][1])]

                    #for t in range(1,num_frames_to_track+1):
                        #idx_neighbor = idx_frame_video+t

                        ## Check if neighbor still in video
                        #if idx_neighbor < len(frames):
                            #frame_list_f.append(os.path.join(video_dir,frames[idx_neighbor][1]))
                            ## Take only those of the current class
                            #detections_neighbors_f.append(detected_boxes[evaluation_set.index(fu[0]+t)][detected_labels[evaluation_set.index(fu[0]+t)] == labels_frame[idx_det]])

                    ####### Backward part
                    #detections_neighbors_b = []
                    #frame_list_b = [os.path.join(video_dir,frames[idx_frame_video][1])]

                    #for t in range(1,num_frames_to_track+1):
                        #idx_neighbor = idx_frame_video-t
                        #if idx_neighbor >= 0:
                            #frame_list_b.append(os.path.join(video_dir,frames[idx_neighbor][1]))
                            ## Take only those of the current class
                            #detections_neighbors_b.append(detected_boxes[evaluation_set.index(fu[0]-t)][detected_labels[evaluation_set.index(fu[0]-t)] == labels_frame[idx_det]])

                    ## Save frames and detections in lists
                    #frame_list_video.append([frame_list_f,frame_list_b])
                    #detections_neighbors_video.append([detections_neighbors_f,detections_neighbors_b])



            ## Track ALL frames and all detections in video with one call
            #bboxes_video, elapsed_time_video = tracker_full_video(hp, run, design, frame_list_video, pos_x_video, pos_y_video, target_w_video, target_h_video, final_score_sz, env)

            #elapsed_time.append(elapsed_time_video)

            ## Computation of TC-FP score
            #frame_counter = 0

            #tc_scores = np.zeros(len(frames_candidate))

            #for fu in frames_candidate:

                #num_good_dets = num_good_dets_video[frame_counter]

                #tc_sum_frame = np.zeros(num_good_dets)
                #tc_neigh_frame = np.zeros(num_good_dets)

                #for idx_det in range(num_good_dets):

                    ## Return and delete from list first element, going in the same order as before
                    #bboxes = bboxes_video.pop(0)
                    #detections_neighbors = detections_neighbors_video.pop(0)
                    #frame_list = frame_list_video.pop(0)

                    ## First position contains forward
                    #bboxes_f = bboxes[0]
                    #frame_list_f = frame_list[0]
                    #detections_neighbors_f = detections_neighbors[0]

                    #for t in range(1,len(frame_list_f)):

                        ## Visualize track and detections
                        ##curr_im = Image.open(frame_list_f[t])
                        ##im_w,im_h = curr_im.size
                        ##vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(convert_boxes_xy(bboxes_f[t]).reshape((1,4)),im_w,im_h))
                        ##vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(detections_neighbors_f[t-1],im_w,im_h),color='green')
                        ##curr_im.show()

                        #tc_neigh_frame[idx_det] += 1

                        ## Check if tracked detection matches any detection in neighbor frame, if any
                        #if len(detections_neighbors_f[t-1]) > 0:
                            #ovTr = np_box_ops.iou(convert_boxes_xy(bboxes_f[t]).reshape((1,4)),detections_neighbors_f[t-1])
                            ## Increment score if it does
                            #if np.max(ovTr) > threshold_track:
                                #tc_sum_frame[idx_det] += 1


                    ## Second position contains backward
                    #bboxes_b = bboxes[1]
                    #frame_list_b = frame_list[1]
                    #detections_neighbors_b = detections_neighbors[1]


                    #for t in range(1,len(frame_list_b)):

                        ### Visualize track and detections
                        ##curr_im = Image.open(frame_list_b[t])
                        ##im_w,im_h = curr_im.size
                        ##vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(convert_boxes_xy(bboxes_b[t]).reshape((1,4)),im_w,im_h))
                        ##vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(detections_neighbors_b[t-1],im_w,im_h),color='green')
                        ##curr_im.show()


                        #tc_neigh_frame[idx_det] += 1

                        ## Check if tracked detection matches any detection in neighbor frame, if any
                        #if len(detections_neighbors_b[t-1]) > 0:
                            #ovTr = np_box_ops.iou(convert_boxes_xy(bboxes_b[t]).reshape((1,4)),detections_neighbors_b[t-1])
                            ## Increment score if it does
                            #if np.max(ovTr) > threshold_track:
                                #tc_sum_frame[idx_det] += 1


                ## Compute and save mean score per frame
                #if num_good_dets > 0:
                    ## Score is normalized count
                    #tc_scores[frame_counter] = 1 - np.mean(tc_sum_frame/tc_neigh_frame)
                #else:
                    ## Frames with no detections don't have TCFP score (inf so they aren't taken)
                    #tc_scores[frame_counter] = -1

                #frame_counter += 1


            #scores_videos.append(tc_scores)
            #idx_videos.append(np.asarray([fc[0] for fc in frames_candidate]))
            ##num_frames.append(len(frames_candidate))

            #print("Current average elapsed time per video: {:.2f}".format(np.mean(elapsed_time)))

    #elapsed_time = time.time() - t_start
    #print("All videos processed in: {:.2f} seconds".format(elapsed_time))

    ## Call selection function
    #indices=top_score_frames_selector(scores_videos, idx_videos,num_neighbors=5,budget=budget)
    #return indices


def select_GraphTC(dataset,videos,candidate_set,evaluation_set,detections,dataset_name='imagenet', data_dir='.', name='emptyName', cycle=1, run=1, neighbors_across=3,
                   neighbors_in=5,budget=3200,mode='FP'):

    # Selector configuration
    threshold_track = 0.7
    num_frames_to_track = 3

    # Tracker configuration
    hp, evaluation, run, env, design = parse_arguments()
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1

    total_frames = len(evaluation_set)
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

        # Get frames in video that are in the candidate set 
        frames_graph = [f for f in frames if f[0] in evaluation_set]
        frames_candidate = [f for f in frames if f[0] in candidate_set and f[2]]

        # Get maximium index of frames in video
        idx_all_frames_video = [f[0] for f in frames_graph]
        max_frame = np.max(idx_all_frames_video)

        if len(frames_candidate) > 0:

            frame_counter = 0

            frame_list_video = []
            frame_paths = []

            pos_x_video = []
            pos_y_video = []
            target_w_video = []
            target_h_video = []

            num_good_dets_video = []

            detections_neighbors_video = []

            local_dets = [[] for i in range(len(frames_graph))]

            for fu in frames_graph:

                frame_paths.append(os.path.join(video_dir,fu[1]))

                idx_frame_video = idx_all_frames_video.index(fu[0])


                print("Adding information about frame in video: {}/{}, overall: {}/{}".format(frame_counter+1,len(frames_graph),overall_frame_counter, total_frames))

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

                    # Save detection information as dictionary
                    local_dets[frame_counter].append({'idx':idx_det,'box':boxes_frame[idx_det],'score':scores_frame[idx_det],'label':labels_frame[idx_det]})

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
                    #detections_neighbors_f = []
                    frame_list_f = [os.path.join(video_dir,frames[idx_frame_video][1])]

                    for t in range(1,num_frames_to_track+1):
                        idx_neighbor = idx_frame_video+t

                        # Check if neighbor still in video
                        if (idx_neighbor < len(frames_graph)) and (abs(fu[0]-frames_graph[idx_neighbor][0])<num_frames_to_track+2) :
                            frame_list_f.append(os.path.join(video_dir,frames_graph[idx_neighbor][1]))
                            # Take only those of the current class
                            #detections_neighbors_f.append(detected_boxes[evaluation_set.index(fu[0]+t)][detected_labels[evaluation_set.index(fu[0]+t)] == labels_frame[idx_det]])

                    ###### Backward part
                    #detections_neighbors_b = []
                    frame_list_b = [os.path.join(video_dir,frames_graph[idx_frame_video][1])]

                    for t in range(1,num_frames_to_track+1):
                        idx_neighbor = idx_frame_video-t
                        #if idx_neighbor >= 0:
                        if (idx_neighbor >= 0) and (abs(fu[0]-frames_graph[idx_neighbor][0])<num_frames_to_track+2):
                            frame_list_b.append(os.path.join(video_dir,frames_graph[idx_neighbor][1]))
                            # Take only those of the current class
                            #detections_neighbors_b.append(detected_boxes[evaluation_set.index(fu[0]-t)][detected_labels[evaluation_set.index(fu[0]-t)] == labels_frame[idx_det]])

                    # Save frames and detections in lists
                    frame_list_video.append([frame_list_f,frame_list_b])
                    #detections_neighbors_video.append([detections_neighbors_f,detections_neighbors_b])



                frame_counter += 1
                overall_frame_counter += 1

            if np.max(num_good_dets_video)==0:

                # No detections in video --> still add frame
                idx_videos.append(np.asarray([fc[0] for fc in frames_candidate]))
                scores_videos.append(np.zeros(len(frames_candidate)))

            else:

                # Track ALL frames and all detections in video with one call
                bboxes_video, elapsed_time_video = tracker_full_video(hp, run, design, frame_list_video, pos_x_video, pos_y_video, target_w_video, target_h_video, final_score_sz, env)

                elapsed_time.append(elapsed_time_video)

                print("Video tracking processed in {:.2f}".format(elapsed_time_video))
                print("Current average elapsed time per video: {:.2f}".format(np.mean(elapsed_time)))

                start_time = time.time()

                # Accumulate detections and tracks in frames
                tracked_dets = [[] for i in range(len(frames_graph))]

                frame_counter = 0

                for idx in range(len(frames_graph)):
                    fu = frames_graph[idx]

                    for idx_det in range(len(local_dets[idx])):
                        bboxes = bboxes_video.pop(0)
                        frame_list = frame_list_video.pop(0)

                        # First position contains forward
                        bboxes_f = bboxes[0]
                        frame_list_f = frame_list[0]

                        for t in range(1,len(frame_list_f)):
                            if idx+t < len(tracked_dets):
                                tracked_dets[idx+t].append({'idx_frame':idx, 'idx_det':idx_det,'box':convert_boxes_xy(bboxes_f[t]).reshape((1,4))})

                        bboxes_b = bboxes[1]
                        frame_list_b = frame_list[1]
                        for t in range(1,len(frame_list_b)):
                            tracked_dets[idx-t].append({'idx_frame':idx, 'idx_det':idx_det,'box':convert_boxes_xy(bboxes_b[t]).reshape((1,4))})

                remaining_tracks = []

                total_dets = np.sum([len(det) for det in local_dets])

                # We need to keep them in two separate arrays because of different types
                det_info = np.zeros((total_dets,3),dtype=int)
                det_all_boxes = np.zeros((total_dets,4))
                det_counter = 0

                # Now associate detections from different frames 
                for idx in range(len(frames_graph)):
                    det_boxes = []
                    for i, det in enumerate(local_dets[idx]):
                        det_boxes.append(det['box'])

                        # Save information of detection 
                        det_info[det_counter,0] = idx
                        det_info[det_counter,1] = i
                        det_all_boxes[det_counter,:] = det['box']
                        det_counter +=1
                        det['assoc']=[]

                    tracked_boxes = []
                    for track in tracked_dets[idx]:
                        tracked_boxes.append(track['box'])

                    tracked_boxes = np.asarray(tracked_boxes).reshape(len(tracked_boxes),4)
                    det_boxes = np.asarray(det_boxes).reshape(len(det_boxes),4)

                    ovTr = np_box_ops.iou(tracked_boxes,det_boxes)

                    # Save what tracks are not going to match any detection
                    if len(det_boxes)>0:
                        remaining_tracks.extend([{'idx_track_frame':idx,'idx_det_frame':tracked_dets[idx][i]['idx_frame'],
                                              'idx_det':tracked_dets[idx][i]['idx_det'],'idx_track':i,'box':tracked_dets[idx][i]['box']} for i,x in enumerate(np.max(ovTr,axis=1) < 0.5) if x])
                    else:
                        # Add all tracks as remaining if there is no detection
                        remaining_tracks.extend([{'idx_track_frame':idx,'idx_det_frame':tracked_dets[idx][i]['idx_frame'],
                                              'idx_det':tracked_dets[idx][i]['idx_det'],'idx_track':i,'box':tracked_dets[idx][i]['box']} for i in range(len(tracked_dets[idx]))])

                    for idx_det in range(len(local_dets[idx])):
                        idx_assoc = ovTr[:,idx_det] > 0.5
                        matching_dets = list(compress(tracked_dets[idx],idx_assoc))
                        for md in matching_dets:
                            local_dets[idx][idx_det]['assoc'].append({'idx_frame':md['idx_frame'],'idx_det':md['idx_det']})



                total_tracks = len(remaining_tracks)
                node_info = np.concatenate((det_info,np.zeros((total_tracks,3), det_info.dtype)),axis=0)
                node_boxes = np.concatenate((det_all_boxes,np.zeros((total_tracks,4), det_info.dtype)),axis=0)
                total_nodes = total_dets+total_tracks

                adj_matrix = np.zeros((total_nodes,total_nodes),dtype=int)

                print('Adding edges between detections')
                # Create edges for det-det
                for i in range(total_dets):
                    idx_frame = node_info[i,0]
                    idx_det = node_info[i,1]
                    for assoc_det in local_dets[idx_frame][idx_det]['assoc']:
                        # Find associated detection in node_info (use det_info)
                        idx_assoc = [i for i,c in enumerate(det_info) if c[0]==assoc_det['idx_frame'] and c[1] == assoc_det['idx_det']]
                        # Set both connections to 1
                        adj_matrix[i,idx_assoc] = 1
                        adj_matrix[idx_assoc,i] = 1

                print('Adding edges between dets and remaining tracks')
                # Create edges between det and tracks 
                track_counter = 0
                for i in range(total_dets,total_nodes):
                    # Fill in node info for track
                    node_info[i,0]=remaining_tracks[track_counter]['idx_track_frame']
                    node_info[i,1]=remaining_tracks[track_counter]['idx_track']
                    node_info[i,2]=1
                    node_boxes[i,:] = remaining_tracks[track_counter]['box']

                    # Add edge with corresponding track (checking in det_info only)
                    idx_orig_det = [j for j,c in enumerate(det_info) if c[0]==remaining_tracks[track_counter]['idx_det_frame'] and c[1] ==remaining_tracks[track_counter]['idx_det']]
                    adj_matrix[i,idx_orig_det] = 1
                    adj_matrix[idx_orig_det,i] = 1


                    track_counter +=1

                # MISSING EDGES BETWEEN TRACKS

                # Create graph for maxflow 
                g = maxflow.Graph[int]()

                # Add nodes
                nodes = g.add_nodes(total_nodes)

                for i in range(total_nodes):

                    # Add terminal edge
                    g.add_tedge(nodes[i],node_info[i,2], 1 if node_info[i,2]==0 else 0)

                    #And add edges between non-terminal nodes
                    edge_list = [i for i,c in enumerate(adj_matrix[i,:]) if c==1]
                    for e in edge_list:
                        g.add_edge(nodes[i],nodes[e],1,1)

                flow = g.maxflow()

                assigned_labels = []
                FP_graph = np.zeros(len(frames_graph),dtype=int)
                FN_graph = np.zeros(len(frames_graph),dtype=int)
                for i in range(total_nodes):
                    assigned_labels.append(g.get_segment(nodes[i]))
                    #Detection
                    if node_info[i,2] == 0 and assigned_labels[-1] == 0:
                        FP_graph[node_info[i,0]] += 1
                    #Track
                    if node_info[i,2] == 1 and assigned_labels[-1] == 1:
                        FN_graph[node_info[i,0]] += 1

                FP = []
                FN = []

                # Count number of FP and FN per frame
                for f in frames_candidate:
                    idx_in_graph = [i for i,c in enumerate(frames_graph) if c[0] == f[0]]
                    FP.append(FP_graph[idx_in_graph[0]])
                    FN.append(FN_graph[idx_in_graph[0]])

                idx_videos.append(np.asarray([fc[0] for fc in frames_candidate]))

                if mode == 'FP':
                    scores_videos.append(np.asarray(FP))
                elif mode == 'FN':
                    scores_videos.append(np.asarray(FN))
                elif mode == 'FPFN':
                    scores_videos.append(np.asarray(FP)+np.asarray(FN))
                else:
                    raise ValueError('Mode for GraphTC not recognized')

                graph_elapsed_time = time.time() - start_time
                print("Graph built and solved in {:.2f}".format(graph_elapsed_time))

    elapsed_time = time.time() - t_start
    print("All videos processed in: {:.2f} seconds".format(elapsed_time))

    # Call selection function
    indices=top_score_frames_selector(scores_videos, idx_videos, data_dir=data_dir, name=name, cycle=cycle, run=run, num_neighbors=neighbors_in, budget=budget, thresh_video=-1)
    return indices



