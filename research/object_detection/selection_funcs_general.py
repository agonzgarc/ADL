import pdb
import random
import numpy as np

import functools
import json
import os
import tensorflow as tf
import imp
import time




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

def select_dummy():

    scores_videos = [[0.2, 0.2, 0.3],[0.1, 0.4]]
    idx_videos = [[12312,1231244,66745],[133,144]]

    # len(indices) = total budget
    indices = select_frames(scores_videos, idx_videos, max_num_frames)

    return indices

#def obtain_indices_best_frames(budget, scores_videos, idx_videos, max_num_frames):
#    return 0

def select_least_confident(dataset,videos,active_set,detections,num_neighbors=5):

        thresh_detection = 0.5

        # We have detections only for the unlabeled dataset, be careful with indexing
        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=num_neighbors)
        unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        predictions = detections['scores']


        scores_videos = []
        idx_videos = []
        num_frames = []

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
                        acf = 1-sel_dets.mean()
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
                num_frames.append(len(frames))

        elapsed_time = time.time() - t_start
        print("All videos processed in:{:.2f} seconds".format(elapsed_time))

        # Javad, call your function here
	indices=top_score_frames_selector(budget, scores_videos, idx_videos)

        return indices


def top_score_frames_selector(budget, scores_videos, idx_videos)


    pdb.set_trace()

    n=2 # neighbourhood
    budget=3200
    number_of_vids=800
    max_vid_length=1000

    SCORES=np.zeros((number_of_vids,max_vid_length))
    CANDIDATES=np.zeros((number_of_vids,max_vid_length))-1
    CANDIDATES_SC=np.zeros((number_of_vids,max_vid_length))-1

    for vid in range(number_of_vids):
    
      length_of_vid=np.random.randint(max_vid_length, size=1)
      print('length_of_vid= ', length_of_vid[0])
      scores=np.random.randint(10, size=length_of_vid)
      SCORES[vid,:]=np.pad(scores, (0,max_vid_length-len(scores)), 'constant', constant_values=-1)
    
    print('--------------------------------------------------------------------------------')
    #--------------------------SORTING THE SCORES IN DESCENDING ORDER-----------------------
    sorted_INDICES=np.flip(np.argsort(SCORES,axis=1),1)  
    sorted_SCORES=np.flip(np.sort(SCORES,axis=1),1)
    print('--------------------------------------------------------------------------------')

    for v in range(0,number_of_vids):
      iter=0
      sorted_indices=sorted_INDICES[v,:]
      sorted_scores=sorted_SCORES[v,:]
      idx_max=sorted_indices[0]
      score_max=sorted_scores[0]
      
      while(score_max>=0):
        print('video= ',v)      
        print('iter= ',iter)
        #print('sorted_indices= ',sorted_indices)
        #print('sorted_scores= ',sorted_scores)
        #print('idx_max= ',idx_max)
        #print('score_max= ',score_max)    
        CANDIDATES[v,iter]=idx_max
        CANDIDATES_SC[v,iter]=score_max
        left=max(idx_max-n,0)
        right=min(idx_max+n,max_vid_length)
        frames_to_remove=np.arange(left,right+1,1)
        #print('frames to remove= ', frames_to_remove)
        IND = np.in1d(sorted_indices, frames_to_remove) #intersection    
        shrinked_indices=sorted_indices[~IND] # removing frames from indices
        shrinked_scores=sorted_scores[~IND] # removing frames from scores    
        #print('shrinked_indices= ',shrinked_indices)
        #print('shrinked_scores= ',shrinked_scores)
        sorted_scores=shrinked_scores
        sorted_indices=shrinked_indices
        if sorted_indices.size != 0:
           idx_max=sorted_indices[0]
           score_max=sorted_scores[0]
        else:
           print('************************************************')
           break
        iter=iter+1    
        print('************************************************')
    
    #----------------SELECTING FRAMES FROM TOP CANDIDATES------------------------
    b=0    
    sel_idx=np.zeros(budget)
    for j in range(0,len(CANDIDATES[0])):    
        for i in range(0,len(CANDIDATES)):
            print('i= ',i, ' j= ',j)
            if CANDIDATES_SC[i,j]>=0:
                sel_idx[b]=CANDIDATES[i,j]
                b=b+1
                if b==budget:
                   break
        if b==budget:
           break        
    print(sel_idx)
    print('length of selected frames = ',len(sel_idx))                
