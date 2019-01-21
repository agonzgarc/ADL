import pdb
import random
import numpy as np

import functools
import json
import os
import tensorflow as tf
import imp





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



def compute_confidence_scores(dataset,videos,active_set,detections,num_neighbors=3):

        thresh_detection = 0.5

        # We have detections only for the unlabeled dataset, be careful with indexing
        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=num_neighbors)
        unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        predictions = detections['scores']


        scores_videos = []
        idx_videos = []

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
                    if len(sel_dets) > 0:
                        acf = sel_dets.mean()
                    else:
                        acf = np.inf
                    avg_conf.append(acf)

                avg_conf = np.asarray(avg_conf)
                # Select frames that achieve minimum
                idx_min = np.where(avg_conf == np.min(avg_conf))
                idx_sel = np.random.choice(idx_min[0])

                indices.append(frames[idx_sel])

                scores_videos.append([-1])
                idx_videos.append([-1])

            else:
                # If we need video to preserve video idx, add dummy list
                scores_videos.append([-1])
                idx_videos.append([-1])

            #print("Selecting frame {} from video {} with idx {}".format(idxR,v,frames[idxR]))
        return indices




