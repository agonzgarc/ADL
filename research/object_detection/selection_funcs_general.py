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

def obtain_indices_best_frames(budget, scores_videos, idx_videos, max_num_frames):
    return 0

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

        return indices




