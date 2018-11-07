import pdb
import random
import numpy as np

import functools
import json
import os
import tensorflow as tf
import imp

from object_detection.utils import np_box_ops
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops

# Tracking module
import siamfc.siamese as siam
from siamfc.tracker import tracker_full_video
from siamfc.parse_arguments import parse_arguments
from siamfc.region_to_bbox import region_to_bbox

from PIL import Image
from object_detection.utils import visualization_utils as vis_utils

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


def augment_to_track(dataset,videos,unlabeled_set,num_neighbors=3):
    """ Augment set of indices in active_set by adding a given number of neighbors
    Arg:
        dataset: structure with information about each frames
        videos: list of video names
        active_set: list of indices of active_set
        num_neighbors: number of neighbors to include
    Returns:
        aug_active_set: augmented list of indices with neighbors
    """
    aug_unlabeled_set = []

    # We need to do this per video to keep limits in check
    for v in videos:
        frames_video = [f['idx'] for f in dataset if f['video'] == v]
        max_frame = np.max(frames_video)
        idx_videos_unlabeled_set = [idx for idx in frames_video if idx in unlabeled_set]
        idx_with_neighbors = [i for idx in idx_videos_unlabeled_set for i in range(idx-num_neighbors,idx+num_neighbors+1) if i >= 0 and i
         <= max_frame ]
        new_idx_with_neighbors = [i for i in idx_with_neighbors if i not in aug_unlabeled_set]
        aug_unlabeled_set.extend(new_idx_with_neighbors)

    return aug_unlabeled_set



def select_full_dataset_per_video(dataset,videos):

    indices = []

    video_list = list(videos)

    all_frames = [[f['idx'],f['video']] for f in dataset if f['verified']]
    total_frames = len(all_frames)

    frames_videos = []
    for v in video_list:
        frames_videos.append([f[0] for f in all_frames if f[1] == v])

    idx_ignore_video = []
    while len(indices) < total_frames:

        for i in range(len(video_list)):
            if i not in idx_ignore_video:
                v = video_list[i]
                fv = frames_videos[i]
                frames = [f for f in fv if f not in indices]
                if len(frames)>0:
                    new_ind = random.choice(frames)
                    indices.append(new_ind)
                    all_frames.remove([new_ind,v])
                else:
                    idx_ignore_video.append(i)

        print("{}/{} frames added. Remaining videos: {}".format(len(indices),total_frames,len(video_list) - len(idx_ignore_video)))
    return indices



def select_full_dataset(dataset):
    indices = [f['idx'] for f in dataset if f['verified']]
    return indices

def select_random(dataset,videos,active_set,budget=788):
    """ Select a random subset of frames from all frames
    Arg:
        dataset: structure with information about each frame
        active_set: list of indices of active_set
    Returns:
        indices: new indices to be added to active_set
    """

    # Remove 5 neighbors around each labeled sample
    aug_active_set = augment_active_set(dataset,videos,active_set,num_neighbors=5)

    frames = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

    return random.sample(frames,budget)

def select_random_video(dataset,videos,active_set):
    """ Select a random frame from each video
    Arg:
        dataset: structure with information about each frame
        videos: list of video names
        active_set: list of indices of active_set
    Returns:
        indices: new indices to be added to active_set
    """

    if active_set:
        ## Temporarily add neighbors to active_set so they are ignored
        #aug_active_set = augment_active_set(dataset,videos,active_set,num_neighbors=5)
        aug_active_set = augment_active_set(dataset,videos,active_set,num_neighbors=5)
    else:
        aug_active_set = active_set

    indices = []
    for v in videos:
        #Select frames in current video
        frames = [f['idx'] for f in dataset if f['video'] == v and f['verified'] ]

        # Remove if already in active set
        #frames = [f for f in frames if f not in active_set]
        # or closer than 5 frames from active set
        frames = [f for f in frames if f not in aug_active_set]

        # If all frames of video are in active set, ignore video
        if len(frames) > 0:
            idxR = random.randint(0,len(frames)-1)
            indices.append(frames[idxR])
        #print("Selecting frame {} from video {} with idx {}".format(idxR,v,frames[idxR]))
    print(len(indices))
    return indices

def compute_entropy(predictions):
    """ Given a list of predictions (class scores with background), it computes
    the entropy of each prediction in the list
    Args:
        predictions: list of predictions. Each element corresponds to an image,
            containing a numpy nd array of shape (num windows, num_classes+1)
    Returns:
        entropies: list of the same dimension, each item is the summary entropy
            for the corresponding image
    """
    # Add more summary measures, now we only have max

    def softmax_pred(x):
        e = np.exp(x)
        return e/np.sum(e,axis=1,keepdims=True)

    def entropy(x):
        return np.sum(-x*np.log(x),axis=1)

    entropies = [np.max(entropy(softmax_pred(i))) for i in predictions]

    return entropies



def select_entropy(dataset,videos,active_set,detections,budget=788):

        # We have detections only for the selected unlabeled set (no neighbors
        # or unverified frames), be careful with indexing
        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=5)

        unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        predictions = detections['scores_with_background']

        num_classes = 3+1

        num_samples = len(unlabeled_set)

        prediction_list = [predictions[i] for i in range(num_samples)]

        ent = np.array(compute_entropy(prediction_list))

        #pdb.set_trace()
        #output_file='entropies.txt'

        #json_str = json.dumps(ent)
        #f = open(output_file,'w')
        #f.write(json_str)
        #f.close()

        # Default sort in ascending order, here we need descending
        idx_sort = ent.argsort()
        idx_sort = list(idx_sort[::-1])

        indices = []
        #new_active_set = []
        #new_active_set.extend(active_set)

        while len(indices) < budget:
            # Get highest entropy value
            try:
                top = idx_sort.pop(0)

                # top is local index in unlabeled, get global
                top_global = unlabeled_set[top]

                # Only add to indices if not if augmented active set
                if top_global not in aug_active_set:
                    indices.append(top_global)
                    # Get video for corresponding new frames
                    v = dataset[top_global]['video']
                    frames_video = [f['idx'] for f in dataset if f['video'] == v]
                    min_frame = np.min(frames_video)
                    max_frame = np.max(frames_video)
                    idx_with_neighbors = [i for i in range(top_global-5,top_global+5+1) if i >= min_frame and i <= max_frame]
                    new_idx_with_neighbors = [i for i in idx_with_neighbors if i not in aug_active_set]
                    aug_active_set.extend(new_idx_with_neighbors)
                    print("Augmenting new active set with added length:{}/{}".format(len(indices),budget))
            except:
                break
        return indices



def select_entropy_video(dataset,videos,active_set,detections):

        indices = []

        # We have detections only for the labeled dataset, be careful with indexing
        #unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=5)
        unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        predictions = detections['scores_with_background']

        num_classes = 3+1

        for v in videos:

            # Select frames in current video
            frames = [f['idx'] for f in dataset if f['video'] == v]

            # Get only those that are not labeled
            frames = [f for f in frames if f in unlabeled_set]

            # If all frames of video are in active set, ignore video
            if len(frames) > 0:
                # Extract corresponding predictions
                det_frames = [predictions[unlabeled_set.index(f)] for f in frames]

                # Compute and summarize entropy
                ent = np.array(compute_entropy(det_frames))

                #idxR = random.randint(0,len(frames)-1)
                idxR = ent.argmax(0)
                indices.append(frames[idxR])
            #print("Selecting frame {} from video {} with idx {}".format(idxR,v,frames[idxR]))
        return indices



def select_entropy_video(dataset,videos,active_set,detections):

        indices = []

        # We have detections only for the labeled dataset, be careful with indexing
        #unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=5)
        unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        predictions = detections['scores_with_background']

        num_classes = 3+1

        for v in videos:

            # Select frames in current video
            frames = [f['idx'] for f in dataset if f['video'] == v]

            # Get only those that are not labeled
            frames = [f for f in frames if f in unlabeled_set]

            # If all frames of video are in active set, ignore video
            if len(frames) > 0:
                # Extract corresponding predictions
                det_frames = [predictions[unlabeled_set.index(f)] for f in frames]

                # Compute and summarize entropy
                ent = np.array(compute_entropy(det_frames))

                #idxR = random.randint(0,len(frames)-1)
                idxR = ent.argmax(0)
                indices.append(frames[idxR])
            #print("Selecting frame {} from video {} with idx {}".format(idxR,v,frames[idxR]))
        return indices

def select_least_confident_video(dataset,videos,active_set,detections):

        indices = []

        # We have detections only for the labeled dataset, be careful with indexing
        aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=5)
        unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

        predictions = detections['scores']

        thresh_detection = 0.5

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
            #print("Selecting frame {} from video {} with idx {}".format(idxR,v,frames[idxR]))
        return indices




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

def filter_detections(boxes,scores,labels,thresh_detection = 0.5):
    idx_good_det = scores > thresh_detection
    return boxes[idx_good_det,:],scores[idx_good_det],labels[idx_good_det]


def select_TCFP_per_video(dataset,videos,data_dir,active_set,detections):

    # Selector configuration
    threshold_track = 0.7
    num_frames_to_track = 3

    # Tracker configuration
    hp, evaluation, run, env, design = parse_arguments()

    final_score_sz = hp.response_up * (design.score_sz - 1) + 1

    # Candidates are verified frames that aren't close to already labeled frames
    aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=5)
    unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set]
    #unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

    total_frames = len([f for f in dataset if f['idx'] in unlabeled_set and f['verified']])
    overall_frame_counter = 0

    # ARE DETECTIONS NMSed?
    detected_boxes = detections['boxes']
    detected_scores = detections['scores']
    detected_labels = detections['labels']

    indices = []
    elapsed_time = []

    # Get only top detections
    for i in range(len(detected_boxes)):
        detected_boxes[i],detected_scores[i],detected_labels[i] = filter_detections(detected_boxes[i],detected_scores[i],detected_labels[i])

    #gt_boxes = groundtruths['boxes']

    for v in videos:

        video_dir = os.path.join(data_dir,'Data','VID','train',v)

        # Select frames in current video (even those with wrong GTs)
        frames = [[f['idx'],f['filename'],f['verified']] for f in dataset if f['video'] == v]

        # Get maximium index of frames in video
        idx_all_frames_video = [f[0] for f in frames]
        max_frame = np.max(idx_all_frames_video)

        # Get only those that are not labeled and verified --> pick from these
        frames_unlabeled = [f for f in frames if f[0] in unlabeled_set and f[2]]

        if len(frames_unlabeled) > 0:

            frame_counter = 0

            frame_list_video = []
            pos_x_video = []
            pos_y_video = []
            target_w_video = []
            target_h_video = []

            num_good_dets_video = []

            detections_neighbors_video = []

            for fu in frames_unlabeled:

                idx_frame_video = idx_all_frames_video.index(fu[0])

                frame_counter += 1
                overall_frame_counter += 1

                #print("Processing frame {}/{} with total idx:{}, video idx {}".format(frame_counter+1,len(frames_unlabeled),fu[0],idx_frame_video))
                print("Adding information about frame in video: {}/{}, overall: {}/{}".format(frame_counter,len(frames_unlabeled),overall_frame_counter, total_frames))

                # ASSUMPTION: for TCFP, we have detections for the whole dataset

                # Get boxes for current frame
                boxes_frame = detected_boxes[fu[0]]
                scores_frame = detected_scores[fu[0]]
                labels_frame = detected_labels[fu[0]]

                #gt_frame = gt_boxes[fu[0]]

                ## Visualization of frame's GT and detections
                #curr_im = Image.open(os.path.join(video_dir,frames[idx_frame_video][1]))
                #im_w,im_h = curr_im.size
                #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_frame,im_w,im_h))
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
                    pos_x_video.append(pos_x)
                    pos_y_video.append(pos_y)
                    pos_y_video.append(pos_y)
                    target_w_video.append(target_w)
                    target_w_video.append(target_w)
                    target_h_video.append(target_h)
                    target_h_video.append(target_h)


                    ###### Forward part
                    detections_neighbors = []
                    frame_list = [os.path.join(video_dir,frames[idx_frame_video][1])]

                    # I can't do this with list comprehension for some reason
                    #frame_list = [frames[i] for i in range(idx_frame_video+1,idx_frame_video+4) if frames[i] in frames]

                    for t in range(1,num_frames_to_track+1):
                        idx_neighbor = idx_frame_video+t

                        # Check if neighbor still in video
                        if idx_neighbor < len(frames):
                            frame_list.append(os.path.join(video_dir,frames[idx_neighbor][1]))
                            # Take only those of the current class
                            detections_neighbors.append(detected_boxes[fu[0]+t][detected_labels[fu[0]+t] == labels_frame[idx_det]])

                    frame_list_video.append(frame_list)
                    detections_neighbors_video.append(detections_neighbors)

                    #bboxes, speed = tracker(hp, run, design, frame_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores, 1)

                    ###### Backward part
                    detections_neighbors = []
                    frame_list = [os.path.join(video_dir,frames[idx_frame_video][1])]

                    for t in range(1,num_frames_to_track+1):
                        idx_neighbor = idx_frame_video-t
                        if idx_neighbor >= 0:
                            frame_list.append(os.path.join(video_dir,frames[idx_neighbor][1]))
                            # Take only those of the current class
                            detections_neighbors.append(detected_boxes[fu[0]-t][detected_labels[fu[0]-t] == labels_frame[idx_det]])

                    frame_list_video.append(frame_list)
                    detections_neighbors_video.append(detections_neighbors)
                    #bboxes, speed = tracker(hp, run, design, frame_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores, 1)




            # Track ALL frames and all detections in video with one call
            bboxes_video, elapsed_time_video = tracker_full_video(hp, run, design, frame_list_video, pos_x_video, pos_y_video, target_w_video, target_h_video, final_score_sz, env)

            elapsed_time.append(elapsed_time_video)

            # Computation of TC-FP score
            frame_counter = 0

            tc_scores = np.zeros(len(frames_unlabeled))

            for fu in frames_unlabeled:

                num_good_dets = num_good_dets_video[frame_counter]

                tc_sum_frame = np.zeros(num_good_dets)
                tc_neigh_frame = np.zeros(num_good_dets)

                for idx_det in range(num_good_dets):

                    # Return and delete from list first element, going in the same order as before
                    bboxes = bboxes_video.pop(0)
                    detections_neighbors = detections_neighbors_video.pop(0)
                    frame_list = frame_list_video.pop(0)

                    for t in range(1,len(frame_list)):

                        # Visualize track and detections
                        #curr_im = Image.open(frame_list[t])
                        #im_w,im_h = curr_im.size
                        #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(convert_boxes_xy(bboxes[t]).reshape((1,4)),im_w,im_h))
                        #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(detections_neighbors[t-1],im_w,im_h),color='green')
                        #curr_im.show()

                        tc_neigh_frame[idx_det] += 1

                        # Check if tracked detection matches any detection in neighbor frame, if any
                        if len(detections_neighbors[t-1]) > 0:
                            ovTr = np_box_ops.iou(convert_boxes_xy(bboxes[t]).reshape((1,4)),detections_neighbors[t-1])
                            # Increment score if it does
                            if np.max(ovTr) > threshold_track:
                                tc_sum_frame[idx_det] += 1


                    bboxes = bboxes_video.pop(0)
                    detections_neighbors = detections_neighbors_video.pop(0)
                    frame_list = frame_list_video.pop(0)

                    for t in range(1,len(frame_list)):

                        ## Visualize track and detections
                        #curr_im = Image.open(frame_list[t])
                        #im_w,im_h = curr_im.size
                        #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(convert_boxes_xy(bboxes[t]).reshape((1,4)),im_w,im_h))
                        #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(detections_neighbors[t-1],im_w,im_h),color='green')
                        #curr_im.show()


                        tc_neigh_frame[idx_det] += 1

                        # Check if tracked detection matches any detection in neighbor frame, if any
                        if len(detections_neighbors[t-1]) > 0:
                            ovTr = np_box_ops.iou(convert_boxes_xy(bboxes[t]).reshape((1,4)),detections_neighbors[t-1])
                            # Increment score if it does
                            if np.max(ovTr) > threshold_track:
                                tc_sum_frame[idx_det] += 1


                # Compute and save mean score per frame
                if num_good_dets > 0:
                    # Score is normalized count
                    tc_scores[frame_counter] = np.mean(tc_sum_frame/tc_neigh_frame)
                else:
                    # Frames with no detections don't have TCFP score (inf so they aren't taken)
                    tc_scores[frame_counter] = np.inf

                frame_counter += 1


            # Select frames that achieve minimum
            idx_min = np.where(tc_scores == np.min(tc_scores))
            idx_sel = np.random.choice(idx_min[0])

            indices.append(frames_unlabeled[idx_sel][0])

            print("Current average elapsed time per video: {:.2f}".format(np.mean(elapsed_time)))

    return indices






def select_TCFN_per_video(dataset,videos,data_dir,active_set,detections):

    # Selector configuration
    threshold_track = 0.5
    threshold_rem_track = 0.5
    num_frames_to_track = 3

    # Tracker configuration
    hp, evaluation, run, env, design = parse_arguments()

    final_score_sz = hp.response_up * (design.score_sz - 1) + 1

    # Candidates are verified frames that aren't close to already labeled frames
    aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=5)
    unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set]
    #unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

    total_frames = len([f for f in dataset if f['idx'] in unlabeled_set and f['verified']])
    overall_frame_counter = 0

    # ARE DETECTIONS NMSed?
    detected_boxes = detections['boxes']
    detected_scores = detections['scores']
    detected_labels = detections['labels']

    indices = []
    elapsed_time = []

    # Get only top detections
    for i in range(len(detected_boxes)):
        detected_boxes[i],detected_scores[i],detected_labels[i] = filter_detections(detected_boxes[i],detected_scores[i],detected_labels[i])

    #gt_boxes = groundtruths['boxes']

    for v in videos:

        video_dir = os.path.join(data_dir,'Data','VID','train',v)

        # Select frames in current video (even those with wrong GTs)
        frames = [[f['idx'],f['filename'],f['verified']] for f in dataset if f['video'] == v]

        # Get maximium index of frames in video
        idx_all_frames_video = [f[0] for f in frames]
        max_frame = np.max(idx_all_frames_video)

        # Get only those that are not labeled and verified --> pick from these
        frames_unlabeled = [f for f in frames if f[0] in unlabeled_set and f[2]]

        if len(frames_unlabeled) > 0:

            frame_counter = 0

            frame_list_video = []
            pos_x_video = []
            pos_y_video = []
            target_w_video = []
            target_h_video = []

            num_good_dets_video = []

            detections_neighbors_video = []

            for fu in frames_unlabeled:

                idx_frame_video = idx_all_frames_video.index(fu[0])

                frame_counter += 1
                overall_frame_counter += 1

                #print("Processing frame {}/{} with total idx:{}, video idx {}".format(frame_counter+1,len(frames_unlabeled),fu[0],idx_frame_video))
                print("Adding information about frame in video: {}/{}, overall: {}/{}".format(frame_counter,len(frames_unlabeled),overall_frame_counter, total_frames))

                # ASSUMPTION: for TCFP, we have detections for the whole dataset

                # Get boxes for current frame
                boxes_frame = detected_boxes[fu[0]]
                scores_frame = detected_scores[fu[0]]
                labels_frame = detected_labels[fu[0]]

                #gt_frame = gt_boxes[fu[0]]

                ## Visualization of frame's GT and detections
                #curr_im = Image.open(os.path.join(video_dir,frames[idx_frame_video][1]))
                #im_w,im_h = curr_im.size
                #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_frame,im_w,im_h))
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
                    pos_x_video.append(pos_x)
                    pos_y_video.append(pos_y)
                    pos_y_video.append(pos_y)
                    target_w_video.append(target_w)
                    target_w_video.append(target_w)
                    target_h_video.append(target_h)
                    target_h_video.append(target_h)


                    ###### Forward part
                    detections_neighbors = []
                    frame_list = [os.path.join(video_dir,frames[idx_frame_video][1])]

                    # I can't do this with list comprehension for some reason
                    #frame_list = [frames[i] for i in range(idx_frame_video+1,idx_frame_video+4) if frames[i] in frames]

                    for t in range(1,num_frames_to_track+1):
                        idx_neighbor = idx_frame_video+t

                        # Check if neighbor still in video
                        if idx_neighbor < len(frames):
                            frame_list.append(os.path.join(video_dir,frames[idx_neighbor][1]))
                            # Take only those of the current class
                            detections_neighbors.append(detected_boxes[fu[0]+t][detected_labels[fu[0]+t] == labels_frame[idx_det]])

                    frame_list_video.append(frame_list)
                    detections_neighbors_video.append(detections_neighbors)

                    ###### Backward part
                    detections_neighbors = []
                    frame_list = [os.path.join(video_dir,frames[idx_frame_video][1])]

                    for t in range(1,num_frames_to_track+1):
                        idx_neighbor = idx_frame_video-t
                        if idx_neighbor >= 0:
                            frame_list.append(os.path.join(video_dir,frames[idx_neighbor][1]))
                            # Take only those of the current class
                            detections_neighbors.append(detected_boxes[fu[0]-t][detected_labels[fu[0]-t] == labels_frame[idx_det]])

                    frame_list_video.append(frame_list)
                    detections_neighbors_video.append(detections_neighbors)


            # Track ALL frames and all detections in video with one call
            bboxes_video, elapsed_time_video = tracker_full_video(hp, run, design, frame_list_video, pos_x_video, pos_y_video, target_w_video, target_h_video, final_score_sz, env)

            elapsed_time.append(elapsed_time_video)

            # Computation of TC-FN score
            frame_counter = 0

            tcfn_scores = np.zeros(len(frames_unlabeled))

            tracked_dets = [[] for i in range(len(frames_unlabeled))]
            local_dets = [[] for i in range(len(frames_unlabeled))]

            frame_paths = [[] for i in range(len(frames_unlabeled))]
            frame_files = [fu[1] for fu in frames_unlabeled]

            for idx in range(len(frames_unlabeled)):
                fu = frames_unlabeled[idx]

                num_good_dets = num_good_dets_video[frame_counter]

                for idx_det in range(num_good_dets):
                    if len(bboxes_video) == 0:
                        pdb.set_trace()
                    bboxes = bboxes_video.pop(0)
                    local_dets[idx].append(convert_boxes_xy(bboxes[0]).reshape((1,4)))

                    frame_list = frame_list_video.pop(0)

                    frame_paths[idx] = frame_list[0]

                    # Forward pass
                    for t in range(1,len(frame_list)):
                        # Get idx of each frame for which we have a detection
                        # It could be that frame is not in unlabeled frames, in this case we don't care and there's no need to add its tracks
                        idx_track = frame_files.index(frame_list[t][-11:]) if frame_list[t][-11:] in frame_files else None

                        # Add tracked detection to corresponding frame
                        if idx_track:
                            tracked_dets[idx_track].append(convert_boxes_xy(bboxes[t]).reshape((1,4)))

                    # Backward pass, we can now discar first box (we have already stored the local det)
                    bboxes = bboxes_video.pop(0)
                    frame_list = frame_list_video.pop(0)

                    for t in range(1,len(frame_list)):
                        # Get idx of each frame for which we have a detection
                        idx_track = frame_files.index(frame_list[t][-11:]) if frame_list[t][-11:] in frame_files else None

                        # Add tracked detection to corresponding frame
                        if idx_track:
                            tracked_dets[idx_track].append(convert_boxes_xy(bboxes[t]).reshape((1,4)))

                frame_counter += 1


            for idx in range(len(frames_unlabeled)):

                tracked_dets_i = np.asarray(tracked_dets[idx]).reshape(len(tracked_dets[idx]),4)
                local_dets_i = np.asarray(local_dets[idx]).reshape(len(local_dets[idx]),4)

                ## Visualize track and detections
                curr_im = Image.open(frame_paths[idx])
                im_w,im_h = curr_im.size
                if tracked_dets[idx]:
                    vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(tracked_dets_i,im_w,im_h))
                if local_dets[idx]:
                    vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(local_dets_i,im_w,im_h),color='green')
                curr_im.show()

                ovTr = np_box_ops.iou(tracked_dets_i,local_dets_i)

                rem_tracks = tracked_dets_i[np.max(ovTr,1)<threshold_track]

                vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(rem_tracks,im_w,im_h),color='orange')
                curr_im.show()
                scores = []
                if len(rem_tracks) > 0:
                    pdb.set_trace()
                    while len(rem_tracks)>1:
                        # Process first of remaining tracks
                        curr_track = rem_tracks[0].reshape(1,4)
                        rem_tracks = rem_tracks[1:].reshape(len(rem_tracks)-1,4)
                        ovTr = np_box_ops.iou(curr_track,rem_tracks)
                        track_group = ovTr > threshold_rem_track
                        scores.append(np.sum(track_group))
                        # Remove group from rem_tracks
                        rem_tracks[(track_group == False)[0]]

                    if len(rem_tracks) == 1:
                        scores.append(1)

                    # Final FN score of frame is...
                    tcfn_scores[idx] = np.mean(scores)

            ## Select frames that achieve minimum
            idx_max = np.where(tcfn_scores == np.max(tcfn_scores))
            idx_sel = np.random.choice(idx_max[0])

            indices.append(frames_unlabeled[idx_sel][0])

            print("Current average elapsed time per video: {:.2f}".format(np.mean(elapsed_time)))

    return indices






