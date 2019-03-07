import pdb
import random
import numpy as np

import functools
import json
import os
#import tensorflow as tf
import imp


from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


from object_detection.utils import np_box_ops
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops
from object_detection.utils import visualization_utils as vis_utils


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
        if data_info['dataset'] == 'imagenet':
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
        elif data_info['dataset'] == 'synthia':
            for path in pF:
                split_path = path.split('/')
                filename = split_path[-1][:-1]
                video = split_path[-4]+'/'+split_path[-3]
                dataset.append({'idx':idx,'filename':filename,'video':video,'verified':True})
                idx+=1
        else:
            raise ValueError('Dataset error: select imagenet or synthia')

    videos = list(set([d['video'] for d in dataset]))
    videos.sort()
    return dataset,videos



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


def visualize_detections(frames, detections):
    detected_boxes = detections['boxes']
    detected_scores = detections['scores']
    detected_labels = detections['labels']

    video_save_dir = save_dir+frames[0]['video']
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)

    for f in frames:
        im_path = os.path.join(data_dir,'Data','VID','train',f['video'],f['filename'])
        curr_im = Image.open(im_path)
        im_w,im_h = curr_im.size
        det_im = detected_boxes[f['idx']]
        vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(det_im[:2,],im_w,im_h),color='green')
        curr_im.show()
        #curr_im.save(video_save_dir+ '/' + f['filename'])

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


def filter_detections(boxes,scores,labels,thresh_detection = 0.5):
    idx_good_det = scores > thresh_detection
    return boxes[idx_good_det,:],scores[idx_good_det],labels[idx_good_det]


def obtain_FP_and_FN(dataset,videos,unlabeled_set,detected_boxes,detected_scores,detected_labels,groundtruth_boxes):
    score_thresh=0.5
    iou_thresh=0.5
    scores_fp_videos = []
    scores_fn_videos = []
    idx_videos = []

    #BOXES = detections['boxes']
    #SCORES= detections['scores']
    #LABELS = detections['labels']
    BOXES = detected_boxes
    SCORES= detected_scores
    LABELS = detected_labels
    gt_boxes = groundtruth_boxes['boxes']
    gt_labels = groundtruth_boxes['labels']

    stat_data={}
    list_of_FNs=[]
    list_of_FPs=[]
    stat_data['FN_info']=[]
    stat_data['FP_info']=[]

    for v in videos:
        # Select frames in current video
        frames = [f['idx'] for f in dataset if f['video'] == v and f['idx'] in unlabeled_set]

        # If all frames of video are in active set, ignore video
        if len(frames)>0:
            # Add video to list if used
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
                                else:
                                    FP[j]+=1
                            else:
                                FP[j]+=1

                        FN[j]=sum(gt_labels_frame_f>-1)
                    else:
                        FN[j]=len(gt_boxes_frame_f)
                else:
                    FP[j]=len(boxes)
                j+=1



            scores_fp_videos.append(FP)
            scores_fn_videos.append(FN)
            idx_videos.append(np.asarray(frames))

        else:
            scores_fp_videos.append([])
            scores_fn_videos.append([])
            idx_videos.append([])

    return scores_fp_videos, scores_fn_videos, idx_videos



def obtain_FP_and_FN_mismatched(dataset,videos,detections_set,detected_boxes,detected_scores,detected_labels,groundtruth_set,groundtruth_boxes):
    score_thresh=0.5
    iou_thresh=0.5
    scores_fp_videos = []
    scores_fn_videos = []
    idx_videos = []

    #BOXES = detections['boxes']
    #SCORES= detections['scores']
    #LABELS = detections['labels']
    BOXES = detected_boxes
    SCORES= detected_scores
    LABELS = detected_labels
    gt_boxes = groundtruth_boxes['boxes']
    gt_labels = groundtruth_boxes['labels']

    stat_data={}
    list_of_FNs=[]
    list_of_FPs=[]
    stat_data['FN_info']=[]
    stat_data['FP_info']=[]


    for v in videos:
        # Select frames in current video
        frames = [f['idx'] for f in dataset if f['video'] == v and f['idx'] in detections_set]

        # If all frames of video are in active set, ignore video
        if len(frames)>0:
            # Add video to list if used
            FN = np.zeros((len(frames)),dtype=int)
            FP = np.zeros((len(frames)),dtype=int)
            TP = np.zeros((len(frames)),dtype=int)
            j=0
            for f in frames:
                anno_ind=detections_set.index(f)
                # Extracting boxes with score greater than threshold
                ind=SCORES[anno_ind] > score_thresh
                boxes=np.array(BOXES[anno_ind])[ind,:]
                labels=LABELS[anno_ind][ind]
                try:
                    gt_ind=groundtruth_set.index(f)
                    gt_labels_frame_f=np.copy(gt_labels[gt_ind])
                    gt_boxes_frame_f=np.copy(gt_boxes[gt_ind])
                except:
                    j+=1
                    continue

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
                                else:
                                    FP[j]+=1
                            else:
                                FP[j]+=1

                        FN[j]=sum(gt_labels_frame_f>-1)
                    else:
                        FN[j]=len(gt_boxes_frame_f)
                else:
                    FP[j]=len(boxes)
                j+=1



            scores_fp_videos.append(FP)
            scores_fn_videos.append(FN)
            idx_videos.append(np.asarray(frames))
        else:
            scores_fp_videos.append([])
            scores_fn_videos.append([])
            idx_videos.append([])


    return scores_fp_videos, scores_fn_videos, idx_videos



