from __future__ import division


import sys
import os
import numpy as np
from PIL import Image
import siamfc.siamese as siam
from siamfc.tracker import tracker_full_video
from siamfc.parse_arguments import parse_arguments
from siamfc.region_to_bbox import region_to_bbox
import imp
import pickle
import tensorflow as tf

from object_detection.utils import visualization_utils as vis_utils

train_dir = '/home/abel/DATA/faster_rcnn/resnet50_coco/checkpoints/tracking_exploration/'
data_dir = '/home/abel/DATA/ILSVRC/'
save_dir = '/home/abel/Documents/graphics/ADL/tracking_evaluation/'

data_info = {'data_dir': data_dir,
          'annotations_dir':'Annotations',
          'label_map_path': './data/imagenetvid_label_map.pbtxt',
          #'set': 'train_150K_clean'}
          'set': 'train_short_clean'}

import pdb

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
    videos = list(set([d['video'] for d in dataset]))
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
        #curr_im.show()
        curr_im.save(video_save_dir+ '/' + f['filename'])

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

def visualize_tracks(video, frames, boxes, idx_det):

    video_save_dir = save_dir+video[-14:]
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)

    im_path = frames[0][0]
    im_name = im_path[-10:-5]
    curr_im = Image.open(im_path)
    im_w,im_h = curr_im.size
    det_im = convert_boxes_xy(boxes[0][0])
    vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(det_im.reshape(1,4),im_w,im_h),color='green')
    curr_im.save(video_save_dir+ '/' + im_name + '-' + str(idx_det) + '.JPEG')

    pdb.set_trace()

    # Plot forward
    for i in range(1,len(boxes[0])):
        im_path = frames[0][i]
        curr_im = Image.open(im_path)
        im_w,im_h = curr_im.size
        det_im = convert_boxes_xy(boxes[0][i])
        vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(det_im.reshape(1,4),im_w,im_h),color='red')
        curr_im.save(video_save_dir+ '/' + im_name + '-' + str(idx_det) + '+' + str(i) + '.JPEG')

    # Plot backward
    for i in range(1,len(boxes[1])):
        im_path = frames[1][i]
        curr_im = Image.open(im_path)
        im_w,im_h = curr_im.size
        det_im = convert_boxes_xy(boxes[1][i])
        vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(det_im.reshape(1,4),im_w,im_h),color='red')
        curr_im.save(video_save_dir+ '/' + im_name + '-' + str(idx_det) + '-' + str(i) + '.JPEG')

if __name__ == "__main__":

    # Selector configuration
    threshold_track = 0.7
    num_frames_to_track = 10

    # Get info about full dataset
    dataset,videos = get_dataset(data_info)

    # Tracker configuration
    hp, evaluation, run, env, design = parse_arguments()

    final_score_sz = hp.response_up * (design.score_sz - 1) + 1

    if os.path.exists(train_dir + 'detections.dat'):
        with open(train_dir + 'detections.dat','rb') as infile:
            ###### pdb remove latinq
            #detected_boxes = pickle.load(infile)
            detections = pickle.load(infile,encoding='latin1')

    detected_boxes = detections['boxes']
    detected_scores = detections['scores']
    detected_labels = detections['labels']


    # Get only top detections
    for i in range(len(detected_boxes)):
        detected_boxes[i],detected_scores[i],detected_labels[i] = filter_detections(detected_boxes[i],detected_scores[i],detected_labels[i])

    for idx_video in range(5):
        v = videos[idx_video]

        video_dir = os.path.join(data_dir,'Data','VID','train',v)

        frames = [f for f in dataset if f['video'] == v]

        idx_all_frames_video = [f['idx'] for f in frames]

        fu = frames[10]['idx']

        idx_frame_video = idx_all_frames_video.index(fu)

        # Get boxes for current frame
        boxes_frame = detected_boxes[fu]
        scores_frame = detected_scores[fu]
        labels_frame = detected_labels[fu]

        num_good_dets = labels_frame.shape[0]

        frame_list_video = []
        pos_x_video = []
        pos_y_video = []
        target_w_video = []
        target_h_video = []

        detections_neighbors_video = []

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
            frame_list_f = [os.path.join(video_dir,frames[idx_frame_video]['filename'])]

            for t in range(1,num_frames_to_track+1):
                idx_neighbor = idx_frame_video+t

                # Check if neighbor still in video
                if idx_neighbor < len(frames):
                    frame_list_f.append(os.path.join(video_dir,frames[idx_neighbor]['filename']))

                    # Take only those of the current class
                    detections_neighbors_f.append(detected_boxes[fu+t][detected_labels[fu+t] == labels_frame[idx_det]])

            ###### Backward part
            detections_neighbors_b = []
            frame_list_b = [os.path.join(video_dir,frames[idx_frame_video]['filename'])]

            for t in range(1,num_frames_to_track+1):
                idx_neighbor = idx_frame_video-t
                if idx_neighbor >= 0:
                    frame_list_b.append(os.path.join(video_dir,frames[idx_neighbor]['filename']))

                    # Take only those of the current class
                    detections_neighbors_b.append(detected_boxes[fu-t][detected_labels[fu-t] == labels_frame[idx_det]])

            # Save frames and detections in lists
            frame_list_video.append([frame_list_f,frame_list_b])
            detections_neighbors_video.append([detections_neighbors_f, detections_neighbors_b])

        # Track ALL frames and all detections in video with one call

        bboxes_video, elapsed_time_video = tracker_full_video(hp, run, design, frame_list_video, pos_x_video, pos_y_video, target_w_video, target_h_video, final_score_sz, env)
        print(bboxes_video)
        for i in range(len(bboxes_video)):
            visualize_tracks(v, frame_list_video[i], bboxes_video[i],i)

    #indices = sel.select_TCFP_per_video(dataset,videos,data_dir,active_set,detections)

