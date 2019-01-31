
import tensorflow as tf
print('Using Tensorflow '+tf.__version__)
import matplotlib.pyplot as plt
import sys
# sys.path.append('../')
import os
import csv
import numpy as np
from PIL import Image
import time
import scipy.io
import sys
import os.path

import pdb

import imp

import siamfc.siamese as siam
from siamfc.visualization import show_frame, show_crops, show_scores
from siamfc.convolutional import set_convolutional
from siamfc.crops import extract_crops_z, extract_crops_x, pad_frame, resize_images

# Configuration for 5 convolutional layers
#_conv_stride = np.array([2,1,1,1,1])
#_filtergroup_yn = np.array([0,1,0,1,1], dtype=bool)
#_bnorm_yn = np.array([1,1,1,1,0], dtype=bool)
#_relu_yn = np.array([1,1,1,1,0], dtype=bool)
#_pool_stride = np.array([2,1,0,0,0]) # 0 means no pool
#_pool_sz = 3
#_bnorm_adjust = True

_conv_stride = np.array([2,1])
_filtergroup_yn = np.array([0,1], dtype=bool)
_bnorm_yn = np.array([1,1], dtype=bool)
_relu_yn = np.array([1,1], dtype=bool)
_pool_stride = np.array([2,1]) # 0 means no pool
_pool_sz = 3
_bnorm_adjust = False



assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), ('These arrays of flags should have same length')
assert all(_conv_stride) >= True, ('The number of conv layers is assumed to define the depth of the network')
_num_layers = len(_conv_stride)

# gpu_device = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

#global graph
#graph= tf.get_default_graph()


def tracker_full_video(hp, run, design, frame_name_list_video, pos_x_video,
                       pos_y_video, target_w_video, target_h_video, final_score_sz, env):

    #g_1 = tf.Graph()
    #with graph.as_default():
    tf.reset_default_graph()
    # Build graph here
    pos_x_ph = tf.placeholder(tf.float64)
    pos_y_ph = tf.placeholder(tf.float64)
    z_sz_ph = tf.placeholder(tf.float64)
    x_sz0_ph = tf.placeholder(tf.float64)
    x_sz1_ph = tf.placeholder(tf.float64)
    x_sz2_ph = tf.placeholder(tf.float64)

    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    # Decode the image as a JPEG file, this will turn it into a Tensor
    image = tf.image.decode_image(image_file, channels=3)
    image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)
    frame_sz = tf.shape(image)
    # used to pad the crops
    if design.pad_with_image_mean:
        avg_chan = tf.reduce_mean(image, axis=(0,1), name='avg_chan')
    else:
        avg_chan = None
    # pad with if necessary
    frame_padded_z, npad_z = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, z_sz_ph, avg_chan)
    frame_padded_z = tf.cast(frame_padded_z, tf.float32)
    # extract tensor of z_crops
    z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x_ph, pos_y_ph, z_sz_ph, design.exemplar_sz)
    frame_padded_x, npad_x = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, x_sz2_ph, avg_chan)
    frame_padded_x = tf.cast(frame_padded_x, tf.float32)
    # extract tensor of x_crops (3 scales)
    x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz0_ph, x_sz1_ph, x_sz2_ph, design.search_sz)
    # use crops as input of (MatConvnet imported) pre-trained fully-convolutional Siamese net
    template_z, templates_x, p_names_list, p_val_list = _create_siamese(os.path.join(env.root_pretrained,design.net), x_crops, z_crops)
    template_z = tf.squeeze(template_z)
    templates_z = tf.stack([template_z, template_z, template_z])
    # compare templates via cross-correlation
    scores_d = _match_templates(templates_z, templates_x, p_names_list, p_val_list)
    # upsample the score maps
    scores = tf.image.resize_images(scores_d, [final_score_sz, final_score_sz],
        method=tf.image.ResizeMethod.BICUBIC, align_corners=True)


    scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    # cosine window to penalize large displacements
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    run_opts = {}

    bboxes_video = []

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    t_start = time.time()

    #with tf.Session(graph=graph) as sess:
    with tf.Session(config=session_config) as sess:
        tf.global_variables_initializer().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Now each element in *_video lists is processed as the previous tracking
        # function, but within the overall session
        for i in range(len(frame_name_list_video)):

            # Get corresponding list for current frame
            pos_x = pos_x_video[i]
            pos_y = pos_y_video[i]
            target_w = target_w_video[i]
            target_h = target_h_video[i]

            context = design.context*(target_w+target_h)
            z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
            x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

            # thresholds to saturate patches shrinking/growing
            min_z = hp.scale_min * z_sz
            max_z = hp.scale_max * z_sz
            min_x = hp.scale_min * x_sz
            max_x = hp.scale_max * x_sz

            templates_z_ = sess.run([templates_z], feed_dict={pos_x_ph: pos_x, pos_y_ph: pos_y, z_sz_ph: z_sz, filename: frame_name_list_video[i][0][0]})
            #image_, templates_z_ = sess.run([image, templates_z], feed_dict={pos_x_ph: pos_x, pos_y_ph: pos_y, z_sz_ph: z_sz, filename: frame_name_list[0]})

            new_templates_z_ = templates_z_

            bboxes = []
            for fb in range(2):
                frame_name_list = frame_name_list_video[i][fb]
                num_frames = np.size(frame_name_list)

                # save first frame position (from ground-truth)
                bboxes_fb = np.zeros((num_frames,4))
                bboxes_fb[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h

                # Get an image from the queue
                for j in range(1, num_frames):
                    scaled_exemplar = z_sz * scale_factors
                    scaled_search_area = x_sz * scale_factors
                    scaled_target_w = target_w * scale_factors
                    scaled_target_h = target_h * scale_factors
                    image_, scores_ = sess.run(
                        [image, scores],
                        feed_dict={
                            pos_x_ph: pos_x,
                            pos_y_ph: pos_y,
                            x_sz0_ph: scaled_search_area[0],
                            x_sz1_ph: scaled_search_area[1],
                            x_sz2_ph: scaled_search_area[2],
                            templates_z: np.squeeze(templates_z_),
                            filename: frame_name_list[j],
                        }, **run_opts)
                    scores_ = np.squeeze(scores_)
                    # penalize change of scale
                    scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
                    scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
                    # find scale with highest peak (after penalty)
                    new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
                    # update scaled sizes
                    x_sz = (1-hp.scale_lr)*x_sz + hp.scale_lr*scaled_search_area[new_scale_id]
                    target_w = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
                    target_h = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
                    # select response with new_scale_id
                    score_ = scores_[new_scale_id,:,:]
                    score_ = score_ - np.min(score_)
                    score_ = score_/np.sum(score_)

                    # apply displacement penalty
                    score_ = (1-hp.window_influence)*score_ + hp.window_influence*penalty
                    pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
                    # convert <cx,cy,w,h> to <x,y,w,h> and save output
                    bboxes_fb[j,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
                    # update the target representation with a rolling average
                    if hp.z_lr>0:
                        new_templates_z_ = sess.run([templates_z], feed_dict={
                                                                        pos_x_ph: pos_x,
                                                                        pos_y_ph: pos_y,
                                                                        z_sz_ph: z_sz,
                                                                        image: image_
                                                                        })

                        templates_z_=(1-hp.z_lr)*np.asarray(templates_z_) + hp.z_lr*np.asarray(new_templates_z_)

                    # update template patch size
                    z_sz = (1-hp.scale_lr)*z_sz + hp.scale_lr*scaled_exemplar[new_scale_id]

                    if run.visualization:
                        show_frame(image_, bboxes_fb[j,:], 1)

                bboxes.append(bboxes_fb)

            bboxes_video.append(bboxes)



        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)
    elapsed_time = time.time() - t_start
    print("Video processed in:{:.2f}".format(elapsed_time))
    plt.close('all')
    return bboxes_video, elapsed_time





# read default parameters and override with custom ones
#def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores, start_frame):
    #num_frames = np.size(frame_name_list)
    ## stores tracker's output for evaluation
    #bboxes = np.zeros((num_frames,4))

    #scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    ## cosine window to penalize large displacements
    #hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    #penalty = np.transpose(hann_1d) * hann_1d
    #penalty = penalty / np.sum(penalty)

    #context = design.context*(target_w+target_h)
    #z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
    #x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

    ## thresholds to saturate patches shrinking/growing
    #min_z = hp.scale_min * z_sz
    #max_z = hp.scale_max * z_sz
    #min_x = hp.scale_min * x_sz
    #max_x = hp.scale_max * x_sz

    ## run_metadata = tf.RunMetadata()
    ## run_opts = {
    ##     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    ##     'run_metadata': run_metadata,
    ## }

    #run_opts = {}

    ## with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #with tf.Session() as sess:
        #tf.global_variables_initializer().run()
        ## Coordinate the loading of image files.
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)
        
        ## save first frame position (from ground-truth)
        #bboxes[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h                

        #image_, templates_z_ = sess.run([image, templates_z], feed_dict={
                                                                        #siam.pos_x_ph: pos_x,
                                                                        #siam.pos_y_ph: pos_y,
                                                                        #siam.z_sz_ph: z_sz,
                                                                        #filename: frame_name_list[0]})
        #new_templates_z_ = templates_z_

        #t_start = time.time()

        ## Get an image from the queue
        #for i in range(1, num_frames):        
            #scaled_exemplar = z_sz * scale_factors
            #scaled_search_area = x_sz * scale_factors
            #scaled_target_w = target_w * scale_factors
            #scaled_target_h = target_h * scale_factors
            #image_, scores_ = sess.run(
                #[image, scores],
                #feed_dict={
                    #siam.pos_x_ph: pos_x,
                    #siam.pos_y_ph: pos_y,
                    #siam.x_sz0_ph: scaled_search_area[0],
                    #siam.x_sz1_ph: scaled_search_area[1],
                    #siam.x_sz2_ph: scaled_search_area[2],
                    #templates_z: np.squeeze(templates_z_),
                    #filename: frame_name_list[i],
                #}, **run_opts)
            #scores_ = np.squeeze(scores_)
            ## penalize change of scale
            #scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
            #scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
            ## find scale with highest peak (after penalty)
            #new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
            ## update scaled sizes
            #x_sz = (1-hp.scale_lr)*x_sz + hp.scale_lr*scaled_search_area[new_scale_id]        
            #target_w = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
            #target_h = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
            ## select response with new_scale_id
            #score_ = scores_[new_scale_id,:,:]
            #score_ = score_ - np.min(score_)
            #score_ = score_/np.sum(score_)
            ## apply displacement penalty
            #score_ = (1-hp.window_influence)*score_ + hp.window_influence*penalty
            #pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
            ## convert <cx,cy,w,h> to <x,y,w,h> and save output
            #bboxes[i,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
            ## update the target representation with a rolling average
            #if hp.z_lr>0:
                #new_templates_z_ = sess.run([templates_z], feed_dict={
                                                                #siam.pos_x_ph: pos_x,
                                                                #siam.pos_y_ph: pos_y,
                                                                #siam.z_sz_ph: z_sz,
                                                                #image: image_
                                                                #})

                #templates_z_=(1-hp.z_lr)*np.asarray(templates_z_) + hp.z_lr*np.asarray(new_templates_z_)
            
            ## update template patch size
            #z_sz = (1-hp.scale_lr)*z_sz + hp.scale_lr*scaled_exemplar[new_scale_id]
            
            #if run.visualization:
                #show_frame(image_, bboxes[i,:], 1)        
                

        #t_elapsed = time.time() - t_start
        #speed = num_frames/t_elapsed

        ## Finish off the filename queue coordinator.
        #coord.request_stop()
        #coord.join(threads) 

        ## from tensorflow.python.client import timeline
        ## trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        ## trace_file = open('timeline-search.ctf.json', 'w')
        ## trace_file.write(trace.generate_chrome_trace_format())

    #plt.close('all')

    #return bboxes, speed



# The following version is redundant in the sense that forward and backward frame lists are completely independent and thus the template for the initial detection is extracted twice
def tracker_full_video_redundant(hp, run, design, frame_name_list_video, pos_x_video,
                       pos_y_video, target_w_video, target_h_video, final_score_sz, env):

    #g_1 = tf.Graph()
    #with graph.as_default():
    tf.reset_default_graph()
    # Build graph here
    pos_x_ph = tf.placeholder(tf.float64)
    pos_y_ph = tf.placeholder(tf.float64)
    z_sz_ph = tf.placeholder(tf.float64)
    x_sz0_ph = tf.placeholder(tf.float64)
    x_sz1_ph = tf.placeholder(tf.float64)
    x_sz2_ph = tf.placeholder(tf.float64)

    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    # Decode the image as a JPEG file, this will turn it into a Tensor
    image = tf.image.decode_image(image_file, channels=3)
    image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)
    frame_sz = tf.shape(image)
    # used to pad the crops
    if design.pad_with_image_mean:
        avg_chan = tf.reduce_mean(image, axis=(0,1), name='avg_chan')
    else:
        avg_chan = None
    # pad with if necessary
    frame_padded_z, npad_z = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, z_sz_ph, avg_chan)
    frame_padded_z = tf.cast(frame_padded_z, tf.float32)
    # extract tensor of z_crops
    z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x_ph, pos_y_ph, z_sz_ph, design.exemplar_sz)
    frame_padded_x, npad_x = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, x_sz2_ph, avg_chan)
    frame_padded_x = tf.cast(frame_padded_x, tf.float32)
    # extract tensor of x_crops (3 scales)
    x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz0_ph, x_sz1_ph, x_sz2_ph, design.search_sz)
    # use crops as input of (MatConvnet imported) pre-trained fully-convolutional Siamese net
    template_z, templates_x, p_names_list, p_val_list = _create_siamese(os.path.join(env.root_pretrained,design.net), x_crops, z_crops)
    template_z = tf.squeeze(template_z)
    templates_z = tf.stack([template_z, template_z, template_z])
    # compare templates via cross-correlation
    scores_d = _match_templates(templates_z, templates_x, p_names_list, p_val_list)
    # upsample the score maps
    scores = tf.image.resize_images(scores_d, [final_score_sz, final_score_sz],
        method=tf.image.ResizeMethod.BICUBIC, align_corners=True)


    scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    # cosine window to penalize large displacements
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    run_opts = {}

    bboxes_video = []

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    t_start = time.time()

    #with tf.Session(graph=graph) as sess:
    with tf.Session(config=session_config) as sess:
        tf.global_variables_initializer().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Now each element in *_video lists is processed as the previous tracking
        # function, but within the overall session
        for i in range(len(frame_name_list_video)):

            # Get corresponding list for current frame
            frame_name_list = frame_name_list_video[i]
            pos_x = pos_x_video[i]
            pos_y = pos_y_video[i]
            target_w = target_w_video[i]
            target_h = target_h_video[i]



            context = design.context*(target_w+target_h)
            z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
            x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

            # thresholds to saturate patches shrinking/growing
            min_z = hp.scale_min * z_sz
            max_z = hp.scale_max * z_sz
            min_x = hp.scale_min * x_sz
            max_x = hp.scale_max * x_sz

            # save first frame position (from ground-truth)
            bboxes = np.zeros((num_frames,4))
            bboxes[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h

            templates_z_ = sess.run([templates_z], feed_dict={pos_x_ph: pos_x, pos_y_ph: pos_y, z_sz_ph: z_sz, filename: frame_name_list[0]})
            #image_, templates_z_ = sess.run([image, templates_z], feed_dict={pos_x_ph: pos_x, pos_y_ph: pos_y, z_sz_ph: z_sz, filename: frame_name_list[0]})

            new_templates_z_ = templates_z_


            num_frames = np.size(frame_name_list)

            # Get an image from the queue
            for j in range(1, num_frames):
                scaled_exemplar = z_sz * scale_factors
                scaled_search_area = x_sz * scale_factors
                scaled_target_w = target_w * scale_factors
                scaled_target_h = target_h * scale_factors
                image_, scores_ = sess.run(
                    [image, scores],
                    feed_dict={
                        pos_x_ph: pos_x,
                        pos_y_ph: pos_y,
                        x_sz0_ph: scaled_search_area[0],
                        x_sz1_ph: scaled_search_area[1],
                        x_sz2_ph: scaled_search_area[2],
                        templates_z: np.squeeze(templates_z_),
                        filename: frame_name_list[j],
                    }, **run_opts)
                scores_ = np.squeeze(scores_)
                # penalize change of scale
                scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
                scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
                # find scale with highest peak (after penalty)
                new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
                # update scaled sizes
                x_sz = (1-hp.scale_lr)*x_sz + hp.scale_lr*scaled_search_area[new_scale_id]
                target_w = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
                target_h = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
                # select response with new_scale_id
                score_ = scores_[new_scale_id,:,:]
                score_ = score_ - np.min(score_)
                score_ = score_/np.sum(score_)

                # apply displacement penalty
                score_ = (1-hp.window_influence)*score_ + hp.window_influence*penalty
                pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
                # convert <cx,cy,w,h> to <x,y,w,h> and save output
                bboxes[j,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
                # update the target representation with a rolling average
                if hp.z_lr>0:
                    new_templates_z_ = sess.run([templates_z], feed_dict={
                                                                    pos_x_ph: pos_x,
                                                                    pos_y_ph: pos_y,
                                                                    z_sz_ph: z_sz,
                                                                    image: image_
                                                                    })

                    templates_z_=(1-hp.z_lr)*np.asarray(templates_z_) + hp.z_lr*np.asarray(new_templates_z_)

                # update template patch size
                z_sz = (1-hp.scale_lr)*z_sz + hp.scale_lr*scaled_exemplar[new_scale_id]

                if run.visualization:
                    show_frame(image_, bboxes[j,:], 1)

            bboxes_video.append(bboxes)



        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)
    elapsed_time = time.time() - t_start
    print("Video processed in:{:.2f}".format(elapsed_time))
    plt.close('all')
    return bboxes_video, elapsed_time





# read default parameters and override with custom ones
#def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores, start_frame):
    #num_frames = np.size(frame_name_list)
    ## stores tracker's output for evaluation
    #bboxes = np.zeros((num_frames,4))

    #scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    ## cosine window to penalize large displacements
    #hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    #penalty = np.transpose(hann_1d) * hann_1d
    #penalty = penalty / np.sum(penalty)

    #context = design.context*(target_w+target_h)
    #z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
    #x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

    ## thresholds to saturate patches shrinking/growing
    #min_z = hp.scale_min * z_sz
    #max_z = hp.scale_max * z_sz
    #min_x = hp.scale_min * x_sz
    #max_x = hp.scale_max * x_sz

    ## run_metadata = tf.RunMetadata()
    ## run_opts = {
    ##     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    ##     'run_metadata': run_metadata,
    ## }

    #run_opts = {}

    ## with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #with tf.Session() as sess:
        #tf.global_variables_initializer().run()
        ## Coordinate the loading of image files.
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)
        
        ## save first frame position (from ground-truth)
        #bboxes[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h                

        #image_, templates_z_ = sess.run([image, templates_z], feed_dict={
                                                                        #siam.pos_x_ph: pos_x,
                                                                        #siam.pos_y_ph: pos_y,
                                                                        #siam.z_sz_ph: z_sz,
                                                                        #filename: frame_name_list[0]})
        #new_templates_z_ = templates_z_

        #t_start = time.time()

        ## Get an image from the queue
        #for i in range(1, num_frames):        
            #scaled_exemplar = z_sz * scale_factors
            #scaled_search_area = x_sz * scale_factors
            #scaled_target_w = target_w * scale_factors
            #scaled_target_h = target_h * scale_factors
            #image_, scores_ = sess.run(
                #[image, scores],
                #feed_dict={
                    #siam.pos_x_ph: pos_x,
                    #siam.pos_y_ph: pos_y,
                    #siam.x_sz0_ph: scaled_search_area[0],
                    #siam.x_sz1_ph: scaled_search_area[1],
                    #siam.x_sz2_ph: scaled_search_area[2],
                    #templates_z: np.squeeze(templates_z_),
                    #filename: frame_name_list[i],
                #}, **run_opts)
            #scores_ = np.squeeze(scores_)
            ## penalize change of scale
            #scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
            #scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
            ## find scale with highest peak (after penalty)
            #new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
            ## update scaled sizes
            #x_sz = (1-hp.scale_lr)*x_sz + hp.scale_lr*scaled_search_area[new_scale_id]        
            #target_w = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
            #target_h = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
            ## select response with new_scale_id
            #score_ = scores_[new_scale_id,:,:]
            #score_ = score_ - np.min(score_)
            #score_ = score_/np.sum(score_)
            ## apply displacement penalty
            #score_ = (1-hp.window_influence)*score_ + hp.window_influence*penalty
            #pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
            ## convert <cx,cy,w,h> to <x,y,w,h> and save output
            #bboxes[i,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
            ## update the target representation with a rolling average
            #if hp.z_lr>0:
                #new_templates_z_ = sess.run([templates_z], feed_dict={
                                                                #siam.pos_x_ph: pos_x,
                                                                #siam.pos_y_ph: pos_y,
                                                                #siam.z_sz_ph: z_sz,
                                                                #image: image_
                                                                #})

                #templates_z_=(1-hp.z_lr)*np.asarray(templates_z_) + hp.z_lr*np.asarray(new_templates_z_)
            
            ## update template patch size
            #z_sz = (1-hp.scale_lr)*z_sz + hp.scale_lr*scaled_exemplar[new_scale_id]
            
            #if run.visualization:
                #show_frame(image_, bboxes[i,:], 1)        
                

        #t_elapsed = time.time() - t_start
        #speed = num_frames/t_elapsed

        ## Finish off the filename queue coordinator.
        #coord.request_stop()
        #coord.join(threads) 

        ## from tensorflow.python.client import timeline
        ## trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        ## trace_file = open('timeline-search.ctf.json', 'w')
        ## trace_file.write(trace.generate_chrome_trace_format())

    #plt.close('all')

    #return bboxes, speed



def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y

# import pretrained Siamese network from matconvnet
def _create_siamese(net_path, net_x, net_z):
    # read mat file from net_path and start TF Siamese graph from placeholders X and Z
    params_names_list, params_values_list = _import_from_matconvnet(net_path)

    # loop through the flag arrays and re-construct network, reading parameters of conv and bnorm layers
    for i in range(_num_layers):
        print('> Layer '+str(i+1))
        # conv
        conv_W_name = _find_params('conv'+str(i+1)+'f', params_names_list)[0]
        conv_b_name = _find_params('conv'+str(i+1)+'b', params_names_list)[0]
        print('\t\tCONV: setting '+conv_W_name+' '+conv_b_name)
        print('\t\tCONV: stride '+str(_conv_stride[i])+', filter-group '+str(_filtergroup_yn[i]))
        conv_W = params_values_list[params_names_list.index(conv_W_name)]
        conv_b = params_values_list[params_names_list.index(conv_b_name)]
        # batchnorm
        if _bnorm_yn[i]:
            bn_beta_name = _find_params('bn'+str(i+1)+'b', params_names_list)[0]
            bn_gamma_name = _find_params('bn'+str(i+1)+'m', params_names_list)[0]
            bn_moments_name = _find_params('bn'+str(i+1)+'x', params_names_list)[0]
            print('\t\tBNORM: setting '+bn_beta_name+' '+bn_gamma_name+' '+bn_moments_name)
            bn_beta = params_values_list[params_names_list.index(bn_beta_name)]
            bn_gamma = params_values_list[params_names_list.index(bn_gamma_name)]
            bn_moments = params_values_list[params_names_list.index(bn_moments_name)]
            bn_moving_mean = bn_moments[:,0]
            bn_moving_variance = bn_moments[:,1]**2 # saved as std in matconvnet
        else:
            bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []
        
        # set up conv "block" with bnorm and activation 
        net_x = set_convolutional(net_x, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i], \
                            bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
                            filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
                            scope='conv'+str(i+1), reuse=False)
        
        # notice reuse=True for Siamese parameters sharing
        net_z = set_convolutional(net_z, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i], \
                            bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
                            filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
                            scope='conv'+str(i+1), reuse=True)    
        
        # add max pool if required
        if _pool_stride[i]>0:
            print('\t\tMAX-POOL: size '+str(_pool_sz)+ ' and stride '+str(_pool_stride[i]))
            net_x = tf.nn.max_pool(net_x, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
            net_z = tf.nn.max_pool(net_z, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))

    print

    return net_z, net_x, params_names_list, params_values_list


def _import_from_matconvnet(net_path):
    mat = scipy.io.loadmat(net_path)
    net_dot_mat = mat.get('net')
    # organize parameters to import
    params = net_dot_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in range(params_values.size)]
    return params_names_list, params_values_list


# find all parameters matching the codename (there should be only one)
def _find_params(x, params):
    matching = [s for s in params if x in s]
    assert len(matching)==1, ('Ambiguous param name found')    
    return matching


def _match_templates(net_z, net_x, params_names_list, params_values_list):
    # finalize network
    # z, x are [B, H, W, C]
    net_z = tf.transpose(net_z, perm=[1,2,0,3])
    net_x = tf.transpose(net_x, perm=[1,2,0,3])
    # z, x are [H, W, B, C]
    Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
    Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))
    # assert B==Bx, ('Z and X should have same Batch size')
    # assert C==Cx, ('Z and X should have same Channels number')
    net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
    net_x = tf.reshape(net_x, (1, Hx, Wx, B*C))
    net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')
    # final is [1, Hf, Wf, BC]
    net_final = tf.concat(tf.split(net_final, 3, axis=3), axis=0)
    # final is [B, Hf, Wf, C]
    net_final = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)
    # final is [B, Hf, Wf, 1]
    if _bnorm_adjust:
        bn_beta = params_values_list[params_names_list.index('fin_adjust_bnb')]
        bn_gamma = params_values_list[params_names_list.index('fin_adjust_bnm')]
        bn_moments = params_values_list[params_names_list.index('fin_adjust_bnx')]
        bn_moving_mean = bn_moments[:,0]
        bn_moving_variance = bn_moments[:,1]**2
        net_final = tf.layers.batch_normalization(net_final, beta_initializer=tf.constant_initializer(bn_beta),
                                                gamma_initializer=tf.constant_initializer(bn_gamma),
                                                moving_mean_initializer=tf.constant_initializer(bn_moving_mean),
                                                moving_variance_initializer=tf.constant_initializer(bn_moving_variance),
                                                training=False, trainable=False)

    return net_final
