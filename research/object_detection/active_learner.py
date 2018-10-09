

"""Extension of the training executal for detection models (train.py)

MORE DETAILS

"""


import pdb
import random
import numpy as np

import functools
import json
import os
import tensorflow as tf
import imp

from object_detection import trainer
from object_detection import evaluator_al as evaluator
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.save_subset_imagenetvid_tf_record import save_tf_record
from object_detection.utils import label_map_util

from PIL import Image
from object_detection.utils import visualization_utils as vis_utils

# Tracking module
import siamfc.siamese as siam
from siamfc.tracker import tracker
from siamfc.parse_arguments import parse_arguments
from siamfc.region_to_bbox import region_to_bbox


tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '/home/abel/DATA/faster_rcnn/resnet101_coco/checkpoints/',
                    'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string('perf_dir', '/home/abel/DATA/faster_rcnn/resnet101_coco/performances/',
                    'Directory to save performance json files.')
flags.DEFINE_string('data_dir', '/home/abel/DATA/ILSVRC/',
                    'Directory that contains data.')
flags.DEFINE_string('pipeline_config_path',
                    #'/home/abel/DATA/faster_rcnn/resnet101_coco/configs/faster_rcnn_resnet101_imagenetvid-active_learning.config',
                    '/home/abel/DATA/faster_rcnn/resnet101_coco/configs/faster_rcnn_resnet101_imagenetvid-active_learning_short.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string('name', 'TCFPshort',
                    'Name of method to run')
flags.DEFINE_string('cycles','10',
                    'Number of cycles')
flags.DEFINE_string('runs','3',
                    'Number of runs for each experiment')
flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

FLAGS = flags.FLAGS



# This should be a custom name per method once we can overwrite fields in
# pipeline_file
data_info = {'data_dir': FLAGS.data_dir,
          'annotations_dir':'Annotations',
          'label_map_path': './data/imagenetvid_label_map.pbtxt',
          #'set': 'train_ALL_clean'}
          'set': 'train_ALL_clean_short'}

# Harcoded keys to retrieve metrics
keyBike = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n03790512'
keyCar = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n02958343'
keyMotorbike = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n02834778'
keyAll = 'PascalBoxes_Precision/mAP@0.5IOU'


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
    videos = set([d['video'] for d in dataset])
    return dataset,videos

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
        # Temporarily add neighbors to active_set so they are ignored
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

def select_entropy_video(dataset,videos,active_set,detections):

        indices = []

        # We have detections only for the labeled dataset, be careful with indexing
        unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

        predictions = detections['scores_with_background']

        num_classes = 3+1

        for v in videos:

            # Select frames in current video
            frames = [f['idx'] for f in dataset if f['video'] == v and f['verified']]

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



def convert_boxes_wh(box):
    """ Detection boxes come as [ymin,xmin,ymax,xmax]
        We need [x,y,w,h] for tracking
    """
    whbox = np.array([box[1],box[0],box[3]-box[1],box[2]-box[0]])
    return whbox


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

def track_detections(dataset,videos,active_set,detections,groundtruths):

    thresh_detection = 0.5

    # Tracker configuration
    hp, evaluation, run, env, design = parse_arguments()

    final_score_sz = hp.response_up * (design.score_sz - 1) + 1


    # Reload module so element are in current graph, look for a better solution to this
    tf.reset_default_graph()
    imp.reload(siam)

    #gtrack = tf.Graph()
    #with gtrack.as_default() as gtr:
        #with gtr.name_scope("gtrack"):
    filename, image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env)

    # We have detections only for the labeled dataset, be careful with indexing
    unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

    # ARE DETECTIONS NMSed?
    detected_boxes = detections['boxes']
    detected_scores = detections['scores']
    detected_labels = detections['labels']

    gt_boxes = groundtruths['boxes']

    for v in videos:

        video_dir = os.path.join(FLAGS.data_dir,'Data','VID','train',v)

        # Select frames in current video (even those with wrong GTs)
        #frames = [f['filename'] for f in dataset if f['video'] == v]
        frames = [[f['idx'],f['filename']] for f in dataset if f['video'] == v and f['verified']]

        # Get maximium index of frames in video
        max_frame = np.max([f[0] for f in frames])

        # Get only those that are not labeled --> pick from these
        frames_unlabeled = [f for f in frames if f[0] in unlabeled_set]

        tc_sum = np.zeros(len(frames_unlabeled))
        tc_contrib = np.zeros(len(frames_unlabeled))

        for fu in frames_unlabeled:

            idx_frame_video = [f[0] for f in frames].index(fu[0])

            # Get boxes for current frame
            boxes_frame = detected_boxes[unlabeled_set.index(fu[0])]
            scores_frame = detected_scores[unlabeled_set.index(fu[0])]
            labels_frame = detected_labels[unlabeled_set.index(fu[0])]

            gt_frame = gt_boxes[unlabeled_set.index(fu[0])]

            pdb.set_trace()

            #curr_im = Image.open(os.path.join(video_dir,frames[idx_frame_video][1]))
            #im_w,im_h = curr_im.size
            #vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_frame,im_w,im_h))
            #curr_im.show()

            # Get surviving detections in frame
            idx_good_det = scores_frame > thresh_detection

            boxes_frame = boxes_frame[idx_good_det,:]
            labels_frame = labels_frame[idx_good_det]

            num_good_dets = labels_frame.shape[0]

            for idx_det in range(num_good_dets):


                # Visualize detection
                #curr_im = Image.open(os.path.join(video_dir,frames[idx_frame_video][1]))

                #vis_utils.visualize_boxes_and_labels_on_image_array(numpy.array(curr_im),boxes_frame[idx_det])

                # Convert [x y x y] to [y x w h]
                curr_box = convert_boxes_wh(boxes_frame[idx_det])

                # Forward frames in the video
                # I can't do this with list comprehension for some reason
                #frame_list = [frames[i] for i in range(idx_frame_video+1,idx_frame_video+4) if frames[i] in frames]
                frame_list = [os.path.join(video_dir,frames[idx_frame_video][1])]
                for t in range(1,10):
                    if idx_frame_video+t <= max_frame:
                        frame_list.append(os.path.join(video_dir,frames[idx_frame_video+t][1]))

                pos_x, pos_y, target_w, target_h = region_to_bbox(curr_box)
                pdb.set_trace()
                bboxes, speed = tracker(hp, run, design, frame_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores, 1)


            for t in range(1,4):
                if idx_frame_video+t <= max_frame:
                    iou = tracked_boxes[t]


        # Two frame lists, one forward, one backward

        # Only go through verified frames
        frames_verified = [f['idx'] for f in dataset if f['video'] == v and f['verified']]

        #for d in frames_verified
            # Select three forward frames


        ## Get only those that are not labeled
        #frames = [f for f in frames if f in unlabeled_set]

        ## If all frames of video are in active set, ignore video
        #if len(frames) > 0:
            ## Extract corresponding predictions
            #det_frames = [predictions[unlabeled_set.index(f)] for f in frames]

            ## Compute and summarize entropy
            #ent = np.array(compute_entropy(det_frames))

            ##idxR = random.randint(0,len(frames)-1)
            #idxR = ent.argmax(0)
            #indices.append(frames[idxR])
        ##print("Selecting frame {} from video {} with idx {}".format(idxR,v,frames[idxR]))
    #return indices



def visualize_detections(dataset, unlabeled_set, detections,groundtruths):
    detected_boxes = detections['boxes']
    detected_scores = detections['scores']
    detected_labels = detections['labels']

    gt_boxes = groundtruths['boxes']

    for f in dataset:
        if f['idx'] in unlabeled_set:
            im_path = os.path.join(FLAGS.data_dir,'Data','VID','train',f['video'],f['filename'])
            curr_im = Image.open(im_path)
            im_w,im_h = curr_im.size
            gt_im = gt_boxes[unlabeled_set.index(f['idx'])]
            vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(gt_im,im_w,im_h))
            det_im = detected_boxes[unlabeled_set.index(f['idx'])]
            vis_utils.draw_bounding_boxes_on_image(curr_im,normalize_box(det_im[:2,],im_w,im_h),color='green')
            curr_im.show()


#def main(_):
if __name__ == "__main__":


    assert FLAGS.train_dir, '`train_dir` is missing.'
    if FLAGS.task == 0: tf.gfile.MakeDirs(FLAGS.train_dir)
    if FLAGS.pipeline_config_path:
      configs = config_util.get_configs_from_pipeline_file(
          FLAGS.pipeline_config_path)
      if FLAGS.task == 0:
        tf.gfile.Copy(FLAGS.pipeline_config_path,
                      os.path.join(FLAGS.train_dir, 'pipeline.config'),
                      overwrite=True)
    else:
      configs = config_util.get_configs_from_multiple_files(
          model_config_path=FLAGS.model_config_path,
          train_config_path=FLAGS.train_config_path,
          train_input_config_path=FLAGS.input_config_path)
      if FLAGS.task == 0:
        for name, config in [('model.config', FLAGS.model_config_path),
                             ('train.config', FLAGS.train_config_path),
                             ('input.config', FLAGS.input_config_path)]:
          tf.gfile.Copy(config, os.path.join(FLAGS.train_dir, name),
                        overwrite=True)


    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']
    eval_config = configs['eval_config']
    eval_input_config = configs['eval_input_config']

    # Save number of test frames, as config is modified with unlabeled set
    num_eval_frames = eval_config.num_examples
    # Also original pointer to tfrecord
    tfrecord_eval = eval_input_config.tf_record_input_reader.input_path[0]

    # Get info about full dataset
    dataset,videos = get_dataset(data_info)

    # Get experiment information from FLAGS
    name = FLAGS.name
    num_cycles = int(FLAGS.cycles)
    num_runs = int(FLAGS.runs)
    num_steps = str(train_config.num_steps)

    output_file = FLAGS.perf_dir + name + 'r' + str(num_runs) + 'c' + str(num_cycles) + '.json'

    # Dictionary to save performance of every run
    performances = {}

    # This is the detection model to be used (Faster R-CNN)
    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    # We need a different one for testing
    eval_model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=False)


    # Input dict function for eval is always the same
    def get_next_eval(config):
       return dataset_builder.make_initializable_iterator(
           dataset_builder.build(config)).get_next()

    label_map = label_map_util.load_labelmap(eval_input_config.label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
          label_map, max_num_classes)

    # Run evaluation once only
    eval_config.max_evals = 1

    # Save path of pre-trained model
    pretrained_checkpoint = train_config.fine_tune_checkpoint

    for r in range(1,num_runs+1):

        # Active set starts empty
        active_set = []

        # Initial model is pre-trained
        train_config.fine_tune_checkpoint = pretrained_checkpoint

        for cycle in range(1,num_cycles+1):

            #### Training of current cycle
            train_dir = FLAGS.train_dir + name + 'run' + str(r) + 'cycle' +  str(cycle) + '/'

            # For first cycle, use random selection
            if ('Rnd' in name) or cycle==1:
                indices = select_random_video(dataset,videos,active_set)
            else:
                if ('Ent' in name):
                    indices = select_entropy_video(dataset,videos,active_set,detected_boxes)
                elif ('TCFP' in name):
                    indices = track_detections(dataset,videos,active_set,detected_boxes,groundtruth_boxes)

            active_set.extend(indices)

            data_info['output_path'] = FLAGS.data_dir + 'AL/tfrecords/' + name + 'run' + str(r) + 'cycle' +  str(cycle) + '.record'
            save_tf_record(data_info,active_set)

            input_config.tf_record_input_reader.input_path[0] = data_info['output_path']


            def get_next(config):
             return dataset_builder.make_initializable_iterator(
                dataset_builder.build(config)).get_next()

            create_input_dict_fn = functools.partial(get_next, input_config)


            env = json.loads(os.environ.get('TF_CONFIG', '{}'))
            cluster_data = env.get('cluster', None)
            cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
            task_data = env.get('task', None) or {'type': 'master', 'index': 0}
            task_info = type('TaskSpec', (object,), task_data)

            # Parameters for a single worker.
            ps_tasks = 0
            worker_replicas = 1
            worker_job_name = 'lonely_worker'
            task = 0
            is_chief = True
            master = ''

            if cluster_data and 'worker' in cluster_data:
            # Number of total worker replicas include "worker"s and the "master".
                worker_replicas = len(cluster_data['worker']) + 1
            if cluster_data and 'ps' in cluster_data:
                ps_tasks = len(cluster_data['ps'])

            if worker_replicas > 1 and ps_tasks < 1:
                raise ValueError('At least 1 ps task is needed for distributed training.')

            if worker_replicas >= 1 and ps_tasks > 0:
            # Set up distributed training.
                server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                     job_name=task_info.type,
                                     task_index=task_info.index)
                if task_info.type == 'ps':
                  server.join()
                  #return

                worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
                task = task_info.index
                is_chief = (task_info.type == 'master')
                master = server.target

            graph_rewriter_fn = None
            if 'graph_rewriter_config' in configs:
                graph_rewriter_fn = graph_rewriter_builder.build(
                    configs['graph_rewriter_config'], is_training=True)


            trainer.train(
              create_input_dict_fn,
              model_fn,
              train_config,
              master,
              task,
              FLAGS.num_clones,
              worker_replicas,
              FLAGS.clone_on_cpu,
              ps_tasks,
              worker_job_name,
              is_chief,
              train_dir,
              graph_hook_fn=graph_rewriter_fn)

            #Save active_set in train dir in case we want to restart training
            with open(train_dir + 'active_set.txt', 'w') as f:
                for item in active_set:
                    f.write('{}\n'.format(item))

            #### Evaluation of trained model on unlabeled set to obtain data
            if 'Rnd' not in name:

               # Get unlabeled set
                data_info['output_path'] = FLAGS.data_dir + 'AL/tfrecords/' + name + 'run' + str(r) + 'cycle' +  str(cycle) + '_unlabeled.record'

                # Remove those with wrong annotations, save some time
                unlabeled_set = [i for i in range(len(dataset)) if i not in active_set]

                # Short version, only save a subset
                #unlabeled_set = unlabeled_set[:100]

                save_tf_record(data_info,unlabeled_set)

                print('Unlabeled frames in the dataset: {}'.format(len(unlabeled_set)))

                # Set number of eval images to number of unlabeled samples and point to tfrecord

                eval_input_config.tf_record_input_reader.input_path[0] = data_info['output_path']
                eval_config.num_examples = len(unlabeled_set)

                eval_train_dir = train_dir + 'eval_train/'

                def get_next_eval_train(config):
                   return dataset_builder.make_initializable_iterator(
                        dataset_builder.build(config)).get_next()

                # Initialize input dict again (necessary?)
                create_eval_train_input_dict_fn = functools.partial(get_next_eval_train, eval_input_config)

                graph_rewriter_fn = None
                if 'graph_rewriter_config' in configs:
                    graph_rewriter_fn = graph_rewriter_builder.build(
                        configs['graph_rewriter_config'], is_training=False)

                # Need to reset graph for evaluation
                tf.reset_default_graph()

                metrics, detected_boxes, groundtruth_boxes = evaluator.evaluate(
                  create_eval_train_input_dict_fn,
                  eval_model_fn,
                  eval_config,
                  categories,
                  train_dir,
                  eval_train_dir,
                  graph_hook_fn=graph_rewriter_fn)

                # Put boxes information somewhere
                #pdb.set_trace()
                #visualize_detections(dataset, unlabeled_set, detected_boxes, groundtruth_boxes)

                print('Done computing detections in training set')


            #### Evaluation of trained model on test set to record performance
            eval_dir = train_dir + 'eval/'

            # Input dict function for eval is always the same
            def get_next_eval(config):
               return dataset_builder.make_initializable_iterator(
                   dataset_builder.build(config)).get_next()

            # Restore eval configuration on test
            eval_config.num_examples = num_eval_frames
            eval_input_config.tf_record_input_reader.input_path[0] = tfrecord_eval
            eval_input_config.tf_record_input_reader.input_path[0] = data_info['output_path']

            # Initialize input dict again (necessary?)
            create_eval_input_dict_fn = functools.partial(get_next_eval, eval_input_config)

            graph_rewriter_fn = None
            if 'graph_rewriter_config' in configs:
                graph_rewriter_fn = graph_rewriter_builder.build(
                    configs['graph_rewriter_config'], is_training=False)

            # Need to reset graph for evaluation
            tf.reset_default_graph()

            metrics,_,_ = evaluator.evaluate(
              create_eval_input_dict_fn,
              eval_model_fn,
              eval_config,
              categories,
              train_dir,
              eval_dir,
              graph_hook_fn=graph_rewriter_fn)


            aps = [metrics[keyAll],[metrics[keyBike], metrics[keyCar],metrics[keyMotorbike]]]


            performances['run'+str(r)+'c'+str(cycle)]= aps

            json_str = json.dumps(performances)
            f = open(output_file,'w')
            f.write(json_str)
            f.close()



            # Update initial model, add latest cycle
            #train_config.fine_tune_checkpoint = train_dir + 'model.ckpt-' + num_steps




#metrics,_,_ = evaluator.evaluate(create_eval_input_dict_fn, eval_model_fn, eval_config, categories, train_dir, eval_dir, graph_hook_fn=graph_rewriter_fn)

#trainer.train(create_input_dict_fn, model_fn, train_config, master, task, FLAGS.num_clones, worker_replicas,FLAGS.clone_on_cpu,ps_tasks, worker_job_name,is_chief,train_dir, graph_hook_fn=graph_rewriter_fn)
