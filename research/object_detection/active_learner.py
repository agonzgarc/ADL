

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
import pickle

from object_detection import trainer
from object_detection import selection_funcs as sel
from object_detection import evaluator_al as evaluator
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.save_subset_imagenetvid_tf_record import save_tf_record
from object_detection.utils import label_map_util
from object_detection.utils import np_box_ops
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops

from pycocotools import mask

from PIL import Image
from object_detection.utils import visualization_utils as vis_utils


tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)
#tf.logging.set_verbosity(tf.logging.WARN)

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
                    #'/home/abel/DATA/faster_rcnn/resnet101_coco/configs/faster_rcnn_resnet101_imagenetvid-active_learning_short.config',
                    '/home/abel/DATA/faster_rcnn/resnet101_coco/configs/faster_rcnn_resnet101_imagenetvid-active_learning-fR5.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
#flags.DEFINE_string('name', 'Rnd-FullDVideoExt',
flags.DEFINE_string('name', 'LstxVid',
                    'Name of method to run')
flags.DEFINE_string('cycles','20',
                    'Number of cycles')
flags.DEFINE_string('epochs','10',
                    'Number of epochs')
flags.DEFINE_string('restart_from_cycle','2',
                    'Cycle from which we want to restart training, if any')
flags.DEFINE_string('run','1',
                    'Number of current run')
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
          'set': 'train_150K_clean'}
          #'set': 'train_ALL_clean_short'}

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

#def nms_detections(boxes,scores,labels,thresh_nms = 0.8):
    #boxlist = np_box_list.BoxList(boxes)
    #boxlist.add_field('scores',scores)

def visualize_detections(dataset, unlabeled_set, detections, groundtruths):
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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

    # Get info about full dataset
    dataset,videos = get_dataset(data_info)

    num_videos = len(videos)

    # Get experiment information from FLAGS
    name = FLAGS.name
    num_cycles = int(FLAGS.cycles)
    run_num = int(FLAGS.run)
    num_steps = str(train_config.num_steps)
    epochs = int(FLAGS.epochs)
    restart_cycle = int(FLAGS.restart_from_cycle)

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


    label_map = label_map_util.load_labelmap(eval_input_config.label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
          label_map, max_num_classes)

    # Run evaluation once only
    eval_config.max_evals = 1

    # Config now indicates directory where we save all first cycle models
    pretrained_checkpoint = train_config.fine_tune_checkpoint


    # Load active set from cycle 0 and point to right model
    if restart_cycle==0:
        train_dir = FLAGS.train_dir + 'R' + str(run_num) + 'cycle0/'
        train_config.fine_tune_checkpoint = train_dir + 'model.ckpt'
    else:
        train_dir = FLAGS.train_dir + name + 'R' + str(run_num) + 'cycle' + str(restart_cycle) + '/'
        # Get actual checkpoint model
        with open(train_dir+'checkpoint','r') as cfile:
            line = cfile.readlines()
            train_config.fine_tune_checkpoint = line[0].split(' ')[1][1:-2]


    active_set = []
    with open(train_dir + 'active_set.txt', 'r') as f:
        for line in f:
            active_set.append(int(line))

    for cycle in range(restart_cycle+1,num_cycles+1):


        #### Evaluation of trained model on unlabeled set to obtain data for selection

        if 'Rnd' not in name and cycle < num_cycles:

            eval_train_dir = train_dir + name + 'R' + str(run_num) + 'cycle' +  str(cycle) + 'eval_train/'
            if os.path.exists(eval_train_dir + 'detections.dat'):
                with open(eval_train_dir + 'detections.dat','rb') as infile:
                ###### pdb remove latinq
                    #detected_boxes = pickle.load(infile)
                    detected_boxes = pickle.load(infile,encoding='latin1')
            else:

                # Get unlabeled set
                data_info['output_path'] = FLAGS.data_dir + 'AL/tfrecords/' + name + 'R' + str(run_num) + 'cycle' +  str(cycle) + '_unlabeled.record'

                # Do not evaluate labeled samples, their neighbors or unverified frames
                aug_active_set =  sel.augment_active_set(dataset,videos,active_set,num_neighbors=5)

                unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]


                # For TCFP, we need to get detections for pretty much every frame,
                # as not candidates can may be used to support candidates
                if ('TCFP' in name):
                    unlabeled_set = [i for i in range(len(dataset))]

                print('Unlabeled frames in the dataset: {}'.format(len(unlabeled_set)))

                save_tf_record(data_info,unlabeled_set)

                # Set number of eval images to number of unlabeled samples and point to tfrecord
                eval_input_config.tf_record_input_reader.input_path[0] = data_info['output_path']
                eval_config.num_examples = len(unlabeled_set)


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

                #visualize_detections(dataset, unlabeled_set, detected_boxes, groundtruth_boxes)
                with open(eval_train_dir + 'detections.dat','wb') as outfile:
                    pickle.dump(detected_boxes,outfile, protocol=pickle.HIGHEST_PROTOCOL)

                print('Done computing detections in training set')


                # Remove tfrecord used for training
                if os.path.exists(data_info['output_path']):
                    os.remove(data_info['output_path'])


        #### Training of current cycle
        train_dir = FLAGS.train_dir + name + 'R' + str(run_num) + 'cycle' +  str(cycle) + '/'

        # Budget for each cycle is the number of videos (0.5% of train set)
        if ('Rnd' in name):
            #indices = select_random_video(dataset,videos,active_set)
            #indices = sel.select_random(dataset,videos,active_set,budget=num_videos)
            indices = sel.select_random_video(dataset,videos,active_set)
        else:
            if ('Ent' in name):
                indices = sel.select_entropy_video(dataset,videos,FLAGS.data_dir,active_set,detected_boxes)
            elif ('Lst' in name):
                indices = sel.select_least_confident_video(dataset,videos,active_set,detected_boxes)
            elif ('TCFP' in name):
                indices = sel.select_TCFP_per_video(dataset,videos,FLAGS.data_dir,active_set,detected_boxes)

        active_set.extend(indices)

        data_info['output_path'] = FLAGS.data_dir + 'AL/tfrecords/' + name + 'R' + str(run_num) + 'cycle' +  str(cycle) + '.record'
        save_tf_record(data_info,active_set)

        input_config.tf_record_input_reader.input_path[0] = data_info['output_path']

        # Set number of steps based on epochs
        train_config.num_steps = epochs*len(active_set)

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

        # Remove tfrecord used for training
        if os.path.exists(data_info['output_path']):
            os.remove(data_info['output_path'])


            # Update initial model, add latest cycle
            #train_config.fine_tune_checkpoint = train_dir + 'model.ckpt-' + num_steps




#metrics, detected_boxes, groundtruth_boxes = evaluator.evaluate( create_eval_train_input_dict_fn, eval_model_fn, eval_config, categories, train_dir, eval_train_dir, graph_hook_fn=graph_rewriter_fn)

#trainer.train(create_input_dict_fn, model_fn, train_config, master, task, FLAGS.num_clones, worker_replicas,FLAGS.clone_on_cpu,ps_tasks, worker_job_name,is_chief,train_dir, graph_hook_fn=graph_rewriter_fn)
