

"""Extension of the training executal for detection models (train.py)

MORE DETAILS

"""


import pdb
import random

import functools
import json
import os
import tensorflow as tf

from object_detection import trainer
from object_detection import evaluator
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.save_subset_imagenetvid_tf_record import save_tf_record
from object_detection.utils import label_map_util

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
flags.DEFINE_string('pipeline_config_path', '/home/abel/DATA/faster_rcnn/resnet101_coco/configs/faster_rcnn_resnet101_imagenetvid-active_learning_short.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

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
          #'output_path': '/home/abel/DATA/ILSVRC/AL/tfrecords/imagenet_train_active_set.record',
          'annotations_dir':'Annotations',
          'label_map_path': './data/imagenetvid_label_map.pbtxt',
          'set': 'train_ALL'}

# Harcoded keys to retrieve metrics 
keyBike = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n03790512'
keyCar = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n02958343'
keyMotorbike = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n02834778'
keyAll = 'PascalBoxes_Precision/mAP@0.5IOU'

perf_dir = '/home/abel/DATA/faster_rcnn/resnet101_coco/performances/'

def get_dataset():
    dataset = []
    path_file = FLAGS.data_dir + '/AL/train_ALL.txt'
    with open(path_file,'r') as pF:
        idx = 0
        for line in pF:
            # Remove trailing \n
            path = line[:-1]
            split_path = path.split('/')
            filename = split_path[-1]
            video = split_path[-3]+'/'+split_path[-2]
            dataset.append({'idx':idx,'filename':filename,'video':video})
            idx+=1
    videos = set([d['video'] for d in dataset])
    return dataset,videos

def selectRandomPerVideo(dataset,videos,active_set):
        indices = []
        for v in videos:
            #Select frames in current video
            frames = [f['idx'] for f in dataset if f['video'] == v]

            # Remove if already in active set
            frames = [f for f in frames if f not in active_set]

            # If all frames of video are in active set, ignore video
            if len(frames) > 0:
                idxR = random.randint(1,len(frames)-1)
                indices.append(frames[idxR])
            #print("Selecting frame {} from video {} with idx {}".format(idxR,v,frames[idxR]))
        return indices


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
    dataset,videos = get_dataset()

    # Make these arguments
    name = 'Rnd_short'
    num_cycles = 5
    num_runs = 3
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

    for r in range(1,num_runs+1):

        # Active set starts empty 
        active_set = []

        for cycle in range(1,num_cycles+1):

            #### Training of current cycle
            train_dir = FLAGS.train_dir + name + 'run' + str(r) + 'cycle' +  str(cycle) + '/'

            indices = selectRandomPerVideo(dataset,videos,active_set)
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


            #### Evaluation of trained model
            eval_dir = train_dir + 'eval/'

            # Initialize input dict again (necessary?)
            create_eval_input_dict_fn = functools.partial(get_next_eval, eval_input_config)

            graph_rewriter_fn = None
            if 'graph_rewriter_config' in configs:
                graph_rewriter_fn = graph_rewriter_builder.build(
                    configs['graph_rewriter_config'], is_training=False)

            #if cycle == 2:
                #pdb.set_trace()
            # Need to reset graph for evaluation
            tf.reset_default_graph()

            metrics = evaluator.evaluate(
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
            train_config.fine_tune_checkpoint = train_dir + 'model.ckpt-' + num_steps




metrics = evaluator.evaluate(create_eval_input_dict_fn, eval_model_fn, eval_config, categories, train_dir, eval_dir, graph_hook_fn=graph_rewriter_fn)

#trainer.train(create_input_dict_fn, model_fn, train_config, master, task, FLAGS.num_clones, worker_replicas,FLAGS.clone_on_cpu,ps_tasks, worker_job_name,is_chief,train_dir, graph_hook_fn=graph_rewriter_fn)
