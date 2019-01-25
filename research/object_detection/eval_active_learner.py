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
from object_detection.utils import np_box_ops
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops

from pycocotools import mask

from PIL import Image
from object_detection.utils import visualization_utils as vis_utils

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.WARN)

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
flags.DEFINE_string('train_dir', '/home/abel/DATA/faster_rcnn/resnet50_coco/checkpoints/',
                    'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string('perf_dir', '/home/abel/DATA/faster_rcnn/resnet50_coco/performances/',
                    'Directory to save performance json files.')
flags.DEFINE_string('data_dir', '/home/abel/DATA/ILSVRC/',
                    'Directory that contains data.')
flags.DEFINE_string('pipeline_config_path',
                    '/home/abel/DATA/faster_rcnn/resnet50_coco/configs/faster_rcnn_resnet50_imagenetvid-active_learning-fR5.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string('name', 'RndxVidFrom0',
                    'Name of method to run')
flags.DEFINE_integer('cycles','20',
                    'Number of cycles')
flags.DEFINE_integer('start_from_cycle','1',
                    'Cycle from which we want to start evaluating')
flags.DEFINE_boolean('during_training','False',
                    'Indicates whether the evaluation is running during training or not')
flags.DEFINE_integer('run','1',
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

# Harcoded keys to retrieve metrics
keyBike = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n03790512'
keyCar = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n02958343'
keyMotorbike = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/n02834778'
keyAll = 'PascalBoxes_Precision/mAP@0.5IOU'

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

    # Get experiment information from FLAGS
    name = FLAGS.name
    num_cycles = FLAGS.cycles
    run_num = FLAGS.run
    num_steps = str(train_config.num_steps)


    output_file = FLAGS.perf_dir + name + 'R' + str(run_num) + 'c' + str(num_cycles) + '.json'

    # Dictionary to save performance of every run
    performances = {}

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

    cycle = int(FLAGS.start_from_cycle)

    train_dir = FLAGS.train_dir + name + 'R' + str(run_num) + 'cycle' +  str(cycle) + '/'
    future_train_dir = FLAGS.train_dir + name + 'R' + str(run_num) + 'cycle' + str(cycle+1) + '/'

    while True:
        #### Evaluation of trained model on test set to record performance
        eval_dir = train_dir + 'eval/'

        # Input dict function for eval is always the same
        def get_next_eval(config):
           return dataset_builder.make_initializable_iterator(
               dataset_builder.build(config)).get_next()

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
        performances['R'+str(run_num)+'c'+str(cycle)]= aps

        # Write current performance
        json_str = json.dumps(performances)
        f = open(output_file,'w')
        f.write(json_str)
        f.close()

        # Done with previous cycle
        if os.path.exists(future_train_dir+'checkpoint'):

            if FLAGS.during_training:
            # When also running during training, we need to run it one last time before moving on
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
                performances['R'+str(run_num)+'c'+str(cycle)]= aps

                # Write current performance
                json_str = json.dumps(performances)
                f = open(output_file,'w')
                f.write(json_str)
                f.close()

            cycle +=1
            train_dir = future_train_dir
            future_train_dir = FLAGS.train_dir + name + 'R' + str(run_num) + 'cycle' + str(cycle+1) + '/'


