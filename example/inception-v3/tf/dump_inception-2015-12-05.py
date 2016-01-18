
import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import h5py
import sys, os, re, time
import math



class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          'imagenet', 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          'imagenet', 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup, self.node_id_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name, node_id_to_uid

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

  def id_to_uid(self, node_id):
    if node_id not in self.node_id_lookup:
      return ''
    return self.node_id_lookup[node_id]


def create_graph(graph_def_pb='classify_image_graph_def.pb'):
  with gfile.FastGFile(graph_def_pb, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


filename_list = [entry.strip().split(' ') for entry in open( 
  '/storage/ImageNet/ILSVRC2012/val_synset.txt', 
  'r'
)]
synset_label_map = [entry.strip().split(' ')[0] for entry in open(
  '/storage/ImageNet/ILSVRC2012/synset_words.txt',
  'r'
)]
synset_label_dic = {}
for id, synset in enumerate(synset_label_map):
  synset_label_dic[synset] = id

path_prefix = '/storage/ImageNet/ILSVRC2012/val/%s'
num_top_predictions = 5

graph_def_pb = 'imagenet/classify_image_graph_def.pb'
create_graph(graph_def_pb)
node_lookup = NodeLookup()

sess = tf.Session(
  config=tf.ConfigProto(
    log_device_placement=False
  )
)
softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

import pdb; pdb.set_trace()
#
# start for 1 sample
#
filepath = 'imagenet/cropped_panda.jpg'
image_data = gfile.FastGFile(filepath, 'rb').read()
predictions = sess.run(softmax_tensor, 
  {'DecodeJpeg/contents:0': image_data})
predictions = np.squeeze(predictions)
top_k = predictions.argsort()[-num_top_predictions:][::-1]

#import pdb; pdb.set_trace()
for node_id in top_k:
  human_string = node_lookup.id_to_string(node_id)
  uid = node_lookup.id_to_uid(node_id)
  score = predictions[node_id]
  print('(predicted: %d, score = %.5f), %s' % (node_id, score, human_string))

#
# dump
#
def make_padding(padding_name, conv_shape):
  if padding_name == "VALID":
    return [0, 0]
  elif padding_name == "SAME":
    return [int(math.ceil(conv_shape[0]/2)), int(math.ceil(conv_shape[1]/2))]
  else:
    sys.exit('Invalid padding name '+padding_name)


def dump_convbn(sess, gname):
  conv = sess.graph.get_operation_by_name(gname + '/Conv2D')

  weights = sess.graph.get_tensor_by_name(gname + '/conv2d_params:0').eval(session=sess)
  padding = make_padding(conv.get_attr("padding"), weights.shape)
  strides = conv.get_attr("strides")

  beta = sess.graph.get_tensor_by_name(gname + '/batchnorm/beta:0').eval(session=sess)
  gamma= sess.graph.get_tensor_by_name(gname + '/batchnorm/gamma:0').eval(session=sess)
  mean = sess.graph.get_tensor_by_name(gname + '/batchnorm/moving_mean:0').eval(session=sess)
  std  = sess.graph.get_tensor_by_name(gname + '/batchnorm/moving_variance:0').eval(session=sess)

  gname = gname.replace("/", "_")
  h5f = h5py.File('dump/'+gname+'.h5', 'w')
  h5f.create_dataset("weights", data=weights)
  h5f.create_dataset("strides", data=strides)
  h5f.create_dataset("padding", data=padding)
  h5f.create_dataset("beta", data=beta)
  h5f.create_dataset("gamma", data=gamma)
  h5f.create_dataset("mean", data=mean)
  h5f.create_dataset("std", data=std)
  h5f.close()


def dump_pool(sess, gname):
  pool = sess.graph.get_operation_by_name(gname)
  ismax = pool.type=='MaxPool' and 1 or 0
  ksize = pool.get_attr("ksize")
  padding = make_padding(pool.get_attr("padding"), ksize[1:3])
  strides = pool.get_attr("strides")

  gname = gname.replace("/", "_")
  h5f = h5py.File('dump/'+gname+'.h5', 'w')
  h5f.create_dataset("ismax", data=[ismax])
  h5f.create_dataset("ksize", data=ksize)
  h5f.create_dataset("padding", data=padding)
  h5f.create_dataset("strides", data=strides)
  h5f.close()


def dump_softmax(sess):
  softmax_w = sess.graph.get_tensor_by_name('softmax/weights:0').eval(session=sess)
  softmax_b = sess.graph.get_tensor_by_name('softmax/biases:0').eval(session=sess)
  h5f = h5py.File('dump/softmax.h5', 'w')
  h5f.create_dataset("weights", data=softmax_w)
  h5f.create_dataset("biases", data=softmax_b)
  h5f.close()



def dump_filters(sess):
  # dump the filters
  dump_convbn(sess, 'conv')
  dump_convbn(sess, 'conv_1')
  dump_convbn(sess, 'conv_2')
  dump_pool(sess,   'pool')
  dump_convbn(sess, 'conv_3')
  dump_convbn(sess, 'conv_4')
  dump_pool(sess,   'pool_1')

  # inceptions with 1x1, 3x3, 5x5 convolutions
  dump_convbn(sess, 'mixed/conv')
  dump_convbn(sess, 'mixed/tower/conv')
  dump_convbn(sess, 'mixed/tower/conv_1')
  dump_convbn(sess, 'mixed/tower_1/conv')
  dump_convbn(sess, 'mixed/tower_1/conv_1')
  dump_convbn(sess, 'mixed/tower_1/conv_2')
  dump_pool(sess,   'mixed/tower_2/pool')
  dump_convbn(sess, 'mixed/tower_2/conv')

  dump_convbn(sess, 'mixed_1/conv')
  dump_convbn(sess, 'mixed_1/tower/conv')
  dump_convbn(sess, 'mixed_1/tower/conv_1')
  dump_convbn(sess, 'mixed_1/tower_1/conv')
  dump_convbn(sess, 'mixed_1/tower_1/conv_1')
  dump_convbn(sess, 'mixed_1/tower_1/conv_2')
  dump_pool(sess,   'mixed_1/tower_2/pool')
  dump_convbn(sess, 'mixed_1/tower_2/conv')

  dump_convbn(sess, 'mixed_2/conv')
  dump_convbn(sess, 'mixed_2/tower/conv')
  dump_convbn(sess, 'mixed_2/tower/conv_1')
  dump_convbn(sess, 'mixed_2/tower_1/conv')
  dump_convbn(sess, 'mixed_2/tower_1/conv_1')
  dump_convbn(sess, 'mixed_2/tower_1/conv_2')
  dump_pool(sess,   'mixed_2/tower_2/pool')
  dump_convbn(sess, 'mixed_2/tower_2/conv')

  # inceptions with 1x1, 3x3(in sequence) convolutions
  dump_convbn(sess, 'mixed_3/conv')
  dump_convbn(sess, 'mixed_3/tower/conv')
  dump_convbn(sess, 'mixed_3/tower/conv_1')
  dump_convbn(sess, 'mixed_3/tower/conv_2')
  dump_pool(sess,   'mixed_3/pool')

  # inceptions with 1x1, 7x1, 1x7 convolutions
  dump_convbn(sess, 'mixed_4/conv')
  dump_convbn(sess, 'mixed_4/tower/conv')
  dump_convbn(sess, 'mixed_4/tower/conv_1')
  dump_convbn(sess, 'mixed_4/tower/conv_2')
  dump_convbn(sess, 'mixed_4/tower_1/conv')
  dump_convbn(sess, 'mixed_4/tower_1/conv_1')
  dump_convbn(sess, 'mixed_4/tower_1/conv_2')
  dump_convbn(sess, 'mixed_4/tower_1/conv_3')
  dump_convbn(sess, 'mixed_4/tower_1/conv_4')
  dump_pool(sess,   'mixed_4/tower_2/pool')
  dump_convbn(sess, 'mixed_4/tower_2/conv')

  dump_convbn(sess, 'mixed_5/conv')
  dump_convbn(sess, 'mixed_5/tower/conv')
  dump_convbn(sess, 'mixed_5/tower/conv_1')
  dump_convbn(sess, 'mixed_5/tower/conv_2')
  dump_convbn(sess, 'mixed_5/tower_1/conv')
  dump_convbn(sess, 'mixed_5/tower_1/conv_1')
  dump_convbn(sess, 'mixed_5/tower_1/conv_2')
  dump_convbn(sess, 'mixed_5/tower_1/conv_3')
  dump_convbn(sess, 'mixed_5/tower_1/conv_4')
  dump_pool(sess,   'mixed_5/tower_2/pool')
  dump_convbn(sess, 'mixed_5/tower_2/conv')

  dump_convbn(sess, 'mixed_6/conv')
  dump_convbn(sess, 'mixed_6/tower/conv')
  dump_convbn(sess, 'mixed_6/tower/conv_1')
  dump_convbn(sess, 'mixed_6/tower/conv_2')
  dump_convbn(sess, 'mixed_6/tower_1/conv')
  dump_convbn(sess, 'mixed_6/tower_1/conv_1')
  dump_convbn(sess, 'mixed_6/tower_1/conv_2')
  dump_convbn(sess, 'mixed_6/tower_1/conv_3')
  dump_convbn(sess, 'mixed_6/tower_1/conv_4')
  dump_pool(sess,   'mixed_6/tower_2/pool')
  dump_convbn(sess, 'mixed_6/tower_2/conv')

  dump_convbn(sess, 'mixed_7/conv')
  dump_convbn(sess, 'mixed_7/tower/conv')
  dump_convbn(sess, 'mixed_7/tower/conv_1')
  dump_convbn(sess, 'mixed_7/tower/conv_2')
  dump_convbn(sess, 'mixed_7/tower_1/conv')
  dump_convbn(sess, 'mixed_7/tower_1/conv_1')
  dump_convbn(sess, 'mixed_7/tower_1/conv_2')
  dump_convbn(sess, 'mixed_7/tower_1/conv_3')
  dump_convbn(sess, 'mixed_7/tower_1/conv_4')
  dump_pool(sess,   'mixed_7/tower_2/pool')
  dump_convbn(sess, 'mixed_7/tower_2/conv')

  # inceptions with 1x1, 3x3, 1x7, 7x1 filters
  dump_convbn(sess, 'mixed_8/tower/conv')
  dump_convbn(sess, 'mixed_8/tower/conv_1')
  dump_convbn(sess, 'mixed_8/tower_1/conv')
  dump_convbn(sess, 'mixed_8/tower_1/conv_1')
  dump_convbn(sess, 'mixed_8/tower_1/conv_2')
  dump_convbn(sess, 'mixed_8/tower_1/conv_3')
  dump_pool(sess,   'mixed_8/pool')

  dump_convbn(sess, 'mixed_9/conv')
  dump_convbn(sess, 'mixed_9/tower/conv')
  dump_convbn(sess, 'mixed_9/tower/mixed/conv')
  dump_convbn(sess, 'mixed_9/tower/mixed/conv_1')
  dump_convbn(sess, 'mixed_9/tower_1/conv')
  dump_convbn(sess, 'mixed_9/tower_1/conv_1')
  dump_convbn(sess, 'mixed_9/tower_1/mixed/conv')
  dump_convbn(sess, 'mixed_9/tower_1/mixed/conv_1')
  dump_pool(sess,   'mixed_9/tower_2/pool')
  dump_convbn(sess, 'mixed_9/tower_2/conv')

  dump_convbn(sess, 'mixed_10/conv')
  dump_convbn(sess, 'mixed_10/tower/conv')
  dump_convbn(sess, 'mixed_10/tower/mixed/conv')
  dump_convbn(sess, 'mixed_10/tower/mixed/conv_1')
  dump_convbn(sess, 'mixed_10/tower_1/conv')
  dump_convbn(sess, 'mixed_10/tower_1/conv_1')
  dump_convbn(sess, 'mixed_10/tower_1/mixed/conv')
  dump_convbn(sess, 'mixed_10/tower_1/mixed/conv_1')
  dump_pool(sess,   'mixed_10/tower_2/pool')
  dump_convbn(sess, 'mixed_10/tower_2/conv')

  dump_pool(sess, "pool_3")
  dump_softmax(sess)


if not os.path.exists("dump"):
  os.makedirs("dump")

import pdb; pdb.set_trace()
dump_filters(sess)

#
# start prediction for ILSVRC12 val. set
#
top_1 = 0
top_5 = 0
for n, item in enumerate(filename_list):
  filepath = path_prefix % item[0]
  label = int(item[1])

  if not gfile.Exists(filepath):
    tf.logging.fatal('File does not exist %s', filepath)
  image_data = gfile.FastGFile(filepath, 'rb').read()

  #import pdb; pdb.set_trace()
  start_predict = time.time()
  predictions = sess.run(softmax_tensor, 
    {'DecodeJpeg/contents:0': image_data})
  elapsed_predict = time.time() - start_predict

  predictions = np.squeeze(predictions)
  top_k = predictions.argsort()[-num_top_predictions:][::-1]

  for k, node_id in enumerate(top_k):
    uid = node_lookup.id_to_uid(node_id)
    if k == 0 and synset_label_dic[uid] == label:
      top_1 += 1.0
      top_5 += 1.0
      break
    if k > 0 and synset_label_dic[uid] == label:
      top_5 += 1.0
      break

  print('%s, top@1: %d/%d = %.4f, top@5: %d/%d = %.4f in %.3f sec.' % \
    (n+1, 
     top_1, n+1, top_1/(n+1)*100, 
     top_5, n+1, top_5/(n+1)*100, 
     elapsed_predict))
  sys.stdout.flush()

