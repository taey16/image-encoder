
import numpy as np
from sklearn.metrics import precision_recall_curve

entries = [entry.strip().split(' ')[1:3] \
  for entry in open('log_china_collar.log.txt', 'r')]

entries = np.array(entries, dtype=np.float32)
entries[:,1] -= 1

import pdb; pdb.set_trace()
precision = dict()
recall = dict()
precision, recall, _ = precision_recall_curve(entries[:,1], entries[:,0])
