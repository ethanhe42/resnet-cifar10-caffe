import sys
import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread

import numpy as np
import lmdb
import caffe

def load_CIFAR_batch(filename, pad=True):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).astype(np.uint8)
    padded = np.zeros((10000, 3, 40, 40), dtype=np.uint8)
    padded[:,:,:,:] = 128
    padded[:,:,4:-4, 4:-4] = X
    Y = np.array(Y, dtype=np.int64) 
    if not pad:
      return X, Y
    return padded, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'), pad=False)
  print Xtr.shape
  print Ytr.shape
  print Xte.shape
  print Yte.shape
  return Xtr, Ytr, Xte, Yte

def py2lmdb(X, y, save_path):
  # Let's pretend this is interesting data

  assert X.dtype == np.uint8
  N = X.shape[0]
  assert N == y.shape[0]
  
  
  # We need to prepare the database for the size. We'll set it 10 times
  # greater than what we theoretically need. There is little drawback to
  # setting this too big. If you still run into problem after raising
  # this, you might want to try saving fewer entries in a single
  # transaction.
  map_size = X.nbytes * 10
  
  env = lmdb.open(save_path, map_size=map_size)
  
  with env.begin(write=True) as txn:
      # txn is a Transaction object
      for i in range(N):
          datum = caffe.proto.caffe_pb2.Datum()
          datum.channels = X.shape[1]
          datum.height = X.shape[2]
          datum.width = X.shape[3]
          datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
          datum.label = int(y[i])
          str_id = '{:08}'.format(i)
  
          # The encode is only essential in Python 3
          txn.put(str_id.encode('ascii'), datum.SerializeToString())


if __name__ == '__main__':
  root = sys.argv[1]
  Xtr, Ytr, Xte, Yte = load_CIFAR10(root)
  paths = [ os.path.join(root, i) for i in ['train', 'test']]
  py2lmdb(Xtr, Ytr, paths[0])
  py2lmdb(Xte, Yte, paths[1])
  for i in paths:
    print 'saved to', i
  
