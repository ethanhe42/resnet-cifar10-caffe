# ResNet 20 56 110 for CIFAR10 with caffe 
1. get cifar10 python version, then create a soft link `ln -s cifar-10-batches-py` here
2. use [data_utils.py](data_utils.py) to generate 4 pixel padded training data and testing data
3. use [net_generator.py](net_generator.py) to generate `solver.prototxt` and `trainval.prototxt`
4. use [train.sh](train.sh) to train it

###### results are consistent with original paper
