wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
python data_utils.py cifar-10-batches-py/
rm cifar-10-python.tar.gz
rm cifar-10-batches-py/*batch*
rm cifar-10-batches-py/readme.html
