set -e
file="cifar-10-python.tar.gz"
if [ ! -f $file ]; then
    wget https://www.cs.toronto.edu/~kriz/$file
fi
tar -xvf $file
python data_utils.py cifar-10-batches-py/
rm cifar-10-python.tar.gz
rm cifar-10-batches-py/*batch*
rm cifar-10-batches-py/readme.html
