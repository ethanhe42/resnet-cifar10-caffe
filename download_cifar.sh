file="cifar-10-batches-py"
if [ ! -f $file.zip ]; then
    wget https://github.com/yihui-he/resnet-cifar10-caffe/releases/download/1.0/$file.zip
fi
tar -xzvf ../cifar-10-batches-py.zip
rm $file.zip

