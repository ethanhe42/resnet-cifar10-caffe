file="cifar-10-batches-py"
wget https://github.com/yihui-he/resnet-cifar10-caffe/releases/download/1.0/$file.zip
mkdir $file
mv $file.zip $file
cd $file
unzip $file
rm $file.zip
cd ..

