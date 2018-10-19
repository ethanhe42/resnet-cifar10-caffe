# ResNet 20/32/44/56/110 for CIFAR10 with caffe 
### Testing
| Model                                                                                                    |  Acc | Claimed Acc|
|:---------------------------------------------------------------------------------------------------------|:-----------:|:-------------:|
| [ResNet-20](https://github.com/yihui-he/resnet-cifar10-caffe/releases/download/1.0/resnet-20_iter_60000.caffemodel) | 40.8%       | 42.3%         |
|  [ResNet-32]()  | %       | %         |
|  [ResNet-44]()  | %       | %         |
| [ResNet-56](https://github.com/yihui-he/resnet-cifar10-caffe/releases/download/1.0/resnet-56_iter_64000.caffemodel)  | 92.8%       | 93.03%         |
|  [ResNet-110]()  | %       | %         |

### Citation
If you find the code useful in your research, please consider citing:

    @InProceedings{He_2017_ICCV,
    author = {He, Yihui and Zhang, Xiangyu and Sun, Jian},
    title = {Channel Pruning for Accelerating Very Deep Neural Networks},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
    }

### Training
```Shell
#build caffe
git clone https://github.com/yihui-he/resnet-cifar10-caffe
./download_cifar.sh
./train.sh [GPUs] [NET]
#eg., ./train.sh 0 resnet-20
#find logs at resnet-20/logs
```
### Visualization
specify caffe path in [cfgs.py](cfgs.py) and use [plot.py]() to generate beautful loss plots.
```Shell
python plot.py PATH/TO/LOGS
```
Results are consistent with original paper. seems there's no much difference between resnet-20 and plain-20. However, from the second plot, you can see that plain-110 have difficulty to converge.

<p float="left">
  <img src="plots/resnet-20__2016-08-14_00-25-56plain_orth20__2016-08-14_15-34-29.png" width="400" />
  <img src="plots/resnet-110__2016-08-15_10-12-25plain110__2016-08-15_10-11-55.png" width="400" /> 
</p>

### How I generate prototxts:
use [net_generator.py](net_generator.py) to generate `solver.prototxt` and `trainval.prototxt`, you can generate resnet or plain net of depth 20/32/44/56/110, or even deeper if you want. you just need to change `n` according to `depth=6n+2`  

### How I generate lmdb data:
create 4 pixel padded training LMDB and testing LMDB, then create a soft link `ln -s cifar-10-batches-py` in this folder.
    - get [cifar10 python version](https://www.cs.toronto.edu/~kriz/cifar.html)
    - use [data_utils.py](data_utils.py) to generate 4 pixel padded training data and testing data. Horizontal flip and random crop are performed on the fly while training.

### Other models in Caffe
[ResNet-ImageNet-Caffe](https://github.com/yihui-he/resnet-imagenet-caffe)  
[Xception-Caffe](https://github.com/yihui-he/Xception-caffe)  

