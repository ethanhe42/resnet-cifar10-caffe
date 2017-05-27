from __future__ import print_function
from caffe.proto import caffe_pb2
import os.path as osp
import sys
import os
# import caffe

class Solver:
    def __init__(self, solver_name=None, folder=None):
        self.solver_name = solver_name
        self.folder = folder
        
        if self.folder is not None:
            self.name = osp.join(self.folder, 'solver.prototxt')
        if self.name is None:
            self.name = 'solver.pt'
        else:
            filepath, ext = osp.splitext(self.name)
            if ext == '':
                ext = '.prototxt'
                self.name = filepath+ext

        self.p = caffe_pb2.SolverParameter()        
        
        class Method:
            nesterov = "Nesterov"
            SGD = "SGD"
            AdaGrad = "AdaGrad"
            RMSProp = "RMSProp"
            AdaDelta = "AdaDelta"
            Adam = "Adam"
        self.method=Method()
        
        class Policy:
            """    - fixed: always return base_lr."""
            fixed = 'fixed'
            """    - step: return base_lr * gamma ^ (floor(iter / step))"""
            """    - exp: return base_lr * gamma ^ iter"""
            """    - inv: return base_lr * (1 + gamma * iter) ^ (- power)"""
            """    - multistep: similar to step but it allows non uniform steps defined by stepvalue"""
            multistep = 'multistep'
            """    - poly: the effective learning rate follows a polynomial decay, to be zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)"""
            """    - sigmoid: the effective learning rate follows a sigmod decay"""
            """      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))"""
        self.policy = Policy()

        class Machine:
            GPU = self.p.GPU
            CPU = self.p.GPU
        self.machine = Machine()

        # defaults

        self.p.test_iter.extend([100])
        self.p.test_interval = 1000
        self.p.test_initialization = True

        self.p.base_lr = 0.1
        self.p.lr_policy = self.policy.multistep
        self.p.stepvalue.extend([32000, 48000])
        self.p.gamma = 0.1
        self.p.momentum = 0.9
        self.p.weight_decay = 0.0001

        self.p.display = 100
        self.p.max_iter = 64000
        self.p.snapshot = 10000
        self.p.snapshot_prefix = osp.join(self.folder, "snapshot/")
        self.p.solver_mode = self.machine.GPU

        self.p.type = self.method.nesterov
        self.p.net = osp.join(self.folder, "trainval.prototxt")

    def write(self):
        dirname = osp.dirname(self.name)
        if not osp.exists(dirname):
            os.mkdir(dirname)
        if not osp.exists(self.p.snapshot_prefix):
            os.mkdir(self.p.snapshot_prefix)
        with open(self.name, 'w') as f:
            f.write(str(self.p))

class Net:
    def __init__(self, name="network"):
        self.net = caffe_pb2.NetParameter()
        self.net.name = name
        self.bottom = None
        self.cur = None
        self.this = None
    
    def setup(self, name, layer_type, bottom=[], top=[], inplace=False):

        self.bottom = self.cur

        new_layer = self.net.layer.add()

        new_layer.name = name
        new_layer.type = layer_type

        if self.bottom is not None and new_layer.type != 'Data':
            bottom_name = [self.bottom.name]
            if len(bottom) == 0:
                bottom = bottom_name
            new_layer.bottom.extend(bottom)
        
        if inplace:
            top = bottom_name
        elif len(top) == 0:
            top = [name]
        new_layer.top.extend(top)

        self.this = new_layer
        if not inplace:
            self.cur = new_layer

    def suffix(self, name, self_name=None):
        if self_name is None:
            return self.cur.name + '_' + name
        else:
            return self_name

    def write(self, name=None, folder=None):
        # dirname = osp.dirname(name)
        # if not osp.exists(dirname):
        #     os.mkdir(dirname)
        if folder is not None:
            name = osp.join(folder, 'trainval.prototxt')
        elif name is None:
            name = 'trainval.pt'
        else:
            filepath, ext = osp.splitext(name)
            if ext == '':
                ext = '.prototxt'
                name = filepath+ext
        with open(name, 'w') as f:
            f.write(str(self.net))

    def show(self):
        print(self.net)
    #************************** params **************************

    def param(self, lr_mult=1, decay_mult=0):
        new_param = self.this.param.add()
        new_param.lr_mult = lr_mult
        new_param.decay_mult = decay_mult

    def transform_param(self, mean_value=128, batch_size=128, scale=.0078125, mirror=1, crop_size=None, mean_file_size=None, phase=None):

        new_transform_param = self.this.transform_param
        new_transform_param.scale = scale
        new_transform_param.mean_value.extend([mean_value])
        if phase is not None and phase == 'TEST':
            return

        new_transform_param.mirror = mirror
        if crop_size is not None:
            new_transform_param.crop_size = crop_size
        

    def data_param(self, source, backend='LMDB', batch_size=128):
        new_data_param = self.this.data_param
        new_data_param.source = source
        if backend == 'LMDB':
            new_data_param.backend = new_data_param.LMDB
        else:
            NotImplementedError
        new_data_param.batch_size = batch_size    

    def weight_filler(self, filler='msra'):
        """xavier"""
        if self.this.type == 'InnerProduct':
            self.this.inner_product_param.weight_filler.type = filler
        else:
            self.this.convolution_param.weight_filler.type = filler
    
    def bias_filler(self, filler='constant', value=0):
        if self.this.type == 'InnerProduct':
            self.this.inner_product_param.bias_filler.type = filler
            self.this.inner_product_param.bias_filler.value = value
        else:
            self.this.convolution_param.bias_filler.type = filler
            self.this.convolution_param.bias_filler.value = value

    def include(self, phase='TRAIN'):
        if phase is not None:
            includes = self.this.include.add()
            if phase == 'TRAIN':
                includes.phase = caffe_pb2.TRAIN
            elif phase == 'TEST':
                includes.phase = caffe_pb2.TEST
        else:
            NotImplementedError


    #************************** inplace **************************
    def ReLU(self, name=None):
        
        self.setup(self.suffix('relu', name), 'ReLU', inplace=True)
    
    def BatchNorm(self, name=None):
        
        self.setup(self.suffix('bn', name), 'BatchNorm', inplace=True)

        self.param(lr_mult=0, decay_mult=0)
        self.param(lr_mult=0, decay_mult=0)
        self.param(lr_mult=0, decay_mult=0)
        batch_norm_param = self.this.batch_norm_param
        #batch_norm_param.use_global_stats = False
        #batch_norm_param.moving_average_fraction = 0.95

    def Scale(self, name=None):
        self.setup(self.suffix('scale', name), 'Scale', inplace=True)
        self.this.scale_param.bias_term = True

    #************************** layers **************************

    def Data(self, source, top=['data', 'label'], name="data", phase=None, **kwargs):
        self.setup(name, 'Data', top=top)

        self.include(phase)

        self.data_param(source)
        self.transform_param(phase=phase, **kwargs)
        
    def Convolution(self, name, bottom=[], num_output=None, kernel_size=3, pad=1, stride=1, decay = True, bias = False, freeze = False):
        self.setup(name, 'Convolution', bottom=bottom, top=[name])
        
        conv_param = self.this.convolution_param
        if num_output is None:
            num_output = self.bottom.convolution_param.num_output

        conv_param.num_output = num_output
        conv_param.pad.extend([pad])
        conv_param.kernel_size.extend([kernel_size])
        conv_param.stride.extend([stride])
        
        if freeze:
            lr_mult = 0
        else:
            lr_mult = 1
        if decay:
            decay_mult = 1
        else:
            decay_mult = 0
        self.param(lr_mult=lr_mult, decay_mult=decay_mult)
        self.weight_filler()

        if bias:
            if decay:
                decay_mult = 2
            else:
                decay_mult = 0
            self.param(lr_mult=lr_mult, decay_mult=decay_mult)
            self.bias_filler()
        
    def SoftmaxWithLoss(self, name='loss', label='label'):
        self.setup(name, 'SoftmaxWithLoss', bottom=[self.cur.name, label])

    def Softmax(self,bottom=[], name='softmax'):
        self.setup(name, 'Softmax', bottom=bottom)

    def Accuracy(self, name='Accuracy', label='label'):
        self.setup(name, 'Accuracy', bottom=[self.cur.name, label])


    def InnerProduct(self, name='fc', num_output=10):
        self.setup(name, 'InnerProduct')
        self.param(lr_mult=1, decay_mult=1)
        self.param(lr_mult=2, decay_mult=0)    
        inner_product_param = self.this.inner_product_param
        inner_product_param.num_output = num_output
        self.weight_filler()
        self.bias_filler()
    
    def Pooling(self, name, pool='AVE', global_pooling=False):
        """MAX AVE """
        self.setup(name,'Pooling')
        if pool == 'AVE':
            self.this.pooling_param.pool = self.this.pooling_param.AVE
        else:
            NotImplementedError
        self.this.pooling_param.global_pooling = global_pooling

    def Eltwise(self, name, bottom1, operation='SUM'):
        bottom0 = self.bottom.name
        self.setup(name, 'Eltwise', bottom=[bottom0, bottom1])
        if operation == 'SUM':
            self.this.eltwise_param.operation = self.this.eltwise_param.SUM
        else:
            NotImplementedError


    #************************** DIY **************************
    def conv_relu(self, name, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.ReLU(relu_name)

    def conv_bn_relu(self, name, bn_name=None, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.BatchNorm(bn_name)
        self.Scale(None)
        self.ReLU(relu_name)

    def conv_bn(self, name, bn_name=None, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.BatchNorm(bn_name)
        self.Scale(None)

    def softmax_acc(self,bottom, **kwargs):
        self.Softmax(bottom=[bottom])

        has_label=None
        for name, value in kwargs.items():
            if name == 'label':
                has_label = value
        if has_label is None:
            self.Accuracy()
        else:
            self.Accuracy(label=has_label)
            

    #************************** network blocks **************************

    def res_func(self, name, num_output, up=False):
        bottom = self.cur.name
        print(bottom)
        self.conv_bn_relu(name+'_conv0', num_output=num_output, stride=1+int(up))
        self.conv_bn(name+'_conv1', num_output=num_output)
        if up:
            self.conv_bn(name+'_proj', num_output=num_output, bottom=[bottom], pad=0, kernel_size=1, stride=2)
            self.Eltwise(name+'_sum', bottom1=name+'_conv1')
        else:
            self.Eltwise(name+'_sum', bottom1=bottom)
    
    def res_group(self, group_id, n, num_output):
        def name(block_id):
            return 'group{}'.format(group_id) + '_block{}'.format(block_id)

        if group_id == 0:
            up = False
        else:
            up = True
        self.res_func(name(0), num_output, up=up)
        for i in range(1, n):
            self.res_func(name(i), num_output)


    #************************** networks **************************
    def resnet_cifar(self, n=3):
        """6n+2, n=3 9 18 coresponds to 20 56 110 layers"""
        num_output = 16
        self.conv_bn_relu('first_conv', num_output=num_output)
        for i in range(3):
            self.res_group(i, n, num_output*(2**i))
        
        self.Pooling("global_avg_pool", global_pooling=True)
        self.InnerProduct()
        self.SoftmaxWithLoss()
        self.softmax_acc(bottom='fc')


if __name__ == '__main__':
    n=18
    pt_folder = osp.join(osp.abspath(osp.curdir), "resnet-%d" % (6*n+2))
    name = 'resnet'+str(n)+'-cifar10'

    solver = Solver(folder=pt_folder)
    solver.write()

    builder = Net(name)
    builder.Data('cifar-10-batches-py/train', phase='TRAIN', crop_size=32)
    builder.Data('cifar-10-batches-py/test', phase='TEST')
    builder.resnet_cifar(n)
    builder.write(folder=pt_folder)

