#!/usr/bin/env python
import inspect
import os
import os.path as osp
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks
import pandas as pd
import cfgs

class Field:
    class x:
        iters = u"#Iters"
        Seconds = u"Seconds"
    class y:
        TrainingLoss = u"TrainingLoss"
        LearningRate = u"LearningRate"
        TestAccuracy = u"TestAccuracy"
        TestLoss = u"TestLoss"

def get_log_parsing_script():
    dirname = osp.join(cfgs.caffe_path, 'tools/extra')
    return dirname + '/parse_log.sh'

def get_log_file_suffix():
    return '.log'
def reader(filename):
    table = pd.read_csv(filename, sep=' +', index_col=0)
    print table.columns
    return table

def plot_chart(path_to_log_list):
    plt.cla()
    plot_xlim=3
    if not osp.exists(cfgs.plots):
        os.mkdir(cfgs.plots)
    path_to_png = ''

    model_names = []
    for path_to_log in path_to_log_list:
        model_name = osp.basename(path_to_log).split('_')[0]
        
        label_name = model_name
        i = 0
        while True:
            if not label_name in model_names:
                model_names.append(label_name)
                break
            else:
                label_name = model_name + str(i)
                i += 1

        path_to_png+=osp.splitext(osp.basename(path_to_log))[0]

        train_file = osp.basename(path_to_log+'.train')
        test_file = osp.basename(path_to_log+'.test')
        new_train_file = osp.join(cfgs.plots, train_file)
        new_test_file = osp.join(cfgs.plots, test_file)
        print train_file
        print new_train_file
        # if not osp.exists(new_test_file) or not osp.exists(new_train_file):
        os.system('%s %s' % (get_log_parsing_script(), path_to_log))
        os.rename(train_file, new_train_file)
        os.rename(test_file, new_test_file)

        table = reader(new_train_file)
        table[table>plot_xlim]=plot_xlim
        table.TrainingLoss.plot(legend=True, label=label_name+' tr')
        # table.LearningRate.plot(legend=True, label=label_name+' lr')
        try:
            table = reader(new_test_file)
            table[table>plot_xlim]=plot_xlim

            table.TestLoss.plot(legend=True, label=label_name+' te')
            table.TestAccuracy.plot(secondary_y=True, legend=True, label=label_name+' acc')
        except:
            # no accuracy
            pass
        os.remove(new_train_file)
        os.remove(new_test_file)
    
    plt.title(' '.join(model_names))
    plt.xlabel(Field.x.iters)
#     plt.ylabel(y_axis_field)
    plt.savefig(os.path.join(cfgs.plots, path_to_png+'.png'))
    plt.show()
    plt.gcf().clear()

if __name__ == '__main__':
    path_to_logs = sys.argv[1:]
    for path_to_log in path_to_logs:
        if not os.path.exists(path_to_log):
            print 'Path does not exist: %s' % path_to_log
            sys.exit()
        if not path_to_log.endswith(get_log_file_suffix()):
            print 'Log file must end in %s.' % get_log_file_suffix()
    ## plot_chart accpets multiple path_to_logs
    plot_chart(path_to_logs)
