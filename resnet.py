import caffe
from caffe import layers as L
from caffe import params as P

def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0, bias_term=False):

    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, bias_term=bias_term)
    conv_bn = L.BatchNorm(conv, in_place=True, batch_norm_param =dict(use_global_stats=use_global_stats))
    conv_scale = L.Scale(conv ,in_place=True, scale_param=dict(bias_term=True))
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def conv_bn_scale(bottom, num_output=64, kernel_size=3, stride=1, pad=0, bias_term=False):

    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, bias_term=bias_term)
    conv_bn = L.BatchNorm(conv, in_place=True, batch_norm_param =dict(use_global_stats=use_global_stats))
    conv_scale = L.Scale(conv, in_place=True, scale_param=dict(bias_term=True))

    return conv, conv_bn, conv_scale


def eltwize_relu(bottom1, bottom2):

    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True)

    return residual_eltwise, residual_eltwise_relu


def residual_branch(bottom, base_output=64):

    branch2a, branch2a_bn, branch2a_scale, branch2a_relu =  conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1) 
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu =  conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1)  
    branch2c, branch2c_bn, branch2c_scale = conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1)  # 4*base_output x n x n

    residual, residual_relu = eltwize_relu(bottom, branch2c) 

    return branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, branch2b_bn, branch2b_scale, branch2b_relu, \
           branch2c, branch2c_bn, branch2c_scale, residual, residual_relu


def residual_branch_shortcut(bottom, stride=2, base_output=64):

    branch1, branch1_bn, branch1_scale = conv_bn_scale(bottom, num_output=4 * base_output, kernel_size=1, stride=stride)

    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1, stride=stride)
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1)
    branch2c, branch2c_bn, branch2c_scale = conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1)

    residual, residual_relu = eltwize_relu(branch1, branch2c)  # 4*base_output x n x n

    return branch1, branch1_bn, branch1_scale, branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, \
           branch2b_bn, branch2b_scale, branch2b_relu, branch2c, branch2c_bn, branch2c_scale, residual, residual_relu


branch_shortcut_string = 'n.res(stage)a_branch1, n.bn(stage)a_branch1, n.scale(stage)a_branch1, \
        n.res(stage)a_branch2a, n.bn(stage)a_branch2a, n.scale(stage)a_branch2a, n.res(stage)a_branch2a_relu, \
        n.res(stage)a_branch2b, n.bn(stage)a_branch2b, n.scale(stage)a_branch2b, n.res(stage)a_branch2b_relu, \
        n.res(stage)a_branch2c, n.bn(stage)a_branch2c, n.scale(stage)a_branch2c,\
        n.res(stage)a, n.res(stage)a_relu = residual_branch_shortcut((bottom), stride=(stride), base_output=(num))'

branch_string = 'n.res(stage)b(order)_branch2a, n.bn(stage)b(order)_branch2a, n.scale(stage)b(order)_branch2a, \
        n.res(stage)b(order)_branch2a_relu, n.res(stage)b(order)_branch2b, n.res(stage)b(order)_branch2b_bn, \
        n.res(stage)b(order)_branch2b_scale, n.res(stage)b(order)_branch2b_relu, n.res(stage)b(order)_branch2c, \
        n.res(stage)b(order)_branch2c_bn, n.res(stage)b(order)_branch2c_scale, n.res(stage)b(order), n.res(stage)b(order)_relu = \
            residual_branch((bottom), base_output=(num))'


class ResNet(object):
    def __init__(self, lmdb_train, lmdb_test, num_output):
        self.train_data = lmdb_train
        self.test_data = lmdb_test
        self.classifier_num = num_output

    def resnet_layers_proto(self, batch_size, phase='TRAIN', stages=(3, 4, 6, 3)):
        """
            (3, 4, 6, 3) for 50 layers; (3, 4, 23, 3) for 101 layers; (3, 8, 36, 3) for 152 layers
        """
        global use_global_stats

        n = caffe.NetSpec()
        if phase == 'TRAIN':
            n.data, n.label = L.Data(source=self.train_data, backend=P.Data.LMDB, batch_size=batch_size,
                                 ntop=2, include=dict(phase=0),
                                 transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=True))

            use_global_stats = False

        else:
            n.data, n.label = L.Data(source=self.test_data, backend=P.Data.LMDB, batch_size=batch_size,
                                 ntop=2, include=dict(phase=1),
                                 transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=False))

            use_global_stats = True

        n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = conv_bn_scale_relu(n.data, num_output=64, kernel_size=7, stride=2, pad=3, bias_term=True) 
        n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x56x56

        for num in xrange(len(stages)):  # num = 0, 1, 2, 3
            for i in xrange(stages[num]):
                if i == 0:
                    stage_string = branch_shortcut_string
                    bottom_string = ['n.pool1', 'n.res2b%s' % str(stages[0] - 1), 'n.res3b%s' % str(stages[1] - 1),
                                     'n.res4b%s' % str(stages[2] - 1)][num]
                else:
                    stage_string = branch_string
                    if i == 1:
                        bottom_string = 'n.res%sa' % str(num + 2)
                    else:
                        bottom_string = 'n.res%sb%s' % (str(num + 2), str(i - 1))
                exec (stage_string.replace('(stage)', str(num + 2)).replace('(bottom)', bottom_string).
                      replace('(num)', str(2 ** num * 64)).replace('(order)', str(i)).
                      replace('(stride)', str(int(num > 0) + 1)))

        exec 'n.pool5 = L.Pooling((bottom), pool=P.Pooling.AVE, global_pooling=True)'.replace('(bottom)', 'n.res5b%s' % str(stages[3] - 1))
        n.classifier = L.InnerProduct(n.pool5, num_output=self.classifier_num)
        n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
        if phase == 'TEST':
            n.accuracy_top1 = L.Accuracy(n.classifier, n.label, include=dict(phase=1))
            n.accuracy_top5 = L.Accuracy(n.classifier, n.label, include=dict(phase=1),accuracy_param=dict(top_k=5))

        return n.to_proto()
