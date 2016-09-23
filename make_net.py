from caffe.proto import caffe_pb2
import resnet

def save_proto(proto, prototxt):
    with open(prototxt, 'w') as f:
        f.write(str(proto))


if __name__ == '__main__':
    
    model = resnet.ResNet('/data/train_lmbd', 'data/test_lmbd', 1000)

    train_proto = model.resnet_layers_proto(64)

    test_proto = model.resnet_layers_proto(64, phase='TEST')

    save_proto(train_proto, 'proto/train.prototxt')

    save_proto(test_proto, 'proto/test.prototxt')
