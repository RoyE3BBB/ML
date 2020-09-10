from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, Activation

class Basic_Block(Layer):
    def __init__(self, filter_num, stride=1):
        super(Basic_Block, self).__init__()
        self.filter_num = filter_num
        self.stride = stride

        # layers imformation
        self.conv1 = Conv2D(filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')

        self.conv2 = Conv2D(filter_num, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')

        # in order to keep the dimention as same as output
        if stride != 1:
            self.identity_mapping = Conv2D(filter_num, kernel_size=(1, 1), stride=stride)
        else:
            self.identity_mapping = lambda x: x

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        identity_mapping = self.identity_mapping(inputs)
        outputs = self.relu2(layers.add([identity_mapping, bn2]))

        return outputs

class Bottle_Neck(Layer):
    def __init__(self, small_filter_num, big_filter_num, stride=1):
        super(Bottle_Neck, self).__init__()
        self.small_fileter_num = small_filter_num
        self.big_filter_num = big_filter_num
        self.stride = stride

        self.conv1 = Conv2D(small_filter_num, kernel_size=(1, 1), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')

        self.conv2 = Conv2D(small_filter_num, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')

        self.conv3 = Conv2D(big_filter_num, kernel_size=(1, 1), strides=1, padding='same')
        self.bn3 = BatchNormalization()
        self.relu3 = Activation('relu')

        if stride != 1:
            self.identity_mapping = Conv2D(big_filter_num, kernel_size=(1, 1), stride=stride)
        else:
            self.identity_mapping = lambda x: x

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 =self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)

        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)

        identity_mapping = self.identity_mapping(inputs)
        outputs = self.relu3(layers.add([identity_mapping, bn3]))

        return outputs