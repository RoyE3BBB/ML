import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense
from MY_DL.CIFAR10.Layer import Basic_Block, Bottle_Neck

SMALL_FILTER_NUM = 8
BIG_FILTER_NUM = 16

class Normal_Model(Model):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.optimizer = tf.keras.optimizers.Adam()
        self.losser = tf.keras.losses.sparse_categorical_crossentropy
        self.me = ['acc']
        self.compile(optimizer=self.optimizer,
                     loss=self.losser,
                     metrics=self.me)

        self.conv1 = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same')
        self.conv2 = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')
        self.conv3 = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same')
        self.bn3 = BatchNormalization()
        self.relu3 = Activation('relu')
        self.averagepool = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')
        self.flatten = Flatten()
        self.dense = Dense(self.num_classes, activation='softmax')

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu3(bn3)
        averagepool = self.averagepool(relu3)
        flatten = self.flatten(averagepool)
        outputs = self.dense(flatten)

        return outputs

class Bottle_Neck_Model(Model):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.optimizer = tf.keras.optimizers.Adam()
        self.losser = tf.keras.losses.sparse_categorical_crossentropy
        self.me = ['accuracy']
        self.compile(optimizer=self.optimizer,
                     loss=self.losser,
                     metrics=self.me)

        self.conv = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same')
        self.block1 = self.build_block(SMALL_FILTER_NUM, BIG_FILTER_NUM, 3)
        self.averagepool = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')
        self.flatten = Flatten()
        self.dense = Dense(self.num_classes, activation='softmax')

    def call(self, inputs):
        conv = self.conv(inputs)
        # bb1 = self.bb1(conv)
        # bb2 = self.bb2(bb1)
        # bb3 = self.bb3(bb2)
        # averagepool = self.averagepool(bb3)
        block1 = self.block1(conv)
        averagepool = self.averagepool(block1)
        flatten = self.flatten(averagepool)
        outputs = self.dense(flatten)

        return outputs

    def build_block(self, small_filter_num, big_filter_num, blocks_num, stride=1):
        block = Sequential()
        block.add(Bottle_Neck(small_filter_num, big_filter_num, stride=stride))
        for _ in range(1, blocks_num):
            block.add(Bottle_Neck(small_filter_num, big_filter_num, 1))

        return block

class Basic_Block_Model(Model):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.optimizer = tf.keras.optimizers.Adam()
        self.losser = tf.keras.losses.sparse_categorical_crossentropy
        self.me = ['accuracy']
        self.compile(optimizer=self.optimizer,
                     loss=self.losser,
                     metrics=self.me)

        # layers information
        self.conv = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same')
        # self.bb1 = Basic_Block(16)
        # self.bb2 = Basic_Block(16)
        # self.bb3 = Basic_Block(16)
        self.block1 = self.build_block(BIG_FILTER_NUM, 3)
        self.averagepool = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')
        self.flatten = Flatten()
        self.dense = Dense(self.num_classes, activation='softmax')

    def call(self, inputs):
        conv = self.conv(inputs)
        # bb1 = self.bb1(conv)
        # bb2 = self.bb2(bb1)
        # bb3 = self.bb3(bb2)
        # averagepool = self.averagepool(bb3)
        block1 = self.block1(conv)
        averagepool = self.averagepool(block1)
        flatten = self.flatten(averagepool)
        outputs = self.dense(flatten)

        return outputs

    def build_block(self, filter_num, blocks_num, stride=1):
        block = Sequential()
        block.add(Basic_Block(filter_num, stride=stride))
        for _ in range(1, blocks_num):
            block.add(Basic_Block(filter_num))

        return block

if __name__ == '__main__':
    # model = Basic_Block_Model(10)
    # model = Normal_Model(10)
    model = Bottle_Neck_Model(10)

    model.build(input_shape=(None, 32, 32, 3))
    print(model.summary())
