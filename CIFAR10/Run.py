import tensorflow as tf
from MY_DL.CIFAR10 import Model
import matplotlib.pyplot as plt


if __name__ == '__main__':
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # model = Model.Basic_Block_Model(10)
    # model = Model.Normal_Model(10)
    model = Model.Basic_Block_Model(10)

    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    history = model .fit(x_train,
                         y_train,
                         epochs=25,
                         batch_size=256,
                         validation_data=(x_test, y_test),
                         shuffle=True)

    plt.figure()
    plt.plot(history.epoch, history.history['val_loss'])
    plt.show()

    plt.figure()
    plt.plot(history.epoch, history.history['val_acc'])
    plt.show()