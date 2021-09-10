import cv2
# 下面俩行是我当时CPU不够尝试的方法，后面充电跑了一下发现不报错了就直接注释掉了嗯
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import tensorflow as tf
import keras
#from tensorflow import keras
#from keras import layers
from keras import layers


def load_mnist(): #读取离线的MNIST.npz文件。
    path = r'mnist.npz' #放置mnist.py的目录，这里默认跟本代码在同一个文件夹之下。
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
(train_image,train_label),(test_image,test_label) = load_mnist()
print(train_image.shape)
print(train_label.shape)

#将image映射为784维向量，并映射为[0,1]之间的浮点数
train_image = train_image.reshape([60000,784])
test_image = test_image.reshape([10000,784])
train_image = train_image.astype("float32") / 255
test_image = test_image.astype("float32") / 255

#将label映射为one-hot-key的形式
num_classes = 10
train_label = keras.utils.to_categorical(train_label, num_classes)
test_label = keras.utils.to_categorical(test_label, num_classes)

print(type(train_image),type(train_label))

# #构建模型
model = keras.Sequential(
    [
        keras.Input(shape=(784,)), #这里(784,)的意思是784维向量构成的batch，省略的是batch的大小
        layers.Dense(20, activation="relu"),
        layers.Dense(20, activation="relu"),
        #layers.Dense(10, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

#模型训练和测试
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
model.fit(x=train_image, y=train_label, batch_size=32, epochs=10, validation_data=(test_image, test_label), verbose=2, callbacks=[tensorboard_callback])
score = model.evaluate(test_image, test_label, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save('model.h5')

