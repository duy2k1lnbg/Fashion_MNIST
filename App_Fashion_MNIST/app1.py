from __future__ import absolute_import, division, print_function
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

'''
    fashion_mnist = keras.datasets.fashion_mnist tải dữ Fashion MNIST gồm  60,000 hình ảnh cho tập huấn luyện và 10,000 hình ảnh cho tập kiểm tra.
'''
fashion_mnist = keras.datasets.fashion_mnist


'''
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    chia tệp đữ liệu vừa tải thành 2 tệp fashion-mnist_train.csv và fashion-mnist_test.csv
'''
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'''
    Mỗi tệp gồm 10 lớp được gán nhãn: 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
'''
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


'''
    train_images.shape sẽ xuất ra (60000, 28, 28), đại diện cho 60.000 hình ảnh thang độ xám 28x28 được sử dụng để đào tạo
'''
train_images.shape

'''
    train_labels sẽ xuất ra một mảng hình dạng (60000,)với các số nguyên nằm trong khoảng từ 0 đến 9, 
    đại diện cho nhãn lớp của mỗi hình ảnh. Ánh xạ giữa nhãn lớp và tên lớp được định nghĩa trong class_names.
'''
train_labels

'''
    test_images.shape sẽ xuất ra (10000, 28, 28), đại diện cho 10.000 hình ảnh thang độ xám 28x28 được sử dụng để thử nghiệm
'''
test_images.shape


'''
    load 1 hình ảnh đầu tiên 
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

three_layer_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

three_layer_model.summary()

three_layer_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

three_layer_model.fit(train_images, train_labels, epochs=20)

# Compute and print the test loss and accuracy
test_loss, test_acc = three_layer_model.evaluate(test_images, test_labels)
print("Model with three layers and ten epochs -- Test loss:", test_loss)
print("Model with three layers and ten epochs -- Test accuracy:", test_acc)

