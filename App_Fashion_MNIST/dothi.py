from __future__ import absolute_import, division, print_function
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
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

three_layer_model.fit(train_images, train_labels, epochs=10)

# Compute and print the test loss and accuracy
test_loss, test_acc = three_layer_model.evaluate(test_images, test_labels)
print("Model with three layers and ten epochs -- Test loss:", test_loss * 100)
print("Model with three layers and ten epochs -- Test accuracy:", test_acc * 100)

train_model = three_layer_model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val))

def create_trace(x, y, ylabel, color):
    trace = go.Scatter(
        x=x, y=y,
        name=ylabel,
        marker=dict(color=color),
        mode="markers+lines",
        text=x
    )
    return trace


def plot_accuracy_and_loss(three_layer_model):
    hist = three_layer_model.history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1, len(acc) + 1))

    trace_ta = create_trace(epochs, acc, "Training accuracy", "Green")
    trace_va = create_trace(epochs, val_acc, "Validation accuracy", "Red")
    trace_tl = create_trace(epochs, loss, "Training loss", "Blue")
    trace_vl = create_trace(epochs, val_loss, "Validation loss", "Magenta")

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Training and validation accuracy', 'Training and validation loss'))
    fig.append_trace(trace_ta, 1, 1)
    fig.append_trace(trace_va, 1, 1)
    fig.append_trace(trace_tl, 1, 2)
    fig.append_trace(trace_vl, 1, 2)
    fig['layout']['xaxis'].update(title='Epoch')
    fig['layout']['xaxis2'].update(title='Epoch')
    fig['layout']['yaxis'].update(title='Accuracy', range=[0, 1])
    fig['layout']['yaxis2'].update(title='Loss', range=[0, 1])

    iplot(fig, filename='accuracy-loss')


plot_accuracy_and_loss(three_layer_model)
