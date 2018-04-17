import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data

# Save an image to a specified location, restoring it from a normalized, inverted and squashed form
def writeImg(img, w, h, name):
    image = img.reshape((w, h))
    temp = np.ones((w, h))
    image = np.multiply((temp - image), 256)
    cv2.imwrite(name, image)

# Creates a set of images with random noise
# count - the number of images to be created
# height, width - the image size
# write_to_folder - a path; if present, the images will be written in a subfloder of this path named random-negative
# returns - the images as a numpy array
def build_negative_samples(count, height, width, write_to_folder = None):
    np.random.seed(1)
    toRet = np.ndarray(shape = [count, height * width], dtype=np.float32)
    toRet[0] = np.zeros(shape=(1, width * height))
    toRet[1] = np.zeros(shape=(1, width * height))
    for i in range(2, count):
        #toRet[i] = np.random.uniform(size=height * width)
        toRet[i] = np.random.choice([0, 0.2, 0.4, 0.6, 0.8, 1], size=height * width, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
        if write_to_folder is not None:
            writeImg(toRet[i], 28, 28, write_to_folder + "//random-negative//img" + str(i) + ".png")
    return toRet

# Creates a new dataset based on MNIST
# dataset - the original dataset
# digit - the index of the digit that is the positive example
# extra_negatives - a numpy array of negative samples to be appended
# write_to_folder - a path; if present, the images will be written in subfloders of this path
def build_modified_mnist(dataset, digit, extra_negatives = None, write_to_folder = None, include_negatives=True):
    addSize = 0 if extra_negatives is None else extra_negatives.shape[0]
    size = int(dataset.labels.shape[0] / 7) + addSize
    labels = np.ndarray(shape=(size, 1), dtype=dataset.labels.dtype)
    images = np.ndarray(shape=(size, dataset.images.shape[1]), dtype=dataset.images.dtype)
    positives = 0
    negatives = addSize
    c = addSize
    np.random.seed(1)
    for i in range(0, addSize):
        labels[i][0] = 0
        images[i] = extra_negatives[i]
    for i in range(0, dataset.labels.shape[0]):
        if dataset.labels[i][digit] == 0 and include_negatives:
            if negatives < positives and np.random.uniform(0, 1, 1) <= 0.5:
                negatives += 1
                labels[c][0] = 0
                images[c] = dataset.images[i]
                if write_to_folder is not None:
                    writeImg(images[c], 28, 28, write_to_folder + "//negative//img" + str(c) + ".png")
                c += 1
        else:
            positives += 2
            labels[c][0] = 1
            images[c] = dataset.images[i]
            if write_to_folder is not None:
                writeImg(images[c], 28, 28, write_to_folder + "//positive//img" + str(c) + ".png")
            c += 1
            if c >= size:
                break
            temp = np.copy(dataset.images[i])
            mask = np.random.choice([0, 1], size=dataset.images[i].shape[0], p=[0.5,0.5])
            temp = np.multiply(temp, mask)
            labels[c][0] = 0.7
            images[c] = temp
            if write_to_folder is not None:
                writeImg(images[c], 28, 28, write_to_folder + "//positive//img-degraded" + str(c) + ".png")

            c += 1
        if c >= size:
            break

    mnist_modified = DataSet(images, labels, one_hot=True, reshape=False)
    return mnist_modified

# Create a TF variable of the specified shape
# Initializes it with 0-centered, normally distributed, stdev = 0.1 random values
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Creates a TF constant
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Creates a TF 2D convolution operation from a layer of neurons and a set of weights
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Creates a TF 2X2 pooling operation for a layer of neurons
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Creates a small convolutional neural network in the following configuration:
# 28X28 (input) -> 8 5X5 convolution layers -> 16 2X2 pooling layer ->
# 200 dense layer -> 1 output
# train_ds - the training dataset
# test_ds - the testing dataset
# save_to - the path where the network will be saved
def build_conv_nn2(train_ds, test_ds, save_to):

    x = tf.placeholder(tf.float32, shape=[None, 784], name = "x")
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    W_conv1 = weight_variable([5, 5, 1, 8])
    b_conv1 = bias_variable([8])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.identity(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 8, 16])
    b_conv2 = bias_variable([16])

    h_conv2 = tf.identity(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 16, 200])
    b_fc1 = bias_variable([200])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([200, 1])
    b_fc2 = bias_variable([1])

    y2pre = tf.identity(tf.matmul(h_fc1, W_fc2) + b_fc2, name="output_pre")
    y2 = tf.nn.sigmoid(y2pre, name="output")

    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y2) + (1 - y_) * tf.log(1 - y2)
                                         , reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
    correct_prediction = tf.logical_not(tf.greater(tf.abs(tf.subtract(y2, y_)), 0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        batch = train_ds.next_batch(10000)

        if i % 5 == 0:
            [train_accuracy, train_loss] = sess.run([accuracy, loss], feed_dict={x: batch[0], y_: batch[1]})
            print('step %d, training accuracy/loss %g / %g' % (i, train_accuracy, train_loss))
        if i %10 == 0:
            [test_accuracy, test_loss] = sess.run([accuracy, loss], feed_dict={x: test_ds.images, y_: test_ds.labels})
            print('step %d, test accuracy/loss %g / %g' % (i, test_accuracy, test_loss))

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    saver = tf.train.Saver()
    saver.save(sess, save_to)

# Creates a neural network in the following configuration:
# 784 inputs -> 200 -> 200 -> 100 -> 50 -> 1 output
# train_ds - the training dataset
# test_ds - the testing dataset
# save_to - the path where the network will be saved
def build_deep_nn(train_ds, test_ds, save_to):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.00)

    x = tf.placeholder(tf.float32, shape=[None, 784], name = "x")
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    W1 = tf.Variable(tf.random_normal([784, 200]))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W1)
    b1 = tf.Variable(tf.zeros([200]))
    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    W11 = tf.Variable(tf.random_normal([200, 200]))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W11)
    b11 = tf.Variable(tf.zeros([200]))
    y11 = tf.nn.sigmoid(tf.matmul(y1, W11) + b11)

    W12 = tf.Variable(tf.random_normal([200, 100]))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W12)
    b12 = tf.Variable(tf.zeros([100]))
    y12 = tf.nn.sigmoid(tf.matmul(y11, W12) + b12)

    W13 = tf.Variable(tf.random_normal([100, 50]))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W13)
    b13 = tf.Variable(tf.zeros([50]))
    y13 = tf.nn.sigmoid(tf.matmul(y12, W13) + b13)


    W2 = tf.Variable(tf.random_normal([50, 1]))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W2)
    b2 = tf.Variable(tf.zeros([1]))
    y2pre = tf.identity(tf.matmul(y13, W2) + b2, name="output_pre")
    y2 = tf.nn.sigmoid(y2pre, name="output")

    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y2) + (1 - y_) * tf.log(1 - y2)
                                         , reduction_indices=[1]))
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss += reg_term
    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
    correct_prediction = tf.logical_not(tf.greater(tf.abs(tf.subtract(y2, y_)), 0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(150):
        batch = train_ds.next_batch(10000)

        [train_accuracy, train_loss] = sess.run([accuracy, loss], feed_dict={x: batch[0], y_: batch[1]})
        print('step %d, training accuracy/loss %g / %g' % (i, train_accuracy, train_loss))

        [test_accuracy, test_loss] = sess.run([accuracy, loss], feed_dict={x: test_ds.images, y_: test_ds.labels})
        print('step %d, test accuracy/loss %g / %g' % (i, test_accuracy, test_loss))

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    saver = tf.train.Saver()
    saver.save(sess, save_to)

# Creates a neural network in the following configuration:
# 784 inputs -> 300 -> 1 output
# train_ds - the training dataset
# test_ds - the testing dataset
# save_to - the path where the network will be saved
def build_2layers_nn(train_ds, test_ds, save_to, layer1_size=300):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0015)

    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    W1 = tf.Variable(tf.random_normal([784, layer1_size]))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W1)
    b1 = tf.Variable(tf.zeros([layer1_size]))
    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_normal([layer1_size, 1]))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W2)
    b2 = tf.Variable(tf.zeros([1]))

    y2pre = tf.identity(tf.matmul(y1, W2) + b2, name="output_pre")
    y2 = tf.nn.sigmoid(y2pre, name="output")

    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y2) + (1 - y_) * tf.log(1 - y2)
                                         , reduction_indices=[1]))
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss += reg_term
    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
    correct_prediction = tf.logical_not(tf.greater(tf.abs(tf.subtract(y2, y_)), 0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        batch = train_ds.next_batch(10000)

        [train_accuracy, train_loss] = sess.run([accuracy, loss], feed_dict={x: batch[0], y_: batch[1]})
        print('step %d, training accuracy/loss %g / %g' % (i, train_accuracy, train_loss))

        [test_accuracy, test_loss] = sess.run([accuracy, loss], feed_dict={x: test_ds.images, y_: test_ds.labels})
        print('step %d, test accuracy/loss %g / %g' % (i, test_accuracy, test_loss))

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    saver = tf.train.Saver()
    saver.save(sess, save_to)



if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    negatives = build_negative_samples(1000, 28, 28)
    mnist_train_modified = build_modified_mnist(mnist.train, 3, negatives, include_negatives = True)
    mnist_test_modified = build_modified_mnist(mnist.test, 3, include_negatives = True)
    build_2layers_nn(mnist_train_modified, mnist_test_modified, save_to="../2layers-5", layer1_size=600)