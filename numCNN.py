import tensorflow as tf
import sys
import numpy
import csv

pic_size = 784 #28*28
train_data = []
train_label = []
test_data = []
test_label = []
learn_speed = 0.01
checkpoint_dir = '../checkpoint/'
train_path = '../train.csv'
test_path = '../test.csv'

def load_data(file_path, data, label):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        if file_path == train_path:
            for row in reader:
                data.append(row[1:])
                tmp = []
                for num in range(0, 10):
                    if row[0] == str(num):
                        tmp.append(1)
                    else:
                        tmp.append(0)
                label.append(tmp)
        elif file_path == test_path:
            for row in reader:
                data.append(row)

def init_weight(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def init_bias(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2D(x, w):
    return tf.nn.conv2d(x, w, strides = [1,1,1,1], padding = "SAME")

def pooling(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")


#testing test.csv
def cnn_test():
    #here should be test.csv, now testing train.csv 
    x, y_label, y_pr, train_op, keep_prbty = train_ops()
    load_data(train_path, train_data, train_label)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            imgs = train_data[5000:6000]
            labels = train_label[5000:6000]
            correct_y_pr = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_pr, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_y_pr, tf.float32)) 
            print("Train accuracy: " + str(accuracy.eval(session = sess, feed_dict = {x:imgs, y_label:labels,keep_prbty:1})))

def recognize():
    x, y_label, recg_result, keep_prbty = recg_ops()
    load_data(test_path, test_data, test_label)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            print("recognizing...")
            saver.restore(sess, ckpt.model_checkpoint_path)
            imgs = test_data[1:100]
            ans = sess.run(recg_result, feed_dict = {x:imgs, keep_prbty:1}).tolist()
            print(ans)
 
def cnn_model():
    with tf.name_scope("reshape"):
        x = tf.placeholder(tf.float32, shape = [None, pic_size])
        y_label = tf.placeholder(tf.float32, shape = [None, 10]) #1-10
        x_image = tf.reshape(x, [-1,28,28,1])
    with tf.name_scope("conv1"):
        w_conv1 = init_weight([5,5,1,32])
        b_conv1 = init_bias([32])
        o_conv1 = tf.nn.relu(conv2D(x_image, w_conv1) + b_conv1)
    with tf.name_scope("pool1"):
        p_conv1 = pooling(o_conv1) 
    with tf.name_scope("conv2"):
        w_conv2 = init_weight([5,5,32,64])
        b_conv2 = init_bias([64])
        o_conv2 = tf.nn.relu(conv2D(p_conv1, w_conv2) + b_conv2)
    with tf.name_scope("pool2"):
        p_conv2 = pooling(o_conv2)
    with tf.name_scope("hiddenfc"):
        w_fc1 = init_weight([7*7*64, 1024])
        b_fc1 = init_bias([1024])
        flat_pool2 = tf.reshape(p_conv2, [-1, 7*7*64])
        o_fc1 = tf.nn.relu(tf.matmul(flat_pool2, w_fc1) + b_fc1)
        #here need dropout?
    with tf.name_scope("dropout"):
        keep_prbty = tf.placeholder(tf.float32)
        dr_output = tf.nn.dropout(o_fc1, keep_prbty)
    with tf.name_scope("outputfc"):
        w_fc2 = init_weight([1024,10])
        b_fc2 = init_bias([10])
        y_pr = tf.matmul(dr_output, w_fc2) + b_fc2
        l2_loss = tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(w_conv2) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(b_fc2)
    return x, y_label, y_pr, l2_loss, keep_prbty

def train_ops():
    x, y_label, y_pr, l2_loss, keep_prbty = cnn_model()
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_pr))
        cross_entropy += 5e-4 * l2_loss 
    return x, y_label, y_pr, tf.train.AdamOptimizer(1e-4).minimize(cross_entropy), keep_prbty 

def recg_ops():
    x, y_label, y_pr, l2_loss, keep_prbty = cnn_model()
    return x, y_label, tf.argmax(y_pr, 1), keep_prbty

def cnn_train():
    load_data(train_path, train_data, train_label)
    x, y_label, y_pr, train_op, keep_prbty = train_ops()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        l = len(train_data)
        for i in range(1, l):
            sys.stdout.write(' Training----> ' + str(i)+'\r')
            sess.run(train_op, feed_dict = {y_label:[train_label[i]], x:[train_data[i]], keep_prbty:0.5})
            if i % 500 == 0:
                saver.save(sess, checkpoint_dir + 'cnn_model.ckpt', global_step = i+1)
                imgs = train_data[i-50:i]
                labels = train_label[i-50:i]
                correct_y_pr = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_pr, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_y_pr, tf.float32)) 
                print("Train accuracy: " + str(accuracy.eval(session = sess, feed_dict = {x:imgs, y_label:labels,keep_prbty:1})))

cnn_train()
#cnn_test()
#recognize()
print("finish train")
