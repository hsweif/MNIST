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

def cnn_train():
    #training loop
    load_data(train_path, train_data, train_label)
    l = 4000 #len(train_data)
    for i in range(1, l):
        sys.stdout.write(' Training----> ' + str(i)+'\r')
        train_step.run(session = sess, feed_dict = {ans_data:[train_label[i]],input_data:[train_data[i]], keep_prbty:0.5})
        if i % 5000 == 0:
            saver.save(sess, checkpoint_dir + 'cnn_model.ckpt', global_step = i+1)

def cnn_test():
    #load_data(test_path, test_data, test_label)
    #ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #if ckpt and ckpt.model_checkpoint_path:
    #    saver.restore(sess, ckpt.model_checkpoint_path)
    #    print("Test accuracy: " + str(accuracy.eval(session = sess, feed_dict = {input_data: train_data[1:100], ans_data:train_label[1:100],keep_prbty:0.5})))
    #print("Test accuracy: " + str(accuracy.eval(session = sess, feed_dict = {input_data: test_data[1:10], ans_data:test_label[1:10],keep_prbty:0.5})))

    #Local test, split train set
    print("Test accuracy: " + str(accuracy.eval(session = sess, feed_dict = {input_data: train_data[5000:8000], ans_data:train_label[5000:8000],keep_prbty:0.5})))


input_data = tf.placeholder("float", shape = [None, pic_size])
ans_data = tf.placeholder("float", shape = [None, 10]) #1-10
#cnn
#First hidden layer
input_image = tf.reshape(input_data, [-1,28,28,1])
w_conv1 = init_weight([5,5,1,32])
b_conv1 = init_bias([32])
output_conv1 = tf.nn.relu(conv2D(input_image, w_conv1) + b_conv1)
pool_1 = pooling(output_conv1) 
#Second hidden layer
w_conv2 = init_weight([5,5,32,64])
b_conv2 = init_bias([64])
output_conv2 = tf.nn.relu(conv2D(pool_1, w_conv2) + b_conv2)
pool_2 = pooling(output_conv2)
#Full Connect Layer
w_fc = init_weight([7*7*64, 1024])
b_fc = init_bias([1024])
flat_pool2 = tf.reshape(pool_2, [-1, 7*7*64])
output_fc = tf.nn.relu(tf.matmul(flat_pool2, w_fc) + b_fc)
#Dropout Layer
keep_prbty = tf.placeholder("float")
dr_output = tf.nn.dropout(output_fc, keep_prbty)
#Output Layer
w_output = init_weight([1024,10])
b_output = init_bias([10])
prediction = tf.nn.softmax(tf.matmul(dr_output, w_output) + b_output)
#Evaluate
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ans_data, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(ans_data, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

#with tf.Session() as sess:
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
cnn_train()
print("finish train")
cnn_test()

print("done.")
