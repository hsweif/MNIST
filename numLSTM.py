import tensorflow as tf
import sys
import numpy
import csv
import random

pic_size = 784 #28*28
train_data = []
train_label = []
test_data = []
test_label = []
checkpoint_dir = './lstmCheckPoint/'
train_path = './train.csv'
test_path = './test.csv'
output_path = './output_lstm.csv'

#Some parameters
batch_size = 100
time_step = 28
input_size = 28
learn_speed = 1e-4 
unit_num = 100
iteration = 20001
output_num = 10 # 0 - 9
testSet_size = 200

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

def write_result(file_path, ans):
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        for item in ans:
            writer.writerow(item)


load_data(train_path, train_data, train_label)

#define placeholder for input and output 
image = tf.placeholder(tf.float32, [None, time_step*input_size])
x = tf.reshape(image, [-1, time_step, input_size])
y = tf.placeholder(tf.int32, [None, output_num])

#define LSTM structure
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = unit_num) #总共有unit_num 个LSTM神经元
outputs, final_state = tf.nn.dynamic_rnn(
        cell = lstm_cell,
        inputs = x,
        initial_state = None,
        dtype = tf.float32,
        time_major = False,
        )

output = tf.layers.dense(inputs=outputs[:,-1,:], units = output_num)
loss = tf.losses.softmax_cross_entropy(onehot_labels = y, logits = output)
train_op = tf.train.AdamOptimizer(learn_speed).minimize(loss)
correct_pr = tf.equal(tf.argmax(y, axis = 1), tf.argmax(output, axis = 1))
accuracy = tf.reduce_mean(tf.cast(correct_pr, 'float'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

l = len(train_data)
sz = l - batch_size - testSet_size
test_x, test_y = train_data[-testSet_size:], train_label[-testSet_size:]
saver = tf.train.Saver()
for i in range(0, iteration):
    sys.stdout.write(' Training----> ' + str(i)+ ' Batch...' + '\r')
    n = random.randint(1, sz)
    train_x, train_y = train_data[n:n+batch_size], train_label[n:n+batch_size] 
    _blank, train_loss = sess.run([train_op, loss], {image:train_x, y:train_y})
    if i % 500 == 0:
        saver.save(sess,checkpoint_dir + 'lstm_model.ckpt', global_step = i+1)
        print()
        train_accuracy = sess.run(accuracy, {image:test_x, y:test_y})
        print('loss: %.5f' % train_loss, '|accuracy: %.5f' % train_accuracy)
        if train_loss < 0.1:
            break

