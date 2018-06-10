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
learn_speed = 1e-3 
checkpoint_dir = './checkpoint/'
train_path = './train.csv'
test_path = './test.csv'
output_path = './output.csv'
summary_path = './SummaryCNN/'

#将数据读入内存，以列表形式储存，元素为长度784的列表
def load_data(file_path, data, label):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        if file_path == train_path:
            for row in reader:
                data.append(row[1:]) #每行第二个以后的字符对应的是长度为784的串，为图片灰度信息
                tmp = []
                for num in range(0, 10): #每行第一个元素是该图片对应的正确数字，将其装换成[0000100000]这类的形式
                    if row[0] == str(num): 
                        tmp.append(1)
                    else:
                        tmp.append(0)
                label.append(tmp)
        elif file_path == test_path:
            for row in reader:
                data.append(row) #测试集没有正确数字，故读入灰度信息即可

def write_result(file_path, ans):
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        for item in ans:
            writer.writerow(item)

def variable_summary(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#初始化边权，从正态分布中取值
def init_weight(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

#初始化边的偏置
def init_bias(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#根据边权和输入创建卷积层，步长为1，使用SAME的方式填充
def conv2D(x, w):
    return tf.nn.conv2d(x, w, strides = [1,1,1,1], padding = "SAME")

#建立一个2*2的池化层，取最大值做池化
def pooling(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")


#testing test.csv
#def cnn_test():
#    #here should be test.csv, now testing train.csv 
#    x, y_label, y_pr, train_op, dropout_fc, dropout_input = train_ops()
#    load_data(train_path, train_data, train_label)
#    with tf.Session() as sess:
#        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#        saver = tf.train.Saver()
#        if ckpt and ckpt.model_checkpoint_path:
#            saver.restore(sess, ckpt.model_checkpoint_path)
#            imgs = train_data[1:1000]
#            labels = train_label[1:1000]
#            correct_y_pr = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_pr, 1))
#            accuracy = tf.reduce_mean(tf.cast(correct_y_pr, tf.float32)) 
#            print("Train accuracy: " + str(accuracy.eval(session = sess, feed_dict = {x:imgs, y_label:labels, dropout_input:1, dropout_fc:1})))

#根据训练完的模型识别测试数据
def recognize():
    x, y_label, recg_result, dropout_fc, dropout_input = recg_ops()
    load_data(test_path, test_data, test_label)
    with tf.Session() as sess:
        #使用Saver获取存储在checkpoint里的模型信息
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            print("recognizing...")
            saver.restore(sess, ckpt.model_checkpoint_path)
            imgs = test_data[1:] #测试数据集的第一行为pixel0, pixel1...等的标记信息，所以test_data[0]舍弃
            #dropout 设定为1，由此识别不对模型产生改动
            result = sess.run(recg_result, feed_dict = {x:imgs, dropout_input:1, dropout_fc:1}).tolist()
        it = 1
        ans = [["ImageId", "Label"]] 
        for item in result:
            ans.append([it, result[it-1]]) 
            it = it+1
        write_result(output_path, ans)
 
#建立卷积神经网路模型
def cnn_model():
    dropout_input = tf.placeholder(tf.float32)
    dropout_fc = tf.placeholder(tf.float32)
    with tf.name_scope("reshape"):
        x = tf.placeholder(tf.float32, shape = [None, pic_size]) #长度为784的灰度信息
        y_label = tf.placeholder(tf.float32, shape = [None, 10]) #1-10, 正确的输出
        x_image = tf.reshape(x, [-1,28,28,1]) #将输入重新编程28*28的格式
        tf.summary.image('input', x_image, 10)
    with tf.name_scope("conv1"):
        w_conv1 = init_weight([5,5,1,32]) #建立一个5*5的Patch，输出32个特征（从正态分布中选取）
        b_conv1 = init_bias([32]) #给予每个特征偏置
        o_conv1 = tf.nn.relu(conv2D(x_image, w_conv1) + b_conv1) #应用ReLU激活函数处理卷积神经元
        variable_summary(w_conv1)
        variable_summary(b_conv1)
    with tf.name_scope("pool1"):
        p_conv1 = pooling(o_conv1) #对ReLU神经元的输出进行池化
    with tf.name_scope("conv2"):
        w_conv2 = init_weight([5,5,32,64]) #第二层卷积层，对从第一层得到的32个输入处理得到64个输出
        b_conv2 = init_bias([64])
        o_conv2 = tf.nn.relu(conv2D(p_conv1, w_conv2) + b_conv2)
    with tf.name_scope("pool2"):
        p_conv2 = pooling(o_conv2)
    with tf.name_scope("hiddenfc"): #隐藏的全连接网路层
        w_fc1 = init_weight([7*7*64, 512]) #第二层池化后的7*7的图有64个特征，加入一个512的全连接神经网路
        b_fc1 = init_bias([512])
        flat_pool2 = tf.reshape(p_conv2, [-1, 7*7*64])
        o_fc1 = tf.nn.relu(tf.matmul(flat_pool2, w_fc1) + b_fc1) #将拍平后的张量乘上权重加上偏置
    with tf.name_scope("dropout"):
        dropout_fc = tf.placeholder(tf.float32)
        dr_output = tf.nn.dropout(o_fc1, dropout_fc) #以概率舍弃全连接层的输出结果，减少过拟合
    with tf.name_scope("outputfc"):
        w_fc2 = init_weight([512,10])
        b_fc2 = init_bias([10])
        y_pr = tf.matmul(dr_output, w_fc2) + b_fc2 #对输出（数字）的预测
        l2_loss = tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(w_conv2) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(b_fc2)
        variable_summary(l2_loss)
    return x, y_label, y_pr, l2_loss, dropout_input, dropout_fc

def train_ops():
    #从建立cnn_model()获得建立的模型信息
    x, y_label, y_pr, l2_loss, dropout_input, dropout_fc = cnn_model()
    with tf.name_scope("loss"):
        #The softmax method here may be removed from tensorflow soon...
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_pr)) #交叉熵作为损失估计
        cross_entropy += 5e-4 * l2_loss 
        tf.summary.scalar('cross_entropy', cross_entropy)
    return x, y_label, y_pr, tf.train.AdamOptimizer(learn_speed).minimize(cross_entropy), dropout_input, dropout_fc 

def recg_ops():
    #从cnn_model()中获取「识别」需要的信息
    x, y_label, y_pr, l2_loss, dropout_input, dropout_fc = cnn_model()
    return x, y_label, tf.argmax(y_pr, 1), dropout_fc, dropout_input

def cnn_train():
    load_data(train_path, train_data, train_label)
    x, y_label, y_pr, train_op, dropout_input, dropout_fc = train_ops() #获取训练模型师会用到的信息
    with tf.Session() as sess:
        saver = tf.train.Saver() #用来储存模型的类
        sess.run(tf.global_variables_initializer())
        testSet_size = 200 #切割训练集最后200个元素作为训练中途检验用的数据
        l = len(train_data) - testSet_size 
        batch_size = 200
        sz = l - batch_size - 1
        for i in range(1, 5000):
            sys.stdout.write(' Training----> ' + str(i)+ ' Batch...' + '\r')
            #随机从保留的训练集中选取batch_size 个数据作为输入训练模型
            t = random.randint(1,sz)  
            sess.run(train_op, feed_dict = {y_label:train_label[t:t+batch_size], x:train_data[t:t+batch_size],dropout_input:0.5, dropout_fc:0.5})
            #每到训练500轮时储存一次当前信息，并测试模型性能
            if i % 500 == 0:
                saver.save(sess, checkpoint_dir + 'cnn_model.ckpt', global_step = i+1)
                imgs = train_data[-testSet_size:] #在训练集中保留的评估用数据
                labels = train_label[-testSet_size:]
                correct_y_pr = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_pr, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_y_pr, tf.float32)) 
                tf.summary.scalar('accuracy', accuracy)
                merged = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
                #dropout = 1,评估行为对模型不造成影响
                print("train accuracy: " + str(accuracy.eval(session = sess, feed_dict = {x:imgs, y_label:labels, dropout_input:1, dropout_fc:1})))


arg_num = len(sys.argv) 
if arg_num == 1 or sys.argv[1] == '0':
    recognize()
elif sys.argv[1] == '1':
    cnn_train()
print("Done!")
