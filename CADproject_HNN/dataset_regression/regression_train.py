import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
c = 40
F1 =64
F2 = 64

keep_prob = tf.placeholder(tf.float32)
xs=tf.placeholder(tf.float32,[None,c])
ys=tf.placeholder(tf.float32,[None,1])
sess=tf.Session()
b = []

train_data = np.loadtxt('dataset/train_zscore.txt')
valid_data = np.loadtxt('dataset/valid_zscore.txt')
print(train_data.shape, valid_data.shape)
train_x = train_data[:,:c]
train_y = train_data[:,c:]
valid_x = valid_data[:,:c]
valid_y = valid_data[:,c:]
def add_layer(input,in_size,out_size,activation_function):
    Weight=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size]))+0.001
    Wx_plus_b=tf.matmul(input,Weight)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
L1=add_layer(xs,c,F1,activation_function=tf.nn.tanh)
L2=add_layer(L1,F1,F2,activation_function=tf.nn.tanh)
dropped = tf.nn.dropout(L2,keep_prob)
prediction = add_layer(L2,F2,1,activation_function=None)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
hubers = tf.losses.huber_loss(ys, prediction,delta=1.0)
hubers_loss = tf.reduce_sum(hubers)

train_step=tf.train.RMSPropOptimizer(0.01).minimize(hubers_loss)

init=tf.global_variables_initializer()
sess.run(init)
batch_size = 512
data_size = len(train_x)
STEPS = 200001
train_lossing = []
valid_lossing = []
for i in range(STEPS):
    start = (i*batch_size)%data_size
    end = min(start + batch_size,data_size)
    sess.run(train_step,feed_dict={xs:train_x[start:end], ys:train_y[start:end],keep_prob: 0.8})
    if i % 2000 == 0:
        train_pre, train_loss = sess.run([prediction,hubers_loss], feed_dict={xs:train_x[:,:],ys:train_y[:,:],keep_prob: 1})
        valid_pre,valid_loss = sess.run([prediction, hubers_loss], feed_dict={xs:valid_x,ys:valid_y,keep_prob: 1})
        train_lossing.append(train_loss)
        valid_lossing.append(valid_loss)
        print("i=", i, "train_loss=", train_loss,"test_loss=", valid_loss)
        np.savetxt('OHF_predict_regression/train_predict.txt', train_pre, fmt="%2.12f")
        np.savetxt('OHF_predict_regression/valid_predict.txt', valid_pre, fmt="%2.12f")


saver=tf.train.Saver()


src='matmul'
#读取测试集
test_data = np.loadtxt('dataset/test_zscore.txt')
print("valiation data shape:",test_data.shape)
test_x = test_data[:,:c].astype('float')
test_y = test_data[:,c:].astype('float')

test_x = test_x[:,:]
test_y = test_y[:,:]

import time
cpu_start = time.clock()
test_y = sess.run(prediction,feed_dict={xs:test_x,ys:test_y,keep_prob:1})
cpu_end = time.clock()
print('cpu:', cpu_end - cpu_start)
print(test_y.shape)
print("test_loss=",sess.run(hubers_loss,feed_dict={xs:test_x,ys:test_y,keep_prob:1}))
np.savetxt('OHF_predict_regression/test_predict.txt',test_y,fmt="%2.12f")

plt.figure()
plt.plot(train_lossing, label='train lossing')
plt.plot(valid_lossing, label='test lossing')
plt.show()
