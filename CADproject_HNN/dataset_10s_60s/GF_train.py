import tensorflow as tf
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
random.seed(1234)


epoch = 500     #Number of iterations
batch_size = 128
num_C = 6    #Number of labels
num_F = 30   #Number of feature
num_fc1 = 128     #First hidden layer size
num_fc2=256   #Second hidden layer size
num_fc3=512   #Third hidden layer size
num_fc4 = 512
learning_rate = 0.01    #Learning rate
regular_rate = 0.001    #L2 regularization parameters

display_step = 100   #Output the current training result every iteration 100 times

model_path = 'GF_model'   #Save path of the model
model_name = 'dense_model'   #The prefix name of the model file
predict_name = 'GF_predict'   #Save path of prediction results
traindata = np.array(np.loadtxt('./dataset/GF_train.txt'))   #Read training set data
X_train = traindata[:, 0:num_F]   #Features of the training set
Y_train = traindata[:, num_F:]    #Label of training set

validdata = np.array(np.loadtxt('./dataset/GF_valid.txt'))    #Read validation set data
X_validation = validdata[:, 0:num_F]       #Features of the validation set
Y_validation = validdata[:, num_F:]        #Label of validation set

'''
The structure of the model.The structure of the model. First, L2 regularization 
is performed on the features, and then three hidden layers are used. The activation 
function is relu.
'''
def inference(input, num_output, training):
    #regularizer = tf.contrib.layers.l2_regularizer(scale=regular_rate)
    regularizer = tf.contrib.layers.l2_regularizer(scale=regular_rate)

    fc1 = tf.layers.dense(input, num_fc1,
                          bias_initializer=tf.constant_initializer(0.1),
                          activation='relu', kernel_regularizer=regularizer)
    fc2 = tf.layers.dense(fc1, num_fc2,
                          bias_initializer=tf.constant_initializer(0.1),
                          activation='relu', kernel_regularizer=regularizer)
    fc3 = tf.layers.dense(fc2, num_fc2,
                          bias_initializer=tf.constant_initializer(0.1),
                          activation='relu', kernel_regularizer=regularizer)
    logit = tf.layers.dense(fc2, num_output)
    return logit


x = tf.placeholder(tf.float32, shape=[None, num_F], name='x_input')   #Input
batch_input = tf.reshape(x, [-1, num_F], name='batch_input')
training_flag = tf.placeholder(tf.bool)
y_ = tf.placeholder(tf.float32, shape=[None, num_C], name='y_output')   #output
y_model = inference(input=batch_input, num_output=num_C, training=training_flag)
y = tf.nn.softmax(y_model)   #Map the output result between 0 and 1, which is the probability of each label.

tf.add_to_collection('output', y_model)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(y_, 1),
                                                               logits=y_model)

#lossing function
loss = tf.reduce_mean(cross_entropy)
l2_loss = tf.losses.get_regularization_loss()
loss += l2_loss
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
predict_label = tf.arg_max(y, 1, name='predict_label')   #Predict result,#Set the label with the highest probability to 1
tf.add_to_collection('predict_label', predict_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')   #the accuracy of model predictions

train = []
valid = []
lossing = []
validloss = []
min_loss = 100
best_valid_accuracy = 0
best_epoch = 0
valid_accuracy_i = 0

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epoch):
        total_batch = int(np.ceil(X_train.shape[0]/batch_size))
        for step in range(total_batch):
            start = (step * batch_size) % int(np.shape(X_train)[0])
            end = min(start+batch_size, int(np.shape(X_train)[0]))

            batch_x = X_train[start:end, 0:num_F]
            batch_y = Y_train[start:end, 0:num_C]

            train_feed = {
                x: batch_x,
                y_: batch_y,
                training_flag: True
            }
            _ = sess.run([train_step], train_feed)

        train_feed1 = {
            x: X_train,
            y_: Y_train,
            training_flag: True
        }

        #The prediction result of the model on the training set
        train_loss, train_accuracy, pre_label = sess.run([loss, accuracy, predict_label], train_feed1)
        train.append(train_accuracy)
        lossing.append(train_loss)

        #The prediction result of the model on the validation set
        valid_feed = {
            x: X_validation,
            y_: Y_validation,
            training_flag: False
        }
        valid_loss, valid_accuracy, pre_valid = sess.run([loss, accuracy, predict_label], valid_feed)
        valid.append(valid_accuracy)
        validloss.append(valid_loss)

        '''
        only save the best model.When the accuracy of the verification set is the highest,
        the results are output, and the prediction results of the training set and the 
        verification set are saved.
        '''
        if valid_accuracy>best_valid_accuracy:
            min_loss = valid_loss
            best_epoch = i
            best_train_accuracy = train_accuracy
            best_valid_accuracy = valid_accuracy
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            saver.save(sess, os.path.join(model_path, model_name), global_step=epoch)
            np.savetxt(predict_name+'/train_predict.txt', pre_label,fmt='%d')
            np.savetxt(predict_name+'/valid_predict.txt', pre_valid,fmt='%d')

        if i % display_step == 0:
            print("epoch={0}, training_loss={1:.8}, valid_loss={2:.8}, training_accuracy={3:.8}, valid accuracy={4:.8}".
                  format(i, train_loss, valid_loss, train_accuracy, valid_accuracy))

    #Output the results of the best training model.
    print('Best epoch:', best_epoch, 'train accuracy:', train_accuracy,
          'valid accuracy:', best_valid_accuracy, 'train loss:', best_train_loss, 'valid loss:',
          best_valid_loss)

'''
Visualize the changes in accuracy and loss functions on the training and validation 
sets during the training process.
'''
plt.plot(train, label="train accuracy")
plt.legend()
plt.plot(valid, label="valid accuracy")
plt.legend()
#plt.savefig(predict_name+'var'+str(num_C)+'_accuracy.eps')
#plt.savefig(predict_name+'var'+str(num_C)+'_accuracy.png')
plt.show()
plt.plot(lossing, label="train loss")
plt.legend()
plt.plot(validloss, label="valid loss")
plt.legend()
#plt.savefig(predict_name+'var'+str(num_C)+'_loss.eps')
#plt.savefig(predict_name+'var'+str(num_C)+'_loss.png')
plt.show()
