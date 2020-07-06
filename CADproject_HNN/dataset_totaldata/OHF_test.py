import numpy as np
import tensorflow as tf
f = 40  #Number of feature
testdata = np.loadtxt('dataset/test_zscore.txt')      #Read test set data

print(testdata.shape)
X_test = testdata[:,:f]    #Features of the training set
Y_test = testdata[:,f:]    #Label of training set

print(X_test.shape,Y_test.shape)
sess = tf.Session()
#Read the training model
ckpt = tf.train.get_checkpoint_state('./OHF_model')
print(ckpt)
new_saver = tf.train.import_meta_graph('./OHF_model/dense_model-500.meta')
new_saver.restore(sess, save_path='./OHF_model/dense_model-500')

graph = tf.get_default_graph()
x = graph.get_operation_by_name('x_input').outputs[0]   #Enter data in the model
y_ = graph.get_operation_by_name('y_output').outputs[0]  #Actual label
pred = graph.get_operation_by_name('predict_label').outputs[0]   #Predict label
test_accuracy = graph.get_operation_by_name('accuracy').outputs[0]   #Prediction accuracy

test_accuracy1, y = sess.run([test_accuracy, pred], feed_dict={x: X_test, y_: Y_test})
print('test accuracy:', test_accuracy1)
np.savetxt('./OHF_predict/test_predict.txt', y,fmt='%.2f')   #Save prediction results
