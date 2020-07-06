import numpy as np
import tensorflow as tf
#import lenet_conv1d_v3   # [1]加载mnist_train.py文件中定义的函数和常量
c = 40
d = 46
testdata = np.array(np.loadtxt('./dataset/test_original.txt'))
X_test = testdata[:, 0:c]
time_test = testdata[:, d:d+6]
print(X_test.shape, X_test[0].reshape([1,c]).shape)
Y_regression = np.array(np.loadtxt('./OHF_predict_regression/test_predict.txt'))
Y_test = Y_regression
#Y_test = label_reset(Y_test, Y_test.shape[0])
print(X_test.shape)

def zscore(F):
    newF = (F - F.mean(axis=0)) / F.std(axis=0)
    return newF

def label_set(time, n):
    result = []
    for i in range(len(time)):
        label = [0] * n
        for j in range(n):
            if time[i][j] == min(time[i]):
                label[j] = 1
                break
        result.append(label)
    return result

def predict(filename, X_test,Y_test):   #根据filename调用相应模型
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state('../'+filename+'/OHF_model')
    print(ckpt)
    new_saver = tf.train.import_meta_graph('../'+filename+'/OHF_model/dense_model-500.meta')
    new_saver.restore(sess, save_path='../'+filename+'/OHF_model/dense_model-500')
    graph = tf.get_default_graph()
    x = graph.get_operation_by_name('x_input').outputs[0]
    y_ = graph.get_operation_by_name('y_output').outputs[0]
    pred = graph.get_operation_by_name('predict_label').outputs[0]
    test_accuracy = graph.get_operation_by_name('accuracy').outputs[0]
    #correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y_test, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy,y = sess.run([test_accuracy, pred], feed_dict={x: X_test,y_:Y_test})
    sess.close()
    return  accuracy, y

index = np.array(range(X_test.shape[0]))
index = index.reshape([X_test.shape[0],1])
X_test = np.hstack((X_test, time_test, index))
np.savetxt('testdata_regression.txt',X_test,fmt='%.6f')
data1 = []
data2 = []
data3 = []
data4 = []
for i in range(testdata.shape[0]):
    if Y_test[i]<=10.:
        data1.append(X_test[i])
    elif Y_test[i]<=60.:
        data2.append(X_test[i])
    elif Y_test[i]<=600.:
        data3.append(X_test[i])
    else:
        data4.append(X_test[i])

data1 = np.array(data1)
X_test1 = np.array(zscore(data1[:,:c]))
Y_test1 = np.array(label_set(data1[:,c:c+6],6))
data2 = np.array(data2)
X_test2 = np.array(zscore(data2[:,:c]))
Y_test2 = np.array(label_set(data2[:,c:c+6],6))
data3 = np.array(data3)
X_test3 = np.array(zscore(data3[:,:c]))
Y_test3 = np.array(label_set(data3[:,c:c+6],6))
data4 = np.array(data4)
X_test4 = np.array(zscore(data4[:,:c]))
Y_test4 = np.array(label_set(data4[:,c:c+6],6))
print('data1.shape',data1.shape)
#np.savetxt('data1.txt',data1,fmt='%.4f')
ac1, result1 = predict('dataset_10s', X_test1, Y_test1)
ac2, result2 = predict('dataset_10s_60s', X_test2,Y_test2)
ac3, result3 = predict('dataset_60s_600s', X_test3,Y_test3)
ac4, result4 = predict('dataset_600s_900s', X_test4,Y_test4)
print('{0:.4f},{1:.4f},{2:.4f},{3:.4f}'.format(ac1,ac2,ac3,ac4))
result1 = result1.reshape([result1.shape[0],1])
result1 = np.hstack((result1, data1[:,-1].reshape([data1.shape[0],1])))
#np.savetxt('result1.txt',result1,fmt='%.4f')
result2 = result2.reshape([result2.shape[0],1])
result2 = np.hstack((result2, data2[:,-1].reshape([data2.shape[0],1])))
result3 = result3.reshape([result3.shape[0],1])
result3 = np.hstack((result3, data3[:,-1].reshape([data3.shape[0],1])))
result4 = result4.reshape([result4.shape[0],1])
result4 = np.hstack((result4, data4[:,-1].reshape([data4.shape[0],1])))
result = np.vstack((result1,result2,result3,result4))
result = result[result[:,1].argsort()]
print(X_test.shape,result.shape)
np.savetxt('./predict_regression.txt',result,fmt='%.4f')


realdata1 = np.loadtxt('../dataset_10s/dataset/OHF_original.txt')
realdata2 = np.loadtxt('../dataset_10s_60s/dataset/OHF_original.txt')
realdata3 = np.loadtxt('../dataset_60s_600s/dataset/OHF_original.txt')
realdata4 = np.loadtxt('../dataset_600s_900s/dataset/OHF_original.txt')
n1 = realdata1.shape[0]-realdata1.shape[0]//6*5
n2 = realdata2.shape[0]-realdata2.shape[0]//6*5
n3 = realdata3.shape[0]-realdata3.shape[0]//6*5
n4 = realdata4.shape[0]-realdata4.shape[0]//6*5
pre1 = result[:n1,:]
pre2 = result[n1:n1+n2,:]
pre3 = result[n1+n2:n1+n2+n3,:]
pre4 = result[n1+n2+n3:,:]
print(pre1.shape[0],pre2.shape[0],pre3.shape[0],pre4.shape[0])
np.savetxt('../dataset_10s/OHF_predict/'
           'test_regression.txt',pre1,fmt='%.4f')
np.savetxt('../dataset_10s_60s/OHF_predict/'
           'test_regression.txt',pre2,fmt='%.4f')
np.savetxt('../dataset_60s_600s/OHF_predict/'
           'test_regression.txt',pre3,fmt='%.4f')
np.savetxt('../dataset_600s_900s/OHF_predict/'
           'test_regression.txt',pre4,fmt='%.4f')


#cumpute the accuracy
count1 =0
t1 = 0
n1 = 0
count2 = 0
t2 = 0
n2 = 0
count3 = 0
t3 = 0
n3 = 0
count4 =0
t4 = 0
n4 = 0
timeout = 0
totaltimeout = 0
for i in range(result.shape[0]):
    if max(X_test[i, c:c+6])<=10:
        if X_test[i,c+int(result[i, 0])]==min(X_test[i,c:c+6]):
            count1 += 1
        t1 += X_test[i, c+int(result[i, 0])]
        n1 +=1
    elif max(X_test[i, c:c+6])<=60:
        if X_test[i,c+int(result[i, 0])]==min(X_test[i,c:c+6]):
            count2 += 1
        t2 += X_test[i, c+int(result[i, 0])]
        n2 += 1
    elif max(X_test[i, c:c+6])<=600:
        if X_test[i,c+int(result[i, 0])]==min(X_test[i,c:c+6]):
            count3 += 1
        t3 += X_test[i, c+int(result[i, 0])]
        n3 += 1
    else:
        if X_test[i,c+int(result[i, 0])]==min(X_test[i,c:c+6]):
            count4 += 1
        t4 += X_test[i, c+int(result[i, 0])]
        n4 += 1
    if X_test[i, c + int(result[i, 0])]==1800:
            timeout += 1
    if max(X_test[i, c :c+6]) == 1800:
        totaltimeout += 1
print('test of 0-10: count:{0},accuracy:{1:.4f},avg time:{2:.2f}'.format(n1,count1/n1,t1/n1))
print('test of 10-60: count:{0},accuracy:{1:.4f},avg time:{2:.2f}'.format(n2,count2/n2,t2/n2))
print('test of 60-600: count:{0},accuracy:{1:.4f},avg time:{2:.2f}'.format(n3,count3/n3,t3/n3))
print('test of 600-1800: count:{0},accuracy:{1:.4f},avg time:{2:.2f}, '
      'timeout count:{3}, total timeout:{4}'.
      format(n4,count4/n4,t4/n4,timeout,totaltimeout))