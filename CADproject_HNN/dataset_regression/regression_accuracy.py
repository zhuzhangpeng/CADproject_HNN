import numpy as np
#train data
train_real = np.loadtxt('dataset/train_zscore.txt')

train_realtime = list(train_real[:,-1])
train_predict = np.loadtxt('OHF_predict_regression/train_predict.txt')
print(train_real.shape, train_predict.shape)
train_predict = list(train_predict)
#tvalid data
valid_real = np.loadtxt('dataset/valid_zscore.txt')
valid_realtime = list(valid_real[:,-1])
valid_predict = list(np.loadtxt('OHF_predict_regression/valid_predict.txt'))
print(valid_real.shape)

#test data
test_real = np.loadtxt('dataset/test_zscore.txt')
test_realtime = list(test_real[:,-1])
test_predict = list(np.loadtxt('OHF_predict_regression/test_predict.txt'))
def pre_accuracy(realtime, predict):
    count = 0 # number of predict right
    for i in range(len(realtime)):
        if realtime[i]<10 and predict[i]<10:
            count +=1
        if realtime[i]>=10 and realtime[i]<60 and predict[i]>=10 and predict[i]<60:
            count += 1
        if realtime[i]>=60 and realtime[i]<600 and predict[i]>=60 and predict[i]<600:
           count += 1
        if realtime[i]>=600 and predict[i]>=600:
             count += 1
    return count/len(realtime)

train_accuracy = pre_accuracy(train_realtime, train_predict)
valid_accuracy = pre_accuracy(valid_realtime, valid_predict)
test_accuracy = pre_accuracy(test_realtime, test_predict)

print('train predict accuracy:{0:.4f},'
      'valid predict accuracy:{1:.4f},'
      'test predict accuracy:{2:.4f} '.
      format(train_accuracy,valid_accuracy, test_accuracy ))
