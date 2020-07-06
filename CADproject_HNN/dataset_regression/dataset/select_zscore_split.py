import numpy as np
def set_label(time):     #make longest time as the label
    label = []
    for i in range(len(time)):
        label.append(max(time[i]))
    return label

def zscore(feature):    #zscore the features
    feature = (feature - feature.mean(axis=0)) / feature.std(axis=0)
    return feature

c = 40
d = 46
n = 6
#train data
train = np.loadtxt('./train_original.txt')
train_time = train[:,d:d+n]
train_label = set_label(list(train_time))
train_label = np.array(train_label)
print(train.shape, train_label.shape)
train_label = train_label.reshape([train.shape[0],1])
train_feature = train[:, :c]
train_feature = zscore(train_feature)
print(train.shape, train_label.shape)
train_data = np.hstack((train_feature, train_label[:,:]))
np.savetxt('./train_zscore.txt', train_data,fmt='%.6f')



#valid data
valid = np.loadtxt('./valid_original.txt')
valid_time = valid[:,d:d+n]
valid_label = set_label(list(valid_time))
valid_label = np.array(valid_label)
print(valid.shape, valid_label.shape)
valid_label = valid_label.reshape([valid.shape[0],1])
valid_feature = valid[:, :c]
valid_feature = zscore(valid_feature)
print(valid.shape, valid_label.shape)
valid_data = np.hstack((valid_feature, valid_label[:,:]))
np.savetxt('./valid_zscore.txt', valid_data,fmt='%.6f')

#test data
test = np.loadtxt('./test_original.txt')
test_time = test[:,d:d+n]
test_label = set_label(list(test_time))
test_label = np.array(test_label)
print(test.shape, test_label.shape)
test_label = test_label.reshape([test.shape[0],1])
test_feature = test[:, :c]
test_feature = zscore(test_feature)
print(test.shape, test_label.shape)
test_data = np.hstack((test_feature, test_label[:,:]))
np.savetxt('./test_zscore.txt', test_data,fmt='%.6f')