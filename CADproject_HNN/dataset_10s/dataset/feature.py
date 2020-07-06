import numpy as np

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

def feature_set(feature,n):
    newfeature = []
    for i in range(len(feature)):
        f = [0]*n
        for j in range(n):
            if feature[i][j]!=0:
                f[j] = 1
        newfeature.append(f)
    return newfeature

def zscore(feature):
    feature = (feature - feature.mean(axis=0)) / feature.std(axis=0)
    return feature

olddata = np.array(np.loadtxt('CF_totaldata.txt'))
print(olddata.shape)
newdata = np.array(np.loadtxt('OHF_original.txt'))
ix = newdata[:,-1]
ix = np.array(ix,dtype=int)
newdata = olddata[ix,:]
np.savetxt('CF_original.txt',newdata,fmt='%.2f')

print(olddata.shape,newdata.shape)
data = newdata
d = 40
f =46
feature = data[:, :d]
feature = zscore(feature)
time = list(data[::, f:f+6])
label = np.array(label_set(time, 6))
data = np.hstack((feature, label))

r, c = data.shape
train_size=r//6*4   #train dataset size
val_size=r//6   #valid dataset size

CF_train=data[0:train_size,:]
CF_valid=data[train_size:train_size+val_size,:]
CF_test=data[train_size+val_size:,:]
np.savetxt('./CF_train.txt', CF_train, fmt='%1.6f')
np.savetxt('./CF_valid.txt', CF_valid, fmt='%1.6f')
np.savetxt('./CF_test.txt', CF_test, fmt='%1.6f')
