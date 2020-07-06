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
data = olddata
d = 40
f =46
oldfeature = list(data[:, :d])
feature = np.array(feature_set(oldfeature,d))
data = np.hstack((feature, data[:,d:]))
a = data[:, :]
print(a.shape)
d = 40
e = 46

tmpa, ix = np.unique(a[:, 0:d], axis=0, return_index=True)
newF = a[ix][:, 0:d]
newL = a[ix][:, d::]
print(newF.shape)
#ix = np.array(ix)
ix = ix.reshape((newF.shape[0], 1))
newa = np.hstack((newF, newL,ix))
np.random.shuffle(newa)
np.savetxt('./OHF_original.txt', newa, fmt='%1.2f')
feature = newa[:,0:d]
time = list(newa[::, f:f+6])
label = np.array(label_set(time, 6))
feature = zscore(feature)
dataM = np.hstack((feature, label))
#split datast
r, c = dataM.shape
train_size=r//6*4   #train dataset size
val_size=r//6   #valid dataset size
OHF_train=dataM[0:train_size,:]
OHF_valid=dataM[train_size:train_size+val_size,:]
OHF_test=dataM[(train_size+val_size):r, :]
np.savetxt('./OHF_train.txt', OHF_train, fmt='%1.6f')
np.savetxt('./OHF_valid.txt', OHF_valid, fmt='%1.6f')
np.savetxt('./OHF_test.txt', OHF_test, fmt='%1.6f')
print(r,c, train_size+val_size, dataM.shape, OHF_train.shape, OHF_valid.shape,OHF_test.shape)

#old 40feature dataset
ix = newa[:, -1]
#ix = ix.reshape([ix.shape[0],1])
ix = np.array(ix, dtype=int)
olddata = olddata[ix,:]
#print('olddata shape:',olddata.shape)
np.savetxt('CF_original.txt',olddata,fmt='%.4f')
oldlabel = np.array(label_set(list(olddata[:, e:e+6]),6))
oldfeature = olddata[:,0:d]
oldfeature = zscore(oldfeature)
oldM = np.hstack((oldfeature, oldlabel))
#split datast
CF_train=oldM[0:train_size,:]
CF_valid=oldM[train_size:train_size+val_size,:]
CF_test=oldM[train_size+val_size:,:]
np.savetxt('./CF_train.txt', CF_train, fmt='%1.6f')
np.savetxt('./CF_valid.txt', CF_valid, fmt='%1.6f')
np.savetxt('./CF_test.txt', CF_test, fmt='%1.6f')

#30 feature dataset
f = 30
g = 36
ix = olddata[:, -1]
ix= np.array(ix, dtype=int)
dataset = np.loadtxt('./GF_totaldata.txt')
dataset = dataset[ix, :]
ix = ix.reshape([ix.shape[0],1])
dataset = np.hstack((dataset, ix))
np.savetxt('GF_original.txt', dataset, fmt='%.4f')
labelset = np.array(label_set(list(dataset[:,g:g+6]),6))
feature = dataset[:,:f]
feature = zscore(feature)
newset = np.hstack((feature, labelset))
#split datast
GF_train=newset[0:train_size,:]
GF_valid=newset[train_size:train_size+val_size,:]
GF_test=newset[train_size+val_size:,:]
np.savetxt('./GF_train.txt', GF_train, fmt='%1.6f')
np.savetxt('./GF_valid.txt', GF_valid, fmt='%1.6f')
np.savetxt('./GF_test.txt', GF_test, fmt='%1.6f')