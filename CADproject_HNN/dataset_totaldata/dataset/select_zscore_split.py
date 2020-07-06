import numpy as np
c = 40
d = 46

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

#train data
train=np.loadtxt('./train_original.txt')
valid = np.loadtxt('./valid_original.txt')
test = np.loadtxt('./test_original.txt')
print(train.shape, valid.shape, test.shape)
dataset = np.vstack((train, valid,test))

np.savetxt('./total_original.txt',dataset,fmt='%1.6f')

print(dataset.shape)
time = dataset[:, d:d+6]
label = label_set(time, 6)
F=dataset[:,0:c]
L=np.array(label)
#L = np.reshape((L.shape[0],1))
newF=(F-F.mean(axis=0))/F.std(axis=0)
newa=np.hstack([newF, L])
print(newa.shape)
#np.savetxt('./total_zscore.txt',newa,fmt='%1.6f')

r, c=newa.shape
d=40
train_size=r//6*4
val_size=r//6
test_size=r-train_size-val_size
train_data=newa[0:train_size,:]
val_data=newa[train_size:train_size+val_size,:]
test_data=newa[train_size+val_size:r,:]
print(train_data.shape,val_data.shape, test_data.shape)
np.savetxt('./train_zscore.txt', train_data, fmt='%1.6f')
np.savetxt('./valid_zscore.txt', val_data, fmt='%1.6f')
np.savetxt('./test_zscore.txt', test_data, fmt='%1.6f')

