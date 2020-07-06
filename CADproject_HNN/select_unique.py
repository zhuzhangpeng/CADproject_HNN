import numpy as np


#The real code
#data is [feature,cell,time] with number of columns [30,6,6]
totaldata = np.array(np.loadtxt('./totaldata_CF.txt'))
time = np.array(np.loadtxt('./totaldata_GF.txt'))
time = time[:,30:]
totaldata = np.hstack((totaldata,time))
count = 0
data1=[]
data2=[]
data3=[]
data4=[]

a = totaldata[:, 0:]
print(a.shape)
d = 40
e = 46
r, c = a.shape
tmpa, ix = np.unique(a[:, 0:d], axis=0, return_index=True)
newF = a[ix][:, 0:d]
newL = a[ix][:, d::]
#ix = np.array(ix)
ix = ix.reshape((newF.shape[0], 1))
newa = np.hstack((newF, newL, ix))
print(newa.shape)
print(np.unique(newa, axis=0).shape)
np.random.shuffle(newa)
print(newa.shape)
print(np.unique(newa, axis=0).shape)
np.savetxt('totaldata_CF_unique.txt', newa, fmt='%1.2f')


data = newa.tolist()
for i in range(len(data)):
    if max(data[i][e:e+6])<=10.0:
        data1.append(newa[i])
    elif max(newa[i][e:e+6])<=60.0:
        data2.append(newa[i])
    elif max(newa[i][e:e+6])<=600.0:
        data3.append(newa[i])
    else:
        data4.append(newa[i])

data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.array(data3)
data4 = np.array(data4)
print('0-10s:{0},10-60s:{1},60-600s:{2},600-900s:{3}'.format(data1.shape,data2.shape,data3.shape,data4.shape))  
np.savetxt('./dataset_10s/dataset/CF_totaldata.txt',data1,fmt='%.2f')
np.savetxt('./dataset_10s_60s/dataset/CF_totaldata.txt',data2,fmt='%.2f')
np.savetxt('./dataset_60s_600s/dataset/CF_totaldata.txt',data3,fmt='%.2f')
np.savetxt('./dataset_600s_900s/dataset/CF_totaldata.txt',data4,fmt='%.2f')

