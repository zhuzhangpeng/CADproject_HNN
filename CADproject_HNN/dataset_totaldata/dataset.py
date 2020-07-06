import numpy as np
#get total dataset
data1 = np.loadtxt('../dataset_10s/dataset/OHF_original.txt')
data2 = np.loadtxt('../dataset_10s_60s/dataset/OHF_original.txt')
data3 = np.loadtxt('../dataset_60s_600s/dataset/OHF_original.txt')
data4 = np.loadtxt('../dataset_600s_900s/dataset/OHF_original.txt')

train = np.vstack((data1[:data1.shape[0]//6*4,:],data2[:data2.shape[0]//6*4,:],
                   data3[:data3.shape[0]//6*4,:], data4[:data4.shape[0]//6*4,:]))
np.random.shuffle(train)

valid = np.vstack((data1[data1.shape[0]//6*4:data1.shape[0]//6*5,:],
                   data2[data2.shape[0]//6*4:data2.shape[0]//6*5,:],
                   data3[data3.shape[0]//6*4:data3.shape[0]//6*5,:],
                   data4[data4.shape[0]//6*4:data4.shape[0]//6*5,:]))
np.random.shuffle(valid)

test = np.vstack((data1[data1.shape[0]//6*5:,:],
                  data2[data2.shape[0]//6*5:,:],
                  data3[data3.shape[0]//6*5:,:],
                  data4[data4.shape[0]//6*5:,:]))
np.random.shuffle(test)

print('train size:{0},valid size:{1},test size:{2},total data size:{3},'
      ' reset data size:{4}'.format(train.shape,valid.shape,test.shape,
             data1.shape[0]+data2.shape[0]+data3.shape[0]+data4.shape[0],
                                    train.shape[0]+valid.shape[0]+test.shape[0]))
np.savetxt('./dataset/train_original.txt',train,fmt='%.2f')
np.savetxt('./dataset/valid_original.txt',valid,fmt='%.2f')
np.savetxt('./dataset/test_original.txt',test,fmt='%.2f')
