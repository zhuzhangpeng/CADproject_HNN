import numpy as np
realdata = np.array(np.loadtxt('./dataset/total_original.txt'))
predict = np.array(np.loadtxt('./OHF_predict/test_predict.txt'))
real = realdata[int(realdata.shape[0])//6*5::,:]
precell = 0
pretime = 0
count = 0
#print(predict.shape,real.shape,predict[0])
c = 40
d = 46
#predict
for i in range(predict.shape[0]):
    for j in range(6):
        if real[i,d+j]==min(real[i,d:d+6]):
            if j==predict[i]:count += 1
    precell = precell+real[i, c+int(predict[i])]
    pretime = pretime + real[i, d+int(predict[i])]
#print('PREDICT: totalcell:',precell,'total time:',pretime, 'avg cell:'
#      ,precell/predict.shape[0], 'avg time:', pretime/predict.shape[0],count/predict.shape[0])

#different time of 0-10, 10-60, 60-600, 600-1800
time1 = 0
count1 = 0
right1 = 0
time2 = 0
count2 = 0
right2 = 0
time3 = 0
count3 = 0
right3 = 0
time4 = 0
count4 = 0
right4 = 0
timeout = 0
for i in range(predict.shape[0]):
    if max(real[i, d:d+6])<=10:
        time1 += real[i, d+int(predict[i])]
        count1 += 1
        if real[i,d+int(predict[i])]==min(real[i, d:d+6]):
            right1 += 1
    elif max(real[i, d:d+6])>10 and max(real[i, d:d+6])<=60:
        time2 += real[i, d + int(predict[i])]
        count2 += 1
        if real[i,d+int(predict[i])]==min(real[i, d:d+6]):
            right2 += 1
    elif max(real[i, d:d+6])>60 and max(real[i, d:d+6])<=600:
        time3 += real[i, d + int(predict[i])]
        count3 += 1
        if real[i,d+int(predict[i])]==min(real[i, d:d+6]):
            right3 += 1
    else:
        time4 += real[i, d + int(predict[i])]
        count4 += 1
        if real[i,d+int(predict[i])]==min(real[i, d:d+6]):
            right4 += 1
        if real[i, d+int(predict[i])]==1800:
            timeout += 1
print('0-10:avg time: {0:.4f},count: {1},accuracy: {2:.4f}'.format(time1/count1,count1,right1/count1))
print('10-60:avg time: {0:.4f},count: {1},accuracy: {2:.4f}'.format(time2/count2,count2,right2/count2))
print('60-600:avg time: {0:.4f},count: {1},accuracy: {2:.4f}'.format(time3/count3,count3,right3/count3))
print('600-1800:avg time: {0:.4f},count: {1},timeout: {2},accuracy:{3:.4f}'.format(time4/count4,count4,timeout,right4/count4))

