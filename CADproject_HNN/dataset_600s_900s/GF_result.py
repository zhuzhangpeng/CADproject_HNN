import numpy as np
realdata = np.array(np.loadtxt('./dataset/GF_original.txt'))   #original dataset
predict = np.array(np.loadtxt('./GF_predict/test_predict.txt'))    #predict labels
real = realdata[int(realdata.shape[0])//6*5::,:]
precell = 0   #predict orders's CAD cells
pretime = 0   # predict orders's CAD time
precount = 0   # predict best orders count
preout = 0     #predict timeout count
print(predict.shape,real.shape,predict[0])
c = 30   # feature count
d = 36   #feature count + orders count
for i in range(predict.shape[0]):
    if real[i, d+int(predict[i])]==min(real[i,d:d+6]):
        precount += 1
    if real[i, d+int(predict[i])]==1800:
        preout += 1
    precell = precell+real[i, c+int(predict[i])]
    pretime = pretime + real[i, d+int(predict[i])]
print('predict result:\n total CAD cell:{0:.2f}, avg CAD cell:{1:.2f}, total time:{2:.2f},'
      'avg time:{3:.2f}, predict accuracy:{4:.4f}, predict timeout:{5}'.
      format(precell, precell/predict.shape[0],pretime, pretime/predict.shape[0],
             precount/predict.shape[0], preout))

n = realdata.shape[0]-realdata.shape[0] // 6 * 5   #dataset
#print(n, realdata.shape)
#result for best order based on min time
CADcell = 0
CADt = 0
for i in range(realdata.shape[0]//6*5, realdata.shape[0]):
    for j in range(6):
        if realdata[i,d+j]==min(realdata[i,d:d+6]):
            CADcell += realdata[i, c+j]
            CADt += realdata[i, d+j]
            break
print('CAD result:\n total CAD cell:{0:.2f}, avg CAD cell:{1:.2f}, total time:{2:.2f}, '
      'avg time:{3:.2f}'.format(CADcell, CADcell/n, CADt, CADt/n))

#svo  result
svocell = 0
svot = 0   # total CAD time
svocount = 0   #best svo orders count
for i in range(realdata.shape[0]//6*5, realdata.shape[0]):
    svolabel = int(realdata[i,d+6])-1
    if realdata[i, d+svolabel]==min(realdata[i, d:d+6]):
        svocount +=1
    svocell += realdata[i, c+svolabel]
    svot += realdata[i, d+svolabel]
print('SVO result:\n total CAD cell:{0:.2f}, avg CAD cell:{1:.2f}, total time:{2:.2f}, '
      'avg time:{3:.2f}, SVO accuracy: {4:.4f}'.format(svocell, svocell/n, svot, svot/n, svocount/n))
