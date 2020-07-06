程序运行步骤：
1.  运行根目录下的 select_unique.py ,根据多相式的最大CAD计算时间对数据进行分箱处理,将分箱后的数据保存到对应的文件夹中
2.  使用不同特征的直接建立模型分类(M-GF,M-CF,M-OHF)：
    (1)  运行dataset_10s,dataset_10s_60s,dataset_60s_600s,dataset_600s_900s文件夹下dataset文件夹中的
         select_zscore_split.py, 对数据进行预处理(根据最短计算时间设置标签, 将特征进行归一化处理),然后将数据划
         分为训练集、验证集、测试集.
    (2)  在dataset_10s,dataset_10s_60s,dataset_60s_600s,dataset_600s_900s文件夹下,依次运行:
            *_train.py, 获取训练模型
            *_test.py,  获取测试集的预测标签
            *_result.py, 获取测试集预测序的平均计算时间,平均CAD胞腔个数等结果。
            * 为CF,GF,OHF,对应文中的三种特征
    注： 若要复现论文中的结果则需要跳过步骤1, 步骤2中的(1),直接运行步骤2-(2)中的*_result.py,
        因为步骤1,2-(1)中与处理数据时对数据做了随机化处理
3.  使用全部数据集直接分类(M-OHF-Total):
    在dataset_totaldata文件夹下：
    (1) 运行dataset.py, 将分箱后的数据直接按照训练集、验证集、测试集重新整合到一起,以保证数据的一致性
    (2) 运行dataset文件夹下的select_zscore_split.py, 对数据进行预处理(根据最短计算时间设置标签,
        将特征进行归一化处理),然后将数据划分为训练集、验证集、测试集.
    (3) 依次运行:
           OHF_train.py, 获取训练模型
           OHF_test.py,  获取测试集的预测标签
           OHF_result.py, 获取测试集预测序的平均计算时间,平均CAD胞腔个数等结果。
    注： 若要复现论文中的结果则需要跳过步骤(1), (2),直接运行步骤(3)中的OHF_result.py,
        因为步骤(1),(2)中与处理数据时对数据做了随机化处理
4.  使用分级神经网络预测变元序：
    在dataset_regression文件夹下：
    (1) 运行dataset.py 将分箱后的数据直接按照训练集、验证集、测试集重新整合到一起,以保证数据的一致性
    (2) 运行dataset文件夹下的select_zscore_split.py, 对数据进行预处理(将最长计算时间设置为标签,
        将特征进行归一化处理),然后将数据划分为训练集、验证集、测试集.
    (3) 依次运行:
           regression_train.py, 获取训练模型,同时获取测试集的预测时间
           regression_accuracy.py,  获取测试集的分箱准确率
           regression_result.py, 根据测试集的预测CAD计算时间,调用相应的分类模型预测变元序,
           并统计测试集预测序的平均计算时间,平均CAD胞腔个数等结果。
    注： 在发表论文后重新运行了regression_train.py, 导致模型变动,覆盖了原模型,因此直接运行
        regression_result.py后不能得到文中完全相同的结果, 但此结果不影响文中的总体结论


数据格式说明：
1. total_eqs.txt: 全部数据集的原始多项式系统

2. totaldata_CF.txt: 全部数据集的CF特征

3. totaldata_GF.txt: 全部数据集的GF特征以及CAD计算结果, 前30列为GF特征,31-36列为全部序的CAD胞腔个数,
   37-42列为全部序的CAD计算时间(所有序按照字典序升序排列, (x1,x2,x3),(x1,x3,x2),(x2,x1,x3),(x2,x3,x1),
   (x3,x1,x2),(x3,x2,x1)), 最后一列为进程序号,可忽略

4. total_CF_unique.txt: 在CF特征的基础上去除特征相同的数据, 前40列为CF特征,41-46列为全部序的CAD胞腔个数,
   47-52列为全部序的CAD计算时间(所有序按照字典序升序排列, (x1,x2,x3),(x1,x3,x2),(x2,x1,x3),(x2,x3,x1),
   (x3,x1,x2),(x3,x2,x1)), 最后一列为数据随机化前在CF_total.txt的数据下标,根据此下标可寻找原始多项式系统

5. dataset_10s,dataset_10s_60s,dataset_60s_600s,dataset_600s_900s四个文件夹下的数据格式均相同,分别如下:
   (1) dataset:GF_totaldata.txt  前30列为多项式初始GF特征,31-36列为所有序的CAD胞腔个数,37-42列为CAD计算时间,
               CF_totaldata.txt  前40列为多项式初始CF特征,41-46列为所有序的CAD胞腔个数,47-52列为CAD计算时间,
                                 最后一列为数据随机前的下标
               OHF_original.txt  根据OHF特征去除特征相同的数据,前40列为多项式初始OHF特征,41-46列为所有序的CAD
                                 胞腔个数,47-52列为CAD计算时间,倒数第二列为数据随机前在CF_total.txt的下标,
                                 最后一列为重新随机前在CF_total.txt中的数据下标
               GF_original.txt   前30列为多项式初始GF特征,31-36列为所有序的CAD胞腔个数,37-42列为CAD计算时间,
                                 最后一列为数据随机前的下标
               CF_original.txt   根据CF特征去除特征相同的数据,前40列为多项式初始OHF特征,41-46列为所有序的CAD胞
                                 腔个数,47-52列为CAD计算时间,最后一列为数据随机前的下标
                                 
               *_train.txt       基于OHF/GF/CF特征的训练集,特征使用zscore方法归一化,最后六列为基于最短计算时间
                                 的数据标签
               *_valid.txt       基于OHF/GF/CF特征的验证集,特征使用zscore方法归一化,最后六列为基于最短计算时间
                                 的数据标签
               *_test.txt        基于OHF/GF/CF特征的测试集,特征使用zscore方法归一化,最后六列为基于最短计算时间
                                 的数据标签

   (2) *_predict:  基于OHF/GF/CF特征的测试集预测标签,其中:
                  train_predict.txt  训练集模型预测标签
                  valid_predict.txt  验证集模型预测标签
                  test_predict.txt   测试集模型预测标签
                                           
   (3) *_model:  基于OHF/GF/CF特征的训练模型
   
6. dataset_regression:   非线性回归网络训练与测试以及分级神经网络的测试
   (1) testdata_regreession.txt   分级神经网络中的测试集,前40列为OHF 特征,41-46列为CAD计算时间,最后一列为数据下标
   (2) predict_regression.txt     分级神经网络模型预测最优序标签,第一列为预测标签,第二列为数据下标
   (3) dataset: *_original.txt    初始训练/验证/测试集,前40列为OHF特征,41-46列为所有序的CAD胞腔个数,47-52列为
                                  CAD计算时间,倒数第二列为数据随机前totaldata_CF.txt的下标,最后一列为重新随机前
                                  在CF_total.txt中的数据下标
                *_zscore.txt      训练/验证/测试集,前40列为归一化之后的OHF 特征,最后一列为最长CAD计算时间
   (4) OHF_predict_regression：   *_predict   回归网络模型的训练/验证/测试集的预测计算时间
   (5) OHF_regression_model       回归网络的训练模型
                       
   
7. dataset_totaldata:  使用全部数据集进行训练与测试(M-OHF-totaldata)
   (1) dataset: total_original.txt 全部初始数据集,前40列为OHF特征,41-46列为所有序的CAD胞腔个数,47-52列为
                                   CAD计算时间,倒数第二列为数据随机前totaldata_CF.txt的下标,最后一列为重新随机前
                                   在CF_total.txt中的数据下标
                *_original.txt     初始训练/验证/测试集,前40列为OHF特征,41-46列为所有序的CAD胞腔个数,47-52列为
                                   CAD计算时间,倒数第二列为数据随机前totaldata_CF.txt的下标
                *_zscore.txt       训练/验证/测试集,前40列为归一化之后的OHF 特征,最后六列为为基于最短计算时间的数据标签

   (2) *_predict:  *_predict       回归网络模型的训练/验证/测试集的预测最优序标签
                                           
   (3) *_model:  训练模型

