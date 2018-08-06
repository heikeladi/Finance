# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 08:43:04 2018

@author: heiser
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import tushare as ts









############
industr_list=ts.get_industry_classified()
industry=np.array(industr_list)
for i in range(len(industry[:,2])-1):
    if(industry[i+1,2]!=industry[i,2]):
        print(i,industry[i,2])


print(industry[1893,2])
print(industry[1843:1894,0])##金融行业

print(industry[1720:1841,2])








###################
def getData(id,start,end,num,flag):
    df = ts.get_hist_data(id,start,end)
    if(not isinstance(df,pd.core.frame.DataFrame)):
        return 0
    #df = (df-np.sum(df)/len(df))/(np.std(df))
    if(flag=="true"):
        df = df[1:num]
    else:
        df = df[:num]
    df1 = np.array(df)
    #df2 = np.array(df.index)
    
    ##df = df.T
    x = []
    for i in range(len(df1)):
        #temp = np.append(df2[i],df1[i])
        temp = df1[i]
        newresult = []
        for item in temp:
            newresult.append(item)
        x.append(newresult)
    x.reverse()
    return x


def getlatestData(id,start,end,num):
    df = ts.get_hist_data(id,start,end)
    if(not isinstance(df,pd.core.frame.DataFrame)):
        return 0
    #df = (df-np.sum(df)/len(df))/(np.std(df))
    df = df[-num:]
    df1 = np.array(df)
    #df2 = np.array(df.index)
    
    ##df = df.T
    x = []
    for i in range(len(df1)):
        #temp = np.append(df2[i],df1[i])
        temp = df1[i]
        newresult = []
        for item in temp:
            newresult.append(item)
        x.append(newresult)
    x.reverse()
    return x


def getDataR(id,start,end,num):
    df = ts.get_hist_data(id,start,end)
    if(not isinstance(df,pd.core.frame.DataFrame)):
        return 0
    df1 = np.array(df)
    x = []
    for i in range(len(df1)):
        temp = df1[i]
        newresult = []
        for item in temp:
            newresult.append(item)
        x.append(newresult)
    
    P=df['close']
    #ʵ������û��end��һ������ݣ�������Ԥ��δ��һ����������ڵ����̼�
    templist=(P.shift(1)-P)/P
    templist = templist[:num]
    templist = np.array(templist)
    templist = templist.tolist()
    templist.reverse()
    tempDATA = []
    for i in range(len(templist)):
        if((i+1)%10!=0):
            pass
        else:
            if(templist[i]>=0.01):
                #tempDATA.append(templist[i])
                tempDATA.append([1,0,0,0])
            elif(templist[i]<0.01 and templist[i]>=0):
                #tempDATA.append(templist[i])
                tempDATA.append([0,1,0,0])
            elif(templist[i]<0 and templist[i]>=-0.01):
                #tempDATA.append(templist[i])
                tempDATA.append([0,0,1,0])
            else:
                #tempDATA.append(templist[i])
                tempDATA.append([0,0,0,1])
            
    y=tempDATA
    return y

#df_sh = ts.get_sz50s()['code']
df_sh =["600848","600016"]
df_sh=industry[1843:1879,0]
df_sh=industry[1720:1841,0]
df_sh=industry[:,0]
enum=0
num=0
fac = []
ret = []
facT = []
retT = []
predFAC = []
for ishare in df_sh:
    print(enum,num)##统计有效股票数
    num+=1
    #ȡ�����260������
    print(ishare)
    newfac = getData(ishare,'2015-08-01','2017-12-01',501,"true")
    newret = getDataR(ishare,'2015-08-01','2017-12-01',501)
    if(newfac == 0 or newret == 0 ):
        continue
    #fac.append(newfac)
    if(len(newfac)==500 and len(newret)==50):
        enum+=1
        print(len(newfac))
        print(len(newret))
        for i in range(len(newfac)):
            fac.append(newfac[i])
        for i in range(len(newret)):
            ret.append(newret[i])
    
    newfacT = getData(ishare,'2017-12-02','2018-05-31',101,"true")
    newretT = getDataR(ishare,'2017-12-02','2018-05-31',101)
    #fac.append(newfac)
    if(newfacT == 0 or newretT == 0 ):
        continue
    if(len(newfacT)==100 and len(newretT)==10):
        for i in range(len(newfacT)):
            facT.append(newfacT[i])
        for i in range(len(newretT)):
            retT.append(newretT[i])
    
    newpredFAC = getData(ishare,'2018-06-02','2018-08-05',10,"false")
    if(newpredFAC == 0):
        continue
    for i in range(len(newpredFAC)):
        predFAC.append(newpredFAC[i])

fac = np.array(fac)
ret = np.array(ret)
meanfac = np.sum(fac, axis=0)/len(fac)
stdfac = np.std(fac, axis=0)
fac = (fac-meanfac)/stdfac

facT = np.array(facT)
retT = np.array(retT)
facT = (facT-meanfac)/stdfac


newf = []
newfa = []
for i in range(len(fac)):
    if((i+1)%10!=0):
        newf.append(fac[i])
    else:
        newf.append(fac[i])
        newfa.append(newf)
        newf = []
fac = np.array(newfa)
newfT = []
newfaT = []
for i in range(len(facT)):
    if((i+1)%10!=0):
        newfT.append(facT[i])
    else:
        newfT.append(facT[i])
        newfaT.append(newfT)
        newfT = []
facT = np.array(newfaT)




predFAC = (predFAC-meanfac)/stdfac



######## count distribution
num1=0
num2=0
num3=0
num4=0
for i in range(ret.shape[0]):
    if(ret[i][0]==1):
        num1+=1
    elif(ret[i][1]==1):
        num2+=1
    elif(ret[i][2]==1):
        num3+=1
    else:
        num4+=1
        
print(num1,num2,num3,num4)        





###########

learning_rate = 0.001
batch_size = 10
print(int(fac.shape[0]))
training_iters = int(fac.shape[0]/batch_size)
display_step = 10

# Network Parameters
n_input = 13
n_steps = 10
n_hidden = 1024
n_classes = 4
dropout = 0.8
# tf Graph input
x = tf.placeholder('float',[None, n_steps, n_input])
y = tf.placeholder('float',[None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# �������
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

# ����²�������
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# ��һ������
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# ������������ 
def alex_net(_X, _weights, _biases, _dropout):
    # ����תΪ����
    _X = tf.reshape(_X, shape=[-1, 10, 13, 1])

    # �����
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # �²�����
    pool1 = max_pool('pool1', conv1, k=2)
    # ��һ����
    norm1 = norm('norm1', pool1, lsize=4)
    # Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # ���
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # �²���
    pool2 = max_pool('pool2', conv2, k=2)
    # ��һ��
    norm2 = norm('norm2', pool2, lsize=4)
    # Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # ���
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # �²���
    pool3 = max_pool('pool3', conv3, k=2)
    # ��һ��
    norm3 = norm('norm3', pool3, lsize=4)
    # Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    # ȫ���Ӳ㣬�Ȱ�����ͼתΪ����
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') 
    # ȫ���Ӳ�
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # ���������
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

# �洢���е��������
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([1024, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# ����ģ��
pred = alex_net(x, weights, biases, keep_prob)

# ������ʧ������ѧϰ����
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

result = tf.argmax(pred,1)

# ��������
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# ��ʼ�����еĹ������
init = tf.global_variables_initializer()

# ����һ��ѵ��
with tf.Session() as sess:
    sess.run(init)
    for tr in range(500):
    #for tr in range(3):
        for i in range(int(len(fac)/batch_size)):
            batch_x = fac[i*batch_size:(i+1)*batch_size].reshape([batch_size,n_steps,n_input])
            batch_y = ret[i*batch_size:(i+1)*batch_size].reshape([batch_size,n_classes])
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
            if(i%50==0):
                print(i,'----',(int(len(fac)/batch_size)))
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y, keep_prob:0.8})
        print("Iter " + str(tr*batch_size) + ", Minibatch Loss= " +"{:.26f}".format(loss) + ", Training Accuracy= " +"{:.26f}".format(acc))
    print("Optimization Finished!") 
    # ������Ծ���
    print("Accuracy in data set")
    #test_data = fac[:batch_size].reshape([batch_size,n_steps,n_input])
    #test_label = ret[:batch_size].reshape([batch_size,n_classes])
    test_data = fac
    test_label = ret
    loss, acc = sess.run([cost, accuracy], feed_dict={x: test_data,y: test_label, keep_prob:1.})
    print("Accuracy= " +"{:.26f}".format(acc))
    
    print("Accuracy out of data set")
    #test_dataT = facT[:len(facT)].reshape([len(facT),n_steps,n_input])
    #test_labelT = retT[:len(facT)].reshape([len(facT),n_classes])
    test_dataT = facT
    test_labelT = retT
    loss, acc = sess.run([cost, accuracy], feed_dict={x: test_dataT,y: test_labelT, keep_prob:1.})
    print("Accuracy= " +"{:.26f}".format(acc))
    
    #pred_dataT = predFAC[:batch_size].reshape([1,n_steps,n_input])
    pred_dataT = predFAC.reshape([-1,n_steps,n_input])
    pred_lable = sess.run([result],feed_dict={x: pred_dataT, keep_prob:1.})
    list_lable = pred_lable[0][0]
    print("list",pred_lable)
    maxindex = np.argmax(list_lable)
    #print("Predict_label is " + str(pred_lable[0][0]))
    if(maxindex==0):
        print("up")
    else:
        print("down")
    sess.close()
