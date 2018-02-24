import numpy as np
import scipy.io as scio
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt

def sigmoid(inX):
  return expit(inX)

def loaddata(year):
  ''' get train data form .mat file'''
  dataFile = 'C:\\Users\\LUC\\Desktop\\2018_MCM-ICM_Problems\\环境\\drive\\%s.mat'% year
  data = scio.loadmat(dataFile)
  data = data[year]
  # print(np.shape(data))
  # >100 label is `fragile` >60 is `vulnerable` <60 is `stable`
  s = np.sum(data, axis=1)
  label = [""]*np.size(s)
  for i in range(np.size(s)):
    if s[i]<60:
      label[i]='stable'
    elif s[i]<100:
      label[i]='vunlerable'
    else:
      label[i] = 'fragile'
  data =  pd.DataFrame(data)
  data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  data = [np.append(example,1.0) for index, example in data.iterrows()]
  # print(label)
  return np.array(data), np.mat(label).T

def randGradDesc1(dataMatrix, classLabels, numIter=150):
    '''this is improvement for randGradDesc which can change alpha during ieration and
    use random datapoint to update weights
    classLabels require column vector, datamtrix require ndarray''' 
    m,n = np.shape(dataMatrix)
    weights = np.random.uniform(0, 1, n)
    # weights = np.ones(n)   # initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01    # apha decreases with iteration, does not 
            randIndex = int(np.random.uniform(0,len(dataIndex)))# go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


voteResult = [0,0,0]
categorylabels = ['stable','vulnerable','fragile']#类别标签

def getW(alpha):
  # this will get weight with logistic regression
  data, labels = loaddata('ind2017')
  trainNum = int(np.around(178*alpha))
  data = pd.DataFrame(data)
  train = data.sample(frac=alpha, replace=False)
  index = train.index.tolist()
  # print(index, len(index))
  # print(len(set(index)))
  train = np.array(train)
  trainStrlabels = labels[index]
  weight1 = []
  for i in range(3):#三类
    labelMat1 = []  
    for j in range(len(trainStrlabels)):#把名称变成0或1的数字  
        if trainStrlabels[j] == categorylabels[i]:  
            labelMat1.append(1)  
        else:
            labelMat1.append(0)
    labelMat1 = np.array(labelMat1).T
    weight1.append(randGradDesc1(train, labelMat1))
  return np.array(weight1)

def test(alpha, weight1):
  # this will classify a feature vector into three kind and tell
  # the accuracy rate of logistic regression trained weight
  data, labels = loaddata('ind2017')
  trainNum = int(np.around(178*alpha))
  testData = data[trainNum+1:,:]
  testlabel = labels[trainNum+1:,:]
  h = [0]*len(testData)
  for j in range(len(testData)):
    voteResult = [0,0,0]
    for i in range(3):
        h[j] = float(sigmoid(np.ndarray.dot(testData[j],weight1[i]))) #得到训练结果
        if h[j] > 0.5 and h[j] <= 1:
            voteResult[i] = voteResult[i]+1+h[j] #由于类别少，为了防止同票，投票数要加上概率值  
        elif h[j] >= 0 and h[j] <= 0.5:
            voteResult[i] = voteResult[i]-1+h[j]  
        else: 
            print('Properbility wrong!')
    h[j] = voteResult.index(max(voteResult))
  print(h)
  labelMat2 = []
  for j in range(len(testlabel)):#把名称变成0或1或2的数字  
    for i in range(3):#三类  
        if np.array(testlabel)[j][0] == categorylabels[i]:  
            labelMat2.append(i);break
  #计算正确率
  error = 0.0
  for j in range(len(testlabel)):
      if h[j] != labelMat2[j]:
          error = error +1
  
  pro = 1 - error / len(testlabel)#正确率
  print(pro)

weight = getW(0.8)
print(weight)
test(0.8, weight)
