import numpy as np
import scipy.io as scio
import classifier as cf

def loaddata():
  dataFile = 'C:\\Users\\LUC\\Desktop\\2018_MCM-ICM_Problems\\环境\\drive\\'
  climate_mat = scio.loadmat(dataFile+'pickdata.mat')['pick']
  fragile = scio.loadmat(dataFile+'pickFragile')['pickFragile']
  return climate_mat, fragile

def cal_corrcoef_mat(climate_mat, fragile_mat):
  ''' climate_mat is 178*5 matrix, fragile_mat is 178*5'''
  m = np.shape(climate_mat)[1]
  n = np.shape(fragile_mat)[1]
  corr_mat = np.zeros((m, n))
  for i in range(m):
    for j in range(n):
      corr_mat[i][j] = np.corrcoef(climate_mat[:,i], fragile_mat[:, j])[0][1]
  return corr_mat

climate_mat, fragile_mat = loaddata()

w = np.mat(cf.getW(0.85))
w = np.delete(w, -1)
print(w)
corr = cal_corrcoef_mat(climate_mat, fragile_mat)
print(corr)
alter_w = corr *w.transpose()
print(alter_w)