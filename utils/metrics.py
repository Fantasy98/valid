import numpy as np

def RMS_error(y_pred,y_true):
  pred = np.mean(y_pred,0)
  true = np.mean(y_true,0)
  error = 100*np.sqrt( np.mean( (pred-true)**2 ) )/np.mean(true)
  return error

def Glob_error(y_pred,y_true):
  pred = np.mean(y_pred,0)
  true = np.mean(y_true,0)
  
  return 100*(np.mean(pred)-np.mean(true))/np.mean(true)
