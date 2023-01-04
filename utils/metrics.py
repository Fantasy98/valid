import numpy as np

def RMS_error(y_pred,y_true):
  if y_pred.shape == y_true.shape and len(y_pred.shape)==2:
    error = 100*np.sqrt( np.mean( (y_pred-y_true)**2 ) )/np.mean(y_true)
    # error = np.sqrt( np.mean( (y_pred-y_true)**2 ) )
    return error
  else:
    print("Expected shape of (256,256)")

def Glob_error(y_pred,y_true):
  if y_pred.shape == y_true.shape and len(y_pred.shape)==2:  
    return 100*(np.mean(y_pred)-np.mean(y_true))/np.mean(y_true)
  else:
    print("Expected shape of (256,256)")

def Fluct_error(y_pred,y_true):
  if y_pred.shape == y_true.shape and len(y_pred.shape)==2:  
    return 100*(np.std(y_pred-np.mean(y_pred))-np.std(y_true-np.mean(y_true)))/np.std(y_true-np.mean(y_true))
  else:
    print("Expected shape of (256,256)")