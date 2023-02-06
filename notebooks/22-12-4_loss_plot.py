import os
import numpy as np 
import matplotlib.pyplot as plt
np_path = "/home/yuning/thesis/models/trained/fresh-glitter-54"
np_file = os.path.join(np_path,"loss.out") 
loss_array = np.loadtxt(np_file)
print("Shape of loss array is {}".format(loss_array.shape))
plt.figure(figsize=(20,15))
plt.semilogy(loss_array[:,0],"r",lw = 5,label="Train Loss")
plt.semilogy(loss_array[:,1],"b",lw = 5,label="Val Loss")
plt.xlabel("Epochs",fontdict={"size":20})
plt.ylabel("RMSE",fontdict={"size":20})
plt.grid()
plt.legend(fontsize =20)
plt.savefig("/home/yuning/thesis/valid/fig/fresh-glitter-54/loss",bbox_inches="tight")
print("Plot Saved")