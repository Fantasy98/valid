from utils.datas import TF2Torch
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_type= "train"
root_path = "/home/yuning/thesis/valid/tensor/"
TF2Torch(root_path,y_plus,var,target,save_type,normalized=False)