import torch 
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
y_plus = 30 
Num_heads = 16 
Num_layers = 4

fig_dir = "fig/23-2-20/"

vit_dir = f"/home/yuning/thesis/valid/models/y_plus_{y_plus}-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_Skip_FCN_CCT_{Num_heads}h_{Num_layers}l_EPOCH=100.pt"
cbam_dir = "/home/yuning/thesis/valid/models/y_plus_30-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_FCN_CBAM_EPOCH=100.pt"
cnn_dir = "/home/yuning/thesis/valid/models/y_plus_30-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_FCN_4_EPOCH=100.pt"
fcn_dir = "/home/yuning/thesis/valid/models/y_plus_30-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_FCN_baseline_EPOCH=100.pt"


vit_cp = torch.load(vit_dir)
cbam_cp = torch.load(cbam_dir)
fcn_cp = torch.load(fcn_dir)
cnn_cp = torch.load(cnn_dir)


checkpoints = [vit_cp,cbam_cp,fcn_cp,cnn_cp]

names= ["ViT","CBAM","FCN","Simple FCN"]
for i,cp in tqdm(enumerate(checkpoints)):
    plt.figure(i)
    plt.semilogy(cp["loss"],lw =2 , c = "b",label= "Train Loss")
    plt.semilogy(cp["val_loss"],lw =2 , c = "r",label= "Validation Loss")
    plt.grid()
    plt.xlabel("Epochs",fontdict={"size":16})
    plt.ylabel("Loss",fontdict={"size":16})
    cost = cp["time"]
    plt.title( names[i]+f" (time used {np.round(cost/3600)}h)",fontdict={"size":18})
    plt.tight_layout()
    plt.legend()
    plt.savefig(fig_dir+"Loss_"+names[i],dpi= 150)