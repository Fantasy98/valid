import torch 
import os
from torch import nn 
from transformer.FCN_CCT import FullSkip_FCN_CCT, FullSkip_Mul_FCN_CCT
from NNs import HeatFormer_mut
from utils.networks import FCN_Pad_Xaiver_gain, FCN_4
from utils.newnets import FCN_Pad_Xaiver_CBAM2
from utils.datas import slice_dir,JointDataset
from utils.plots import Plot_2D_snapshots
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.metrics import RMS_error,Glob_error,Fluct_error,PCC
from scipy import stats
import matplotlib.pyplot as plt 
from utils.plots import Plot_2D_snapshots,Plot_multi

device = ("cuda" if torch.cuda.is_available() else "cpu")

model_name = "vit_mul_16h_4l"
Prs = [0.025,0.2,0.71,1]
prs = ["0025","02","071","1"]

Num_heads = 16
Num_layers = 4

for Pr, pr in zip(Prs,prs):
    var=['u_vel',"v_vel","w_vel",f"pr{Pr}"]
    target=[f'pr{Pr}_flux']
    normalized=False
    y_plus=30


    save_types= ["train","test","validation"]
    root_path = "/home/yuning/thesis/tensor"
    test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
    print(test_path)

    if Pr == 0.025:
        test_data1 = torch.load(test_path+"/test1.pt")
        test_data2 = torch.load(test_path+"/test2.pt")


        test_x = torch.cat([test_data1.x,test_data2.x])
        test_y = torch.cat([test_data1.y,test_data2.y])
        test_ds = JointDataset(test_x,test_y)

    else:
        test_ds = torch.load(test_path+"/test.pt")

    test_dl = DataLoader(test_ds,shuffle=False,batch_size=1)



    # model = HeatFormer_mut(num_heads=Num_heads,num_layers=Num_layers)
    # model = FullSkip_FCN_CCT(num_heads=Num_heads,num_layers=Num_layers,dropout=0.1)
    # model = FCN_Pad_Xaiver_CBAM2(256,256,4,3,8)
    # model = FCN_Pad_Xaiver_gain(256,256,4,3,8)
    # model = FCN_4(256,256,4,3,8)
    model = FullSkip_Mul_FCN_CCT(num_heads=Num_heads,num_layers=Num_layers)
    # checkpoint_path = "/home/yuning/thesis/valid/models/y_plus_30-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_Skip_CN_FCN_CCT_16h_4l_EPOCH=100.pt"
    # checkpoint_path = "/home/yuning/thesis/valid/models/y_plus_15-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_Skip_FCN_CCT_16h_4l_EPOCH=100.pt"
    # checkpoint_path = "/home/yuning/thesis/valid/models/y_plus_15-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_FCN_baseline_EPOCH=100.pt"
    # checkpoint_path = "/home/yuning/thesis/valid/models/y_plus_15-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_FCN_4_EPOCH=100.pt"
    # checkpoint_path = f"/home/yuning/thesis/valid/models/y_plus_{y_plus}-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_Skip_FCN_CCT_{Num_heads}h_{Num_layers}l_EPOCH=100.pt"
    checkpoint_path = f"/home/yuning/thesis/valid/models/y_plus_{y_plus}-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_Skip_Mul_FCN_CCT_{Num_heads}h_{Num_layers}l_EPOCH=100.pt"
    # checkpoint_path = f"/home/yuning/thesis/valid/models/y_plus_{y_plus}-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_FCN_CBAM_EPOCH=100.pt"
    # checkpoint_path = "/home/yuning/thesis/valid/models/y_plus_30-VARS-prall_u_vel_v_vel_w_vel-TARGETS-prall_flux_FCN_4_EPOCH=100.pt"
    checkpoint = torch.load(checkpoint_path)

    print(checkpoint.keys())
    model_state = checkpoint["model"]
    loss = checkpoint["loss"]
    val_loss = checkpoint["val_loss"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.eval()
    # print(model.eval())
    # tokenizer_model = model.CCTEncoder
    # tokenizer_model.to(device)
    model.to(device)

    # from utils.networks import FCN_Pad_Xaiver_gain
    # model_name = "y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_EPOCH=100_state_dict"
    # model = FCN_Pad_Xaiver_gain(256,256,4,3,8)

    # model_state = torch.load("/home/yuning/thesis/valid/models/{}.pt".format(model_name))
    # model.load_state_dict(model_state)
    # model.cuda()
    # model.eval()
    # print(f"INFO the model state dict has been loaded and to device {device}")


    Y = []; Pred =[]; Tokens = []
    print("INFO: begin to predict")
    i = 0
    for batch in tqdm(test_dl):
        x,y = batch
        x = x.cuda().float()
        y = y.cuda().double()

        with torch.no_grad():
            pred = model(x).double()
            # token = tokenizer_model(x).double()
        
        Pred.append(pred.detach().cpu().numpy().squeeze())
        # Tokens.append(token.detach().cpu().numpy().squeeze())
        Y.append(y.detach().cpu().numpy().squeeze())
        i+= 1
        
        

    Pred = np.array(Pred)
    Tokens = np.array(Tokens)
    Y = np.array(Y)

    print(f"INFO: predict finish, the prediction data has size{Pred.shape}")
    # print(f"INFO: predict finish, the tokens output data has size{Tokens.shape}")
    print(f"INFO: predict finish, the ground truth data has size{Y.shape}")

    np.savez_compressed("pred/"+f"y{y_plus}_all-pr{pr}_{model_name}.npz",
                        pred = Pred,
                        # tokens = Tokens,
                        y = Y)

    print("INFO: The prediction data has been saved")
    RMS_all = []
    Glob_all = []
    Flcut_all = []
    PCC_all = []

    for i in tqdm(range(len(test_dl))):
        RMS  = RMS_error(Pred[i,:,:],Y[i,:,:])
        RMS_all.append(RMS)

        Glob_Error  = Glob_error(Pred[i,:,:],Y[i,:,:])
        Glob_all.append(Glob_Error)

        Fluct  = Fluct_error(Pred[i,:,:],Y[i,:,:])
        Flcut_all.append(Fluct)
        pcc = PCC(Pred[i,:,:],Y[i,:,:])
        PCC_all.append(pcc)

    RMS = np.array(RMS_all).mean()
    Glob_Error = np.array(Glob_all).mean()
    Fluct = np.array(Flcut_all).mean()
    pcc = np.array(PCC_all).mean()


    # RMS  = RMS_error(Pred[0,:,:].mean(0),Y[:407,:,:].mean(0))
    # Glob_Error  = Glob_error(Pred.mean(0),Y.mean(0))
    # Fluct  = Fluct_error(Pred.mean(0),Y.mean(0))
    # pcc = PCC(Pred[:407,:,:].mean(0),Y[:407,:,:].mean(0))
    print(f"RMS {RMS}")
    print(f"Glob {Glob_Error}")
    print(f"Fluct {Fluct}")
    print(f"pcc {pcc}")



