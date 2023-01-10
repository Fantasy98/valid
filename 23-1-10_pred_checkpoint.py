torch.manual_seed(1024)
device = ("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

model_path = "/storage3/yuning/thesis/models/23-1-10/epoch50.pt"
model = torch.load(model_path)
model.to(device)
print(model)
# model.eval()
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
save_types= ["train","test","validation"]
root_path = "/storage3/yuning/thesis/tensor/"
test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
print(test_path)

test_dl = DataLoader(torch.load(test_path+"/test.pt"),batch_size=1,shuffle=True)

RMSErrors = []
GlobErrors = []
with torch.no_grad():
    for batch in test_dl:
        x,y = batch
        x = x.float().to(device); y = y.float().squeeze().numpy()
        pred = model(x)
        pred = pred.float().squeeze().cpu().numpy()

        rms_error = RMS_error(pred,y)
        glb_error = Glob_error(pred,y)
        fluct_error = Fluct_error(pred,y)
        print(rms_error)
        print(glb_error)
        print(fluct_error)
        RMSErrors.append(rms_error)
        GlobErrors.append(glb_error)
        break

# Plot_2D_snapshots(pred,"/storage3/yuning/thesis/fig/23-1-10/pred50_test")
# Plot_2D_snapshots(y,"/storage3/yuning/thesis/fig/23-1-10/target50_test")
# Plot_2D_snapshots((pred-y)/y,"/storage3/yuning/thesis/fig/23-1-10/error50_test")
