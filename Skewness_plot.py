import numpy as np 
import matplotlib.pyplot as plt 
import argparse


parser = argparse.ArgumentParser(description='Setting Parameters')
parser.add_argument('--y', default= 30, type=int, help='Wall normal distance')
parser.add_argument('--o', default= 3, type=int, help='Order to plot')
args = parser.parse_args()
fig_dir = "fig/23-3-14/"

y_plus = args.y

DNS = []
FCN = []
CBAM = []
VIT = []
CNN = []
PR  = ["0025","02","071","1"]

for pr in PR:
    data_dir = f"pred/y{y_plus}_pr{pr}_skew_5.npz"
    print(f"Load data from {data_dir}")
    skewness = np.load(data_dir)
    vit = skewness["vit"]
    cbam = skewness["cbam"]
    fcn = skewness["fcn"]
    cnn = skewness["cnn"]
    dns = skewness["dns"]
    
    DNS.append(dns)
    FCN.append(fcn)
    VIT.append(vit)
    CBAM.append(cbam)
    CNN.append(cnn)

DNS = np.array(DNS)
FCN = np.array(FCN)
VIT = np.array(VIT)
CBAM = np.array(CBAM)
CNN = np.array(CNN)
print(DNS.shape,FCN.shape,VIT.shape,CBAM.shape)


pr_label = [0.025,0.2,0.71,1]
order = args.o
plt.figure(order)
plt.plot(pr_label,DNS[:,order-3],"o-",c="red",lw = 1.5,markersize=10,label="DNS")
plt.plot(pr_label,VIT[:,order-3],"s-",c="orange",lw = 1.5,markersize=10,label="VIT")
plt.plot(pr_label,FCN[:,order-3],"x-",c="blue",lw = 1.5,markersize=10,label="FCN")
plt.plot(pr_label,CBAM[:,order-3],"v-",c="green",lw = 1.5,markersize=10,label="CBAM")
plt.plot(pr_label,CNN[:,order-3],"+-",c="cyan",lw = 1.5,markersize=10,label="Simple FCN")
# plt.yscale("log")


plt.xticks(pr_label)
plt.xlabel("Pr",fontdict={"fontsize":16})

if order == 4:
    plt.ylabel("$<(q_w'/q_{w,rms})^{4}>$",fontdict={"fontsize":16})
    plt.title("Flatteness",fontdict={"size":18,"weight":"bold"})
    plt.yticks(ticks=[0,2,4,6,8,10],labels=[0,2,4,6,8,10])
if order ==3:
    plt.ylabel("$<(q_w'/q_{w,rms})^{3}>$",fontdict={"fontsize":16})
    plt.title("Skewness",fontdict={"size":18,"weight":"bold"})
    plt.yticks(ticks=[0,2],labels=[0,2])

plt.legend()
plt.savefig(fig_dir+f"y{y_plus}_prall_order{order}",dpi=150,bbox_inches="tight")



