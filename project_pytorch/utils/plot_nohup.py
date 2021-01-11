import matplotlib.pyplot as plt
import re
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
plt.style.use("seaborn-paper")


f_train = open("p4_nohup_train.out", "r")
f_val = open("p4_nohup_val.out", "r")

f_train_contents = f_train.readlines()
f_val_contents = f_val.readlines()


# for train
# train_x = []
# train_acc1 = []
# train_acc5 = []
# train_loss = []

# for i, line in enumerate(f_train_contents):

#     l = float(re.findall(r"Loss (.+?) ", line)[0])
#     p1 = float(re.findall(r"Prec@1 (.+?) ", line)[0])
#     p5 = float(re.findall(r"Prec@5 (.+?) ", line)[0])

#     train_x.append(i)
#     train_loss.append(l)
#     train_acc1.append(p1)
#     train_acc5.append(p5)


def plot_standard(x, y, x_legend: str, y_legend: str, title: str):
    plt.figure()
    x_smooth = np.linspace(np.array(x).min(), np.array(x).max(), 1000)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    plt.plot(x, y, color='red', label='Original')
    plt.plot(x_smooth, y_smooth, color='blue', label='Fitting')
    plt.title(title, fontsize=16)
    plt.xlabel(x_legend, fontsize=16)
    plt.ylabel(y_legend, fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    plt.clf()


def plot_val(x, y, x_legend: str, y_legend: str, title: str):
    plt.figure()
    x_smooth = np.linspace(np.array(x).min(), np.array(x).max(), 10000)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    # plt.plot(x, y, color='red', label='Original')
    plt.plot(x_smooth, y_smooth, color='blue', label='Fitting')
    plt.title(title, fontsize=16)
    plt.xlabel(x_legend, fontsize=16)
    plt.ylabel(y_legend, fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    plt.clf()


# plot_standard(train_x, train_loss, x_legend="Steps", y_legend="Loss", title="O3: Loss of Training Process")
# plot_standard(train_x, train_acc1, x_legend="Steps", y_legend="Acc1", title="O3: Top1 Accuracy of Training Process")
# plot_standard(train_x, train_acc5, x_legend="Steps", y_legend="Acc5", title="O3: Top5 Accuracy of Training Process")


# for val
val_x = []
val_acc1 = []
val_acc5 = []
val_loss = []

for i, line in enumerate(f_val_contents):

    l = float(re.findall(r"Loss (.+?) ", line)[0])
    p1 = float(re.findall(r"Prec@1 (.+?) ", line)[0])
    p5 = float(re.findall(r"Prec@5 (.+?) ", line)[0])

    val_x.append(i)
    val_loss.append(l)
    val_acc1.append(p1)
    val_acc5.append(p5)


plot_val(val_x, val_loss, x_legend="Steps", y_legend="Loss", title="O2: Loss of Validation Process")
# plot_val(val_x, val_acc1, x_legend="Steps", y_legend="Acc1", title="O2: Top1 Accuracy of Validation Process")
# plot_val(val_x, val_acc5, x_legend="Steps", y_legend="Acc5", title="O2: Top5 Accuracy of Validation Process")
