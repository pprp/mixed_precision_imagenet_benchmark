import os
import csv
import matplotlib.pyplot as plt
import numpy as np

path1 = r"2021-1-25-可视化\tmp1_exp1\run-.-tag-train_batch_acc1.csv"
path2 = r"2021-1-25-可视化\run-Dec26_21-04-30_DESKTOP-U86PSQQ-tag-train_batch_acc1.csv"


def get_values(path):
    values = []
    with open(path) as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            else:
                a, b, c = row
                c = float(c)
                values.append(c)
    return values


values1 = get_values(path1)
values2 = get_values(path2)

assert len(values1) == len(values2)

print(len(values1))

# draw
x = np.arange(0, len(values1))
plt.plot(x, values1)
plt.plot(x, values2)
plt.show()

values1 = np.array(values1)
values2 = np.array(values2)

np.save("exp1.npy", values1)
np.save("exp2.npy", values2)
