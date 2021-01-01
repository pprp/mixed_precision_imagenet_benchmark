import os
import csv
import matplotlib.pyplot as plt 
import numpy as np

path = r"C:\Users\nudtdpj\Desktop\run-.-tag-train_batch_acc1.csv"

values = []

with open(path) as f:
    csv_reader = csv.reader(f)
    for i, row in enumerate(csv_reader):
        if i == 0:
            continue
        else:
            a,b,c = row 
            a,b,c = float(a),float(b),float(c)
            values.append(c)

# draw
x = np.arange(0,len(values))
plt.plot(x, values)
plt.show()
