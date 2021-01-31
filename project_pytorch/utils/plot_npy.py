import numpy as np 


path1 = 'exp1.npy'
path2 = 'exp2.npy'
path3 = 'exp3.npy'
path4 = 'exp4.npy'

exp1 = np.load(path1)
exp2 = np.load(path2)
exp3 = np.load(path3)
exp4 = np.load(path4)

print(len(exp1),len(exp2),len(exp3),len(exp4))

exp3_idx = np.random.choice(np.arange(len(exp3)),size=1000)
exp3_idx = np.sort(exp3_idx)
exp3 = exp3[exp3_idx]


exp4_idx = np.random.choice(np.arange(len(exp4)),size=1000)
exp4_idx = np.sort(exp4_idx)
exp4 = exp4[exp4_idx]


# print(len(exp3))

# exp3 = np.random.choice(exp3, size=1000)
# exp4 = np.random.choice(exp4, size=1000)

import matplotlib.pyplot as plt 

x = np.arange(1,1001)

plt.plot(x,exp1,color='r', label='O0')
plt.plot(x,exp2,color='b', label='O1')
plt.plot(x,exp3,color='y', label='O2')
plt.plot(x,exp4,color='g', label='O3')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('top1 acc')
plt.title('Comparision between different level setting of mixed precision')
plt.show()