import matplotlib.pyplot as plt


def training(train_acc, valid_acc):
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(valid_acc, label="Validation Accuracy")
    plt.legend(frameon=False)
    plt.savefig("results.png")