import matplotlib.pyplot as plt

def plot(epoch, train, label = "Accuracy"):
    plt.figure()
    plt.plot(epoch, train, label="Training")
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.show()