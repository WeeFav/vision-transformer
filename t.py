import matplotlib.pyplot as plt

train_loss_list = [0] * 10
test_loss_list = [1] * 10
train_accuracy_list = [2] * 100
test_accuracy_list = [3] * 100


for i in range(10)
    # graph loss
    plt.figure(1)
    plt.legend('', frameon=False)
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.legend()
    plt.savefig("./loss")

    # graph accuracy
    plt.figure(2)
    plt.legend('', frameon=False)
    plt.plot(train_accuracy_list, label="train accuracy")
    plt.plot(test_accuracy_list, label="test accuracy")
    plt.legend()
    plt.savefig("./accuracy")

    plt.close(1)
    plt.close(2)
