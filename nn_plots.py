import matplotlib.pyplot as plt
import numpy as np

global figure_location
figure_location = "plots"

def plotting(epochs, costs, train_acc, test_acc, lr):
    epoch_count = int(np.mean(epochs[-1]))

    fig = plt.figure()
    ax1 = plt.axes()

    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor = 'tab:red')

    ax1.plot(epochs, costs)
    ax1.set_xticks(range(0, epoch_count+int(epoch_count/10), int(epoch_count/10)))
    ax1.set_ylabel('cost')
    ax1.set_xlabel('epochs')
    ax1.set_title("Learning rate = " + str(lr))

    ax2.plot(epochs, train_acc, "r", linestyle="-")
    ax2.plot(epochs, test_acc, "r", linestyle=":")
    ax2.set_ylabel('Mean absolute error')

    plt.savefig(figure_location+"/cost_over_epochs.png", dpi=300, bbox_inches='tight')