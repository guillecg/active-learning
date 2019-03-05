import matplotlib.pyplot as plt

def plot_comparison(scores):
    iter_num = list(scores.index + 1)

    plt.figure(figsize=(9, 5))

    # Plot Distance vs number of clusters
    for idx, column in enumerate(scores.columns):
        plt.plot(iter_num, scores[column], label=column)
    plt.xlabel('Iteration number')
    plt.ylabel('Accuracy') # TODO: change score
    plt.legend(bbox_to_anchor=(1, 1), shadow=True, fontsize='x-large')

    plt.show()
