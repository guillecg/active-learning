import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style='darkgrid')

def plot_comparison(scores):
    iter_num = list(scores.index + 1)

    plt.figure(figsize=(9, 5))
    plt.xlabel('Iteration number')
    plt.ylabel('Accuracy') # TODO: change score

    # Use 'dashes' argument to avoid error while plotting more than 6 columns
    ax = sns.lineplot(data=scores, palette='viridis', dashes=False)

    # Locate legend
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height])
    # ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # Add maximum score line
    ax.axhline(y=1.0, xmin=0.0, xmax=max(iter_num), color='r')

    plt.show()
