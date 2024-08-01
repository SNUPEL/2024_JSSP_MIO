import matplotlib.pyplot as plt
import numpy as np

def show_evolution(makespan, score, title, text=None, path=None):
    """
    n_division : 몇 개로 구분해 보여줄 것인가?
    n_generation : 한 구분 당 몇 개의 세대를 보여줄 것인가?
    """
    plt.figure(figsize=(12, 8))
    score = np.array(score)
    plt.scatter(makespan[:100], score[:100], color='blue', alpha=0.3, s=4)
    plt.scatter(makespan[100:], score[100:], color='red', alpha=0.1, s=4)
    correlation_matrix = np.corrcoef(makespan, score)
    pearson_corr = correlation_matrix[0, 1]
    plt.ylabel('Spearman Footrule Distance')
    plt.xlabel('makespan')
    plt.title('Pearson Correlation :' + str(round(pearson_corr, 4)))
    plt.suptitle(title)
    if path is not None:
        plt.savefig(path + title+'.png', dpi=200)
    else:
        plt.savefig(title+'.png', dpi=200)

    # plt.show()