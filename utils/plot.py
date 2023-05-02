import os
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('science')


def plot_results(dataset_name, model_name, Is, train_size, initials_mean_accs_all_Is,
                 final_mean_accs_all_Is, crossval_mean_accs_all_Is, output_file_name):
    x = np.array(Is)
    fig, ax = plt.subplots(figsize=(7, 5))

    final_max_index = np.argmax(final_mean_accs_all_Is)
    crossval_max_index = np.argmax(crossval_mean_accs_all_Is)

    ax.plot(x[final_max_index], round(final_mean_accs_all_Is[final_max_index], 7), 'ro-', label="")
    ax.plot(
        x[crossval_max_index],
        round(crossval_mean_accs_all_Is[crossval_max_index],
              7),
        'ro', label="maximum point")

    ax.plot(x, [round(x, 7) for x in final_mean_accs_all_Is], 'r--',
            label=f"$N:{train_size*100}\%$ " + "$E[\\text{acc}_{\\text{SRPM}}]$", alpha=0.5)
    ax.plot(x, [round(x, 7) for x in crossval_mean_accs_all_Is], 'r:',
            label=f"$N:{train_size*100}\%$ " + "$E[\\text{acc}_{\\text{STCV}}]$", alpha=0.5)
    ax.plot(x, [round(x, 7) for x in initials_mean_accs_all_Is], 'r-',
            label=f"$N:{train_size*100}\%$ " + "$E[\\text{acc}_{0}]$", alpha=0.5)

    ax.grid()

    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.06),
        bbox_transform=plt.gcf().transFigure,
        ncol=2,
        fontsize=15
    )

    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_title(f'dataset - {dataset_name}, model - {model_name}', fontsize=15)
    ax.set_ylabel('expected accuracy', fontsize=15)
    ax.set_xlabel('$i$', fontsize=15)

    fig.tight_layout(rect=(0.03, 0.02, 1, 1))
    fig.savefig(f'{output_file_name}', bbox_inches='tight', dpi=600)
