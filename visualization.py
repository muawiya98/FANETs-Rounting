
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def visualize_measures(performance_logger, last, id):
    measures = list(filter(lambda x: type(x[1]) == list, performance_logger.__dict__.items()))
    half_measures_num = (len(measures) + 1) // 2  # Use ceiling division to handle odd number of measures

    fig, axs = plt.subplots(half_measures_num, 2, figsize=(15, 5 * half_measures_num))
    fig.suptitle("Network Performance Measures", fontsize=16, weight='bold')

    # Set a style for better-looking plots
    sns.set_style("whitegrid")

    for idx, (measure_name, measure_data) in enumerate(measures):
        row = idx // 2
        col = idx % 2
        ax = axs[row, col] if half_measures_num > 1 else axs[col]

        sns.lineplot(data=measure_data, ax=ax, color='#A0CBE2')

        if measure_name == 'e2e_delay_list':
            title = 'End to End Delay'
            ylabel = 'Delay (ms)'
        elif measure_name == 'arrived_packets_list':
            title = 'Arrived Packets'
            ylabel = 'Number of Packets'
        elif measure_name == 'energy_consumption':
            title = 'Energy Consumption'
            ylabel = 'Energy (units)'
        elif measure_name == 'pdr':
            title = 'Packet Delivery Ratio (PDR)'
            ylabel = 'PDR'
        else:
            title = f"Dropped Packets: {measure_name.replace('dropped_packets', '').replace('_', ' ').strip()}"
            ylabel = 'Number of Packets'

        ax.set_title(title, fontsize=10, weight='bold')
        ax.set_xlabel('Time', fontsize=8, weight='bold')
        ax.set_ylabel(ylabel, fontsize=8, weight='bold')

        # Improve x-axis ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend([title], loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust to make room for the main title

    if not last:
        plt.savefig(f"measures_{id}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"measures_{id}.svg", bbox_inches='tight')
    else:
        plt.savefig(f"measures_final.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"measures_final.svg", bbox_inches='tight')

    # plt.show()
    plt.close(fig)