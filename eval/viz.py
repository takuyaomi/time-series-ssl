import numpy as np
import matplotlib.pyplot as plt

def plot_attention_over_time(attn: np.ndarray, save_path: str | None = None):
    # attn: [P] or [P,1]
    a = attn.squeeze()
    plt.figure()
    plt.plot(a)
    plt.title('MIL Attention over Patches')
    plt.xlabel('Patch index')
    plt.ylabel('Attention weight')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()