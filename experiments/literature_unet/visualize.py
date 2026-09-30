import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize():
    os.makedirs('figures', exist_ok=True)
    for snr in [float('inf'), 20, 10, 5]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Prediction Examples - SNR {snr}')
        plt.savefig(f'figures/prediction_examples_snr{snr}.png')
        plt.close()
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('4-PLD vs 3-PLD')
    plt.savefig('figures/4pld_vs_3pld.png')
    plt.close()

if __name__ == '__main__':
    # visualize()
    pass
