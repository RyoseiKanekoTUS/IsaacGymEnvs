import torch
import numpy as np
import matplotlib.pyplot as plt


class CNNVisualizer():

    def __init__(self, path):

        self.path = path
        self.model = torch.load(self.path)

        self.policy = self.model['policy']
        self.get_layers()

    def get_layers(self):
        
        self.conv_1 = self.policy['d_feture_extractor.0.weight'].cpu()
        self.conv_2 = self.policy['d_feture_extractor.3.weight'].cpu()
        self.conv_3 = self.policy['d_feture_extractor.6.weight'].cpu()

    def visualize(self, layer):

        layer = np.array(layer) # shape(8, 1, 9, 9)
        fig, ax = plt.subplots(layer.shape[1], layer.shape[0])
        print(len(ax.shape))

        for input_ch in range(layer.shape[1]):
            for output_ch in range(layer.shape[0]):

                array = layer[output_ch, input_ch, :,:].reshape(layer.shape[2], layer.shape[3])
                plt.subplot(layer.shape[1], layer.shape[0], (1 + input_ch + output_ch*input_ch))
                if len(ax.shape) == 2:
                    ax[input_ch, output_ch].imshow(array, cmap='gray')
                else:
                    ax[output_ch].imshow(array, cmap='gray')
        plt.show()


if __name__ == '__main__':

    path = 'skrl_runs/DoorHook/conv_ppo/BEST_levorg_devel_additional_2/checkpoints/best_agent.pt'
    Visualizer = CNNVisualizer(path)
    Visualizer.visualize(Visualizer.conv_2)
    # print(Visualizer.conv_2.shape)