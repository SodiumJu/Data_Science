import torch #YR
import numpy as np
from torch.nn.modules.module import Module



class PruningModule(Module):
    DEFAULT_PRUNE_RATE = {
        'conv1': 84,
        'conv2': 38,
        'conv3': 35,
        'conv4': 37,
        'conv5': 37,
        'fc1': 9,
        'fc2': 9,
        'fc3': 25
    }

    def _prune(self, module, threshold):

        #################################
        # TODO:
        #    1. Use "module.weight.data" to get the weights of a certain layer of the model
        #    2. Set weights whose absolute value is less than threshold to 0, and keep the rest unchanged
        #    3. Save the results of the step 2 back to "module.weight.data"
        #    --------------------------------------------------------
        #    In addition, there is no need to return in this function ("module" can be considered as call by
        #    reference)
        #################################

        weight_dev = module.weight.device
        # Convert Tensors to numpy and calculate
        tensor = module.weight.data.cpu().numpy()
        new_tensor = np.where(abs(tensor) < threshold, 0, tensor)
        # Apply new weight and mask
        module.weight.data = torch.from_numpy(new_tensor).to(weight_dev)
        #module.mask.data = torch.from_numpy(new_mask).to(mask_dev)

    def prune_by_percentile(self, q=DEFAULT_PRUNE_RATE):

        ########################
        # TODO
        # 	For each layer of weights W (including fc and conv layers) in the model, obtain the (100 - q)th percentile
        # 	of absolute W as the threshold, and then set the absolute weights less than threshold to 0 , and the rest
        # 	remain unchanged.
        ########################
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3', 'conv1' ,'conv2' ,'conv3' ,'conv4' ,'conv5']:
            # Calculate percentile value
                tensor = module.weight.data.cpu().numpy()
                percentile_value = np.percentile(np.abs(tensor), (100-q[name]))
                print(f'Pruning with threshold : {percentile_value:.4f} for layer {name}')
                self._prune(module, percentile_value)
        # Prune the weights and mask

    def prune_by_std(self, s=0.25):
    	# YR for conv layer constant
        conv_r = 0.1
        for name, module in self.named_modules():

            #################################
            # TODO:
            #    Only fully connected layers were considered, but convolution layers also needed
            #################################
            #,'conv1','conv2','conv3','conv4','conv5'
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy())*s
                print(f'Pruning with threshold : {threshold:.4f} for layer {name}')
                self._prune(module, threshold)

            elif name in ['conv1','conv2','conv3','conv4','conv5']:
                threshold = np.std(module.weight.data.cpu().numpy())*s*conv_r
                print(f'Pruning with threshold : {threshold:.4f} for layer {name}')
                self._prune(module, threshold)





