import numpy as np
import torch

# Seeds
torch.manual_seed(0)
np.random.seed(0)

class SpectrogramMasker:
    def __init__(self,
                 spec,
                 num_mask,
                 mask_width):
        self.spec = spec
        self.num_mask = num_mask
        self.mask_width = mask_width

    def random_mask_byframe(self, spec = None, num_mask = None, mask_width = None):
        if spec == None:
            spec = self.spec
        if num_mask == None:
            num_mask = self.num_mask
        if mask_width == None:
            mask_width = self.mask_width

        new_spec = spec.clone().detach()

        for _ in range(num_mask):
            for i in range(spec.shape[0]):
                if spec.shape[3] > mask_width:
                    start = np.random.randint(0, spec.shape[3]-mask_width)
                    
                    new_spec[i, :, :, start: start+mask_width] = 0
        return new_spec

    def random_mask_bypatch(spec = None):
        if spec == None:
            spec = self.spec
        #TODO
        return spec


def mixup(self, signal, label):
    # Choose another signal/label randomly
    
    
    # Select a random number alpha from the given beta distribution
    alpha = 0.3
    # Mixup the signal accordingly
    lam = np.random.beta(alpha, alpha, signal.shape[0])
    X = lam.reshape(signal.shape[0], 1, 1, 1)
    y = lam.reshape(signal.shape[0], 1)
    X_l, y_l = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
    
    # Mix all batches
    mixup_signal, mixup_label = signal.flip(dims=[0]), label.flip(dims=[0])
    signal = X_l * signal.to('cuda') + (1 - X_l) * mixup_signal.float().to('cuda')
    label = y_l * label.to('cuda') + (1 - y_l) * mixup_label.float().to('cuda')
    

        
    return signal, label

def random_pick_frames(signal, num_frames = 384):
    # Randomly pick frames, usually smaller than whole sample frames, to save the GPU resource
    # signal.shape[0] == batch_size
    # signal.shape[1] == number of audio channels (mono = 1)
    # signal.shape[2] == number of freq bands in spectrogram
    # signal.shape[3] == number of frames in spectrogram

    new_signal = np.zeros((signal.shape[0], 1, signal.shape[2], num_frames), dtype=np.float32)
    new_signal = torch.tensor(new_signal)
    for i in range(signal.shape[0]):
        if signal.shape[3] > num_frames:
            start = np.random.randint(0, signal.shape[3]-num_frames)
            new_signal[i] = signal[i, :, :, start: start+num_frames]
    return new_signal