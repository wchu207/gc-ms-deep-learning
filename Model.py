import torch
from torch.nn import Conv2d, ConvTranspose2d
import torch.nn.functional as F

class MSNetAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(MSNetAutoEncoder, self).__init__()
        self.encoder = MSNetEncoder()
        self.decoder = MSNetDecoder()

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded

class MSNetEncoder(torch.nn.Module):
    def __init__(self):
        super(MSNetEncoder, self).__init__()
        self.reduce_conv1 = Conv2d(1, 1, (11, 11), 2)
        self.reduce_conv2 = Conv2d(1, 1, (9, 9), 1)
        self.reduce_conv3 = Conv2d(1, 1, (9, 9), 1)

    def forward(self, X):
        r1 = F.relu(self.reduce_conv1(X))
        r2 = F.relu(self.reduce_conv2(r1))
        r3 = F.relu(self.reduce_conv3(r2))
        return r3
        

class MSNetDecoder(torch.nn.Module):
    def __init__(self):
        super(MSNetDecoder, self).__init__()
        self.transpose_conv1 = ConvTranspose2d(1, 1, (9, 9), 1)
        self.transpose_conv2 = ConvTranspose2d(1, 1, (9, 9), 1)
        self.transpose_conv3 = ConvTranspose2d(1, 1, (11, 11), 2, output_padding=(1, 1))

    def forward(self, X):
        t1 = F.relu(self.transpose_conv1(X))
        t2 = F.relu(self.transpose_conv2(t1))
        t3 = F.relu(self.transpose_conv3(t2))
        return t3