import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, nz=100, nc=1):
        """GAN generator.
        
        Args:
          nz:  Number of elements in the latent code. latentDIM
          ngf: Base size (number of channels) of the generator layers.
          nc:  Number of channels in the generated images.
        """
        super(Generator, self).__init__()

        self.base = nn.Sequential(nn.ConvTranspose1d(1, 4, kernel_size=256, stride=2, bias=False, padding=1),
                                  nn.ConvTranspose1d(4, 8, kernel_size=71, stride=2, bias=False, padding=1),
                                  nn.ConvTranspose1d(8, 16, kernel_size=64, stride=2, bias=False, padding=1),
                                  nn.ConvTranspose1d(16, 32, kernel_size=36, stride=2, bias=False, padding=1))
        
        self.conv2 = nn.ConvTranspose1d(32, nc, kernel_size=34, stride=2, bias=False, padding=0)
        self.rnn_layer = nn.LSTM(input_size=100, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
        self.act = nn.Tanh()

    def forward(self, z, verbose=False):
        """Generate signals by transforming the given noise tensor.
        
        Args:
          z of shape (batch_size, nz, 1, 1): Tensor of noise samples. We use the last two singleton dimensions
                          so that we can feed z to the generator without reshaping.
          verbose (bool): Whether to print intermediate shapes (True) or not (False).
        
        Returns:
          out of shape (batch_size, nc, 28, 28): Generated images.
        """
        out,_ = self.rnn_layer(z)
        out = self.base(out)
        out = self.conv2(out)
        out = self.act(out)
        if verbose:
            print(out.shape)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_size = 9000, nc=1, ndf=8):
        """GAN discriminator.
        
        Args:
          nc:  Number of channels in the generator output.
          ndf: number of channels of the discriminator layers.
        """

        super(Discriminator, self).__init__()
        
        self.base = nn.Sequential(nn.Conv1d(nc, ndf, kernel_size=256, stride=4, bias=False, padding=1),
                          nn.LeakyReLU(0.2),
                          nn.Conv1d(ndf, 2*ndf, kernel_size=128, stride=3, bias=False, padding=2),
                          nn.LeakyReLU(0.2),
                          nn.Conv1d(2*ndf, 4*ndf, kernel_size=64, stride=3, bias=False, padding=1),
                          nn.LeakyReLU(0.2))
        
        self.rnn_layer = nn.LSTM(input_size=209, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        self.base2 = nn.Sequential(nn.Conv1d(32, nc, kernel_size=8, stride=2, bias=False, padding=0),
                          nn.Linear(29, 16),
                          nn.Linear(16, nc),
                          nn.Sigmoid())

    def forward(self, x, verbose=False):
        """Classify given images into real/fake.
        
        Args:
          x of shape (batch_size, 1, 90000): Signal to be classified.
        
        Returns:
          out of shape (batch_size,): Probabilities that signals are real. All elements should be between 0 and 1.
        """
        out = self.base(x)
        #print(out.shape)
        out,_ = self.rnn_layer(out)
        
        #out = out[:,-1,:]
        #print(out.shape)
        out = self.base2(out)

        return out.squeeze()



def generator_loss(netD, fake_samples):
    """Loss computed to train the generator.

    Args:
      netD: The discriminator whose forward function takes inputs of shape (batch_size, 1, 9000)
         and produces outputs of shape (batch_size, 1). <- real of fake
      fake_samples of shape (batch_size, 1, 9000): Fake images produces by the generator.

    Returns:
      loss: The mean of the binary cross-entropy losses computed for all the samples in the batch.
    """

    batch_size = fake_samples.shape[0]
    netD = netD.to(fake_samples.device)
    pred = netD.forward(fake_samples)
    targets = (torch.ones(batch_size) * real_label).to(fake_samples.device)

    loss = F.binary_cross_entropy(pred, targets) #does the mean by default
    return loss

def discriminator_loss(netD, real_samples, fake_samples):
    """Loss computed to train the discriminator.

    Args:
      netD: The discriminator.
      real_samples of shape (batch_size, 1, 9000): Real samples.
      fake_samples of shape (batch_size, 1, 9000): Fake samples produces by the generator.

    Returns:
      d_loss_real: The mean of the binary cross-entropy losses computed on the real_samples.
      D_real: Mean output of the discriminator for real_samples. This is useful for tracking convergence.
      d_loss_fake: The mean of the binary cross-entropy losses computed on the fake_samples.
      D_fake: Mean output of the discriminator for fake_samples. This is useful for tracking convergence.
    """

    batch_size = fake_samples.shape[0]
    netD = netD.to(fake_samples.device)

    pred_fake = netD.forward(fake_samples)
    targets_fake = (torch.ones(batch_size) * fake_label).to(fake_samples.device)
    d_loss_fake = F.binary_cross_entropy(pred_fake, targets_fake)
    D_fake = torch.mean(pred_fake)

    pred_real = netD.forward(real_samples)
    targets_real = (torch.ones(batch_size) * real_label).to(fake_samples.device)
    d_loss_real = F.binary_cross_entropy(pred_real, targets_real)
    D_real = torch.mean(pred_real)


    return d_loss_real, D_real, d_loss_fake, D_fake