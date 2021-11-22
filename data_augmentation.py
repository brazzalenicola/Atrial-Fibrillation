import pandas as pd
import preprocessing
import GAN
import torch
import os
import zipfile
from torch.utils.data import Dataset, DataLoader
import torchinfo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the data folder location
data_dir = 'data'.encode()

# Extract data from zip file
if not os.path.exists(data_dir):
    with zipfile.ZipFile('data.zip', 'r') as f:
        f.extractall('.')

reference_df = pd.read_csv('data/REFERENCE-v3.csv', names=['mat', 'label'], index_col=0)

# Replace 'N' with 0, 'A' with 1, 'O' with 2 and '~' with 3
reference_df['label'] = reference_df['label'].astype(str) #labels 'A','N','O','~'
cate = pd.Categorical(reference_df['label'])
reference_df['label_code'] = cate.codes #corresponding label code 


batch_size = 64

# 'N' -> 1, 'A'  ->  0, 'O'  ->  2 and '~'  ->  3
trainset_noisy = preprocessing.ImbalancedDataset(ref = reference_df, data_dir = data_dir, label_code=3)
noisy_loader = DataLoader(trainset_noisy, batch_size=batch_size, shuffle=True, num_workers=2)


trainset_afib = preprocessing.ImbalancedDataset(ref = reference_df, data_dir = data_dir, label_code=0)
afib_loader = DataLoader(trainset_afib, batch_size=batch_size, shuffle=False, num_workers=2)


"""## GAN for Data Augmentation

## Generator and generator loss
"""

nz = 100
netG = GAN.Generator(nz=nz, nc=1)

netG = netG.to(device)
real_label = 1.
fake_label = 0.

summary(netG)

"""##Discriminator and generator-discriminator losses"""

netD = GAN.Discriminator(nc=1, ndf=8)
netD = netD.to(device)

summary(netD)

"""## GAN Training

`D_real` and `D_fake`. When it is hard for the discriminator to separate real and fake images, their values are close to 0.5.
"""

epochs = 1000
opt_g = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# GAN TRAINING 
for ep in range(epochs):
    g_losses, d_losses = [], []
    D_real_list, D_fake_list = [], []
    for i, data in enumerate(afib_loader):
        inputs = data['ecg'].to(device)
        labels = data['label'].to(device).long()

        batch_size = labels.shape[0]

        z = torch.randn(batch_size, 1, 100, device=device)

        inputs = inputs.unsqueeze(1)
        fake_inp = netG(z)
        opt_d.zero_grad()
        d_loss_real, D_real, d_loss_fake, D_fake = GAN.discriminator_loss(netD, inputs, fake_inp)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward(retain_graph=True)
        opt_d.step()

        opt_g.zero_grad()
        g_loss = generator_loss(netD, fake_inp)
        g_loss.backward()
        opt_g.step()
        

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        D_real_list.append(D_real.item())
        D_fake_list.append(D_fake.item())
    if ep % 100 == 0:
        print('Epoch {}'.format(ep+1))
        print(' - Mean g_loss: {}'.format(round(np.mean(g_losses), 4)))
        print(' - Mean d_loss: {}'.format(round(np.mean(d_losses), 4)))
        print(' - Mean D_real: {}'.format(round(np.mean(D_real_list), 4)))
        print(' - Mean D_fake: {}'.format(round(np.mean(D_fake_list), 4)))

#VISUAL TEST
z = torch.randn(100, 1, 100, device=device)
samples = netG(z)

samples = samples.squeeze()
samples = samples.to('cpu')
test_sample = samples[4,:].detach().numpy()

preprocessing.plot_ecg(test_sample, '~')

torch.save(netG.state_dict(), '../gen_afib.pth')