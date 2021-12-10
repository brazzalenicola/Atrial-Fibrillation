import pandas as pd
import preprocessing
import GAN
import torch
import os
import zipfile
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
import torchinfo
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (16,10)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.fontsize'] = 'large'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the data folder location
data_dir = 'data'.encode()

# Extract data from zip file
if not os.path.exists(data_dir):
    with zipfile.ZipFile('data.zip', 'r') as f:
        f.extractall('.')

reference_df = pd.read_csv('data/REFERENCE-v3.csv', names=['mat', 'label'], index_col=0)

# Replace 'N' with 1, 'A' with 0, 'O' with 2 and '~' with 3
reference_df['label'] = reference_df['label'].astype(str) #labels 'A','N','O','~'
cate = pd.Categorical(reference_df['label'])
reference_df['label_code'] = cate.codes #corresponding label code

# Keep 20% of the data out for validation
train_reference_df, val_reference_df = train_test_split(reference_df, test_size=0.2, stratify=reference_df['label'], random_state=seed)

# Count the elements in the sets
num_train_data_normal = sum(train_reference_df['label_code'] == 1)
num_train_data_afib   = sum(train_reference_df['label_code'] == 0)
num_train_data_abnor = sum(train_reference_df['label_code'] == 2)
num_train_data_noisy   = sum(train_reference_df['label_code'] == 3)

num_val_data_normal   = sum(val_reference_df['label_code'] == 1)
num_val_data_afib     = sum(val_reference_df['label_code'] == 0)
num_val_data_abnor = sum(val_reference_df['label_code'] == 2)
num_val_data_noisy   = sum(val_reference_df['label_code'] == 3)

print('TRAIN SET')
print('\tNormal ECG: {} ({:.2f}%)'.format(num_train_data_normal, 100 * num_train_data_normal / len(train_reference_df)))
print('\tAfib ECG: {} ({:.2f}%)'.format(num_train_data_afib, 100 * num_train_data_afib / len(train_reference_df)))
print('\tAbnormal ECG: {} ({:.2f}%)'.format(num_train_data_abnor, 100 * num_train_data_abnor / len(train_reference_df)))
print('\tNoisy ECG: {} ({:.2f}%)'.format(num_train_data_noisy, 100 * num_train_data_noisy / len(train_reference_df)))

print('VALIDATION SET')
print('\tNormal ECG: {} ({:.2f}%)'.format(num_val_data_normal, 100 * num_val_data_normal / len(val_reference_df)))
print('\tAfib ECG: {} ({:.2f}%)'.format(num_val_data_afib, 100 * num_val_data_afib / len(val_reference_df)))
print('\tAbnormal ECG: {} ({:.2f}%)'.format(num_val_data_abnor, 100 * num_val_data_abnor / len(val_reference_df)))
print('\tNoisy ECG: {} ({:.2f}%)'.format(num_val_data_noisy, 100 * num_val_data_noisy / len(val_reference_df)))

batch_size = 32

# 'N' -> 1, 'A'  ->  0, 'O'  ->  2 and '~'  ->  3
trainset_noisy = preprocessing.ImbalancedDataset(ref = reference_df, data_dir = data_dir, label_code=3)
noisy_loader = DataLoader(trainset_noisy, batch_size=batch_size, shuffle=True, num_workers=2)


trainset_afib = preprocessing.ImbalancedDataset(ref = reference_df, data_dir = data_dir, label_code=0)
afib_loader = DataLoader(trainset_afib, batch_size=batch_size, shuffle=True, num_workers=2)

"""
## Generator"""

nz = 100
netG = GAN.Generator(nz=nz, nc=1)

netG = netG.to(device)
real_label = 1.
fake_label = 0.

summary(netG)

"""##Discriminator


"""

netD = GAN.Discriminator(nc=1, ndf=8)
netD = netD.to(device)

summary(netD)

"""## GAN TRAINING

Here we can choose between afib_loader to learn to generate new afib samples or noisy_loader to learn noisy samples.
"""

"""## GAN Training"""
loader = afib_loader #or noisy_loader
epochs = 1000
opt_g = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# GAN TRAINING 
for ep in range(epochs):
    g_losses, d_losses = [], []
    D_real_list, D_fake_list = [], []
    for i, data in enumerate(loader):
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
        D_real_list.append(D_real.item()) #When it is hard for the discriminator to separate real and fake images, their values are close to 0.5.
        D_fake_list.append(D_fake.item())
    if ep % 100 == 0:
        print('Epoch {}'.format(ep+1))
        print(' - Mean g_loss: {}'.format(round(np.mean(g_losses), 4)))
        print(' - Mean d_loss: {}'.format(round(np.mean(d_losses), 4)))
        print(' - Mean D_real: {}'.format(round(np.mean(D_real_list), 4)))
        print(' - Mean D_fake: {}'.format(round(np.mean(D_fake_list), 4)))

torch.save(netG.state_dict(), '../gen_afib.pth')
#OR
#torch.save(netG.state_dict(), '../gen_noisy.pth')

"""## CREATION OF NEW NOISY SAMPLES"""

model = GAN.Generator(nz=100, nc=1)
model.load_state_dict(torch.load('gen_noisy.pth', map_location=torch.device('cpu')))
model.eval()

z = torch.randn(279, 1, 100, device=device)
samples = netG(z)

visual_test = samples[12,:].squeeze().detach().numpy()
plt.figure()
plt.plot(visual_test)
plt.xlim([5000, 8000])
plt.title('Synthetic Noisy ECG')
#plt.savefig("synthetic_noisy.png")

file_names = list()
for i in range(8529, 8529+279):
    file_names.append("A0"+str(i))

label = np.repeat('~', 279)
label_code = np.repeat(3, 279)

data = {'mat': file_names, 'label': label, 'label_code': label_code}
# Create Noisy DataFrame.
df = pd.DataFrame(data)
df_noisy = df.set_index('mat')

"""### Save .mat files containing the generated examples"""

for i in range(8529, 8529+279):
    file_name = "afib_data/"+"A0"+str(i)+'.mat'
    sampl = samples[i,:].detach().numpy()
    mdic = {"val": sampl, "label": 'A'}
    savemat(file_name, mdic)

"""## CREATION OF NEW AFIB SAMPLES"""

model = GAN.Generator(nz=100, nc=1)
model.load_state_dict(torch.load('gen_afib.pth', map_location=torch.device('cpu')))
model.eval()

z = torch.randn(758, 1, 100, device=device)
samples = netG(z)

visual_test = samples[2,:].squeeze().detach().numpy()
plt.figure()
plt.plot(visual_test)
plt.xlim([5000, 8000])
plt.title('Synthetic Afib ECG')
#plt.savefig("synthetic_afib.png")

file_names = list()
for i in range(8808, 8808+758):
    file_names.append("A0"+str(i))
label = np.repeat('A', 758)
label_code = np.repeat(0, 758)

data_afib = {'mat': file_names, 'label': label, 'label_code': label_code}
# Create AFib DataFrame.
df = pd.DataFrame(data_afib)
df_afib = df.set_index('mat')

"""### Save .mat files containing the generated examples"""

for i in range(8808, 8808+758):
    file_name = "afib_data/"+"A0"+str(i)+'.mat'
    sampl = samples[i,:].detach().numpy()
    mdic = {"val": sampl, "label": '~'}
    savemat(file_name, mdic)

"""### Saving new data to the disk and the reference dataframes"""

df_afib.to_csv('df_afib.csv',index=True)
df_noisy.to_csv('df_noisy.csv',index=True)

#!zip -r /content/afib_data.zip /content/afib_data

#!zip -r /content/noisy_data.zip /content/noisy_data