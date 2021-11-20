import pandas as pd
import numpy as np
import os
import random
import zipfile
import torch
from torch.utils.data import data_utils, Dataset, DataLoader
import torch
from torchinfo import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, accuracy_score, precision_recall_curve
import preprocessing #NEW_FILE
import GAN #NEW_FILE
import RNN #NEW_FILE
import data_augmentation #NEW_FILE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed)

# Define the data folder location
data_dir = 'data'.encode()

# Extract data from zip file
if not os.path.exists(data_dir):
    with zipfile.ZipFile('data.zip', 'r') as f:
        f.extractall('.')

# Create a dataframe containing all the files and the labels from the reference file
reference_df = pd.read_csv('data/REFERENCE-v3.csv', names=['mat', 'label'], index_col=0)

# Replace 'N' with 0, 'A' with 1, 'O' with 2 and '~' with 3
reference_df['label'] = reference_df['label'].astype(str) #labels 'A','N','O','~'
cate = pd.Categorical(reference_df['label'])
reference_df['label_code'] = cate.codes #corresponding label code 

"""# Utils function"""

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

file_name = 'A00001'
reference_df[reference_df.index==file_name].iloc[0]['label']


# Examples

# Plot a Normal ECG
file_name = 'A00001'.encode()
normal_data, lbl = preprocessing.load_data(file_name, data_dir)
preprocessing.plot_ecg(normal_data, lbl)

# Plot an Afib ECG
file_name = 'A00005'.encode()
afib_data, lbl  = lpreprocessing.oad_data(file_name, data_dir)
preprocessing.plot_ecg(afib_data, lbl)

# Plot an Abnormal ECG
file_name = 'A00013'.encode()
abn_data, lbl  = preprocessing.load_data(file_name, data_dir)
preprocessing.plot_ecg(abn_data, lbl )

# Plot an Noisy ECG
file_name = 'A00022'.encode()
noisy_data,lbl = preprocessing.load_data(file_name, data_dir)
preprocessing.plot_ecg(noisy_data, lbl )

"""# Preprocessing"""

file_name = 'A00001'.encode()

# Load the data and apply the baseline wander removal   
data,lbl  = preprocessing.load_data(file_name, data_dir)
filt_data = preprocessing.baseline_wander_removal(data)

preprocessing.plot_ecg(data,lbl)
preprocessing.plot_ecg(filt_data,lbl)

"""## Normalisation"""
    
file_name = 'A00001'.encode()

# Load the data, apply the baseline wander removal and normalize the data
data,lbl = preprocessing.load_data(file_name, data_dir)
filt_data = preprocessing.baseline_wander_removal(data)
norm_data = preprocessing.normalize_data(filt_data)

preprocessing.plot_ecg(filt_data,lbl)
preprocessing.plot_ecg(norm_data,lbl)

"""## Random Crop

for name in list(train_reference_df.index[:100]):
    data = load_data(name.encode(), data_dir)
    print('File: {} - Shape: {}'.format(name, data.shape))
"""

#Example
file_name = 'A02095'.encode()
data, lbl = preprocessing.load_data(file_name, data_dir)

preprocessing.plot_ecg(filt_data,lbl)

"""
for i in range(3):
    plt.figure()   
    # plot the cropped signal
    rc = random_crop(data, center_crop=False)
    print(rc.shape)
    plt.plot(rc)
    plt.title('Crop {}'.format(i+1))
"""

# Example
file_name = 'A00001'.encode()
data, lbl = preprocessing.load_and_preprocess_data(file_name, data_dir)

preprocessing.plot_ecg(filt_data,lbl)


"""#Dataset creation"""

batch_size = 32

train_set = preprocessing.PhysioNetDataset(ref = train_reference_df, data_dir = data_dir)
val_set = preprocessing.PhysioNetDataset(ref = val_reference_df, data_dir = data_dir)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)

"""#Model """

input_size = 9000
input_shape = (input_size, 1)
output_size = 4

rnn_model = RNN.RNN(input_size, output_size).to(device)
summary(rnn_model)

optimizer = torch.optim.NAdam(rnn_model.parameters(), lr=0.001)
criterion = RNN.FocalLoss()

epochs = 80

train_loss, val_loss = [], []
train_acc, val_acc = [], []
for epoch in range(epochs):
    train_running_loss = 0.0
    val_running_loss = 0.0
    train_correct = 0
    val_correct = 0
    rnn_model.train()
    for i, data in enumerate(train_loader):
        inputs = data['ecg'].to(device)
        labels = data['label'].to(device).long()
        #bs = labels.shape[0]

        optimizer.zero_grad()
        outputs = rnn_model(inputs)
        outputs = outputs.squeeze()
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item() * outputs.shape[0]

        _, predicted = torch.max(outputs, dim=1)
        train_correct += torch.sum(predicted == labels)

    train_loss.append(train_running_loss / len(train_set))
    train_acc.append(train_correct.float().item() / len(train_set))
    print("Epoch:", epoch+1, "Loss:", train_loss[-1], "Accuracy:", train_acc[-1])

model = RNN.RNN(input_size, output_size)
model.load_state_dict(torch.load('rnn1.pth', map_location=torch.device('cpu')))
model.eval()

acc = train_acc
loss = train_loss

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training Loss')

"""# Evaluation"""

val_loss = []
val_acc = []
val_running_loss = 0.0
val_correct = 0
labels_predicted = []
true_label = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(val_loader):
        inputs = data['ecg'].to(device)
        labels = data['label'].to(device).long()
        true_label.append(labels)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        val_running_loss += loss.item() * outputs.shape[0]

        _, predicted = torch.max(outputs, dim=1)
        labels_predicted.append(predicted) 
        val_correct += torch.sum(predicted == labels)

    val_loss.append(val_running_loss / len(val_set))
    val_acc.append(val_correct.float().item() / len(val_set))
    print("Loss:", val_loss[-1], "Accuracy:", val_acc[-1])

import sklearn

true_labels = np.concatenate(true_label)
labels_predicted = np.concatenate(labels_predicted)

conf_matrix = sklearn.metrics.confusion_matrix(true_labels, labels_predicted)

def evaluate_metrics(confusion_matrix):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(ACC) # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    return ACC_macro, ACC, TPR, TNR, PPV

acc_macr, acc, TPR, TNR, PPV = evaluate_metrics(conf_matrix) 
print("macro accuracy:", acc_macr)
print("accuracy:", acc)

print(TPR)
print(TNR)
print(PPV)
