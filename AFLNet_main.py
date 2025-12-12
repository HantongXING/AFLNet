import _pickle as cPickle
import numpy as np
import random
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from CLDNN import DNET_DANN
from CLDNN import GNET
from teachermodel import EMATeacher
from selfsup.ntx_ent_loss import NTXentLoss

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
warnings.filterwarnings('ignore')


# ============================================================
# Utility functions
# ============================================================
def to_onehot(yy):
    """
    Convert label indices to one-hot encoding.
    """
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


# ============================================================
# Data augmentation methods
# Shape of data: [batch_size, 2, 128]
# ============================================================
def aug_mask(data):
    """
    Random temporal masking augmentation.
    """
    if np.random.rand(1) > 0.5:
        mask = torch.ones(data.shape[0], 2, 128)
        index = 4 * np.round(31 * np.random.rand(data.shape[0], 12))
        for m in range(data.shape[0]):
            mask[m, :, index[m, :]] = 0
            mask[m, :, index[m, :] + 1] = 0
            mask[m, :, index[m, :] + 2] = 0
            mask[m, :, index[m, :] + 3] = 0
        feed_data = torch.mul(data, mask)
    else:
        feed_data = data
    return feed_data


def aug_reverse(data):
    """
    Signal inversion augmentation.
    """
    if np.random.rand(1) > 0.5:
        reverse_data = -data
    else:
        reverse_data = data
    return reverse_data


def aug_resize(data):
    """
    Amplitude scaling augmentation.
    """
    if np.random.rand(1) > 0.5:
        sample_return = data.copy()
        s = random.randint(8, 12)
        I = data[:, 0, :] * s * 0.1
        Q = data[:, 1, :] * s * 0.1
        sample_return[:, 0, :] = I.copy()
        sample_return[:, 1, :] = Q.copy()
    else:
        sample_return = data
    return sample_return


def aug_shift(data):
    """
    Circular time shift augmentation.
    """
    if np.random.rand(1) > 0.5:
        sample_return = data.copy()
        s = random.randint(1, 126)
        sample_return[:, :, 127 - s:128] = data[:, :, 0:s + 1]
        sample_return[:, :, 0:127 - s] = data[:, :, s + 1:]
    else:
        sample_return = data
    return sample_return


def aug_rot(data):
    """
    Phase rotation augmentation in IQ domain.
    """
    if np.random.rand(1) > 0.5:
        sample_return = data.copy()
        in_array = [
            np.pi / 8, np.pi / 2, np.pi * 5 / 6, np.pi * 2 / 3,
            np.pi / 6, np.pi * 3 / 5, np.pi * 5 / 8,
            np.pi * 3 / 2, np.pi
        ]
        t = random.choice(in_array)
        c = np.cos(t)
        s = np.sin(t)
        I = c * data[:, 0, :] + s * data[:, 1, :]
        Q = (-s) * data[:, 0, :] + c * data[:, 1, :]
        sample_return[:, 0, :] = I.copy()
        sample_return[:, 1, :] = Q.copy()
    else:
        sample_return = data
    return sample_return


# ============================================================
# Dataset paths
# ============================================================
path_awgn = '/data2/RML/Datasets/Data0124/AWGN.dat'
path_rician = '/data2/RML/Datasets/Data0124/Rayleigh_complexer.dat'


# ============================================================
# Load target domain dataset (Rician / Rayleigh)
# ============================================================
a = open(path_rician, 'rb')
Xd = cPickle.load(a, encoding='iso-8859-1')

snrs, mods = map(
    lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))),
    [1, 0]
)

X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))

X = np.vstack(X)
np.random.seed(2016)

train_idx = list(np.random.choice(range(0, X.shape[0]), size=X.shape[0], replace=False))
X_test = X[train_idx]
X_SNR_test = list(map(lambda x: snrs.index(lbl[x][1]), train_idx))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))


# ============================================================
# Load source domain dataset (AWGN)
# ============================================================
b = open(path_awgn, 'rb')
Xd = cPickle.load(b, encoding='iso-8859-1')

snrs, mods = map(
    lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))),
    [1, 0]
)

X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))

X = np.vstack(X)
np.random.seed(2016)

train_idx = list(np.random.choice(range(0, X.shape[0]), size=X.shape[0], replace=False))
X_train = X[train_idx]
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))


# ============================================================
# DataLoader construction
# ============================================================
train_set_all = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
test_set_all = TensorDataset(
    torch.tensor(X_test),
    torch.tensor(Y_test),
    torch.tensor(X_SNR_test)
)

train_loader_all = DataLoader(train_set_all, batch_size=64, shuffle=True)
test_loader_all = DataLoader(test_set_all, batch_size=64)
test_loader = DataLoader(test_set_all, batch_size=64)


# ============================================================
# Model, optimizer, and loss setup
# ============================================================
generator = GNET().cuda()
discriminator = DNET_DANN().cuda()

teacher = EMATeacher(generator, alpha=0.9, pseudo_label_weight='prob')

optimizer = optim.Adam([
    {"params": generator.parameters(), "lr": 1e-3},
    {"params": discriminator.parameters(), "lr": 1e-3}
])

criterion = nn.CrossEntropyLoss()
contrast_loss = NTXentLoss()

len_dataloader = len(train_loader_all)


# ============================================================
# Entropy-based loss (Information Maximization)
# ============================================================
def Entropy(input_):
    """
    Compute entropy of softmax outputs.
    """
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


# ============================================================
# Training loop
# ============================================================
precesion_previous = 0

for epoch in range(200):
    for i, (sdata, tdata) in enumerate(zip(train_loader_all, test_loader_all)):

        # Domain adaptation schedule
        p = float(i + epoch * len_dataloader) / 200 / len_dataloader
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        train_data, label = sdata
        test_data, _, _ = tdata

        # ----------------------------------------------------
        # Strong target-domain augmentation pipeline
        # ----------------------------------------------------
        reverse_test = torch.tensor(aug_reverse(test_data.numpy()))
        reseize_test = torch.tensor(aug_resize(reverse_test.numpy()))
        shift_test = torch.tensor(aug_shift(reseize_test.numpy()))
        rot_test = torch.tensor(aug_rot(shift_test.numpy()))
        mask_test = aug_mask(rot_test).cuda()

        # ----------------------------------------------------
        # Data preparation
        # ----------------------------------------------------
        train_data = torch.reshape(train_data, [-1, 2, 128]).cuda()
        label = label.cuda().long()
        test_data = test_data.cuda()

        teacher.update_weights(generator, epoch * len_dataloader + 1)

        # Domain labels
        domain_label_s = torch.zeros(len(train_data)).long().cuda()
        domain_label_t = torch.ones(len(test_data)).long().cuda()

        # ----------------------------------------------------
        # Forward passes
        # ----------------------------------------------------
        feat_s, logits_s, _, _ = generator(train_data)
        feat_t, logits_t, _, projection1 = generator(test_data)
        _, logits_mask_t, _, projection2 = generator(mask_test)

        # Pseudo-labels from EMA teacher
        logit_t_pseudo, pseudo_prob_t = teacher(test_data)

        # ----------------------------------------------------
        # Self-training loss (pseudo-label weighted CE)
        # ----------------------------------------------------
        mask_ce = F.cross_entropy(logits_mask_t, logit_t_pseudo, reduction='none')
        masking_loss_value = torch.mean(pseudo_prob_t * mask_ce)

        # ----------------------------------------------------
        # Domain adversarial loss
        # ----------------------------------------------------
        domain_output_s = discriminator(feat_s, alpha=alpha)
        domain_output_t = discriminator(feat_t, alpha=alpha)

        # ----------------------------------------------------
        # Confidence-based auxiliary loss (IM loss)
        # ----------------------------------------------------
        softmax_out = nn.Softmax(dim=1)(logits_mask_t)
        entropy_loss = torch.mean(Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

        # ----------------------------------------------------
        # Similarity-based auxiliary loss (contrastive)
        # ----------------------------------------------------
        cont_loss = contrast_loss(projection1, projection2)

        # ----------------------------------------------------
        # Supervised classification loss
        # ----------------------------------------------------
        loss_ce = criterion(logits_s, torch.argmax(label, dim=1))
        loss_domain_s = criterion(domain_output_s, domain_label_s)
        loss_domain_t = criterion(domain_output_t, domain_label_t)

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------
        loss = (
            loss_ce +
            loss_domain_s +
            loss_domain_t +
            masking_loss_value +
            entropy_loss +
            cont_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(epoch, i, loss.item(), cont_loss.item())

    # ========================================================
    # Evaluation
    # ========================================================
    generator.eval()
    correct, fals = 0, 0
    cmt = torch.zeros(11, 11, 20, dtype=torch.int16)

    for i, data in enumerate(test_loader):
        testdata, testlabel, SNR = data
        testdata = torch.reshape(testdata, [-1, 2, 128]).cuda()
        testlabel = testlabel.cuda()

        _, outputs, _, _ = generator(testdata)
        _, label = torch.max(testlabel, 1)
        _, predict = torch.max(outputs, 1)

        for k in range(len(predict)):
            if predict[k] == label[k]:
                correct += 1
            else:
                fals += 1
            cmt[label[k]][predict[k]][SNR[k]] += 1

    generator.train()
    precision = correct / (correct + fals)
    print('acc current:', precision)

    if precision > precesion_previous:
        torch.save(generator, 'AFLNet.pth')
        precesion_previous = precision

    print('max acc:', precesion_previous)
    for j in range(20):
        num = sum(cmt[k, k, j] for k in range(11))
        print(int(num) / int(sum(sum(cmt[:, :, j]))))


# ============================================================
# Final evaluation using best model
# ============================================================
net = torch.load('AFLNet.pth')
net.eval()

correct, fals = 0, 0
cmt = torch.zeros(11, 11, 20, dtype=torch.int16)

for i, data in enumerate(test_loader):
    testdata, testlabel, SNR = data
    testdata = torch.reshape(testdata, [-1, 2, 128]).cuda()
    testlabel = testlabel.cuda()

    _, outputs, _, _ = net(testdata)
    _, label = torch.max(testlabel, 1)
    _, predict = torch.max(outputs, 1)

    for k in range(len(predict)):
        if predict[k] == label[k]:
            correct += 1
        else:
            fals += 1
        cmt[label[k]][predict[k]][SNR[k]] += 1

print('Avg acc:', correct / (correct + fals))
for j in range(20):
    num = sum(cmt[k, k, j] for k in range(11))
    print(int(num) / int(sum(sum(cmt[:, :, j]))))
