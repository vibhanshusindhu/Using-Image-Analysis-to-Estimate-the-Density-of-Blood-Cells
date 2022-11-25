import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import ops
from focal_loss.focal_loss import FocalLoss

import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import glob
import pickle

class PredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(36, 64, num_layers = 2, bidirectional = False)

        self.predNet = nn.Sequential(
          nn.Linear(64, 128),
          nn.Linear(128, 128),
          nn.Linear(128, 1)
        )

    def forward(self, x, lengths):
        ##pack sequence
        ##print(x.shape, lengths.shape, lengths.data)
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        ##print("PACKED EMBEDDED", packed_embedded.data.shape)
        #packed_output, hn = self.rnn(packed_embedded)
        ##unpack sequence
        ##print("PACKED OUTPUT", packed_output.data.shape)
        #output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=-1, total_length=350)  
        ##print("OUTPUT", output.shape)

        output, hn = self.rnn(x)
        return self.predNet(output), lengths #output_lengths
        #return torch.sigmoid(self.predNet(output)), output_lengths
    
    def predict(self, x):
        end_of_wbcs = []
        output, hn = self.rnn(x)
        output = model.predNet(output)
        seq_len = x.shape[0]

        res = torch.sigmoid(output)
        predicted_values = res.reshape(-1)
        predicted_values = np.around(predicted_values.numpy())
        of_wbc = np.where(predicted_values == 1)[0]
        if len(of_wbc) == 0:
            return [0] * seq_len

        end_of_wbc = of_wbc[0]  
        while len(of_wbc) != 0 and end_of_wbc < seq_len - 1:
            end_of_wbcs.append(end_of_wbc)
            input = x[end_of_wbc+1:]
            output, hn = self.rnn(input)
            output = model.predNet(output)
            res = torch.sigmoid(output)
            predicted_values = res.reshape(-1)
            predicted_values = np.around(predicted_values.numpy())
            of_wbc = np.where(predicted_values == 1)[0]
            if len(of_wbc) > 0:
                end_of_wbc = of_wbc[0] + end_of_wbc + 1
                end_of_wbc = min(seq_len - 1, end_of_wbc)

        if end_of_wbc not in end_of_wbcs:
            end_of_wbcs.append(end_of_wbc)
        predicted_values = np.zeros(seq_len, dtype=np.uint8)
        for idx in end_of_wbcs:
            predicted_values[idx] = 1
        return predicted_values

# TODO: try to find a more suitable weight or an arternative
# loss function.
class WeightedMSELoss(nn.Module):
    
    def __init__(self, wt):
        self.weight = wt
        super(WeightedMSELoss,self).__init__()
    
    def forward(self, outputs, labels):
        weight = torch.ones_like(outputs)
        weight[labels == 1] = self.weight
        return (weight * (torch.sigmoid(outputs) - labels) ** 2).mean()

class RNNDataset(Dataset):
    def __init__(self, file_names):
        btches = []
        y_btches = []
        lengths = []
        for pickeled_file_path in file_names:
            with open(pickeled_file_path, 'rb') as f:
                tup = pickle.load(f)
                if len(tup) == 3:
                    video_points, y_points, length = tup
                    btches.append(video_points)
                    y_btches.append(np.array(y_points))
                    lengths.append(length)
                else:
                    video_points, y_points = tup
                    btches.append(video_points)
                    y_btches.append(np.array(y_points))
        self.x = btches
        self.y = y_btches
        self.lengths = lengths

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if len(self.lengths) > 0:
            return self.x[idx], self.y[idx], self.lengths[idx]
        return self.x[idx], self.y[idx]

#wts = [random.uniform(0, 1) for i in range(0, 20)]
#wts = np.linspace(0.81, 0.9, 10)
wts = range(75, 80)
for wt in wts:
    print("WT", wt)

    good_file_names = glob.glob("synth_data_pkl/*.pkl")

    train_set_length = int(len(good_file_names)* 0.8)
    train_file_names = random.sample(good_file_names, train_set_length)
    test_file_names = [x for x in good_file_names if x not in train_file_names]

    training_data = RNNDataset(train_file_names)
    test_data = RNNDataset(test_file_names)
    train_dataloader = DataLoader(training_data, batch_size=12, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=12, shuffle=True)

    val_file_names = glob.glob("synth_data_pkl_test/*.pkl")
    val_data = RNNDataset(val_file_names)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

    total_pos = 0
    total_neg = 0
    for batch, y, seq_lengths in train_dataloader:
        inds = np.where(y.reshape(-1) == 0)[0]
        total_neg = total_neg + len(inds)
        inds = np.where(y.reshape(-1) == 1)[0]
        total_pos = total_pos + len(inds)
    pos_weight = total_neg / total_pos

    model = PredModel()
    criterion = WeightedMSELoss(wt)
    #criterion = nn.BCEWithLogitsLoss(reduction='mean')  #pos_weight=torch.tensor(pos_weight)
    #criterion = nn.BCELoss(reduction='mean')
    #criterion = torchvision.ops.focal_loss.sigmoid_focal_loss()
    #criterion = FocalLoss(gamma = 0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    epoch_train_losses = []
    epoch_test_losses = []
    for epoch in range(0, 3):
        model.train()
        train_losses = []
        test_losses = []
        for batch, y, seq_lengths in train_dataloader:
            input = batch.permute(2, 0, 1).float()
            #print("INPUT", input.shape)
            res, output_lengths = model(input, seq_lengths)
            #print("RES", res.shape)

            inds = np.where(y.reshape(-1) == 1)[0]
            if len(inds) == 0:
                print(y)
            predicted = res.reshape(-1)[inds]
            target = y.reshape(-1).float()[inds]

            pos_num = len(target)

            inds = np.where(y.reshape(-1) == 0)[0]
            if len(inds) == 0:
                print(y)
            sorted, indices = torch.sort(res.reshape(-1), descending=True)

            predicted = torch.cat((predicted, res.reshape(-1)[indices < pos_num]))
            target = torch.cat((target, torch.zeros_like(res.reshape(-1)[indices < pos_num])))

            #loss = torchvision.ops.focal_loss.sigmoid_focal_loss(predicted, target, reduction = 'mean') # alpha = wt
            loss = criterion(predicted, target)
            #print(loss.data, loss)
            #print("predicted", predicted)
            #print("target", target)
            train_losses.append(loss.data.item())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for batch, y, seq_lengths in test_dataloader:
                input = batch.permute(2, 0, 1).float()
                res, output_lengths = model(input, seq_lengths)
                inds = np.where(y.reshape(-1) >= 0)[0]
                predicted = res.reshape(-1)[inds]
                target = y.reshape(-1).float()[inds]
                #loss = torchvision.ops.focal_loss.sigmoid_focal_loss(predicted, target, reduction = 'mean') # alpha = wt
                loss = criterion(predicted, target)
                if np.isnan(loss.data.item()):
                    print("PREDICTED", predicted)
                    print("TARGET", target)
                    
                test_losses.append(loss.data.item())
        epoch_train_losses.append(np.mean(train_losses))
        epoch_test_losses.append(np.mean(test_losses))
        print(epoch, epoch_train_losses[-1], epoch_test_losses[-1])

    all_predicted_values = []
    all_target_values = []
    model.eval()
    with torch.no_grad():
        for batch, y in val_dataloader:
            end_of_wbcs = []
            input = batch.permute(2, 0, 1).float()
            predicted_values = model.predict(input)
            all_predicted_values.extend(predicted_values)
            target_values = y.reshape(-1).float()
            all_target_values.extend(target_values.numpy())

    print(np.sum(all_predicted_values), np.sum(all_target_values), len(all_target_values))

    accuracy = accuracy_score(all_target_values, all_predicted_values)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    comms = 0
    for idx, value in enumerate(all_target_values):
        if all_target_values[idx] == 1.0 and (all_target_values[idx] == all_predicted_values[idx]):
            comms = comms + 1
    print("TP", comms)

    comms = 0
    for idx, value in enumerate(all_target_values):
        if all_target_values[idx] == 0.0 and (all_target_values[idx] == all_predicted_values[idx]):
            comms = comms + 1
    print("TN", comms)

    #print()

    print(len(all_target_values), len(all_predicted_values))
    tn, fp, fn, tp = confusion_matrix(all_target_values, all_predicted_values).ravel()
    print("Confusion matrix:\n", "TN", str(tn), "FP", str(fp), "FN", str(fn), "TP", str(tp))
