import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt


def PrepareDataset(speed_matrix, BATCH_SIZE=40, seq_len=10, pred_len=1, train_propotion=0.7, valid_propotion=0.2):
    """ Prepare training and testing datasets and dataloaders.

    Convert speed/volume/occupancy matrix to training and testing dataset.
    The vertical axis of speed_matrix is the time axis and the horizontal axis
    is the spatial axis.

    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    time_len = speed_matrix.shape[0]

    max_speed = speed_matrix.max().max()
    speed_matrix = speed_matrix / max_speed

    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix.iloc[i:i + seq_len].values)
        speed_labels.append(speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)

    # shuffle and split the dataset to training and testing datasets
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, max_speed


if __name__ == "__main__":
#     data = 'inrix'
    data = 'loop'
# /Users/prasadmaduranga/higher studies/research/Finland Paper/GNN_RNN/Graph_Convolutional_LSTM-master/LoopData
#     directory = '../../Data_Warehouse/Data_network_traffic/'
    directory = '/Users/prasadmaduranga/higher studies/research/Finland Paper/GNN_RNN/Graph_Convolutional_LSTM-master/LoopData/Seattle-Loop-Data-2015/'
    if data == 'inrix':
        speed_matrix =  pd.read_pickle( directory + 'inrix_seattle_speed_matrix_2012')
        A = np.load(directory + 'INRIX_Seattle_2012_A.npy')
        FFR_5min = np.load(directory + 'INRIX_Seattle_2012_reachability_free_flow_5min.npy')
        FFR_10min = np.load(directory + 'INRIX_Seattle_2012_reachability_free_flow_10min.npy')
        FFR_15min = np.load(directory + 'INRIX_Seattle_2012_reachability_free_flow_15min.npy')
        FFR_20min = np.load(directory + 'INRIX_Seattle_2012_reachability_free_flow_20min.npy')
        FFR_25min = np.load(directory + 'INRIX_Seattle_2012_reachability_free_flow_25min.npy')
        FFR = [FFR_5min, FFR_10min, FFR_15min, FFR_20min, FFR_25min]
    elif data == 'loop':
        speed_matrix =  pd.read_pickle( directory + 'speed_matrix_2015')
        A = np.load( directory + 'Loop_Seattle_2015_A.npy')
        FFR_5min = np.load( directory + 'Loop_Seattle_2015_reachability_free_flow_5min.npy')
        FFR_10min = np.load( directory + 'Loop_Seattle_2015_reachability_free_flow_10min.npy')
        FFR_15min = np.load( directory + 'Loop_Seattle_2015_reachability_free_flow_15min.npy')
        FFR_20min = np.load( directory + 'Loop_Seattle_2015_reachability_free_flow_20min.npy')
        FFR_25min = np.load( directory + 'Loop_Seattle_2015_reachability_free_flow_25min.npy')
        FFR = [FFR_5min, FFR_10min, FFR_15min, FFR_20min, FFR_25min]


train_dataloader, valid_dataloader, test_dataloader, max_speed = PrepareDataset(speed_matrix)
inputs, labels = next(iter(train_dataloader))
[batch_size, step_size, fea_size] = inputs.size()
input_dim = fea_size
hidden_dim = fea_size
output_dim = fea_size


def TrainModel(model, train_dataloader, valid_dataloader, learning_rate=1e-5, num_epochs=300, patience=10,
               min_delta=0.00001):
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    model.cpu()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    use_gpu = torch.cuda.is_available()

    interval = 100
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []

    cur_time = time.time()
    pre_time = time.time()

    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0

    for epoch in range(num_epochs):
        #         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #         print('-' * 10)

        trained_number = 0

        valid_dataloader_iter = iter(valid_dataloader)

        losses_epoch_train = []
        losses_epoch_valid = []

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            model.zero_grad()

            outputs = model(inputs)

            loss_train = loss_MSE(outputs, torch.squeeze(labels))

            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)

            optimizer.zero_grad()

            loss_train.backward()

            optimizer.step()

            # validation
            try:
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)

            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            outputs_val = model(inputs_val)

            loss_valid = loss_MSE(outputs_val, torch.squeeze(labels_val))
            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)

            # output
            trained_number += 1

        avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid) / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break

        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format( \
            epoch, \
            np.around(avg_losses_epoch_train, decimals=8), \
            np.around(avg_losses_epoch_valid, decimals=8), \
            np.around([cur_time - pre_time], decimals=2), \
            is_best_model))
        pre_time = cur_time
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]


def TestModel(model, test_dataloader, max_speed):
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.MSELoss()

    tested_batch = 0

    losses_mse = []
    losses_l1 = []

    for data in test_dataloader:
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # rnn.loop()
        hidden = model.initHidden(batch_size)

        outputs = None
        outputs = model(inputs)

        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        loss_mse = loss_MSE(outputs, torch.squeeze(labels))
        loss_l1 = loss_L1(outputs, torch.squeeze(labels))

        losses_mse.append(loss_mse.cpu().data.numpy())
        losses_l1.append(loss_l1.cpu().data.numpy())

        tested_batch += 1

        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                tested_batch * batch_size, \
                np.around([loss_l1.data[0]], decimals=8), \
                np.around([loss_mse.data[0]], decimals=8), \
                np.around([cur_time - pre_time], decimals=8)))
            pre_time = cur_time
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed

    print('Tested: L1_mean: {}, L1_std : {}'.format(mean_l1, std_l1))
    return [losses_l1, losses_mse, mean_l1, std_l1]


class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, output_last=True):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(LSTM, self).__init__()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

        self.output_last = output_last

    def step(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        return Hidden_State, Cell_State

    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)

        if self.output_last:
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)
            return Hidden_State
        else:
            outputs = None
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)
                if outputs is None:
                    outputs = Hidden_State.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State


lstm = LSTM(input_dim, hidden_dim, output_dim, output_last = True)
lstm, lstm_loss = TrainModel(lstm, train_dataloader, valid_dataloader, num_epochs = 1)
lstm_test = TestModel(lstm, test_dataloader, max_speed )