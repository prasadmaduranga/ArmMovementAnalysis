# model input : hand and body feature including angles and distant measures
# feature vector size : (2 hands x (15 angles + 6 distance measures)) + (6 body angles + 2 body measures)
# total feature vector size : 56
import pandas
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
from sklearn.metrics import confusion_matrix, accuracy_score
import src.util.plot_util as plot_util
from torch_geometric.nn import GCNConv, ChebConv
import gsom
import datetime
from torch_geometric.data import Data, Batch
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

OUTPUT_CLASSES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
data_file = '../data/video_graph_features_0.csv'
meta_data_file = '../data/sign_lang_meta_data.csv'
model_archive_file = '../models/ChebConvLSTM_userclassfication_400sample_2023_02_20_V1.pt'

feature_encode_file = '../data/encoded_features_fclayer_single_sign.csv'
gsom_plot_file = "../results/gsom_user_classification_traindata_plot_single_sign_train_100_smooth_50_sf_0_86_{}".format(
    datetime.datetime.now().strftime("%H_%M_%S"))
gsom_mappoints_file = "../results/gsom_out_user_classification_traindata_single_sign_train_100_smooth_50_sf_0_86_{}.csv".format(
    datetime.datetime.now().strftime("%H_%M_%S"))

edge_index = torch.tensor(
    [[0, 1, 2, 3, 4, 3, 2, 1, 0, 5, 6, 7, 8, 7, 6, 5, 0, 9, 10, 11, 12, 11, 10, 9, 0, 13, 14, 15, 16, 15, 14, 13, 0, 17,
      18, 19, 20, 19, 18, 17],
     [1, 2, 3, 4, 3, 2, 1, 0, 5, 6, 7, 8, 7, 6, 5, 0, 9, 10, 11, 12, 11, 10, 9, 0, 13, 14, 15, 16, 15, 14, 13, 0, 17,
      18, 19, 20, 19, 18, 17, 0]],
    dtype=torch.long)
node_feature_in = 4
node_feature_out = 4
df = pandas.DataFrame()

if __name__ == "__main__":
    df = pd.read_csv(data_file)
    meta_data = pd.read_csv(meta_data_file)

    # Select only right handed videos
    right_handed_sign_list = meta_data.query("hand == 'r'")['id']
    df = df.query("SIGN in @right_handed_sign_list & SIGN == 5")

    # Select right hand data columns + video id + Sign(output class)
    df = df.iloc[:, np.r_[0:2, 91:175]].dropna()


def PrepareDataset(df, BATCH_SIZE=2, seq_len=84, train_propotion=0.7, valid_propotion=0.2):
    # ['VIDEO_ID', 'USER_ID', 'ITERATION', 'FRAME_SQ_NUMBER', 'TIMESTAMP', 'CURRENT_POS_AVI_RATIO', 'HANDEDNESS','-- FEATURE DATA -- ','SIGN']

    for col in df.columns[2:]:
        if df[col].mean() == 0 and df[col].std() == 0:
            continue
        df[col] = (df[col] - df[col].mean()) / (df[col].std())

    frame_feature_sequences, video_labels = [], []
    max = df.groupby('VIDEO_ID').count().max().max()
    df['USER_ID'] = df['USER_ID'].astype('int') - 1

    for video_id in df['VIDEO_ID'].unique():
        frames = df.query("VIDEO_ID == @video_id").iloc[:, 2:]
        user = df.query("VIDEO_ID == @video_id")['USER_ID'].values[0]
        pad_count = max - len(frames)

        frame_feature_sequences.append(np.pad(frames.values, [(pad_count, 0), (0, 0)], 'constant', constant_values=0))

        video_labels.append(user)

    frame_feature_sequences, video_labels = np.asarray(frame_feature_sequences), np.asarray(video_labels)

    # shuffle the input
    permutation = np.random.permutation(frame_feature_sequences.shape[0])
    frame_feature_sequences = frame_feature_sequences[permutation]
    video_labels = video_labels[permutation]

    sample_size = frame_feature_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = frame_feature_sequences[:train_index], video_labels[:train_index]
    valid_data, valid_label = frame_feature_sequences[train_index:valid_index], video_labels[train_index:valid_index]
    test_data, test_label = frame_feature_sequences[valid_index:], video_labels[valid_index:]
    all_data, all_label = frame_feature_sequences[:], video_labels[:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    all_data, all_label = torch.Tensor(all_data), torch.Tensor(all_label)

    ndarray = np.array(train_data)
    torch.from_numpy(ndarray)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    all_dataset = utils.TensorDataset(all_data, all_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    all_dataloader = utils.DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, all_dataloader


def TrainModel(model, train_dataloader, valid_dataloader, learning_rate=1e-5, num_epochs=300, patience=10,
               min_delta=0.00001):
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    model.cpu()

    loss_crossEntropy = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    use_gpu = torch.cuda.is_available()

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

            loss_train = loss_crossEntropy(outputs, labels.long())
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

            loss_valid = loss_crossEntropy(outputs_val, torch.squeeze(labels_val.long()))
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


def TestModel(model, test_dataloader, midgetter=None):
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()

    tested_batch = 0

    losses_CE = []
    y_true = []
    y_pred = []
    encoded_feats = []

    for data in test_dataloader:
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        hidden = model.initHidden(batch_size)

        outputs = model(inputs)
        if midgetter:
            mid_outputs, outputs = mid_getter(inputs)
            encoded_feats.extend(torch.cat([mid_outputs['fc'], labels.long().unsqueeze(dim=1)], dim=1).tolist())
        else:
            outputs = model(inputs)

        loss_crossEntropy = torch.nn.CrossEntropyLoss()
        loss_CE = loss_crossEntropy(outputs, labels.long())
        losses_CE.append(loss_CE.cpu().data.numpy())

        tested_batch += 1

        _, preds = torch.max(outputs, dim=1)
        acc = torch.sum(preds == labels).float() / labels.size(0)

        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

        print('btach accuracy {} , batch id {}'.format(acc, (tested_batch - 1)))
        if tested_batch % 10 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {} ,  time: {}'.format( \
                tested_batch * batch_size, \
                np.around([loss_CE.data], decimals=8), \
                np.around([cur_time - pre_time], decimals=8)))
            pre_time = cur_time

    losses_CE = np.array(losses_CE)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print('Tested: Cross Entropy losses: {} '.format(losses_CE))
    print('accuracy_score {}'.format(acc))
    print('confusion_matrix {}'.format(cm))

    plot_util.plot_confusion_matrix(cm, classes=[], title='Confusion matrix')
    plt.show()

    return [losses_CE], encoded_feats


class GCN_LSTM_Classifier(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, output_last=True, output_classes=1):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(GCN_LSTM_Classifier, self).__init__()

        self.cheb_conv = ChebConv(in_channels=node_feature_in, out_channels=node_feature_out, node_dim=1, K=2)
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_classes)
        self.softmax = nn.Softmax(dim=1)

        self.output_last = output_last
        self.output_classes = output_classes

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
        batch_size = inputs.size(0)  # number of videos
        time_step = inputs.size(1)  # number of frames in the video
        Hidden_State, Cell_State = self.initHidden(batch_size)

        if self.output_last:
            for i in range(time_step):
                frame_features = torch.squeeze(inputs[:, i:i + 1, :])

                if ((frame_features == 0).all()):
                    continue
                node_features = frame_features.reshape(batch_size,
                                                       int(frame_features.shape[1] / node_feature_in),
                                                       node_feature_in)
                output_graph = self.cheb_conv(node_features, edge_index)
                step_tensor = output_graph.reshape(batch_size, output_graph.shape[1] * node_feature_out)
                Hidden_State, Cell_State = self.step(step_tensor, Hidden_State, Cell_State)

            out = self.fc(Hidden_State)
            return self.softmax(torch.squeeze(out))

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


def ApplyGSOM(df, spredFactor=0.83, featureSpace=1):
    # featureSpace: number of dimensions in the feature vector
    np.random.seed(1)
    data_training = df.iloc[:, :-1]
    gsom_map = gsom.GSOM(spredFactor, featureSpace, max_radius=2)
    # 0.7 is good spread factor value
    gsom_map.fit(data_training.to_numpy(), 100, 50)
    map_points = gsom_map.predict(df, "user_id")
    gsom.plot(map_points, "user_id", gsom_map=gsom_map, file_name=gsom_plot_file)
    map_points.to_csv(gsom_mappoints_file, index=False)

    print("gsom complete")


# to load from a saved params, set train_model to False
train_model = False
skip_test = False

train_dataloader, valid_dataloader, test_dataloader, all_dataloader = PrepareDataset(df)
inputs, labels = next(iter(train_dataloader))
[batch_size, step_size, fea_size] = inputs.size()
input_dim = fea_size
hidden_dim = fea_size
output_dim = fea_size

# LSTM model with angle and distance measures
lstm = GCN_LSTM_Classifier(input_dim, hidden_dim, output_dim, output_last=True, output_classes=len(OUTPUT_CLASSES))

if train_model:
    lstm, lstm_loss = TrainModel(lstm, train_dataloader, valid_dataloader, num_epochs=200)
    torch.save(lstm.state_dict(), model_archive_file)
else:
    lstm.load_state_dict(torch.load(model_archive_file))

mid_getter = MidGetter(lstm, return_layers={'fc': 'fc'}, keep_output=True)
# lstm_test, encoded_feats = TestModel(lstm, test_dataloader, mid_getter)

if skip_test:
    encoded_feats = pd.read_csv(feature_encode_file)
else:
   lstm_test, encoded_feats = TestModel(lstm, all_dataloader, mid_getter)
   encoded_feats = pd.DataFrame(encoded_feats,
                             columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'user_id'])
   encoded_feats.to_csv(feature_encode_file,index=False)

ApplyGSOM(encoded_feats, spredFactor=0.83, featureSpace=10)

# lstm_test = TestModel(lstm, test_dataloader)
