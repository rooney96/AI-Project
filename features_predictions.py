import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

BATCH_SIZE = 16
EPOCHS = 800

df_features = pd.read_excel('feature_prediction_data\\selected_feature.xlsx')
df_labels = pd.read_excel('feature_prediction_data\\LastThreeSeasons.xlsx')

le = LabelEncoder()
df_features = df_features.dropna()
df_labels = df_labels.dropna()


def min_max(val, old_min, old_max, new_min, new_max):
    old_diff = old_max - old_min
    new_diff = new_max - new_min
    new_val = (val - old_min) * (new_diff / old_diff) + new_min

    return new_val


def z_score(val, mean, std):
    score = (val - mean) / std

    return score


class DataNormalizer:
    def __init__(self):
        self.data = None

    def __call__(self, data):
        self.data = data
        self.fix_numerical_distributions()  # fix the distributions of the numerical data
        return self.data

    def fix_numerical_distributions(self):
        normalize_by_min_max = ['avg3_home_red_cards', 'avg3_away_red_cards', 'last3_home_points_earned',
                                'last3_away_points_earned', 'avg_direct_home_goals_scored',
                                'avg_direct_away_goals_scored', 'avg_direct_home_red_cards',
                                'avg_direct_away_red_cards', 'direct_home_points', 'direct_away_points',
                                'HY', 'AY', 'HR', 'AR']
        normalize_by_z_score = ['avg3_home_goals_scored', 'avg3_away_goals_scored',
                                'avg3_home_goals_conceded', 'avg3_away_goals_conceded',
                                'avg3_home_shots', 'avg3_away_shots', 'avg3_home_fouls',
                                'avg3_away_fouls', 'avg3_home_corners', 'avg3_away_corners',
                                'avg3_home_yellow_cards', 'avg3_away_yellow_cards',
                                'avg_home_att_pac', 'avg_away_att_pac', 'avg_home_def_pac',
                                'avg_away_def_pac', 'avg_home_phy', 'avg_away_phy',
                                'home_att_away_def_diff', 'away_att_home_def_diff', 'home_away_mid_diff',
                                'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']

        self.min_max_transform(normalize_by_min_max)
        self.z_score_transform(normalize_by_z_score)

        return self

    def min_max_transform(self, features):

        for feature in features:
            if feature not in self.data.columns.values:
                continue
            old_min, old_max = self.data.iloc[:, :][feature].min(), \
                               self.data.iloc[:, :][feature].max()
            self.data[feature] = self.data[feature].apply(lambda v: min_max(v, old_min, old_max, 0, 1))

        return self

    def z_score_transform(self, features):

        for feature in features:
            if feature not in self.data.columns.values:
                continue
            cur_mean, cur_std = self.data.iloc[:, :][feature].mean(), \
                                self.data.iloc[:, :][feature].std()
            self.data[feature] = self.data[feature].apply(lambda v: z_score(v, cur_mean, cur_std))

        return self

    def get_data(self):
        return self.data


distributions_normalizer = DataNormalizer()
df_features = distributions_normalizer(df_features)
df_labels = distributions_normalizer(df_labels)

features = df_features.iloc[:, :].values
labels = df_labels.iloc[:, :].values


class FootballDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode

        if self.mode == 'train':
            self.inp = features
            self.oup = labels
        else:
            self.inp = features

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        if self.mode == 'train':
            input = torch.Tensor(self.inp[idx])
            output = torch.Tensor(self.oup[idx])
            return {'inp': input,
                    'oup': output,
                    }
        else:
            inpt = torch.Tensor(self.inp[idx])
            return {'inp': input
                    }


data = FootballDataset()
data_train = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        # clamp with min = 0 is ReLu
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, D_out = 16, 31, 12

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss()


def runNN1(optimizer, model):
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in data_train:
            x_train, y_train = batch['inp'], batch['oup']
            # print(x_train)
            # print(y_train)

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x_train)

            # Compute and print loss
            loss = criterion(y_pred, y_train)
            total_loss += loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch == 0:
            print(epoch + 1, total_loss)
        if epoch % 100 == 99:
            print(epoch + 1, total_loss)


def findBestLR1():
    lrs = (0.001, 0.01, 0.05, 0.1)
    for lr in lrs:
        # Construct our model by instantiating the class defined above
        model = DynamicNet(D_in, H, D_out)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        print("lr=" + str(lr))
        runNN1(optimizer, model)


def findBestSize():
    sizes = (256, 16, 32, 64, 128)
    for s in sizes:
        # Construct our model by instantiating the class defined above
        model = DynamicNet(D_in, s, D_out)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        print("size=" + str(s))
        runNN1(optimizer, model)


input_size = 31
hidden_sizes = [128, 64]
output_size = 12
# Build a feed-forward network
criterion = torch.nn.MSELoss()


def runNN2(optimizer, model):
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in data_train:
            x_train, y_train = batch['inp'], batch['oup']
            # print(x_train)
            # print(y_train)

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x_train)

            # Compute and print loss
            loss = criterion(y_pred, y_train)
            total_loss += loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch == 0:
            print(epoch + 1, total_loss)
        if epoch % 100 == 99:
            print(epoch + 1, total_loss)


def findBestLR2():
    lrs = (0.001, 0.01, 0.05, 0.1)
    for lr in lrs:
        model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                              nn.ReLU(),
                              nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                              nn.ReLU(),
                              nn.Linear(hidden_sizes[1], output_size),
                              nn.Softmax(dim=1))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        print("lr=" + str(lr))
        runNN2(optimizer, model)


BATCH_SIZE = 16


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(31, 32)
        self.b1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 64)
        self.b2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.b3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 12)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.b1(x)
        x = self.relu(self.fc2(x))
        x = self.b2(x)
        x = self.relu(self.fc3(x))
        x = self.b3(x)
        x = F.sigmoid(self.fc4(x))

        return x


def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    return loss


criterion = nn.MSELoss()


def runNN3(optm, model):
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch in data_train:
            x_train, y_train = batch['inp'], batch['oup']
            # x_train = x_train.to(device)
            # y_train = y_train.to(device)
            loss = train(model, x_train, y_train, optm, criterion)
            epoch_loss += loss
        if epoch == 0:
            print('Epoch {} Loss : {}'.format((epoch + 1), epoch_loss))
        if epoch % 100 == 99:
            print('Epoch {} Loss : {}'.format((epoch + 1), epoch_loss))


def findBestLR3():
    lrs = (0.001, 0.01, 0.05, 0.1)
    for lr in lrs:
        model = Network()
        optm = Adam(model.parameters(), lr=lr)
        print(lr)
        runNN3(optm, model)


if __name__ == '__main__':
    findBestSize()

#
# class DynamicNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out, N):
#         super(DynamicNet, self).__init__()
#         self.N = N
#         self.input_linear = torch.nn.Linear(D_in, H)
#         self.middle_linear = torch.nn.Linear(H, H)
#         self.output_linear = torch.nn.Linear(H, D_out)
#
#     def forward(self, x):
#         # clamp with min = 0 is ReLu
#         h_relu = self.input_linear(x).clamp(min=0)
#         for _ in range(random.randint(0, 3)):
#             h_relu = self.middle_linear(h_relu).clamp(min=0)
#         y_pred = self.output_linear(h_relu)
#         return y_pred
#
#     def run(self):
#         criterion = torch.nn.MSELoss()
#         optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
#         loss_results = []
#         for epoch in range(self.N):
#             total_loss = 0
#             for batch in data_train:
#                 x_train, y_train = batch['inp'], batch['oup']
#
#                 # Forward pass: Compute predicted y by passing x to the model
#                 y_pred = self(x_train)
#
#                 # Compute and print loss
#                 loss = criterion(y_pred, y_train)
#                 total_loss += loss.item()
#
#                 # Zero gradients, perform a backward pass, and update the weights.
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             if epoch % 100 == 99:
#                 loss_results.append(total_loss)
#                 # print(loss_results)
#
#         print(loss_results)
#         print(loss_results, sorted(loss_results)[0])
#         # return self, sorted(loss_results)[0]
#
#
# class Network(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#         self.fc1 = nn.Linear(31, 32)
#         self.b1 = nn.BatchNorm1d(32)
#         self.fc2 = nn.Linear(32, 64)
#         self.b2 = nn.BatchNorm1d(64)
#         self.fc3 = nn.Linear(64, 32)
#         self.b3 = nn.BatchNorm1d(32)
#         self.fc4 = nn.Linear(32, 12)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.b1(x)
#         x = self.relu(self.fc2(x))
#         x = self.b2(x)
#         x = self.relu(self.fc3(x))
#         x = self.b3(x)
#         x = self.relu(self.fc4(x))
#
#         return x
#
#     def run(self):
#         optm = Adam(self.parameters(), lr=0.01)
#         criterion = nn.MSELoss()
#         loss_results = []
#         for epoch in range(EPOCHS):
#             epoch_loss = 0
#             for batch in data_train:
#                 x_train, y_train = batch['inp'], batch['oup']
#
#                 optm.zero_grad()
#                 output = self(x_train)
#                 loss = criterion(output, y_train)
#                 loss.backward()
#                 optm.step()
#
#                 epoch_loss += loss.item()
#             if epoch % 100 == 99:
#                 loss_results.append(epoch_loss)
#                 # print(epoch_loss)
#
#         # print(loss_results, sorted(loss_results)[0])
#         print(loss_results)
#         print(loss_results, sorted(loss_results)[0])
#         # return self, sorted(loss_results)[0]
#
#
# def run_feed_forward_network():
#     input_size = 31
#     hidden_sizes = [128, 64]
#     output_size = 12
#     # Build a feed-forward network
#     model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
#                           nn.ReLU(),
#                           nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#                           nn.ReLU(),
#                           nn.Linear(hidden_sizes[1], output_size),
#                           nn.Softmax(dim=1))
#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
#     loss_results = []
#     for epoch in range(EPOCHS):
#         total_loss = 0
#         for batch in data_train:
#             x_train, y_train = batch['inp'], batch['oup']
#
#             # Forward pass: Compute predicted y by passing x to the model
#             y_pred = model(x_train)
#
#             # Compute and print loss
#             loss = criterion(y_pred, y_train)
#             total_loss += loss.item()
#
#             # Zero gradients, perform a backward pass, and update the weights.
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         if epoch % 100 == 0:
#             loss_results.append(total_loss)
#     print(loss_results)
#     print(loss_results, sorted(loss_results)[0])
#     # print(loss_results, sorted(loss_results)[0])
#     # return model, sorted(loss_results)[0]
#
#
# def train(model, x, y, optimizer, criterion):
#     model.zero_grad()
#     output = model(x)
#     loss = criterion(output, y)
#     loss.backward()
#     optimizer.step()
#
#     return loss
#
#
# def get_best_model():
#     N, D_in, H, D_out = 800, 31, 64, 12
#     # models = [DynamicNet(D_in, H, D_out, N).run(), run_feed_forward_network(), Network().run()]
#     # DynamicNet(D_in, H, D_out, N).run()
#     # run_feed_forward_network()
#     Network().run()
#     # print loss for each model
#     # print([m[1] for m in models])
#     #
#     # models = models.sort(key=lambda m: m[1])
#     # print(models[0][1])
#     # torch.save(models[0][0], 'network_model')
#
#
# if __name__ == '__main__':
#     get_best_model()
