import argparse
import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.models import vit_b_16
import dill
import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from TCN import TemporalConvNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from iTransformer import iTransformer
import warnings
warnings.filterwarnings('ignore')
class Music_Data(Dataset):
    def __init__(self,data_30,data_3 ):
        # data_30=data_30.values
        self.features_30=np.array(data_30[:,0:-3,None])
        self.labels=data_30[:,-1]
        self.features_3= data_3[:,:-2].reshape(data_3.shape[0],57,10)
        if data_30[:,-2].all()==data_3[:,-2].all() and data_30[:,-1].all()==data_3[:,-1].all():
            print('Split correct')
        else:
            print('Caution for data split')

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        x_30=self.features_30[idx,:].astype(np.float32)
        y=self.labels[idx]
        x_3=self.features_3[idx,:,:].astype(np.float32)
        x_3=x_3.transpose(1,0)
        return x_30, x_3, y


class CNN(nn.Module):
    def __init__(self, num_classes,dim_input=13,dim_hidden=128,dim_output=128,kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(dim_input, dim_hidden, kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = nn.Conv1d(dim_hidden, dim_hidden, kernel_size=kernel_size, stride=1, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128*323, dim_output)
        # self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 64 * 16 * 16)
        x=x.flatten(1)
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout,length_MC=1):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(num_channels[-1]*length_MC, 64)
        self.rl= nn.ReLU()
        self.linear2 = nn.Linear(64, output_size)
        self.sof = nn.Softmax()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x)
        # output=output[:,-1,:]
        output = self.rl(self.linear1(output.flatten(1)))
        output = self.linear2(output).double()
        return output
class CombinedModel(nn.Module):
    def __init__(self, cnn_model, vit_model, num_classes):
        super(CombinedModel, self).__init__()
        self.cnn_model = cnn_model
        self.vit_model = vit_model
        self.fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, mfcc, image):
        cnn_features = self.cnn_model(mfcc)
        vit_features = self.vit_model(image)
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        output = self.fc(combined_features)
        return output
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = True  # GPU、网络结构固定，可设置为True
    # torch.backends.cudnn.deterministic = True  # 固定网络结构
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # path = os.path.join(self.save_path, 'best_network.pth')
        path = self.save_path
        torch.save(model, path, pickle_module=dill)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def train(data, model, criterion, optm, batch_size=64, device=torch.device("cuda:0")):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for x1,x2, y in tqdm(data,desc=f"Epoch {epoch + 1}/{epochs} - Training"):
        model.zero_grad()
        input1,input2, labels = x1.to(device),x2.to(device), y.to(device)
        optm.zero_grad()
        outputs = model(input2)
        loss = criterion(outputs, labels)
        loss.backward()
        optm.step()
        soft=nn.Softmax()
        outputs=soft(outputs)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * input1.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(data.dataset)
    epoch_acc = running_corrects / len(data.dataset)
    return epoch_loss, epoch_acc

def evaluate(data, model, batch_size=64, device=torch.device("cuda:0")):
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    all_preds = []
    all_labels = []
    for x1,x2, y in tqdm(data):
        model.zero_grad()
        with torch.no_grad():
            input1,input2, labels = x1.to(device),x2.to(device), y.to(device)
            # optm.zero_grad()
            outputs = model(input2)
            loss = criterion(outputs, labels)
            # loss.backward()
            # optm.step()
            soft=nn.Softmax()
            outputs=soft(outputs)
            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * input1.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    epoch_loss = val_running_loss / len(data.dataset)
    epoch_acc = val_running_corrects / len(data.dataset)
    return epoch_loss, epoch_acc, [precision, recall, f1]

def proceess_data(df_30,df):
    df['prefix'] = df['filename'].apply(lambda x: x.rsplit('.', 2)[0])
    # 动态生成聚合规则
    agg_dict = {col: list for col in df.columns if col not in [ 'filename','length', 'prefix']}#
    agg_dict['length'] = 'first'  # 假设每组的length都相同，取第一个值
    agg_dict['label'] = 'first'
    # 按前缀分组并合并数据
    result = df.groupby('prefix').agg(agg_dict).reset_index()
    encoder_file = LabelEncoder()
    encoder_label = LabelEncoder()
    # 对每个字符串列进行编码，并直接覆盖原始列
    result['file_encode'] = encoder_file.fit_transform(result['prefix'])
    result['label_encode']  = encoder_label.fit_transform(result['label'])

    df_30['file_encode'] = encoder_file.transform(df_30['filename'])
    df_30['label_encode']  = encoder_label.transform(df_30['label'])
    def str_to_list(s):
        return [s] * 10
    # 应用这个函数到第一列
    # result['label_encode'] = result['label_encode'].apply(str_to_list)
    # result['file_encode'] = result['file_encode'].apply(str_to_list)
    for i in range(1000):
        if len(result.iloc[i, 2])<10:
            for j in range(1,len(result.iloc[i, 0:-4])):
                result.iloc[i, j].append(np.average(result.iloc[i, j]))
    result=result.drop(columns=['length','prefix'])
    df_30=df_30.drop(columns=['length','filename'])
    # tmp=np.array(result.iloc[:,0:].values)
    # tmp = [num for sublist in tmp for num in sublist]
    # tmp= np.array(tmp).reshape(result.values.shape[0], 59, 10)
    return df_30,result

if __name__ == "__main__":
    seeds = 42
    same_seeds(seeds)
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument('--data_dir', type=str, default='./dataset', help='数据集的路径')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    # transform = ToTensor()
    # train_dataset = AudioDataset(folder='dataset/gtzan/train', transform=transform)
    # valid_dataset = AudioDataset(folder='dataset/gtzan/validation', transform=transform)
    # test_dataset = AudioDataset(folder='dataset/gtzan/test', transform=transform)

    data_30=pd.read_csv('data_music/features_30_sec.csv')
    data_30['filename'] = [col[:-4] if col.endswith('.wav') else col for col in data_30['filename'] ]
    # data_30=data_30.drop(columns=['length'])
    data_3=pd.read_csv('data_music/features_3_sec.csv')
    data_30,data_3=proceess_data(data_30,data_3)
    # tmp = [num for sublist in data_3[:,1:-1] for num in sublist]
    # tmp= np.array(tmp).reshape(tmp.shape[0], 57, 10)
    # data_3=pd.DataFrame()
    # print(data_3.shape,type(data_3))
    # a=data_3[0,1]
    # print(type(data_3[0,1]),type(a[0]))
    # dataset = Music_Data(data_30,data_3)
    #
    # # Split dataset into train, valid, test sets
    # train_size = int(0.7 * len(dataset))
    # valid_size = int(0.15 * len(dataset))
    # test_size = len(dataset) - train_size - valid_size

    def split_data(df, label_column, samples_per_class, train_size=0.70, val_size=0.15):
        train_list = []
        val_list = []
        test_list = []

        # 分组并按比例划分
        for label_value, group in df.groupby(label_column):
            if len(group) >= samples_per_class:
                group = group.sample(n=samples_per_class, random_state=42)
                train, temp = train_test_split(group, test_size=30, random_state=42)
                val, test = train_test_split(temp, test_size=(val_size/(1.0-train_size)), random_state=42)
                train_list.append(train)
                val_list.append(val)
                test_list.append(test)

        train_df = pd.concat(train_list).reset_index(drop=True)
        val_df = pd.concat(val_list).reset_index(drop=True)
        test_df = pd.concat(test_list).reset_index(drop=True)

        return train_df.values, val_df.values, test_df.values

    train_df_30, val_df_30, test_df_30 = split_data(data_30, 'label', 100)
    scaler_30 = StandardScaler()
    train_df_30[:,0:-3] = scaler_30 .fit_transform(train_df_30[:,0:-3])
    val_df_30[:,0:-3] = scaler_30 .transform(val_df_30[:,0:-3])
    test_df_30[:,0:-3] = scaler_30 .transform(test_df_30[:,0:-3])
    # X_test = scaler.transform(X_test)

    train_df_3, val_df_3, test_df_3= split_data(data_3, 'label', 100)
    tmp = [num for sublist in train_df_3[:,0:-3] for num in sublist]
    train_x3= np.array(tmp).reshape(train_df_3[:,0:-3].shape[0], -1)

    # tmp_x= np.array(tmp).reshape(train_df_3[:,0:-3].shape[0], -1)
    # tmp_x=tmp_x.reshape(train_df_3[:,0:-3].shape[0],57,-1)
    # if train_x.all()==tmp_x.all():
    #     print('Split properly')
    # else:
    #     print("Error")
    tmp = [num for sublist in val_df_3[:,0:-3] for num in sublist]
    val_x3= np.array(tmp).reshape(val_df_3[:,0:-3].shape[0], -1)
    tmp = [num for sublist in test_df_3[:,0:-3] for num in sublist]
    test_x3= np.array(tmp).reshape(test_df_3[:,0:-3].shape[0], -1)
    scaler_3 = StandardScaler()
    train_x3 = scaler_3 .fit_transform(train_x3)
    val_x3 = scaler_3 .transform(val_x3)
    test_x3 = scaler_3 .transform(test_x3)
    train_x3=np.concatenate((train_x3,train_df_3[:,-2:]),axis=1)
    val_x3=np.concatenate((val_x3,val_df_3[:,-2:]),axis=1)
    test_x3=np.concatenate((test_x3,test_df_3[:,-2:]),axis=1)
    # if train_df_30[:,-2].all()==train_df_3[:,-2].all() and val_df_30[:,-2].all()==val_df_3[:,-2].all() and test_df_30[:,-2].all()==test_df_3[:,-2].all():
    #     print("数据集划分正确")
    # else:
    #     print('Error')

    train_dataset=Music_Data(train_df_30,train_x3)
    valid_dataset=Music_Data(val_df_30,val_x3)
    test_dataset=Music_Data(test_df_30,test_x3)

    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 10

    model = iTransformer(
    num_variates = 57,
    lookback_len = 10,                  # or the lookback length in the paper
    dim = 128,                          # model dimensions
    depth = 6,                          # depth
    heads = 8,                          # attention heads
    dim_head = 64,                      # head dimension
    pred_length = (15),                 # can be one prediction, or many
    num_class=10,    
    num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
    use_reversible_instance_norm = True # use reversible instance normalization, proposed here https://openreview.net/forum?id=cGDAkQo1C0p . may be redundant given the layernorms within iTransformer (and whatever else attention learns emergently on the first layer, prior to the first layernorm). if i come across some time, i'll gather up all the statistics across variates, project them, and condition the transformer a bit further. that makes more sense
    ).to(device)
    # model = TCN(input_size=57, num_channels=[64, 64, 64],output_size=10, kernel_size=3, dropout=0.2,length_MC=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optm = optim.Adam(model.parameters(), lr=learning_rate)
    optm_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optm, mode="min", factor=0.5, patience=4, verbose=True
    )
    model_name = "iTransformer"
    model_save = f"model_save/{model_name}.pt"
    train_losses, valid_losses = [], []
    earlystopping = EarlyStopping(model_save, patience=20, delta=0.0001)

    need_train = True
    # need_train = False

    if need_train:
        try:
            for epoch in range(epochs):
                time_start = time.time()
                train_loss, train_acc = train(
                    data=train_loader,
                    model=model,
                    criterion=criterion,
                    optm=optm,
                    batch_size=batch_size,
                )
                valid_loss, valid_acc, _ = evaluate(
                    data=valid_loader, model=model, batch_size=batch_size
                )
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                optm_schedule.step(-valid_acc)
                earlystopping(-valid_acc, model)  # 保存验证集最优模型
                print(
                    "\n{}:| end of epoch {:3d} | time: {:5.2f}s |\n Loss_train {:5.4f} | Acc_train {:5.4f} \n| Loss_valid {:5.4f} | Acc_valid {:5.4f}| lr {:5.4f}".format(
                        model_name,
                        epoch,
                        (time.time() - time_start),
                        train_loss,
                        train_acc,
                        valid_loss,
                        valid_acc,
                        optm.state_dict()["param_groups"][0]["lr"],
                    ),
                    flush=True,
                )
                if earlystopping.early_stop:
                    print("Early stopping")
                    break  # 跳出迭代，结束训练
        except KeyboardInterrupt:
            print("Training interrupted by user")
        plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
        plt.plot(np.arange(len(valid_losses)), valid_losses, label="valid rmse")
        plt.legend()  # 显示图例
        plt.xlabel("epoches")
        # plt.ylabel("epoch")
        plt.title("Train_loss&Valid_loss")
        plt.show()
with open(model_save, "rb") as f:
    model = torch.load(f, pickle_module=dill)
model = model.to(device)
test_loss, test_acc, metrics_list = evaluate(
    data=test_loader, model=model, batch_size=batch_size
)
print(
    "{}: \n| ACC_test {:5.4f}| Pre_test {:5.4f}| "
    "Recall_test {:5.4f}| F1_test {:5.4f}| ".format(
        model_name, test_acc, metrics_list[0], metrics_list[1], metrics_list[2]
    )
)
