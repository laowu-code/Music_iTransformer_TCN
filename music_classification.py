import argparse
import os
import time
from torchvision.models import vit_b_16
import dill
import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from vit_pytorch import ViT
# class AudioDataset(Dataset):
#     def __init__(self, folder, transform=None):
#         self.folder = folder
#         self.transform = transform
#         self.files = [os.path.join(folder, f) for f in os.listdir(folder)]
#         self.labels = [f.split('.')[0] for f in os.listdir(folder)]
#         self.label_to_idx = {label: index for index, label in enumerate(set(self.labels))}
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         file_path = self.files[idx]
#         label = self.labels[idx]
#         y, sr = librosa.load(file_path, sr=None)
#         mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#         log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
#         if self.transform:
#             log_mel_spectrogram = self.transform(log_mel_spectrogram)
#
#         label_idx = self.label_to_idx[label]
#         return log_mel_spectrogram, label_idx
class AudioDataset(Dataset):
    def __init__(self, folder, transform=None, target_length=30):
        self.folder = folder
        self.transform = transform
        self.files = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.labels = [f.split('.')[0] for f in os.listdir(folder)]
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.target_length = target_length * 22050  # Assuming 22050 Hz sample rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        y, sr = librosa.load(file_path, sr=22050)  # Ensure consistent sample rate
        y = self._pad_or_trim(y, self.target_length)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        if self.transform:
            log_mel_spectrogram = self.transform(log_mel_spectrogram)

        label_idx = self.label_to_idx[label]
        return log_mel_spectrogram, label_idx

    def _pad_or_trim(self, y, target_length):
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')
        return y

class MusicDataset(Dataset):
    def __init__(self, audio_folder, image_folder, transform=None, target_length=30):
        self.audio_folder = audio_folder
        self.image_folder = image_folder
        self.transform = transform
        self.audio_files = []
        self.labels = []
        self._load_files()
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.target_length = target_length * 22050  # Assuming 22050 Hz sample rate

    def _load_files(self):
        for label in os.listdir(self.audio_folder):
            audio_dir = os.path.join(self.audio_folder, label)
            image_dir = os.path.join(self.image_folder, label)
            for audio_file in os.listdir(audio_dir):
                self.audio_files.append(os.path.join(audio_dir, audio_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        image_path = os.path.join(self.image_folder, label, os.path.basename(audio_path).replace('.wav', '.png').replace('.','',1))
        # Load and preprocess audio
        y, sr = librosa.load(audio_path, sr=22050)
        y = self._pad_or_trim(y, self.target_length)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # mfcc = np.expand_dims(mfcc, axis=0)

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label_idx = self.label_to_idx[label]
        return mfcc, image, label_idx

    def _pad_or_trim(self, y, target_length):
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')
        return y



class Transformer(nn.Module):
    def __init__(self, num_classes, input_dim=128, nhead=8, num_encoder_layers=3):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(input_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        # self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
        patch_size, dim_vit, depth, heads, mlp_dim = 32, 128, 6, 6, 256
        self.vit = ViT(
            image_size=256,
            patch_size=patch_size,
            num_classes=10,
            dim=dim_vit,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.2,
            emb_dropout=0.2
        ).to(device)
        # self.vit.heads = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

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
# class CombinedModel(nn.Module):
#     def __init__(self, cnn_model, transformer_model, num_classes):
#         super(CombinedModel, self).__init__()
#         self.cnn_model = cnn_model
#         self.transformer_model = transformer_model
#         self.fc = nn.Linear(num_classes * 2, num_classes)
#
#     def forward(self, x):
#         cnn_features = self.cnn_model(x.unsqueeze(1))  # Adding channel dimension for CNN
#         transformer_features = self.transformer_model(x)
#         combined_features = torch.cat((cnn_features, transformer_features), dim=1)
#         output = self.fc(combined_features)
#         return output

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
        outputs = model(input1, input2)
        loss = criterion(outputs, labels)
        loss.backward()
        optm.step()
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
            outputs = model(input1, input2)
            loss = criterion(outputs, labels)
            # loss.backward()
            # optm.step()
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

# for epoch in range(num_epochs):
#     combined_model.train()
#     train_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = combined_model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item() * inputs.size(0)
#
#     train_loss /= len(train_loader.dataset)
#
#     combined_model.eval()
#     valid_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in valid_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = combined_model(inputs)
#             loss = criterion(outputs, labels)
#             valid_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     valid_loss /= len(valid_loader.dataset)
#     valid_accuracy = correct / total
#
#     print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    audio_folder = 'dataset/gtzan_10/genres_original'
    image_folder = 'dataset/gtzan_10/images_original'
    dataset = MusicDataset(audio_folder, image_folder, transform=transform)

    # Split dataset into train, valid, test sets
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 10
    cnn_model = CNN(num_classes)
    # transformer_model = Transformer(num_classes)
    vit_model=ViTModel(num_classes)

    # patch_size, dim_vit, depth, heads, mlp_dim = 32, 256, 3, 8, 512
    # model = ViT(
    #     image_size=256,
    #     patch_size=patch_size,
    #     num_classes=num_classes,
    #     dim=dim_vit,
    #     depth=depth,
    #     heads=heads,
    #     mlp_dim=mlp_dim,
    #     dropout=0.2,
    #     emb_dropout=0.2
    # ).to(device)

    model = CombinedModel(cnn_model, vit_model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optm = optim.Adam(model.parameters(), lr=learning_rate)
    optm_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optm, mode="min", factor=0.5, patience=4, verbose=True
    )
    model_name = "CNN_ViT"
    model_save = f"model_save/{model_name}.pt"
    train_losses, valid_losses = [], []
    earlystopping = EarlyStopping(model_save, patience=20, delta=0.0001)

    # need_train = True
    need_train = False

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
