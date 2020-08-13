import torch
import torch.optim as optim
from torch.utils import data
import numpy as np
from torch import nn
import os

path_prefix = 'resource'


def load_training_data(path="training_label.txt"):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(' ') for line in lines]
    if "training_label" in path:
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        return lines


def load_testing_data(path="testing_data"):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        x = ["".join(line.strip('\n').split(",")[1:]) for line in lines[1:]]
        x = [sen.split(" ") for sen in x]
    return x


def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    return torch.sum(torch.eq(outputs, labels)).item()


# NLP基础，将句子数字化(编码)，此处是最简单的用词频数编码，不会区分句子中词的顺序，可用word embedding优化
class BOW():
    def __init__(self, max_len=10000):
        self.wordfreq = {}
        self.vector_size = max_len
        self.word2idx = {}

    # 将一个句子编码成向量，通过前面出现比较多的高频词的频率代替一个句子
    def bow(self, train_sentences, test_sentences):
        for sentence in train_sentences + test_sentences:
            for word in sentence:
                if word in self.wordfreq.keys():
                    self.wordfreq[word] += 1
                else:
                    self.wordfreq[word] = 1
        self.wordfreq = sorted(self.wordfreq.items(), key=lambda x: x[1], reverse=True)
        if self.vector_size > len(self.wordfreq):
            self.vector_size = len(self.wordfreq)
        # 只编码前面比较高频的词
        for idx, (word, freq) in enumerate(self.wordfreq):
            if idx == self.vector_size:
                break
            self.word2idx[word] = len(self.word2idx)
        self.train_bow_list = np.zeros((len(train_sentences), self.vector_size))
        for idx, sentence in enumerate(train_sentences):
            for word in sentence:
                if word in self.word2idx.keys():
                    self.train_bow_list[idx][self.word2idx[word]] += 1
        return torch.FloatTensor(self.train_bow_list)


class TwitterDataset(data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class LSTM_Net(nn.Module):
    def __init__(self, embedding_dim):
        super(LSTM_Net, self).__init__()
        #  此处是简单的DNN，非LSTM
        self.classifier = nn.Sequential(nn.Linear(embedding_dim, 512),
                                        nn.Linear(512, 128),
                                        nn.Linear(128, 1),
                                        nn.Sigmoid())

    def forward(self, inputs):
        x = self.classifier(inputs.float())
        return x


def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    model.train()  # 將model的模式設為train，這樣optimizer就可以更新model的參數
    criterion = nn.BCELoss()  # 定義損失函數，這裡我們使用binary cross entropy loss
    t_len = len(train)
    v_len = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 將模型的參數給optimizer，並給予適當的learning rate
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 這段做training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)  # device為"cuda"，將inputs轉成torch.cuda.LongTensor
            labels = labels.to(device,
                               dtype=torch.float)  # device為"cuda"，將labels轉成torch.cuda.FloatTensor，因為等等要餵進criterion，所以型態要是float
            optimizer.zero_grad()  # 由於loss.backward()的gradient會累加，所以每次餵完一個batch後需要歸零
            outputs = model(inputs)  # 將input餵給模型
            outputs = outputs.squeeze()  # 去掉最外面的dimension，好讓outputs可以餵進criterion()
            loss = criterion(outputs, labels)  # 計算此時模型的training loss
            loss.backward()  # 算loss的gradient
            optimizer.step()  # 更新訓練模型的參數
            correct = evaluation(outputs, labels)  # 計算此時模型的training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                epoch + 1, i + 1, t_len, loss.item(), correct * 100 / batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_len, total_acc / t_len * 100))

        # 這段做validation
        model.eval()  # 將model的模式設為eval，這樣model的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)  # device為"cuda"，將inputs轉成torch.cuda.LongTensor
                labels = labels.to(device,
                                   dtype=torch.float)  # device為"cuda"，將labels轉成torch.cuda.FloatTensor，因為等等要餵進criterion，所以型態要是float
                outputs = model(inputs)  # 將input餵給模型
                outputs = outputs.squeeze()  # 去掉最外面的dimension，好讓outputs可以餵進criterion()
                loss = criterion(outputs, labels)  # 計算此時模型的validation loss
                correct = evaluation(outputs, labels)  # 計算此時模型的validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_len, total_acc / v_len * 100))
            if total_acc > best_acc:
                # 如果validation的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                # torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_len*100))
                torch.save(model, "{}/ckpt_bow".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc / v_len * 100))
        print('-----------------------------------------------')
        model.train()  # 將model的模式設為train，這樣optimizer就可以更新model的參數（因為剛剛轉成eval模式）


def main():
    path_prefix = 'resource'

    # 通過torch.cuda.is_available()的回傳值進行判斷是否有使用GPU的環境，如果有的話device就設為"cuda"，沒有的話就設為"cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 處理好各個data的路徑
    train_with_label = os.path.join(path_prefix, 'training_label.txt')
    testing_data = os.path.join(path_prefix, 'testing_data.txt')

    # 定義句子長度、要不要固定embedding、batch大小、要訓練幾個epoch、learning rate的值、model的資料夾路徑
    batch_size = 512
    epoch = 15
    lr = 0.001
    model_dir = path_prefix  # model directory for checkpoint model

    train_x, y = load_training_data(train_with_label)
    test_x = load_testing_data(testing_data)

    # 對input跟labels做預處理
    max_len = 1200
    b = BOW(max_len=max_len)
    train_x = b.bow(train_x, test_x)
    # import pdb
    # pdb.set_trace()
    y = [int(label) for label in y]
    y = torch.LongTensor(y)

    # 製作一個model的對象
    model = LSTM_Net(embedding_dim=max_len)
    model = model.to(device)  # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)

    # 把data分為training data跟validation data(將一部份training data拿去當作validation data)
    X_train, X_val, y_train, y_val = train_x[:190000], train_x[190000:], y[:190000], y[190000:]

    # 把data做成dataset供dataloader取用
    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    # 把data 轉成 batch of tensors
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=8)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=8)

    # 開始訓練
    training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)


if __name__ == '__main__':
    main()
