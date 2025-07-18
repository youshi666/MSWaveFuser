import torch
import torch.nn as nn
from exp.exp_basic import Exp_basic
from torch import optim
from tqdm import tqdm
import numpy as np
import os
from utils.metrics import evaluation_metrics

class Exp_train(Exp_basic):
    def __init__(self, args):
        super(Exp_train, self).__init__(args)
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_loader, valid_loader, save_dir):
        txt_name = f"{self.args.model_name}_{self.args.data}_result.txt"
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        for epoch in range(self.args.epochs):
            trian_loss = []

            self.model.train()
            for i, (inputs, labels) in enumerate(
                    tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.args.epochs}', ncols=100)):
                if self.args.data == "NASA":
                    inputs = inputs.permute(0, 2, 1)
                    labels = labels.unsqueeze(-1)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                trian_loss.append(loss.item())
            trian_loss = np.average(trian_loss)

            vali_loss = []
            self.model.eval()
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(valid_loader):
                    if self.args.data == "NASA":
                        inputs = inputs.permute(0, 2, 1)
                        labels = labels.unsqueeze(-1)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    vali_loss.append(loss.item())
            vali_loss = np.average(vali_loss)
            print("Epoch: {0}, Train_Loss: {1:.7f}, Vali_Loss: {2:.7f}".format(
                epoch + 1, trian_loss, vali_loss))

        torch.save(self.model.state_dict(), os.path.join(save_dir, f"{self.args.model_name}.pth"))

    def test(self, test_loader, save_dir):
        self.model.load_state_dict(torch.load(os.path.join(save_dir, f"{self.args.model_name}.pth"), weights_only=True))
        self.model.eval()

        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                if self.args.data == "NASA":
                    inputs = inputs.permute(0, 2, 1)
                    labels = labels.unsqueeze(-1)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                all_outputs.extend(outputs.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

        preds = all_outputs
        trues = all_labels
        MAE, MSE, MAPE, RMSE, R2 = evaluation_metrics(np.array(preds), np.array(trues))

        txt_name = f"{self.args.model_name}_{self.args.data}_result.txt"
        with open(os.path.join(save_dir, txt_name), "a") as f:
            if(self.args.data == "NASA1"):
                f.write(f"\t Battery:{test_loader.dataset.test_batteries}  MAE: {MAE:.7f}, MSE: {MSE:.7f}, MAPE: {MAPE:.7f}, RMSE: {RMSE:.7f}, R2: {R2:.7f}\n") #Battery:{test_loader.dataset.test_batteries}
            else:
                f.write(f"\t MAE: {MAE:.7f}, MSE: {MSE:.7f}, MAPE: {MAPE:.7f}, RMSE: {RMSE:.7f}, R2: {R2:.7f}\n")
            f.close()
        return preds,trues


