import argparse
import os
import hydra
import matplotlib.pyplot as plt
import torch
from utils.seed import set_seed
from utils.savefile import create_save_dir
from dataloader.Get_loader import get_loader
from exp.exp_train import Exp_train

def get_args():
    parser = argparse.ArgumentParser(description='Battery Project')
    parser.add_argument('--random_seed', type=int, default=2025)
    parser.add_argument('--hydra_cfg_path', type=str, default='./configs', help='Hydra配置文件路径')
    parser.add_argument('--config_name', type=str, default='train_config', help='配置文件名')
    parser.add_argument('--data', type=str, default='NASA',choices=['NASA','XJTU','MIT','MAITIAN'])
    parser.add_argument('--input_type', type=str, default='charge',choices=['charge', 'partial_charge', 'handcraft_features'])
    parser.add_argument('--test_battery_id', type=int, default=1,help='test battery id, 1-8 for XJTU (1-15 for batch-2), 1-5 for MIT')
    parser.add_argument('--normalized_type', type=str, default='minmax', choices=['minmax', 'standard'])
    parser.add_argument('--minmax_range', type=tuple, default=(0, 1), choices=[(0, 1), (-1, 1)])
    parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--test_file', type=str, default='25_1C1C_features_35.csv')
    parser.add_argument('--test_battery', type=str, default='Battery_14#')
    parser.add_argument('--feature_processing_in',type=int, default=33)
    parser.add_argument('--maitian_type', type=str, default='charge', choices=['charge', 'discharge'])
    parser.add_argument('--feature_num', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='MSWaveFuser')
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--device', type=int, default=0, help='0,1')

    parser.add_argument('--save_path', type=str, default='./outputs')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
    set_seed(args.random_seed)
    save_dir = create_save_dir(args.save_path)

    train_loader, test_loader, valid_loader = get_loader(args)


    exp = Exp_train(args)
    exp.train(train_loader, valid_loader, save_dir)


    print("\n=====================START TEST==========================")

    pred, true = exp.test(test_loader, save_dir)

    fig_path = os.path.join(save_dir, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.plot(range(len(pred)), pred, color='black', label='Predicted')
    plt.plot(range(len(true)), true, color='red', label='True')
    plt.xlabel('CYCLE')
    plt.ylabel('SOH')
    plt.title('SOH Estimation')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(fig_path, 'result_battery.jpg'))

    print("Done")

