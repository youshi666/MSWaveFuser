from dataloader.MIT_loader import MITDdataset
from dataloader.XJTU_loader import XJTUDdataset
from dataloader.nasa_loader import get_nasa_dataloader
from dataloader.CALCE_loader import CALCEdataset

def get_loader(args):
    if args.data == 'NASA':
        train_loader, test_loader, valid_loader = get_nasa_dataloader(args)
        return train_loader, test_loader, valid_loader
    elif args.data == 'CALCE':
        data_loader = CALCEdataset(args).get_data()
        train_loader = data_loader['train']
        valid_loader = data_loader['valid']
        test_loader = data_loader['test']
        return train_loader, test_loader, valid_loader
    elif args.data == 'XJTU':
        loader = XJTUDdataset(args)
    elif args.data == 'MIT':
        loader = MITDdataset(args)
    else:
        raise ValueError("Invalid data name")
    if args.input_type == 'charge':
        data_loader = loader.get_charge_data(test_battery_id=args.test_battery_id)
    elif args.input_type == 'partial_charge':
        data_loader = loader.get_partial_data(test_battery_id=args.test_battery_id)
    else:
        data_loader = loader.get_features(test_battery_id=args.test_battery_id)
    train_loader = data_loader['train']
    valid_loader = data_loader['valid']
    test_loader = data_loader['test']
    return train_loader, test_loader, valid_loader