from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

import sys
sys.path.append('/scratch/ondemand28/charlieguo/tsf/icu-patient-status-forecasting/src')
from labtest_pred import LabValueDatasetMIMICJerry # TODO refactor here
import h5py
import numpy as np
import json

def load_dict_from_hdf5(filename):
    data_dict = {}
    with h5py.File(filename, "r") as hf:
        # Load datasets
        for key in hf.keys():
            data = hf[key][:]
            # Try converting numeric datasets back to arrays directly
            try:
                data_dict[key] = data
            except Exception as e:
                print(f"Error loading dataset {key}: {e}")  

        # Load attributes and attempt to preserve original data types
        for key in hf.attrs.keys():
            # Deserialize JSON data
            attr_data = json.loads(hf.attrs[key])

            # Direct assignment of the deserialized data
            # This ensures that lists and dictionaries are preserved as such
            data_dict[key] = attr_data

            # Special handling for elements that were originally numpy arrays
            # You may add specific conditions to identify these cases, such as checking
            # for specific keys or data patterns that are known to represent arrays

            # Example condition: converting lists to numpy arrays if they represent numeric data
            # This is a simplistic approach; you'll need to adjust based on your specific data characteristics
            if isinstance(attr_data, list) and all(
                isinstance(x, (int, float, list)) for x in attr_data
            ):
                try:
                    data_dict[key] = np.array(attr_data, dtype=object)
                except ValueError:
                    # Handle the case of ragged sequences; keep as list if conversion fails
                    pass

    return data_dict

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

filename = "/scratch/ondemand28/charlieguo/tsf/icu-patient-status-forecasting/gdrive/unnorm_time_vars_sm0111.h5"
data = load_dict_from_hdf5(filename)


def data_provider(args, flag):
    Data = data_dict[args.data] # dict of data, key: specific dataset, value: torch dataset, currently custom
    timeenc = 0 if args.embed != 'timeF' else 1 #  1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else: # e.g. train
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # data_set = Data(
    #     root_path=args.root_path,
    #     data_path=args.data_path,
    #     flag=flag,
    #     size=[args.seq_len, args.label_len, args.pred_len], # 336, 48, 96
    #     features=args.features,
    #     target=args.target,
    #     timeenc=timeenc,
    #     freq=freq
    # )

    window_size = 24
    # Assuming the dataset is already initialized
    data_set = LabValueDatasetMIMICJerry(data_dict=data, config={
            "min_num_vals_past": args.seq_len,
            "min_num_vals_future": args.pred_len,
            "window_size": window_size
        }, split=flag
        )

    # breakpoint()
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
