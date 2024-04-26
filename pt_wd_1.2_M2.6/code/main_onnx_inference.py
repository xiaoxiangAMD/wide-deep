#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
#*******************************************************************************
import torch
import tqdm
import numpy as np
import onnx
import onnxruntime as rt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from torchfm.dataset.criteo import CriteoDataset
from torchfm.model.wd import WideAndDeepModel
import time
import random
seed = 123
torch.manual_seed(seed)
random.seed(seed)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_dataset(name, path):
    if name == 'criteo':
        return CriteoDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)

def test_onnx(model,wd_output_name, data_loader, device):
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            ort_inputs = {model.get_inputs()[0].name: to_numpy(fields)}
            y = model.run([wd_output_name],ort_inputs)[0]
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)

def get_latency(model,wd_output_name,data_loader,device):
    ##model.eval()
    data_len= len(data_loader.dataset)
    start_time = time.time()  
    with torch.no_grad():
        for fields, _ in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields = fields.to(device)
            ort_inputs = {model.get_inputs()[0].name: to_numpy(fields)}
            model.run([wd_output_name],ort_inputs)[0]
    inference_time = time.time() - start_time
    latency = inference_time / data_len 
    print('Inference latency: {:.8f} ms'.format(latency * 1000))  
def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         model_path):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset)
    wd = rt.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    wd_output_name = wd.get_outputs()[0].name

    auc = test_onnx(wd, wd_output_name,test_data_loader, device)
    print(f'test auc: {auc}')
    train_length = int(len(dataset) * 1)
    valid_length = int(len(dataset) * 0)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    batch_size=2048
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    get_latency(wd,wd_output_name,train_data_loader,device)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', default='./data/train.txt',help='./data/train.txt')
    parser.add_argument('--model_name', default='wd')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt/wd.pt')
    parser.add_argument('--model_path', default='wd.pt')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.model_path)
