import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import torchvision
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils import save_config_file, save_checkpoint
import logging
import pandas as pd

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
print("model_names: ", model_names)
parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='pathology-train',
                    help='dataset name')
#  choices=['stl10', 'cifar10', 'pathologytrain', 'pathologyvalidation']
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=2, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--checkpoint', default=False, type=bool, help='Using checkpoint or not')

resolution = "40x"


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        print("Here??????")
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset("pathology-train", args.n_views, finetune=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, drop_last=False)

    # validation_dataset = dataset.get_dataset("pathology-validation", args.n_views, finetune=True)
    # validation_loader = torch.utils.data.DataLoader(
    #     validation_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, drop_last=False)

    # TODO check if the sequence of images are correct!
    model = torchvision.models.resnet34(pretrained=False, num_classes=2).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    checkpoint = torch.load('./runs/Jul26_05-09-12_mahtab-gpu-2/best_checkpoint_0500.pth.tar', map_location=args.device)
    # checkpoint = torch.load('./runs/Feb18_19-42-42_mahtab-gpu-2/best_checkpoint_0500.pth.tar', map_location=args.device)
    state_dict = checkpoint['state_dict']
    #       model.load_state_dict(checkpoint['state_dict'])

    #     for k in list(state_dict.keys()):
    #         print(k)
    #         if k.startswith('backbone.'):
    #            if k.startswith('backbone') and not k.startswith('backbone.fc'):
    #             # remove prefix
    #             state_dict[k[len("backbone."):]] = state_dict[k]
    #         del state_dict[k]
    print("++++")
    log = model.load_state_dict(state_dict, strict=False)
    print(log.missing_keys)
    #     assert log.missing_keys == ['fc.weight', 'fc.bias']
    print(log)
    print("---")
    print(log.missing_keys)

    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print("--------")
    first = True
    all_hidden_representation = np.array([[]])
    for counter, (x_batch, y_batch) in enumerate(train_loader):
        print(counter)
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)
        hidden_representation = model(x_batch)
        fine_HR = hidden_representation.cpu().detach().numpy()
        fine_HR = np.reshape(fine_HR, (fine_HR.shape[0], fine_HR.shape[1]))
        if first:
            all_hidden_representation = fine_HR
            print(all_hidden_representation[0][:100])
            print(all_hidden_representation[1][:100])
            print(all_hidden_representation[2][:100])
            print(all_hidden_representation[3][:100])
            print(all_hidden_representation[4][:100])
            first = False
            print(all_hidden_representation.shape)
        #             print(0/0)

        else:
            all_hidden_representation = np.append(all_hidden_representation, fine_HR, 0)
    #        print(all_hidden_representation.shape)

    #        for h in range(len(hidden_representation)):
    #            hhh = []
    #            for h2 in range(len(hidden_representation[h])):
    #                hhh.append(hidden_representation[h][h2].item())
    #                print(hidden_representation[0])
    #                print(0/0)
    #            all_hidden_representation.append(hhh)

    all_hidden_representation = np.array(all_hidden_representation)
    # with open("/Users/mahtabfarrokh/PycharmProjects/pythonProject/JHU/output/NormalizedJHU_trainedOnCPCTR_" + resolution + "_HR2_sampled_200_128x128_pretrained.npy",
    #           'wb') as f:
    with open(
            "/Users/mahtabfarrokh/PycharmProjects/pythonProject/CPCTR_FULL_DATA/FULL_DATA 2/output/Last_NormalizedJHU_trainedonCPCTR_" + resolution + "_HR2_sampled_200_128x128_pretrained.npy",
            'wb') as f:
        np.save(f, all_hidden_representation)

    print("Done.. ", counter)


if __name__ == "__main__":
    main()
