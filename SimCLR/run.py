import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import torchvision

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
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
parser.add_argument('--epochs', default=500, type=int, metavar='N',
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

parser.add_argument('--out_dim', default=1000, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--checkpoint', default=False, type=bool, help='Using checkpoint or not')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    #     validation_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    #     validation_loader = torch.utils.data.DataLoader(
    #         validation_dataset, batch_size=args.batch_size, shuffle=True,
    #         num_workers=args.workers, pin_memory=True, drop_last=True)
    validation_loader = ""
    if 0:
        print("Loading checkpoint...")
        model = torchvision.models.resnet34(pretrained=False, num_classes=2).to(args.device)
        print("_____________________>>>>")
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        checkpoint = torch.load('./runs/Jan24_23-46-07_mahtab-gpu-2/best_checkpoint_0500.pth.tar',
                                map_location=args.device)
        state_dict = checkpoint['state_dict']
        #       model.load_state_dict(checkpoint['state_dict'])

        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone'):
                    # remove prefix
                    print("HERE!")
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]

        #       state_dict['fc.bias']  = state_dict['fc.2.bias']
        #       state_dict['fc.weight']  = state_dict['fc.2.weight']
        log = model.load_state_dict(state_dict, strict=False)
        print("===")
        print(log)
        print("===")
        print(log.missing_keys)
        assert log.missing_keys == ['fc.weight', 'fc.bias']
        optimizer.load_state_dict(state_dict)
        print(log.missing_keys)
        print(log)
    else:
        print("No checkpoint...", args.out_dim, args.arch)
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    print("!!!!!!!!!!!!!!!!!!!!!")
    print(model)
    print("!!!!!!!!!!!!!!!!!!!!!")

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader, validation_loader)


if __name__ == "__main__":
    main()
