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
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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

    train_dataset = dataset.get_dataset("pathology-train", args.n_views, finetune=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=False)

    validation_dataset = dataset.get_dataset("pathology-validation", args.n_views, finetune=True)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=False)

    model = torchvision.models.resnet50(pretrained=False, num_classes=2).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    
#     checkpoint = torch.load('./runs/Oct07_20-08-39_mahtab-gpu-2/best_checkpoint_0200.pth.tar', map_location=args.device)
    checkpoint = torch.load('./runs/Oct08_19-32-43_mahtab-gpu-2/best_checkpoint_0500.pth.tar', map_location=args.device)
    state_dict = checkpoint['state_dict']
#       model.load_state_dict(checkpoint['state_dict'])
      
        
    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
          if k.startswith('backbone') and not k.startswith('backbone.fc'):
            # remove prefix
            state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]
        
    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    
    print(model)
    
    epochs = 200
    train_acc_all = []
    eval_acc_all = []
    loss_all = []
    loss_all_eval = []
    fff = 0
    for epoch in range(epochs):
        fff += 1
        top1_train_accuracy = 0
        train_loss = 0
        c = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            c += 1
#             x_batch = np.asarray([np.asarray(t) for t in x_batch])
#             x_batch = torch.from_numpy(x_batch)
#             y_batch = np.asarray([np.asarray(t) for t in y_batch])
#             y_batch = torch.from_numpy(y_batch)
            
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            train_loss += loss
            
            top1 = accuracy(logits, y_batch, topk=(1,1))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(counter, c)
        top1_train_accuracy /= (counter + 1)
        train_loss /= (counter + 1)
        train_acc_all.append(top1_train_accuracy.item())
        loss_all.append(train_loss)
        top1_accuracy = 0
        top5_accuracy = 0
        print("===================================")
        print("Let's evaluate...")
        eval_loss = 0 
        for counter, (x_batch, y_batch) in enumerate(validation_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            eval_loss += loss
            top1, top5 = accuracy(logits, y_batch, topk=(1, 1))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        eval_acc_all.append(top1_accuracy.item())
        loss_all_eval.append(eval_loss)
        print(
            f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        print("Loss Train: ", loss_all[-1], " Loss validation: ", loss_all_eval[-1]) 
        # save plots
        x1 = [i for i in range(len(train_acc_all))]
        plt.figure()
        plt.plot(x1, train_acc_all, label="train", color="b")
        plt.plot(x1, eval_acc_all, label="evaluation", color="r")
        plt.xlabel('# Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
        plt.savefig('/home/farrokh/Multi_Res_BYOL/SimCLR-master/plots/Finetuned_acc.png')
        plt.close()
        plt.figure()
        plt.plot(x1, loss_all, label="train", color="b")
        plt.plot(x1, loss_all_eval, label="evaluation", color="r")
        plt.xlabel('# Epoch')
        plt.legend(['Train', 'Validation'])
        plt.ylabel('NCE Loss')
        #                 plt.savefig('/content/drive/MyDrive/simclr/SimCLR-master-pytorch/plots/acc.png')
        plt.savefig('/home/farrokh/Multi_Res_BYOL/SimCLR-master/plots/Finetuned_loss.png')
        plt.close()

        if fff % 5 == 4:
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(args.epochs)
            save_root = "/home/farrokh/Multi_Res_BYOL/SimCLR-master/runs/"
            print(checkpoint_name)
            save_checkpoint({
                'epoch': args.epochs,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(save_root, checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {save_root}.")


if __name__ == "__main__":
    main()
