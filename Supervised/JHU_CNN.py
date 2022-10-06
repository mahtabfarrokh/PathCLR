import torch
from PIL import Image
from torchvision import transforms
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
import numpy as np 
import torch.nn as nn
from random import shuffle
from sklearn.metrics import confusion_matrix
import copy

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
skf = StratifiedKFold(10)
CPCTR = pd.read_csv("./Info_final.csv")


def shuffle_list(*ls):
  l =list(zip(*ls))

  shuffle(l)
  return zip(*l)

def compute_CI(nums):
    avg = sum(nums) / len(nums)
    s = 0
    for i in nums:
        s += (i - avg) * (i - avg)

    s /= (len(nums) - 1)
    s = s ** 0.5
    e = (1.96 * s) / (len(nums) ** 0.5)
    print("CI: ", avg, " + ", e)
    return avg, e


def result_by_patient(y_pred, y_true, case_id):
    df = pd.DataFrame(
        {"caseid": list(case_id), "votes": y_pred, "labels": list(y_true)})

    final_vote = df.groupby(['caseid'], as_index=False).max() 

    final_vote["labels"] = final_vote["labels"].astype(int) 
    final_vote["votes"] = final_vote["votes"].astype(int)
    cm = confusion_matrix(final_vote["labels"], final_vote["votes"])
    accuracy = (cm[0][0] + cm[1][1]) / len(final_vote["labels"])
    return accuracy, cm


def open_images(image_names, labels, case_ids):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    images_dataset = []
    for i in range(len(image_names)):
        im = Image.open("./JHU/ALL/" + image_names[i] + "H&E.tif")
        input_tensor = preprocess(im)
        images_dataset.append([input_tensor, int(labels[i])])

    return images_dataset



df = pd.DataFrame({"caseID": list(CPCTR["CaseID"][:]), "Reccured": list(CPCTR["Reccured"][:])})
df["caseID"] = df["caseID"].astype(int)
df2 = df.drop_duplicates()
f = 0
for train_index, test_index in skf.split(df2["caseID"], df2["Reccured"]):
    print("======================")
    print("fold:", f)
   
    f += 1
    case_ids_set_train = df2.iloc[train_index]["caseID"]
    case_ids_set_validation = df2.iloc[test_index]["caseID"]
    image_name_train = []
    label_train = []
    id_train = []
    image_name_validation = []
    label_validation = []
    id_validation = []
    for j in range(len(CPCTR["ImageName"][:])):
        if  int(CPCTR["CaseID"][j]) in set(case_ids_set_validation):
            image_name_validation.append(CPCTR["ImageName"][j])
            label_validation.append(CPCTR["Reccured"][j])
            id_validation.append(CPCTR["CaseID"][j])
        else:
            image_name_train.append(CPCTR["ImageName"][j])
            label_train.append(CPCTR["Reccured"][j])
            id_train.append(CPCTR["CaseID"][j])

    image_name_train, id_train, label_train = shuffle_list(image_name_train, id_train, label_train)
    image_name_validation, id_validation, label_validation = shuffle_list(image_name_validation, id_validation, label_validation)
    trainset = open_images(image_name_train, label_train, id_train)
    testset = open_images(image_name_validation, label_validation, id_validation)
   
    trainloader =torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=8)
    testloader =torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_acc = 0
    wait_acc = 0 
    
    if torch.cuda.is_available():
        model.to('cuda')

    for epoch in range(200):
        running_loss = 0 
        running_corrects = 0
        if wait_acc > 5:  
            print("=============>>>>")
            print("Best RES: ", best_acc)
            break
        for (idx, batch) in enumerate(trainloader):
            inputs, labels = batch    
            # Transfer to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss_func = nn.CrossEntropyLoss()
                loss =  loss_func(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            print("=> running_loss: ", running_loss, "running_corrects: ", running_corrects)

        exp_lr_scheduler.step()
        with torch.set_grad_enabled(False):
            outputs = []
            for (idx, batch) in enumerate(testloader):
                inputs, labels = batch    
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                preds = torch.argmax(output, dim=1)
                outputs.extend([t.item() for t in preds])

            acc, cm = result_by_patient(outputs, label_validation, id_validation)
          
            if best_acc > acc: 
                wait_acc += 1
            else: 
                best_model_wts = copy.deepcopy(model.state_dict())
                wait_acc = 0
                best_acc = acc
                 
        print("Best RES: ", best_acc) 
        epoch_loss = running_loss / len(image_name_train)
        epoch_acc = running_corrects.double() / len(image_name_train)
        print(f'{"train"} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')



