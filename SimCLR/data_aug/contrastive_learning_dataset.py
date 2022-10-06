from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from PIL import Image
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import io
from utils import save_checkpoint
import cv2 as cv
# import data_aug.stain_utils as stain_utils
# import data_aug.stainNorm_Macenko as stainNorm_Macenko

resolution = "40x"


class Pathology_dataset(Dataset):
    """Pathology dataset."""

    def __init__(self, csv_file, root_dir, finetune=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print(csv_file)
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.finetune = finetune
        self.dataset = self.read_patches()
        # self.i1 = stain_utils.read_image('./data_aug/reference.tif')

    def __len__(self):
        return len(self.landmarks_frame["ImageName"])

    #         return 1000

    def read_patches(self):
        labels_csv = self.landmarks_frame
        if not self.finetune:
            labels_csv = labels_csv.sample(frac=1).reset_index(drop=True)
            print("Shuffled!")
        # n_selected = 1000
        # if len(labels_csv["ImageName"]) < n_selected:
        #     n_selected = len(labels_csv["ImageName"])
        n_selected = len(labels_csv["ImageName"])
        #         n_selected = 500
        print("Total images: ", n_selected)

        patches = []
        labels_patches = []
        image_name = []
        for i in range(n_selected):
            im = self.root_dir + labels_csv["ImageName"][i]
            labels_patches.append(int(labels_csv["Reccured"][i]))
            # imarray = np.array(im)
            patches.append(im)
            image_name.append(labels_csv["ImageName"][i])

        print("Done reading patches...", len(patches))
        patches, labels_patches = patches, np.array(labels_patches)
        # patches, labels_patches = np.array(patches), np.array(labels_patches)
        if not self.finetune:
            patches, labels_patches = shuffle(patches, labels_patches)
            print("Shuffled!")
        final_ds = {"image": patches, "label": labels_patches, "imagename": image_name}
        d = pd.DataFrame(final_ds)
        if self.finetune:
            #             d.to_csv("./balanced/ALL/JHU_train_"+ resolution +"_HR_sampled_200_128x128_pretrained.csv")
            d.to_csv(
                "/Users/mahtabfarrokh/PycharmProjects/pythonProject/CPCTR_FULL_DATA/FULL_DATA 2/output/Last_NormalizedJHU_trainedonCPCTR_" + resolution + "_HR_sampled_200_128x128_pretrained.csv")
            print("Saved..")
        return final_ds

    def __getitem__(self, idx):
        path = self.dataset["image"][idx]

        image = Image.open(self.dataset["image"][idx])

        # normalizer = stainNorm_Macenko.Normalizer()
        # normalizer.fit(self.i1)
        # image = normalizer.transform(image)
        #         image = Image.fromarray(normalized)

        label = self.dataset["label"][idx]

        if self.transform:
            image = self.transform(image)

        sample = (image, label)
        #         image.close()
        return sample


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views, finetune=False):
        print(name)
        project_path = "/Users/mahtabfarrokh/PycharmProjects/pythonProject/JHU/"
        # project_path = "/Users/mahtabfarrokh/PycharmProjects/pythonProject/CPCTR_FULL_DATA/FULL_DATA 2/"
        # labels_path = project_path + "balanced_5folds/ALL/Classification_Train_128patch_40x.csv"
        labels_path = project_path + "Classification_ALL_128patch_40x.csv"

        # images_path = project_path + "CPCTR_Normalized/"
        images_path = project_path + "JHU_normalized/"
        print(images_path)
        print(labels_path)
        if finetune:
            train_examples = Pathology_dataset(csv_file=labels_path, root_dir=images_path, finetune=finetune,
                                               transform=transforms.ToTensor())
        else:
            train_examples = Pathology_dataset(csv_file=labels_path, root_dir=images_path, finetune=finetune,
                                               transform=ContrastiveLearningViewGenerator(
                                                   self.get_simclr_pipeline_transform(128),
                                                   n_views))

        #         labels_path = project_path + "Classification_Validation_128patch_" +  resolution + "_info.csv"
        #         images_path = project_path + "Validation_patches_" + resolution + "/"
        #         if finetune:
        #             validation_examples = Pathology_dataset(csv_file=labels_path, root_dir=images_path, finetune= finetune, transform=transforms.ToTensor())
        #         else:
        #             validation_examples = Pathology_dataset(csv_file=labels_path, root_dir=images_path, finetune= finetune, transform=ContrastiveLearningViewGenerator(
        #                                                                   self.get_simclr_pipeline_transform(128),
        #                                                                   n_views))
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                          'pathology-train': lambda: train_examples,
                          #                           'pathology-validation': lambda: validation_examples,
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
