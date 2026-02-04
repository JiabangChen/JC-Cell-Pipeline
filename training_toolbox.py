import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
import math
import torch
from torchvision.transforms.v2 import functional as F
from torchvision import utils
import torchvision
from PIL import Image
import numpy as np


def plot_loss_curves(train_loss_per_epoch, val_loss_per_epoch, title="Loss vs. Epoch", fold_number = None, hyper_setnum = 0, model_type = "DINO"):
    """
    train_loss_per_epoch: list/array，each object is the average training loss in an epoch
    val_loss_per_epoch:   list/array，each object is the average validating loss in an epoch
    """
    assert fold_number is not None, "need to input fold number!"
    assert len(train_loss_per_epoch) == len(val_loss_per_epoch), \
        f"the length of training loss and validating loss are different: {len(train_loss_per_epoch)} vs {len(val_loss_per_epoch)}!"

    epochs = range(1, len(train_loss_per_epoch) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_per_epoch, label="Train Loss", marker = 'o', markevery = 1)
    plt.plot(epochs, val_loss_per_epoch, label="Val Loss", marker = 's', markevery = 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(title + "_" + model_type + "_" + str(hyper_setnum) + "_" + str(fold_number) + ".png", bbox_inches="tight", dpi=150)
    plt.show()


def get_cosine_with_warmup_tail(optimizer, num_warmup_steps, num_training_steps, min_lr_factor=0.1, num_cycles=0.5):
    """
    Cosine decay with warmup and a fixed min_lr_factor (tail factor).
    min_lr_factor = lr_min / lr_max
    This is used to prevent the lr from decaying to 0
    """
    def lr_lambda(current_step):
        # warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return min_lr_factor
        # progress after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

        return min_lr_factor + (1 - min_lr_factor) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def replace_classifier(model):
    # the name of the potential prediction layer, users can extend this list based on their models
    classifier_names = ['fc', 'classifier', 'head', 'heads', 'logits']

    for name in classifier_names:
        if hasattr(model, name):
            last_layer = getattr(model, name)

            if isinstance(last_layer, nn.Linear):
                in_features = last_layer.in_features
                setattr(model, name, nn.Identity())
                return model, in_features

            elif isinstance(last_layer, nn.Sequential):
                if len(last_layer) > 1 and isinstance(last_layer[-2], nn.Dropout):
                    last_layer[-2] = nn.Identity()
                if isinstance(last_layer[-1], nn.Linear):
                    in_features = last_layer[-1].in_features
                    last_layer[-1] = nn.Identity()
                    setattr(model, name, last_layer)
                    return model, in_features

    raise ValueError("No known classifier layer found in this model.")


def inspect_model_and_optimizer(model, optimizer, logger):
    logger.info("===== Trainable Parameters in Model =====")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"[TRAINABLE] {name} | shape: {tuple(param.shape)} | numel: {param.numel():,}")
            total_params += param.numel()
    logger.info(f"Total trainable parameters in model: {total_params:,}")
    logger.info("=========================================\n")

    logger.info("===== Optimizer Parameter Groups =====")
    for i, group in enumerate(optimizer.param_groups):
        lr = group.get("lr", None)
        wd = group.get("weight_decay", None)
        logger.info(f"-- Group {i}: lr={lr}, weight_decay={wd}")
        group_param_names = []
        for p in group["params"]:
            for name, param in model.named_parameters():
                if p is param:
                    group_param_names.append(name)
        for name in group_param_names:
            logger.info(f"    {name}")
        logger.info(f"  -> Total params in group {i}: {sum(p.numel() for p in group['params']):,}")
    logger.info("=======================================")


def compute_mean_std(ds, channels):  # calculate the mean and std for each channel in the dataset
    n_pixels = 0
    mean = torch.zeros(channels)
    mean_square = torch.zeros(channels)
    for image in ds:
        pixels = image.shape[1] * image.shape[2]
        n_pixels += pixels
        for channel in range(channels):
            mean[channel] += image[channel, :, :].mean()
            mean_square[channel] += (image[channel, :, :] ** 2).sum()
    mean = mean / len(ds)
    mean_square = mean_square / n_pixels
    std = torch.sqrt(mean_square - mean ** 2)

    return mean, std


class ResizeWithPadding:
    def __init__(self, size, fill=0):
        """
        size: tuple (width, height) target size
        fill: pixel value to fill
        """
        self.target_width, self.target_height = size
        self.fill = fill

    def __call__(self, img):
        # obtain the size of original image
        orig_width, orig_height = img.size

        # use the smaller ratio to scale the width and height equally.
        width_ratio = self.target_width / orig_width
        height_ratio = self.target_height / orig_height
        if width_ratio <= height_ratio:
            new_width = int(orig_width * width_ratio + 0.1)  # plus 0.1 to prevent the float error
            new_height = int(orig_height * width_ratio)
        else:
            new_width = int(orig_width * height_ratio)
            new_height = int(orig_height * height_ratio + 0.1)

        # resize
        img = F.resize(img, [new_height, new_width])  # the resize in F needs the format of input as (height, width)

        # calculate padding size
        pad_left = (self.target_width - new_width) // 2
        pad_top = (self.target_height - new_height) // 2
        pad_right = self.target_width - new_width - pad_left
        pad_bottom = self.target_height - new_height - pad_top

        # 添加 padding
        img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)

        assert img.size[0] == self.target_width and img.size[1] == self.target_height, 'Output Image size is incorrect!'

        return img


# save the validate image to target folders
def save_validation_images(val_paths, dataset, cancer_path, normal_path):
    for i in range(len(val_paths)):
        for j in range(len(dataset.samples)):
            if dataset.samples[j][0] == val_paths[i] and dataset.samples[j][1] == 0:
                utils.save_image(dataset[j][0],
                                 cancer_path + val_paths[i].split('\\')[-1].split('.')[0] + "_val.png")
                break
            elif dataset.samples[j][0] == val_paths[i] and dataset.samples[j][1] == 1:
                utils.save_image(dataset[j][0],
                                 normal_path + val_paths[i].split('\\')[-1].split(".")[0] + "_val.png")
                break


def generate_save_train_images(train_paths, dataset, cancer_path, normal_path, num_variations_per_image_0, num_variations_per_image_1, transform_augmented, logger):
    # original training images
    train_origin_cancer_dataset = []
    train_origin_normal_dataset = []
    for i in range(len(train_paths)):
        for j in range(len(dataset.samples)):
            if dataset.samples[j][0] == train_paths[i] and dataset.samples[j][1] == 0:
                train_origin_cancer_dataset.append(
                    {'image': dataset[j][0], 'filename': train_paths[i].split('\\')[-1].split('.')[0]})
                break
            elif dataset.samples[j][0] == train_paths[i] and dataset.samples[j][1] == 1:
                train_origin_normal_dataset.append(
                    {'image': dataset[j][0], 'filename': train_paths[i].split('\\')[-1].split('.')[0]})
                break

    # Initialize an empty list to store the augmented images
    augmented_images_class_0 = []
    augmented_images_class_1 = []

    # Image augmentation
    for image in train_origin_cancer_dataset:
        for i in range(num_variations_per_image_0):
            augmented_images_class_0.append(
                {'image': transform_augmented(image['image']), 'filename': image['filename']})

    for image in train_origin_normal_dataset:
        for i in range(num_variations_per_image_1):
            augmented_images_class_1.append(
                {'image': transform_augmented(image['image']), 'filename': image['filename']})

    # save training dataset (original + augmentation)
    if num_variations_per_image_0 > 0:
        for i in range(len(augmented_images_class_0)):
            utils.save_image(augmented_images_class_0[i]['image'],
                             cancer_path + str(int(i / num_variations_per_image_0)) + "_" + str(
                                 i % num_variations_per_image_0) + "_" + augmented_images_class_0[i][
                                 'filename'] + "_aug.png")

    if num_variations_per_image_1 > 0:
        for i in range(len(augmented_images_class_1)):
            utils.save_image(augmented_images_class_1[i]['image'],
                             normal_path + str(int(i / num_variations_per_image_1)) + "_" + str(
                                 i % num_variations_per_image_1) + "_" + augmented_images_class_1[i][
                                 'filename'] + "_aug.png")

    # save original images (after resize)
    for i in range(len(train_origin_cancer_dataset)):
        utils.save_image(train_origin_cancer_dataset[i]['image'],
                         cancer_path + str(i) + "_" + train_origin_cancer_dataset[i][
                             'filename'] + "_original.png")

    for i in range(len(train_origin_normal_dataset)):
        utils.save_image(train_origin_normal_dataset[i]['image'],
                         normal_path + str(i) + "_" + train_origin_normal_dataset[i][
                             'filename'] + "_original.png")

    logger.info(
        f"we have generate {len(augmented_images_class_0)} augmented cancer cell images and {len(augmented_images_class_1)} augmented normal cell images.")
    logger.info(
        f'Totally we have {len(train_origin_cancer_dataset) + len(augmented_images_class_0)} cancer cell images and {len(train_origin_normal_dataset) + len(augmented_images_class_1)} normal cell images for training')


def read_data(test_path, validation_path, train_path, transform_whole_dataset):
    # read test data
    test_dataset = torchvision.datasets.ImageFolder(test_path, transform=transform_whole_dataset)
    # read validation data
    val_dataset = torchvision.datasets.ImageFolder(validation_path, transform=transform_whole_dataset)
    # read train data
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transform_whole_dataset)

    return test_dataset, val_dataset, train_dataset


def save_whole_image(val_dataset, test_dataset, validation_whole_path, test_whole_path, transform_inverse):
    # save test and validate image (un-shuffle, easy to find which one is misclassified)
    for i in range(len(val_dataset)):
        utils.save_image(transform_inverse(val_dataset[i][0]),
                         validation_whole_path + str(i) + "_" + str(val_dataset[i][1]) + "_" +
                         val_dataset.samples[i][0].split('\\')[-1].split('.')[0] + ".png")
    for i in range(len(test_dataset)):
        utils.save_image(transform_inverse(test_dataset[i][0]),
                         test_whole_path + str(i) + "_" + str(test_dataset[i][1]) + "_" +
                         test_dataset.samples[i][0].split('\\')[-1].split('.')[0] + ".png")


def compute_mean_std_noblue(ds, channels):  # calculate the mean and std for each channel in the dataset
    n_pixels = 0
    mean = torch.zeros(channels)
    mean_square = torch.zeros(channels)
    for image in ds:
        pixels = image.shape[1] * image.shape[2]
        n_pixels += pixels
        for channel in range(channels):
            mean[channel] += image[channel, :, :].mean()
            mean_square[channel] += (image[channel, :, :] ** 2).sum()
    mean = mean / len(ds)
    mean_square = mean_square / n_pixels
    std = torch.sqrt(mean_square - mean ** 2)

    mean_blue = torch.zeros(1)  # add the mean = 0 for blue channel
    std_blue = torch.ones(1)  # std = 1, so the blue channel will not be altered by Normalization
    mean = torch.cat([mean, mean_blue])
    std = torch.cat([std, std_blue])

    return mean, std


def resize_image(image, output_size):
    # Check if image has valid size
    if image.size[0] is None or image.size[1] is None:
        print("Image size is invalid")
        return None, None

    # Check if image size exceeds maximum allowed
    max_pixels = Image.MAX_IMAGE_PIXELS
    if max_pixels is not None and image.size[0] * image.size[1] > max_pixels:
        print("Image size exceeds maximum allowed")
        return None, None

    # Resize the image
    resized_image = image.resize(output_size)

    # Return resized image and original size
    return np.array(resized_image), image.size