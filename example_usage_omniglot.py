import torch
import tqdm
import torchvision.transforms as transforms

from data import ConvertToThreeChannels, FewShotLearningDatasetParallel
import os
image_height = 28
image_width = 28
image_channels = 1

os.environ['DATASET_DIR'] = 'datasets'

if image_channels == 3:
    transforms = [transforms.Resize(size=(image_height, image_width)), transforms.ToTensor(),
                  ConvertToThreeChannels(),
                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
elif image_channels == 1:
    transforms = [transforms.Resize(size=(image_height, image_width)), transforms.ToTensor()]

data = FewShotLearningDatasetParallel(dataset_name='omniglot_dataset',
                                      indexes_of_folders_indicating_class=[-3, -2],
                                      train_val_test_split=[0.73982737361, 0.13008631319, 0.13008631319],
                                      labels_as_int=False, transforms=transforms, num_classes_per_set=5,
                                      num_support_sets=10,
                                      num_samples_per_support_class=1, num_channels=3,
                                      num_samples_per_target_class=5, seed=0, sets_are_pre_split=False,
                                      load_into_memory=False, set_name='train', num_tasks_per_epoch=500,
                                      overwrite_classes_in_each_task=False, class_change_interval=1)

dataloader = torch.utils.data.DataLoader(data, batch_size=2, num_workers=4)

with tqdm.tqdm(total=250) as pbar:
    for item in dataloader:
        x_support_set_task, x_target_set_task, y_support_set_task, y_target_set_task, x_task, y_task = item
        pbar.update(1)