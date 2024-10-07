import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


from engine import train_one_epoch, evaluate # if it fails to execute, then open coco_eval.py and comment out the statement: import torch._six
import utils
import transforms as T

root = '/l/vision/jolteon_ssd/mdreza/drake_teaching/cs195_fall24_detection/'


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# ------------------------------------------------------------------------------------------------------
#                                   dataset to store detector information
# ------------------------------------------------------------------------------------------------------
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path    = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path   = os.path.join(self.root, "PedMasks", self.masks[idx])
        img         = Image.open(img_path).convert("RGB")
        mask        = Image.open(mask_path) # note that we haven't converted the mask to RGB because each color corresponds to a different instance with 0 being background



        mask        = np.array(mask)
        obj_ids     = np.unique(mask) # instances are encoded as different colors
        obj_ids     = obj_ids[1:]     # first id is the background, so remove it

        # split the color-encoded mask into a set of binary masks
        masks       = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs    = len(obj_ids)
        boxes       = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes       = torch.as_tensor(boxes, dtype=torch.float32)
        labels      = torch.ones((num_objs,), dtype=torch.int64) # there is only one class
        masks       = torch.as_tensor(masks, dtype=torch.uint8)

        image_id    = torch.tensor([idx])
        area        = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd     = torch.zeros((num_objs,), dtype=torch.int64) # suppose all instances are not crowd

        target              = {}
        target["boxes"]     = boxes
        target["labels"]    = labels
        target["masks"]     = masks
        target["image_id"]  = image_id
        target["area"]      = area
        target["iscrowd"]   = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    
    
# ------------------------------------------------------------------------------------------------------
#                                   use our dataset and defined transformations
# ------------------------------------------------------------------------------------------------------
from torch.utils.data import Subset # for partitioning the folder in making train+test splits
torch.manual_seed(1)

dataset_test      = PennFudanDataset(root + '/PennFudanPed/', get_transform(train=False)) 
dataset_train     = PennFudanDataset(root + '/PennFudanPed/', get_transform(train=True))  


# split the dataset in train and test set
random_indices    = torch.randperm(len(dataset_train)).tolist()
total_test_image  = 50                                                            # you can adjust the number of samples you want to include in the TEST SPLIT
dataset_test      = Subset(dataset_test,  random_indices[-total_test_image:])     # place the last 50 randomly shuffled images in the TEST SPLIT
dataset_train     = Subset(dataset_train, random_indices[:-total_test_image])     # place rest of the images in the TRAIN SPLIT

print(dataset_test)


# define training and test data loaders

# you don't need to shuffle the test samples because they don't affect the network training
data_loader_test  = torch.utils.data.DataLoader(
                      dataset_test, batch_size=1, shuffle=False, num_workers=0,
                      collate_fn=utils.collate_fn)

# but you should (or must) shuffle the training examples
data_loader_train = torch.utils.data.DataLoader(
                      dataset_train, batch_size=2, shuffle=True, num_workers=0,
                      collate_fn=utils.collate_fn)


# ------------------------------------------------------------------------------------------------------
#                                   define the detector model
# ------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn        # import the pretrained model
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # you need to adjust the total number of output predictors (eg, i) pedestrian ii) backbround), as they might differ from those in pretrained dataset MS COCO which had 91 classes



class FasterRCNN(nn.Module):

  def __init__(self, num_classes):
    super(FasterRCNN, self).__init__()

    self.model = fasterrcnn_resnet50_fpn(pretrained=True)                               # load the detection model pre-trained on MS-COCO; retain the network backbone but replace the predictor head for fine-tuning
    
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features                   # get the number of input features for the classifier
    
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)    # replace the pre-trained predictor head with a new one (both FastRCNN and FasterRCNN use the same predictor)

  def forward(self, x, target=None):
    output = self.model(x, target)
    return output


# ------------------------------------------------------------------------------------------------------
#                                   train the detector
# ------------------------------------------------------------------------------------------------------


from torch.optim.lr_scheduler import StepLR

device          = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_epochs      = 10

# our dataset has two classes only - background and person
num_classes     = 2

model = FasterRCNN(num_classes)

model.to(device)

# construct an optimizer for fine-tuning
params          = [p for p in model.parameters() if p.requires_grad]
optimizer       = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
lr_scheduler    = StepLR(optimizer, step_size=3, gamma=0.1)




# start traininig or fine-tuning

for epoch in range(num_epochs):
    
    # we imported this method from the utility scripts for detection we downloaded earlier: /content/engine.py
    
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10) # train for one epoch, printing every 10 iterations
    
    lr_scheduler.step() # update the learning rate
    
    evaluate(model, data_loader_test, device=device) # evaluate on the test dataset