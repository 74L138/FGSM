import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import glob
device = torch.device("cuda")

#打印一张图片
def imgshow(image):
    res = cv2.cvtColor(dataset[9],cv2.COLOR_BGR2RGB)
    plt.imshow(res)
    plt.show()

class ImgDataset(Dataset):
    def __init__(self,x,y=None,transform=None):
        self.x=x
        self.y=y
        if y is not None:
            self.y=torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return 200
    def __getitem__(self, item):
        X=self.x[item]
        if self.transform is not None:
            X=self.transform(X)
        if self.y is not None:
            Y= self.y[item]
            return X,Y
        else:
            return X

def readfile(path, label=None):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 224, 224, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(224, 224))
        if label is not None:
          y[i] = int(file.split("_")[0])
    if label is not None:
      return x, y
    else:
      return x

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225],inplace=False)
])

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    # print(sign_data_grad)
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

if __name__ == '__main__':
    workspace_dir = '.\data'
    dataset = readfile(os.path.join(workspace_dir,'images'))
    # imgshow(dataset[9])
    dfs = pd.read_csv("./data/labels.csv")
    dfs = dfs.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv("./data/categories.csv")
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # label = []
    # for df in dfs:
    #     label.append(label_name[df])
    dataset1 = ImgDataset(dataset,dfs,train_transform)
    data_loader = DataLoader(dataset1,batch_size=1,shuffle=False)
    success_example  =  []

    model1 = models.vgg16(pretrained=True)
    model1.cuda()
    model1.eval()
    epsilon = 0.1

    adv_examples = []
    wrong,success,fail  = 0,0,0
    for data,label in data_loader:
        data, target = data.to(device), label.to(device)
        data_raw = data
        data.requires_grad = True
        # 输出是对应的种类
        output = model1(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            wrong += 1
            continue

        loss = F.nll_loss(output, target)
        model1.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 将加入了噪声的图片输入到model中看是否能识别.可以识别fail+1
        output = model1(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            fail +=1
        else:
            success +=1
            if len(adv_examples) < 5:
                adv_ex = perturbed_data * torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1) + torch.tensor(
                    [0.485, 0.456, 0.406],device=device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                data_raw = data_raw * torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1) + torch.tensor(
                    [0.485, 0.456, 0.406],device=device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), data_raw, adv_ex))
    final_suc = (success / (wrong + success + fail))

    print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, success, len(data_loader), final_suc))

    #印图片出来看看
epsilons = [0.1]
cnt = 0
plt.figure(figsize=(30, 30))
for j in range(len(adv_examples)):
    cnt += 1
    plt.subplot(1, len(adv_examples) * 2, cnt)
    plt.xticks([], [])
    plt.yticks([], [])
    if j == 0:
        plt.ylabel("Eps: {}".format(epsilons), fontsize=14)
    orig, adv, orig_img, ex = adv_examples[j]
    # plt.title("{} -> {}".format(orig, adv))
    plt.title("original: {}".format(label_name[orig].split(',')[0]))
    orig_img = np.transpose(orig_img, (1, 2, 0))
    plt.imshow(orig_img)
    cnt += 1
    plt.subplot(len(epsilons), len(adv_examples) * 2, cnt)
    plt.title("adversarial: {}".format(label_name[adv].split(',')[0]))
    ex = np.transpose(ex, (1, 2, 0))
    plt.imshow(ex)
plt.tight_layout()
plt.show()










