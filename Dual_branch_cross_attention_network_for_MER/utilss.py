import os
import sys
import json
import pickle
import random
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


# def train_one_epoch(model, optimizer, data_loader, device, epoch):
def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_function):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(torch.device("cuda"))  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)  # 进度条
    # print("length",len(data_loader))

    for step, data in enumerate(data_loader):
        images_1, images_2, labels = data  # 需要获取两张图片

        # pred = model(images.to(device))
        # pred_classes = torch.max(pred, dim=1)[1]
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        #
        # loss = loss_function(pred, labels.to(device))
        # loss.backward()
        # accu_loss += loss.detach()

        sample_num += images_1.shape[0]
        pred = model(images_1.to(device), images_2.to(device))
        pred_classes = torch.max(pred, dim=1)[1]  # 取最有可能的分类的索引值
        # pred_classes_2 = torch.max(pred_2, dim=1)[1]
        accu_num += torch.eq(pred_classes,
                             labels.to(device)).sum()  # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
        f1 = metrics.f1_score(labels, pred_classes.detach().cpu().numpy(), average="macro")
        # print(f1)
        UAR = recall_score(labels, pred_classes.cpu().numpy(), average="macro", zero_division=1)
        # print(UAR)
        UF1 = (2 * UAR * f1) / (UAR + f1)
        # print(UF1)

        # f1 = metrics.f1_score(labels, pred_classes, average="micro")
        # print("f1", f1)
        # UAR = recall_score(labels, pred_classes, average="micro")
        # print("UAR", UAR)
        # UF1 = (2 * UAR * f1) / (UAR + f1)
        # print("UF1", UF1)
        loss = loss_function(pred, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accu_loss += loss.detach()  # .detach返回的是tensor类型     .item返回的是数值

        # data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                        accu_loss.item() / (step + 1),
        #                                                                        accu_num.item() / sample_num)
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, f1:{:.3F}, UAR:{:.3f}, UF1: {:.3f}".format(epoch,
                                                                                                                    accu_loss.item() / (step + 1),
                                                                                                                    accu_num.item() / sample_num,
                                                                                                                    f1,
                                                                                                                    UAR,
                                                                                                                    UF1)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # optimizer.step()
        # optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, model


# @torch.no_grad()
def evaluate(model, data_loader, device, epoch, loss_function):
    model.eval()
    # mymodel.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images_1, images_2, labels = data  # 需要获取两张图片

            sample_num += images_1.shape[0]

            pred = model(images_1.to(device), images_2.to(device))
            # pred_2 = model2(images_2.to(device))  # 应该是两张图片

            pred_classes = torch.max(pred, dim=1)[1]

            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            # accu_num += torch.eq(pred_classes_2, labels_2.to(device)).sum()

            f1 = metrics.f1_score(labels, pred_classes.detach().cpu().numpy(), average="weighted")
            # print(f1)
            UAR = recall_score(labels, pred_classes.cpu().numpy(), average="weighted", zero_division=1)
            # print(UAR)
            UF1 = (2 * UAR * f1) / (UAR + f1)
            # print(UF1)

            # f1 = metrics.f1_score(labels, pred_classes, average="micro")
            # print("f1", f1)
            # UAR = recall_score(labels, pred_classes, average="micro")
            # print("UAR", UAR)
            # UF1 = (2 * UAR * f1) / (UAR + f1)
            # print("UF1", UF1)


            # f1 = metrics.f1_score(label, pred_classes.detach().cpu().numpy(), average=None)
            # UF1 = f1
            # UAR = recall_score(label, pred_classes, average=None, zero_division=0)

            loss = loss_function(pred, labels.to(device))

            accu_loss += loss.detach()  # .detach返回的是tensor类型     .item返回的是数值

            # data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
            #                                                                        accu_loss.item() / (step + 1),
            #                                                                        accu_num.item() / sample_num)
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f},  f1:{:.3F}, UAR:{:.3f}, UF1: {:.3f}".format(epoch,
                                                                                                                        accu_loss.item() / (step + 1),
                                                                                                                        accu_num.item() / sample_num,
                                                                                                                        f1,
                                                                                                                        UAR,
                                                                                                                        UF1)


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
