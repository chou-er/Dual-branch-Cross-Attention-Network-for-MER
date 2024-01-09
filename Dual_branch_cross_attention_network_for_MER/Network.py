import argparse
from collections import OrderedDict
import torch
import torch.optim as optim
from torch import device
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils import data

from dataset import Dataset_2flow
from dataset import get_2flow_paths_and_labels
from FusionBlock import Fusioncls

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    tb_writer = SummaryWriter("logs")
    images_paths, labels = get_2flow_paths_and_labels(r"E:\5class\OPT_CSV\CASME2_flow.csv")

    data_transform = transforms.Compose([transforms.ToTensor()])
    # 读图片
    dataset_2flow = Dataset_2flow(root="E:/5class",
                                  images_paths=images_paths,
                                  img_labels=labels,
                                  transform=data_transform,
                                  get_aux=True,
                                  aux=None)

    train_dataset_2flow_length = int(len(dataset_2flow) * 0.8)
    val_dataset_2flow_length = int(len(dataset_2flow) * 0.2)
    train_dataset_2flow, val_dataset_2flow = torch.utils.data.random_split(dataset_2flow,
                                                                           lengths=[train_dataset_2flow_length,
                                                                                    val_dataset_2flow_length])

    batch_size = args.batch_size

    nw = 8
    print('Using {} dataloader workers every process'.format(nw))

    train_loader_2flow = torch.utils.data.DataLoader(train_dataset_2flow,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw
                                                     )

    val_loader_2flow = torch.utils.data.DataLoader(val_dataset_2flow,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=nw
                                                   )


    Swin_T = torch.load(
        r"E:\My_Work\S_T_mobileViT_2flow_cross_attention\Swin_transformer_second2flow\Pre-training weight\swin_tiny_patch4_window7_224.pth")[
        "model"]

    key_list = list()
    for key in Swin_T:
        key_list.append(key)
    for key in key_list:
        new_key = "model1" + "." + key
        Swin_T[new_key] = Swin_T[key]
    for key in list(Swin_T.keys()):
        if "model1" not in key:
            del Swin_T[key]


    MobileViT = torch.load(
        r"E:\My_Work\S_T_mobileViT_2flow_cross_attention\Swin_transformer_second2flow\Pre-training weight\mobilevit_xs.pt")
    key_list_M = list()
    for key in MobileViT:
        key_list_M.append(key)
    for key in key_list_M:
        new_key = "model2" + "." + key
        MobileViT[new_key] = MobileViT[key]
    for key in list(MobileViT.keys()):
        if "model2" not in key:
            del MobileViT[key]

    Modify_dict = OrderedDict()
    Modify_dict.update(Swin_T)
    Modify_dict.update(MobileViT)

    Modify_dict.pop("model1.head.weight")
    Modify_dict.pop("model1.head.bias")
    Modify_dict.pop("model2.classifier.fc.weight")
    Modify_dict.pop("model2.classifier.fc.bias")

    model = Fusioncls().to(device)


    model_param = model.state_dict()
    model_param.update(Modify_dict)
    model.load_state_dict(model_param, strict=False)


    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)


    from utilss import train_one_epoch, evaluate
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc, model1 = train_one_epoch(model=model,
                                                        optimizer=optimizer,
                                                        data_loader=train_loader_2flow,
                                                        device=device,
                                                        epoch=epoch,
                                                        loss_function=loss_function)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader_2flow,
                                     device=device,
                                     epoch=epoch,
                                     loss_function=loss_function)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model1.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args(args=[])

    main(opt)
