from model.ResNet101 import ResNet101
from datasets.ppe_io import PPE
from torch.utils.data import random_split, DataLoader, ConcatDataset, SubsetRandomSampler
from datasets.transform import preprocess_img
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import binary_cross_entropy_with_logits
import numpy as np
from evaluate_map import compute_map
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from lxml import etree as ET
import shutil
from torchvision.models import resnet101
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
import time
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
# parser.add_argument('-gpu_id', type=str, default='0')
parser.add_argument('-data_root', type=str, default='integrated-above', choices=['above_cleaned_k=3,s=5', 'above_cleaned','above_all', 'integrated-complete', 'complete_views', 'above_view', 'integrated-above'])
parser.add_argument('-load_model', type=str, default="ppe_res101_professional_finished.pt")
parser.add_argument('-save_model', type=str, default= "saved_model")
parser.add_argument('-freeze', action='store_true')
parser.add_argument('-num_epoch', type=int, default=50)
parser.add_argument('-target', type=str, default='rc_nc_ma', choices=['rc_nc_ma','ca_ea_ma', 'rc_nc_ma'])
parser.add_argument('-cv_mode', action='store_true')
args = parser.parse_args()

root = "/home/beomseok/ppe_data/PPE_Profession_finished"
device = torch.device("cuda")

def main(rank, world_size):
    setup(rank, world_size)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    img_dir_train = os.path.join(root, args.data_root, "img_train")
    label_dir_train = os.path.join(root, args.data_root, "label_train")
    img_root_eval = os.path.join(root, args.data_root, "img_test")
    label_root_eval = os.path.join(root, args.data_root, "label_test")

    batch_size = 32
    load_model = args.load_model
    save_dir = args.save_model
    num_epoch = args.num_epoch
    target = args.target
    target_list = str(target).split("_")
    freeze_opt = args.freeze
    cross_validation_opt = args.cv_mode


    print("==========options provided==========")
    # print('gpu id:',args.gpu_id)
    print('img root:', args.data_root)
    # print(os.path.isdir(img_root_eval))
    # print(os.path.isdir(label_root_eval))
    print('model load directory:', load_model)
    print('model save directory:',save_dir)
    print("freeze option {}".format(freeze_opt))
    print("number of epoch: {}".format(num_epoch))
    print("target: {}".format(target))
    print("cross validation option: {}".format(cross_validation_opt))


    dataset_train_val = PPE(img_root=img_dir_train,
                    img_shape=(120, 160, 3),
                    label_root=label_dir_train,
                    mode="train",
                    target=target,
                    random_crop=False)

    dataset_test = PPE(img_root=img_root_eval,
                        img_shape=(120, 160, 3),
                        label_root=label_root_eval,
                        mode="test",
                        target=target,
                        random_crop=False)


    print("number of train dataset: {}".format(len(dataset_train_val)))
    print("number of test dataset: {}".format(len(dataset_test)))

    if cross_validation_opt:
        k=5
        k_fold_cv(dataset_train_val, k, load_model, save_dir, num_epoch, target_list, freeze_opt)

    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train_val, num_replicas=world_size,rank=rank) #!ddp
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, num_replicas=world_size,rank=rank)

        data_train_loader = DataLoader(dataset_train_val, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=batch_size, #!ddp
                            collate_fn=dataset_train_val.detection_collate, drop_last=True, pin_memory=True)
        data_val_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=batch_size,
                                collate_fn=dataset_test.detection_collate, pin_memory=True)

        if "resnet101.pth" in load_model:
            model = ResNet101(num_classes=1000)
            model.load_state_dict(torch.load(load_model), strict=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 3)
        else:
            model = ResNet101(num_classes=3)
            model.load_state_dict(torch.load(load_model), strict=False)
        
        model.to(rank) #!ddp
        model = DDP(model, device_ids=[rank], output_device=rank)

        if freeze_opt:
            for name, param in model.named_parameters():
                if "layer4.2" in name or "fc" in name:
                    param.required_grad = True
                    print(f'{name} is trainable')
                else:
                    param.required_grad = False

        train(data_train_loader, data_val_loader, model, save_dir, num_epoch, rank)
        # eval_model(model, data_val_loader, save_dir, target_list)


def train(data_loader, data_loader_eval, model, save_dir, num_epoch, rank):
    print('=========================================')
    print("start training")   
    train_losses = []
    val_losses = []
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    start_time = time.time()

    for epoch in range(num_epoch+1):
        iterate = 0
        for data, bboxes, labels, _ in data_loader:
            iterate += 1
            data = preprocess_img(data.type(torch.float32))

            data, bboxes, labels = data.to(rank), bboxes.to(rank), labels.to(rank)

            optimizer.zero_grad()

            logits = model(data, bboxes)
            loss = binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()

            if iterate % 100 == 0:
                print("epoch {} iterate {} train loss {}" .format(epoch, iterate, loss.item()))
        train_losses.append(loss.item())
        
        if epoch%5 == 0 or epoch == 100: 
            save_model = save_dir + "/model-ep{}.pth".format(epoch)
            torch.save(model.module.state_dict(), save_model)
        print("model is saved in", save_dir)

        with torch.no_grad():
            for data, bboxes, labels, _ in data_loader_eval:
                data = preprocess_img(data.type(torch.float32))
                data, bboxes, labels = data.to(rank), bboxes.to(rank), labels.to(rank)
                logits = model(data, bboxes)
                val_loss = binary_cross_entropy_with_logits(logits, labels)
            val_losses.append(val_loss.item())

        print("=========================================")
        print("epoch {} train loss {:.5f} val loss {:.5f}" .format(epoch, train_losses[epoch], val_losses[epoch]))
        print("=========================================")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"training time: {total_time:.3f} seconds")
    
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title("Loss loss curve")
    plt.savefig(os.path.join(save_dir, 'loss_graph.png'))
    return mean(train_losses), mean(val_losses)



def eval_model(model, data_loader_eval, save_dir, target_list):
    prediction_all = None
    gt_all = None
    num_classes = len(target_list)
    print('=========================================')
    print("start eval")
    vid_num = 0
    with torch.no_grad():
        for data in data_loader_eval:
            vid_num += 1
            inputs, bboxes, labels, _ = data
            inputs = inputs.type(torch.float32)
            inputs = preprocess_img(inputs).to(device)
            bboxes = bboxes.to(device)

            pred = model(inputs, bboxes)

            pred = pred.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            if prediction_all is None:
                prediction_all = pred
                gt_all = labels
            else:
                prediction_all = np.row_stack((prediction_all, pred))
                gt_all = np.row_stack((gt_all, labels))

    AP, prec_all, rec_all = compute_map(np.array(prediction_all), np.array(gt_all), num_classes)
    print(prec_all.shape)
    print(rec_all.shape)
    mAP = np.mean(AP)
    for i in range(prec_all.shape[0]):
        print(target_list[i], 'AP = {}'.format(AP[i]))
    print("mAP = {}".format(mAP))
    plot_graph(AP, prec_all, rec_all, save_dir, target_list)
    return mAP, AP[0], AP[1], AP[2]

def plot_graph(AP, prec_all, rec_all, save_dir, target_list):
    plt.figure()
    for i in range(prec_all.shape[0]):
        plt.plot(rec_all[i], prec_all[i], label ='{} AP = {:.5f}'.format(target_list[i], AP[i]))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("model performance: mAP = {:.3f}".format(np.mean(AP)))
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'precision-recall-curve.png'))


def k_fold_cv(dataset, k, load_model, save_dir, num_epoch, target_list, freeze_opt):
    if "resnet101.pth" in load_model:
        model = ResNet101(num_classes=1000)
        model.load_state_dict(torch.load(load_model), strict=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)

    else:
        model = ResNet101(num_classes=3)
        model.load_state_dict(torch.load(load_model), strict=False)

    #free last few layers
    if freeze_opt:
        for name, param in model.named_parameters():
            if "layer4.2" in name or "fc" in name:
                param.required_grad = True
                print(f'{name} is trainable')
            else:
                param.required_grad = False
                # print(f'{name} is frozen')

    dataset_size = len(dataset)
    fold_size = dataset_size // k
    indices = list(range(dataset_size))
    # print(indices)
    train_losses, val_losses, mAPs, AP0s, AP1s, AP2s = [], [], [], [], [], []

    for fold in range(k):
        print('==================fold {}=================='.format(fold))
        model = ResNet101(num_classes=1000)
        model.load_state_dict(torch.load(load_model, map_location=device), strict=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        model.to(device)

        train_indices = indices[:fold*fold_size] + indices[(fold+1)*fold_size:]
        val_indices = indices[fold * fold_size:(fold + 1) * fold_size]

        train_sampler = SubsetRandomSampler(train_indices)
        train_loader =  DataLoader(dataset, batch_size=2, sampler=train_sampler, num_workers=2, 
                                collate_fn=dataset.detection_collate, drop_last=True, pin_memory=True) #collate_fn=dataset.detection_collate, pin_memory=True
        

        val_sampler = SubsetRandomSampler(val_indices)
        val_loader =  DataLoader(dataset, batch_size=2, sampler=val_sampler, num_workers=2, 
                                collate_fn=dataset.detection_collate, drop_last=True, pin_memory=True)

        path = save_dir + "/fold{}".format(fold)
        if not(os.path.exists(path)):
            os.mkdir(path)
        save_dir_fold = path
        train_mean_loss, val_mean_loss = train(train_loader, val_loader, model, save_dir_fold, num_epoch)
        train_mean_loss, val_mean_loss= round(train_mean_loss, 5), round(val_mean_loss, 5)
        print('average training loss for fold{}: {:.5f}'.format(k, train_mean_loss))
        print('average validation loss for fold{}: {:.5f}'.format(k, val_mean_loss))
        train_losses.append(train_mean_loss)
        val_losses.append(val_mean_loss)

        mAP, AP0, AP1, AP2 = eval_model(model, val_loader, save_dir_fold, target_list)
        mAP = round(mAP, 5)
        AP0 = round(AP0, 5)
        AP1 = round(AP1, 5)
        AP2 = round(AP2, 5)
        mAPs.append(mAP)
        AP0s.append(AP0)
        AP1s.append(AP1)
        AP2s.append(AP2)

    print('train loss for each fold: {}'.format(train_losses))
    print('validation loss for each fold: {}'.format(val_losses))
    print('mAP for each fold: {}'.format(mAPs))
    print('{} AP for each fold: {}'.format(target_list[0], AP0s))
    print('{} AP for each fold: {}'.format(target_list[1], AP1s))
    print('{} AP for each fold: {}'.format(target_list[2], AP2s))
        

def pre_generate_labels(model, dataset_pre_generate, model_name):

    import json

    assert os.path.isfile(model_name), "no model predict a hammer!!"
    model.load_state_dict(torch.load(model_name))

    activity_list = ["rc", "nc", "ma"] #rc_nc_ma

    prediction_dict = {}

    vid_num = 0
    with torch.no_grad():
        for data in dataset_pre_generate:

            vid_num += 1
            imgs, bboxes, _, img_dirs_ = data
            
            print(img_dirs_)
            img_dir = img_dirs_[0]
            dir_name, file_name = os.path.split(img_dir)
            new_dir_name = dir_name.replace('img_test', 'labels_test')
            new_file_name = file_name[:-4]+'.xml'
            xml_dir = os.path.join(new_dir_name, new_file_name)
            
            print(xml_dir)
            annotation_file = os.path.join(xml_dir)
            xmltree = ET.parse(annotation_file)
            size = xmltree.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)
            # print(w)
            # break


            frame_name = os.path.split(img_dirs_[0])[-1][:-4]
            print(vid_num)

            bboxes_vis = bboxes.data.cpu().numpy()
            inputs = imgs.type(torch.float32)
            inputs = preprocess_img(inputs)

            pred = model(inputs.cuda(), bboxes.cuda())
            pred = pred.data.cpu().numpy()
            # pred = sigmoid(pred)
            pred = np.where(pred > 0.5, 1, 0) #! confidence threshold

            tl = 3

            imgs_copied = imread(img_dirs_[0])

            img_name = os.path.split(img_dirs_[0])[-1][:-4]

            prediction_dict[img_name] = {'bboxes': [], 'scores': []}

            for i in range(len(bboxes_vis)):
                _, x_min, y_min, x_max, y_max = bboxes_vis[i]
                prediction_dict[img_name]['bboxes'].append([x_min.astype(np.float), y_min.astype(np.float), x_max.astype(np.float), y_max.astype(np.float)])
                prediction_dict[img_name]['scores'].append(pred[i].astype(np.float).tolist())

                x_min = (x_min) * w / 160
                x_max = (x_max) * w / 160

                y_min = y_min * h / 120
                y_max = y_max * h / 120

                c1 = (int(x_min), int(y_min))
                c2 = (int(x_max), int(y_max))

                if x_max - x_min == 10 and y_max - y_min == 10:
                    continue
                cv2.rectangle(imgs_copied, c1, c2, (0, 255, 0), tl)

                label_vis = "h"
                pred_i = pred[i]
                for j in range(len(pred_i)):
                    if pred_i[j] == 1:
                        label_vis += "_{}".format(activity_list[j])

                cv2.putText(imgs_copied, label_vis, (c1[0], c1[1] - 2), 0, tl / 4, [255, 0, 0], thickness=2,
                            lineType=cv2.LINE_AA)

            cv2.imwrite("{}/{}.jpg".format("outputs", img_name), imgs_copied[:, :, ::-1])
            print("finished %d" % vid_num)
    print(prediction_dict)
    with open("PPE_complete_views_preds_160.json", "w") as f:
        json.dump(prediction_dict, f)
    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)