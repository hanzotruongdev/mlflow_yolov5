"""
Inference code for YOLOv5
Offline evaluation pipeline
"""
import mlflow
import mlflow.pytorch
mlflow.set_tracking_uri('http://mlflow-tracking.vinbrain.net:8899')


import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
from models.common import Conv, DWConv
from collections import namedtuple

print('*'*30, os.getcwd())



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument("--task", type=str, default= 'test')
    parser.add_argument('--conf_thres', type=float, default=0.001)
    parser.add_argument('--iou_thres', type=float, default=0.6)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_txt', action='store_true')
    parser.add_argument('--save_conf', action='store_true')
    parser.add_argument('--save_json', action='store_true')
    parser.add_argument('--single_cls', action='store_true')
    
    opt = parser.parse_args()
    
    with mlflow.start_run(experiment_id=3, run_name="test yolo model"):

        """
        Configuration
        """
        data_yaml   = 'data/bdd100k.yaml'
        weight = 'checkpoints/best.pt'
        batch_size = 32
        imgsz=640
        device='cuda:0'

        """
        load model
        """
    #     model = torch.load(weight, map_location='cuda:0')['model'].float().fuse().eval()


        model_name = 'yolov5 detection'
        model_version = 2
        model = mlflow.pytorch.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        ).float().fuse().eval()

        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        model.half()
        model.eval()

        """
        load data
        """
        with open(opt.data) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        nc = data['nc']  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()


        # Dataloader
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half()) # run once
        path = data['test']
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)[0]

        """
        define metrics
        """
        confusion_matrix = ConfusionMatrix(nc=nc)
        names = {k: v for k, v in enumerate(model.names)}
        s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'mean f1')
        p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        stats = []
        seen = 0

        """
        # run inference
        """
        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
            img = img.to(device, non_blocking=True)
            img = img.half() # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width

            with torch.no_grad():
                # Run model
                t = time_synchronized()
                inf_out, train_out = model(img, augment=opt.augment)  # inference and training outputs
                t0 += time_synchronized() - t

                # Run NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
                lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_txt else []  # for autolabelling
                t = time_synchronized()
                output = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=lb)
                t1 += time_synchronized() - t

            # Statistics per image
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path = Path(paths[si])
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))


        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=None, save_dir=None, names=names)
            p, r, f1, ap50, ap = p[:, 0], r[:, 0], f1[:,0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, mf1, map50, map = p.mean(), r.mean(), f1.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        print(s)
        pf = '%20s' + '%12.3g' * 7  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map, mf1))


        mlflow.log_metric('map50', map50)
        mlflow.log_metric('map', map)
        mlflow.log_metric('f1', mf1)


        # Print speeds
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)