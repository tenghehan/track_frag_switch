import argparse
import os
import os.path as osp

from iou import bbox_iou
from tqdm import tqdm

import cv2
import numpy as np
import json


def write_imgs(pid2img, pid2gtid, dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    border = 5

    height = 256

    for pid, img_list in tqdm(pid2img.items()):
        # img_cv_list = [cv2.copyMakeBorder(cv2.imread(img_path), border, border, border, border, cv2.BORDER_CONSTANT)
        #                for img_path in img_list]
        gtid_list = pid2gtid[pid]
        img_cv_list = []
        for i, img_path in enumerate(img_list):
            gt_id = gtid_list[i]
            img = cv2.imread(img_path)
            # print(img_path)
            img_h = img.shape[0]
            img_w = img.shape[1]
            width = int(img_w * height * 1.0 / img_h)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT)
            cv2.putText(img, str(gt_id), (2, 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            img_cv_list.append(img)
        img_all = np.concatenate(img_cv_list, axis=1)
        cv2.imwrite(osp.join(dir_path, '{}.png'.format(pid)), img_all)


def write_json(pid2gtid, path):
    with open(path, 'w') as outfile:
        json.dump(pid2gtid, outfile)


def read_in_gt(file_path):
    gt_results = []
    for line in open(file_path):
        info = line.split(',')
        frame_id = int(info[0])
        gt_id = int(info[1])
        # bbox: tlwh
        bbox = [int(info[2]), int(info[3]), int(info[4]), int(info[5])]
        consider = int(info[6])
        type = int(info[7])
        visiblity = float(info[8])
        # visiblity：some wrong visiblity in gt.txt, fuck!
        if consider == 0 or type != 1:
            continue
        info_dict = {
            'frame_id': frame_id,
            'gt_id': gt_id,
            'bbox': bbox
        }
        gt_results.append(info_dict)
    return gt_results


def read_in_track_output(file_path):
    track_results = []
    for line in open(file_path):
        info = line.split(',')
        frame_id = int(info[0])
        track_id = int(info[1])
        # bbox: tlwh
        bbox = [int(info[2]), int(info[3]), int(info[4]), int(info[5])]
        info_dict = {
            'frame_id': frame_id,
            'track_id': track_id,
            'bbox': bbox
        }
        track_results.append(info_dict)
    return track_results


def calculate_gt_id(frame_id, id):
    track_bbox = []
    for info in track_results:
        if info['frame_id'] == frame_id and info['track_id'] == id:
            track_bbox = info['bbox']
            break
    assert track_bbox != [], 'frame_id: {}, id: {} tracking result can not be found'.format(frame_id, id)

    gt_id = -1
    max_iou = 0
    for info in gt_results:
        if frame_id != info['frame_id']:
            continue
        gt_bbox = info['bbox']
        iou = bbox_iou(track_bbox, gt_bbox)
        assert iou >= 0 and iou <=1, 'something wrong with iou calculation'
        if iou > max_iou:
            max_iou = iou
            gt_id = info['gt_id']

    # if max_iou < 0.5:
    #     gt_id = -1
    return gt_id


def process_imgs(images_path):

    def sort_two_list(x, y):
        xy = [(xi, yi) for xi, yi in zip(x, y)]
        sorted_xy = sorted(xy)
        sorted_x = [xi for xi, _ in sorted_xy]
        sorted_y = [yi for _, yi in sorted_xy]
        return sorted_x, sorted_y

    imgs_filenames = os.listdir(images_path)
    pid2img = {}
    pid2gtid = {}
    for img_filename in tqdm(imgs_filenames):
        id = int(img_filename.split('.')[0].split('_')[0])
        frame_id = int(img_filename.split('.')[0].split('_')[2])
        if id in pid2img.keys():
            pid2img[id].append(osp.join(images_path, img_filename))
        else:
            pid2img[id] = []
            pid2img[id].append(osp.join(images_path, img_filename))

        gt_id = calculate_gt_id(frame_id, id)
        if id in pid2gtid.keys():
            pid2gtid[id].append(gt_id)
        else:
            pid2gtid[id] = []
            pid2gtid[id].append(gt_id)

    for pid, img_list in pid2img.items():
        id_list = pid2gtid[pid]
        img_list, id_list = sort_two_list(img_list, id_list)
        pid2img[pid] = img_list
        pid2gtid[pid] = id_list

    return pid2img, pid2gtid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--track_output_path", type=str, required=True)
    parser.add_argument("--gt_file_path", type=str, required=True)
    parser.add_argument("--save_images_path", type=str, required=True)
    parser.add_argument("--save_json_path", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    track_results = read_in_track_output(args.track_output_path)
    gt_results = read_in_gt(args.gt_file_path)

    pid2img, pid2gtid = process_imgs(args.images_path)

    write_imgs(pid2img, pid2gtid, args.save_images_path)
    write_json(pid2gtid, args.save_json_path)





