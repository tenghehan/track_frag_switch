import argparse
import json
from collections import defaultdict
from itertools import groupby


def read_in_json(path):
    with open(path, 'r') as load_f:
        track_dict = json.load(load_f)
    return track_dict


def write_json(track_dict, path):
    with open(path, 'w') as outfile:
        json.dump(track_dict, outfile)


def process_track_result(track_dict):
    min_len = 6

    processed_track_dict = {}
    for pid, id_list in track_dict.items():
        processed_track_dict[pid] = []
        groups = [(id, len(list(segment))) for id, segment in groupby(id_list)]
        if len(groups) == 1:
            processed_track_dict[pid].append(groups[0][0])
        for id, length in groups:
            # print((id, length))
            if length >= min_len and id != -1:
                processed_track_dict[pid].append(id)
    return processed_track_dict


def cal_id_fragmentation_switch(track_dict):
    all_ids = set()
    seg_num = 0
    track_num_by_seg = defaultdict(int)
    for pid, id_list in track_dict.items():
        all_ids.update(id_list)
        seg = len(set(id_list))
        seg_num += seg
        track_num_by_seg[seg] += 1

    frag = (seg_num - len(all_ids)) * 1.0 / len(all_ids)
    switch = seg_num * 1.0 / len(track_dict)

    return frag, switch, track_num_by_seg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--processed_json_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    origin_tracking_dict = read_in_json(args.json_path)

    processed_tracking_dict = process_track_result(origin_tracking_dict)
    write_json(processed_tracking_dict, args.processed_json_path)

    frag, switch, track_num_by_seg = cal_id_fragmentation_switch(processed_tracking_dict)
    print('frag:', frag)
    print('switch:', switch)
    print('track num by #seg:')
    for seg, track_cnt in sorted(track_num_by_seg.items()):
        print(f'{seg},{track_cnt}')