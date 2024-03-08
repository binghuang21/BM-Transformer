#/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib

matplotlib.use('Agg')
import visbeat3 as vb
import os
import os.path as osp
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def makedirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def frange(start, stop, step=1.0):
    while start < stop:
        yield start
        start += step


def process_all_videos(args):
    out_json = {}
    for i, video_name in enumerate(os.listdir(args.video_dir)):
        if '.mp4' not in video_name:
            continue
        print('%d/%d: %s' % (i, len(os.listdir(args.video_dir)), video_name))
        metadata = process_video(video_name, args)
        out_json[video_name] = metadata

    json_str = json.dumps(out_json, indent=4)
    with open(osp.join(args.video_dir, 'metadata.json'), 'w') as f:
        f.write(json_str)


def process_video(video_path, args):

    vb.Video.getVisualTempo = vb.Video_CV.getVisualTempo

    video = os.path.basename(video_path)
    vlog = vb.PullVideo(name=video, source_location=osp.join(video_path), max_height=360)
    vbeats = vlog.getVisualBeatSequences(search_window=None)[0]

    tempo = vlog.getVisualTempo()
    print("Tempo is", tempo)
    vbeats_list = []
    for vbeat in vbeats:
        i_beat = np.round(vbeat.start / 60 * tempo * 4)
        vbeat_dict = {
            'start_time': vbeat.start,
            'bar'       : int(i_beat // 16),
            'tick'      : int(i_beat % 16),
            'weight'    : vbeat.weight
        }
        if vbeat_dict['tick'] % args.resolution == 0:  # only select vbeat that lands on the xth tick
            vbeats_list.append(vbeat_dict)
    print('%d / %d vbeats selected' % (len(vbeats_list), len(vbeats)))

    # npz = np.load("flow/" + video.replace('.mp4', '.npz'), allow_pickle=True)
    npz = np.load("flow/0814.npz",  allow_pickle=True)
    print(npz.keys())
    flow_magnitude_list = npz['flow']    # NOTE:zhe li yong le di yi ge dai ma
    print("flow shape = ", flow_magnitude_list.shape)
    fps = np.round(vlog.n_frames() / float(vlog.getDuration()))
    fpb = int(np.round(fps * 4 * 60 / tempo))  # frame per bar

    fmpb = []  # flow magnitude per bar
    temp = np.zeros((len(flow_magnitude_list)))
    for i in range(0, len(flow_magnitude_list), fpb):
        mean_flow = np.mean(flow_magnitude_list[i: min(i + fpb, len(flow_magnitude_list))])
        fmpb.append(float(mean_flow))
        temp[i: min(i + fpb, len(flow_magnitude_list))] = mean_flow

    return {
        'duration'              : vlog.getDuration(),
        'tempo'                 : tempo,
        'vbeats'                : vbeats_list,
        'flow_magnitude_per_bar': fmpb,
    }


if __name__ == '__main__':
    # vb.SetAssetsDir('.' + os.sep + 'VisBeatAssets' + os.sep)

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='/mnt/0814.mp4')
    parser.add_argument('--resolution', type=int, default=1)
    args = parser.parse_args()

    metadata = process_video(args.video, args)
    print("metadata len = ", metadata)
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)
    print("saved to metadata.json")