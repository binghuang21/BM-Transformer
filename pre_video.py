#/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib
import math
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

import os
import skvideo.io
from tqdm import tqdm
from dictionary_mix import preset_event2word


vbeat_weight_percentile = [0, 0.22890276357193542, 0.4838207191278801, 0.7870981363596372, 0.891160136856027,
                           0.9645568135300789, 0.991241869205911, 0.9978208223154553, 0.9996656159745393,
                           0.9998905521344276]
fmpb_percentile = [0.008169269189238548, 0.020344337448477745, 0.02979462407529354, 0.041041795164346695,
                   0.07087484002113342, 0.10512548685073853, 0.14267262816429138, 0.19095642864704132,
                   0.5155120491981506, 0.7514784336090088, 0.9989343285560608, 1.2067525386810303, 1.6322582960128784,
                   2.031705141067505, 2.467430591583252, 2.8104422092437744]

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



TIME_PER_BAR = 2  # 暂定2s一小节

# 文字参数
ORG = (50, 50)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR = (0, 0, 255)
THICKNESS = 2


def dense_optical_flow(method, video_path, params=[], to_gray=False):
    #	print(video_path)
    assert os.path.exists(video_path)
    metadata = skvideo.io.ffprobe(video_path)
    print("video loaded successfully")
    frame, time = metadata['video']['@avg_frame_rate'].split('/')
    fps = round(float(frame) / float(time))

    # Read the video and first frame
    video = skvideo.io.vread(video_path)[:]
    n_frames = len(video)  # 总帧数
    old_frame = video[0]

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)

    flow_magnitude_list = []
    for i in tqdm(range(1, n_frames)):
        # Read the next frame
        new_frame = video[i]

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)

        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)
        flow_magnitude = np.mean(np.abs(flow))
        flow_magnitude_list.append(flow_magnitude)

        # Update the previous frame
        old_frame = new_frame

    frame_per_bar = TIME_PER_BAR * fps
    flow_magnitude_per_bar = []
    temp = np.zeros((len(flow_magnitude_list)))
    for i in np.arange(0, len(flow_magnitude_list), frame_per_bar):
        mean_flow = np.mean(flow_magnitude_list[int(i): min(int(i + frame_per_bar), len(flow_magnitude_list))])
        flow_magnitude_per_bar.append(mean_flow)
        temp[int(i): min(int(i + frame_per_bar), len(flow_magnitude_list))] = mean_flow

    # np.savez(os.path.join(flow_dir, os.path.basename(video_path).split('.')[0] + '.npz'),
    #          flow=np.asarray(flow_magnitude_list))

    # return optical_flow, flow_magnitude_list
    return flow_magnitude_per_bar, flow_magnitude_list




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
        if vbeat_dict['tick'] % 1 == 0:  # only select vbeat that lands on the xth tick
            vbeats_list.append(vbeat_dict)
    print('%d / %d vbeats selected' % (len(vbeats_list), len(vbeats)))

    # NOTE:optical_flow code
    method = cv2.calcOpticalFlowFarneback
    params = [0.5, 3, 15, 3, 5, 1.2, 0]  # default Farneback's algorithm parameters
    optical_flow, flow_magnitude_list = dense_optical_flow(method, video_path, params, to_gray=True)
    flow_magnitude_list = np.asarray(flow_magnitude_list)
    print("flow shape = ", flow_magnitude_list.shape)
    # npz = np.load("flow/" + video.replace('.mp4', '.npz'), allow_pickle=True)
    # print(npz.keys())
    # flow_magnitude_list = npz['flow']
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


RESOLUTION = 16
DIMENSION = {
    'beat'    : 0,
    'density' : 1,
    'strength': 2,
    'i_beat'  : 3,
    'n_beat'  : 4,
    'p_beat'  : 5,
}
N_DIMENSION = len(DIMENSION)


def _cal_density(flow_magnitude):
    for i, percentile in enumerate(fmpb_percentile):
        if flow_magnitude < percentile:
            return i
    return len(fmpb_percentile)


def _cal_strength(weight):
    for i, percentile in enumerate(vbeat_weight_percentile):
        if weight < percentile:
            return i
    return len(vbeat_weight_percentile)


def _get_beat_token(beat, strength, i_beat, n_beat):
    l = [[0] * N_DIMENSION]
    l[0][DIMENSION['beat']] = preset_event2word['beat']['Beat_%d' % beat]
    l[0][DIMENSION['strength']] = strength
    l[0][DIMENSION['i_beat']] = i_beat
    l[0][DIMENSION['n_beat']] = n_beat
    l[0][DIMENSION['p_beat']] = round(float(i_beat) / n_beat * 100) + 1
    return l


def _get_bar_token(density, i_beat, n_beat):
    l = [[0] * N_DIMENSION]
    l[0][DIMENSION['beat']] = preset_event2word['beat']['Bar']
    l[0][DIMENSION['density']] = density + 1
    l[0][DIMENSION['i_beat']] = i_beat
    l[0][DIMENSION['n_beat']] = n_beat
    l[0][DIMENSION['p_beat']] = round(float(i_beat) / n_beat * 100) + 1
    return l


def metadata2numpy(metadata):
    vbeats = metadata['vbeats']
    fmpb = metadata['flow_magnitude_per_bar']
    n_beat = int(math.ceil(float(metadata['duration']) / 60 * float(metadata['tempo']) * 4))

    n_bars = 0  # 已添加 bar token 个数
    l = []
    for vbeat in vbeats:
        # add bar token
        while int(vbeat['bar']) >= n_bars:
            i_beat = n_bars * RESOLUTION
            l += _get_bar_token(density=_cal_density(fmpb[n_bars]), i_beat=i_beat, n_beat=n_beat)
            n_bars += 1
        # add beat token
        i_beat = int(vbeat['bar']) * RESOLUTION + int(vbeat['tick'])
        l += _get_beat_token(beat=int(vbeat['tick']), strength=_cal_strength(vbeat['weight']), i_beat=i_beat,
                             n_beat=n_beat)
    # add empty bars
    while n_bars < len(fmpb):
        i_beat = n_bars * RESOLUTION
        l += _get_bar_token(density=_cal_density(fmpb[n_bars]), i_beat=i_beat, n_beat=n_beat)
        n_bars += 1

    return np.asarray(l, dtype=int)


if __name__ == '__main__':
    # vb.SetAssetsDir('.' + os.sep + 'VisBeatAssets' + os.sep)

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='/mnt/0814.mp4')
    parser.add_argument('--resolution', type=int, default=1)
    parser.add_argument('--out_dir', default="./")
    args = parser.parse_args()

    metadata = process_video(args.video, args)
    print("success!")
    print(metadata)
    video_name = os.path.basename(args.video)
    input_numpy = metadata2numpy(metadata)

    target_path = os.path.join(args.out_dir, video_name.replace('.mp4', '.npz'))
    np.savez(target_path, input=input_numpy)
    print("saved to " + str(target_path))
    # generate(input_numpy, out_dir, args)
    # with open("metadata.json", "w") as f:
    #     json.dump(metadata, f)
    # print("saved to metadata.json")