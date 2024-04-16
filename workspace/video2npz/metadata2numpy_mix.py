# /usr/bin/env python
# -*- coding: UTF-8 -*-
import json
import os
import math
import argparse

import numpy as np

from dictionary_mix import preset_event2word
from stat_mix import vbeat_weight_percentile, fmpb_percentile

RESOLUTION = 16
DENSITY_THRESHOLD = 0.2
DIMENSION = {
    'tempo': 0,
    'bar-beat': 1,
    'type': 2,
    'strength': 3,
    'density': 4,
    'p_time': 5,
}
N_DIMENSION = len(DIMENSION)


TYPE_CLASS = {
    'EOS': 0,
    'Emotion': 1,
    'Metrical': 2,
    'Note': 3,
    'Rhythm': 4,
}

# event template
compound_event = {
    'tempo': 0,    # add tempo
    'chord': 0,
    'bar-beat': 0,    # add
    'type': 0,    # add
    'pitch': 0,
    'duration': 0,
    'velocity': 0,
    'strength': 0,    # add
    'density': 0,   # add
    'p_time' :0,    # add
    'emotion': 0
}

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

def _get_strength_token(tempo, strength, n_bars, p_time):
    l = [[0] * N_DIMENSION]
    l[0][DIMENSION['tempo']] = tempo
    l[0][DIMENSION['strength']] = strength
    l[0][DIMENSION['bar-beat']] = n_bars
    l[0][DIMENSION['type']] = TYPE_CLASS['Rhythm']
    l[0][DIMENSION['p_time']] = p_time
    return l

def _get_density_token(tempo, density, n_bars, p_time):
    l = [[0] * N_DIMENSION]
    l[0][DIMENSION['tempo']] = tempo
    l[0][DIMENSION['density']] = density
    l[0][DIMENSION['bar-beat']] = n_bars
    l[0][DIMENSION['type']] = TYPE_CLASS['Rhythm']
    l[0][DIMENSION['p_time']] = p_time
    return l

def _get_bar_token(tempo, n_bars, p_time):
    l = [[0] * N_DIMENSION]
    l[0][DIMENSION['tempo']] = tempo
    l[0][DIMENSION['bar-beat']] = n_bars
    l[0][DIMENSION['type']] = TYPE_CLASS['Metrical']
    l[0][DIMENSION['p_time']] = p_time
    return l


def metadata2numpy(metadata):
    vbeats = metadata['vbeats']
    fmpb = metadata['flow_magnitude_per_bar']
    duration = metadata['duration']
    tempo = metadata['tempo']
    n_beat = int(math.ceil(float(metadata['duration']) / 60 * float(metadata['tempo']) * 4))

    n_bars = 0  # 已添加 bar token 个数
    l = []
    bar_l = []
    density = 0
    for vbeat in vbeats:
        # add bar token
        while int(vbeat['bar']) >= n_bars:
            if len(bar_l) != 0 :
                l += _get_density_token(tempo=tempo, density=density, n_bars=density_bar, p_time=bar_time)
                l += bar_l
            density = 0
            bar_l = []
            p_time = int(vbeat['start_time'] * 100 // duration)
            bar_time = p_time
            density_bar = n_bars
            l+= _get_bar_token(tempo=tempo, n_bars=density_bar, p_time=bar_time)
            n_bars += 1
        # add beat token
        p_time = int(vbeat['start_time'] * 100 // duration)
        if vbeat['weight'] >= DENSITY_THRESHOLD:
            density += 1
            bar_l += _get_strength_token(tempo=tempo, strength=_cal_strength(vbeat['weight']), n_bars=density_bar, p_time=p_time)    # mo ni yin fu qiang du

    if len(bar_l) != 0:
        l += _get_density_token(tempo=tempo, density=density, n_bars=density_bar, p_time=bar_time)
        l += bar_l

    return np.asarray(l, dtype=int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default="/mnt/guided/plan2data/video2npz/inference/")
    parser.add_argument('--video', default="/mnt/guided/plan2data/video2npz/videos")
    parser.add_argument('--metadata', default="/mnt/guided/plan2data/video2npz/metadata") 
    parser.add_argument("--is_path", required=True)
    args = parser.parse_args()
    is_path = args.is_path
    if is_path == '1':
        video_path = args.video
        meta_path = args.metadata
        for flie in os.listdir(video_path):
            mfile = os.path.join(meta_path, flie.split('.')[0] + '.json')
            with open(mfile) as f:
                metadata = json.load(f)
            target_path = os.path.join(args.out_dir, flie.replace('.mp4', '.npz'))
            try:
                print('processing to save to %s' % target_path)
                input_numpy = metadata2numpy(metadata)
                np.savez(target_path, input=input_numpy)
                print("saved to " + str(target_path))
            except:
                print("---------------wrong: ", target_path)

    else:
        video_name = os.path.basename(args.video)

        with open(args.metadata) as f:
            metadata = json.load(f)

        target_path = os.path.join(args.out_dir, video_name.replace('.mp4', '.npz'))

        print('processing to save to %s' % target_path)
        input_numpy = metadata2numpy(metadata)
        np.savez(target_path, input=input_numpy)
        print("saved to " + str(target_path))

