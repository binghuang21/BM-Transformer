# /usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import visbeat3 as vb
import os.path as osp
from matplotlib.pyplot import MultipleLocator

import json
import os
import math
import argparse

import numpy as np

# from dictionary_mix import preset_event2word
vbeat_weight_percentile = [0, 0.22890276357193542, 0.4838207191278801, 0.7870981363596372, 0.891160136856027,
                           0.9645568135300789, 0.991241869205911, 0.9978208223154553, 0.9996656159745393,
                           0.9998905521344276]
fmpb_percentile = [0.008169269189238548, 0.020344337448477745, 0.02979462407529354, 0.041041795164346695,
                   0.07087484002113342, 0.10512548685073853, 0.14267262816429138, 0.19095642864704132,
                   0.5155120491981506, 0.7514784336090088, 0.9989343285560608, 1.2067525386810303, 1.6322582960128784,
                   2.031705141067505, 2.467430591583252, 2.8104422092437744]


# flow_dir = '/mnt/guided/plan2data/video2npz/flow/'


def makedirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def frange(start, stop, step=1.0):
    while start < stop:
        yield start
        start += step


def find_tempo(video_tempo, args):
    dictionary = pickle.load(open('./dictionary.pkl', 'rb'))
    event2word, word2event = dictionary
    a = list(event2word['tempo'].keys())  # event2word['tempo']['Tempo_101']
    b = []
    for i in range(1, len(a)):
        if 'Tempo_' in a[i]:
            b.append(int(a[i].split('_')[1]))
        else:
            continue
    return min(b, key=lambda m: abs(m - video_tempo))  # fixme gai wei tempo dui ying de value


def process_video(video_path, is_tempo):

    vb.Video.getVisualTempo = vb.Video_CV.getVisualTempo  # ??????

    video = os.path.basename(video_path)
    vlog = vb.PullVideo(name=video, source_location=osp.join(video_path), max_height=360)  # ????
    vbeats = vlog.getVisualBeatSequences(search_window=None)[0]

    video_tempo = vlog.getVisualTempo()
    if int(is_tempo) == 0:
        tempo = find_tempo(video_tempo,)  # dui ying dao zi dian li, zui jie jin de tempo, yi ji gai tempo de bian ma
    else:
        tempo = int(is_tempo)   # 直接改成如果是0则自动提取，如果不是0则节奏就为is_tempo
    print("Tempo is", tempo)
    vbeats_list = []
    for vbeat in vbeats:
        print("vbeat.start type = ", type(vbeat.start))
        i_beat = round(
            vbeat.start / 60 * tempo * 4)  # cong kai shi dao xian zai de tempo(beat) shu * 4, zai zhe shi tick shu
        vbeat_dict = {
            'start_time': vbeat.start,  # vbeat kai shi de shi jian (s)
            'bar': int(i_beat // 16),  # dang qian dui ying de di ji ge xiao jie
            'tick': int(i_beat % 16),  # dang qian dui ying de di ji ge tick
            'weight': vbeat.weight
        }  # fixme zhe ge bar de ding yi shi gen ju ti qu dao de tempo ding yi de
        if vbeat_dict['tick'] % 1 == 0:  # only select vbeat that lands on the xth tick
            vbeats_list.append(vbeat_dict)
    print('%d / %d vbeats selected' % (len(vbeats_list), len(vbeats)))


    return {
        'duration': vlog.getDuration(),
        'tempo': tempo,  # shi pin kuai man, ji bei jing yin yue su du kuai man
        'vbeats': vbeats_list,
        # 'flow_magnitude_per_bar': fmpb,
    }


RESOLUTION = 16
# DENSITY_THRESHOLD = 0.2  # yu zhi: da yu zhe ge yu zhi de cai bei zhi dao(kong zhi sheng cheng) yuan 0.2
DIMENSION = {
    'tempo': 0,
    'bar-beat': 1,
    'type': 2,
    'strength': 3,
    'density': 4,
    'p_time': 5,
}
N_DIMENSION = len(DIMENSION)

# todo ru guo geng gai le lei xing, yao dui ying dictionary xiu gai
TYPE_CLASS = {
    'EOS': 0,
    'Emotion': 1,
    'Metrical': 2,
    'Note': 3,
    'Rhythm': 4,
}

# event template
compound_event = {
    'tempo': 0,  # fixme add tempo
    'chord': 0,
    'bar-beat': 0,  # add
    'type': 0,  # add
    'pitch': 0,
    'duration': 0,
    'velocity': 0,
    'strength': 0,  # add
    'density': 0,  # add
    'p_time': 0,  # add
    'emotion': 0
}


# def _cal_density(flow_magnitude):
#     for i, percentile in enumerate(fmpb_percentile):  # yong bai fen bi lai ding ji
#         if flow_magnitude < percentile:  # fmpb_percentile shi zi ji ding yi de qiang du deng ji
#             return i
#     return len(fmpb_percentile)


def _cal_strength(weight):
    for i, percentile in enumerate(vbeat_weight_percentile):  # fixme gai bian vbeat_weight_percentile de fem ji
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


def metadata2numpy(metadata, density_threshold):
    vbeats = metadata['vbeats']  # zai tick shang de shi pin jie zou
    # fmpb = metadata['flow_magnitude_per_bar']
    duration = metadata['duration']
    tempo = metadata['tempo']
    # shi pin de zong jie pai shu * 4 (zhe li shi shi pin de zong tick shu)
    # n_beat = int(math.ceil(float(metadata['duration']) / 60 * float(metadata['tempo']) * 4))

    n_bars = 0  # 已添加 bar token 个数
    l = []
    bar_l = []
    density = 0
    for vbeat in vbeats:
        # add bar token
        while int(vbeat['bar']) >= n_bars:  # dang qian suo chu de vbeat zai dang qian bar nei
            # i_beat = n_bars * RESOLUTION    # dang qian bar dui ying de tick shu
            if len(bar_l) != 0:
                l += _get_density_token(tempo=tempo, density=density, n_bars=density_bar, p_time=bar_time)
                l += bar_l
            density = 0
            bar_l = []
            p_time = int(vbeat['start_time'] * 100 // duration)
            bar_time = p_time
            density_bar = n_bars
            l += _get_bar_token(tempo=tempo, n_bars=density_bar, p_time=bar_time)
            # if vbeat['weight'] >= DENSITY_THRESHOLD:
            #     density += 1
            #     bar_l.append(_get_strength_token(strength=_cal_strength(vbeat['weight']), p_time=p_time))   # fmpb[n_bars] shi dang qian xiao jie de guang liu qiang du
            n_bars += 1  # yong guang liu dai ti mo ni yin fu mi du
        # add beat token
        p_time = int(vbeat['start_time'] * 100 // duration)  # mei yi ge shi pin jie zou dui ying de bi lv
        if vbeat['weight'] >= density_threshold:
            density += 1
            bar_l += _get_strength_token(tempo=tempo, strength=_cal_strength(vbeat['weight']), n_bars=density_bar,
                                         p_time=p_time)  # mo ni yin fu qiang du

    if len(bar_l) != 0:
        l += _get_density_token(tempo=tempo, density=density, n_bars=density_bar, p_time=bar_time)
        l += bar_l  # deng suo you dou jia jin qu zhi hou, zui hou yi xiao jie de ye jia jin qu

    return np.asarray(l, dtype=int)  # fixme ci ming ling shi bian wei shu zu, dan shi mu qian shi zi dian lei xing


if __name__ == '__main__':
    vb.SetAssetsDir('.' + os.sep + 'VisBeatAssets' + os.sep)

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str,
                        default='/home/bing/CODE/111.mp4')  # NOTE: need
    # todo: geng huan shu ju ji hou yao zhu yi geng gai zi dian wei zhi
    parser.add_argument('--is_tempo', type=int,
                        default=0)  # NOTE: need. 1 shi yong hu zhi ding tempo, 0 shi cai yong shi pin tempo
    parser.add_argument('--my_tempo', type=int, default=101)  # NOTE: need.
    parser.add_argument("--is_path", type=int, default=0)
    parser.add_argument('--out_dir', default="/home/bing/CODE/create-usr-demo/inference")

    args = parser.parse_args()

    is_path = args.is_path
    if is_path == '1':
        video_path = args.video
        mfile_path = args.meta_data
        for flie in os.listdir(video_path):
            file_path = os.path.join(video_path, flie)

            metadata = process_video(file_path, args)
            try:
                with open(os.path.join(mfile_path, flie.split('.')[0] + '.json'), "w") as f:
                    json.dump(metadata, f)
                print("saved to", os.path.join(mfile_path, flie.split('.')[0] + '.json'))
            except:
                print("wrong: ", file_path)
    else:
        # video = args.video
        metadata = process_video(args.video, args)
        video_name = os.path.basename(args.video)
        target_path = os.path.join(args.out_dir, video_name.replace('.mp4', '.npz'))
        print('processing to save to %s' % target_path)
        # input_numpy = metadata2numpy(metadata_temp)
        input_numpy = metadata2numpy(metadata, args)
        np.savez(target_path, input=input_numpy)
        print("saved to " + str(target_path))
