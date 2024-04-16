# /usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib
import pickle
matplotlib.use('Agg')
import visbeat as vb
import os
import os.path as osp
import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

flow_dir = '/mnt/guided/plan2data/video2npz/flow/'


def makedirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def frange(start, stop, step=1.0):
    while start < stop:
        yield start
        start += step


# def process_all_videos(args):
#     out_json = {}
#     for i, video_name in enumerate(os.listdir(args.video_dir)):
#         if '.mp4' not in video_name:
#             continue
#         print('%d/%d: %s' % (i, len(os.listdir(args.video_dir)), video_name))
#         metadata = process_video(video_name, args)
#         out_json[video_name] = metadata
#
#     json_str = json.dumps(out_json, indent=4)
#     with open(osp.join(args.video_dir, 'metadata.json'), 'w') as f:
#         f.write(json_str)

def find_tempo(video_tempo, args):
    dictionary = pickle.load(open(args.dictionary, 'rb'))
    event2word, word2event = dictionary
    a = list(event2word['tempo'].keys())  # event2word['tempo']['Tempo_101']
    b = []
    for i in range(1, len(a)):
        if 'Tempo_' in a[i]:
            b.append(int(a[i].split('_')[1]))
        else:
            continue
    return min(b, key=lambda m: abs(m - video_tempo))    # fixme gai wei tempo dui ying de value


def process_video(video_path, args):
    figsize = (32, 4)  # zhi ding chang kuan, dan wei wei ying cun
    dpi = 200
    xrange = (0, 95)    # x zhou de zuo tu fan wei
    x_major_locator = MultipleLocator(2)    # zhi ding yi ke du jian ge

    vb.Video.getVisualTempo = vb.Video_CV.getVisualTempo    # ??????

    video = os.path.basename(video_path)
    vlog = vb.PullVideo(name=video, source_location=osp.join(video_path), max_height=360)    # ????
    vbeats = vlog.getVisualBeatSequences(search_window=None)[0]

    video_tempo, beats = vlog.getVisualTempo()
    if args.is_tempo == 0:
        tempo = find_tempo(video_tempo, args)    #dui ying dao zi dian li, zui jie jin de tempo, yi ji gai tempo de bian ma
    else:
        tempo = args.my_tempo
    print("Tempo is", tempo)
    # fixme mu qian shi shi pin jie zou jue ding yin yue jie zou
    #  ru guo yin yue gu ding jie zou, yao chong xin ji suan dang qian zhen suo chu de bar he tick de wei zhi
    vbeats_list = []
    for vbeat in vbeats:
        i_beat = round(vbeat.start / 60 * tempo * 4)    # cong kai shi dao xian zai de tempo(beat) shu * 4, zai zhe shi tick shu
        vbeat_dict = {
            'start_time': vbeat.start,   # vbeat kai shi de shi jian (s)
            'bar': int(i_beat // 16),    # dang qian dui ying de di ji ge xiao jie
            'tick': int(i_beat % 16),    # dang qian dui ying de di ji ge tick
            'weight': vbeat.weight
        }    # fixme zhe ge bar de ding yi shi gen ju ti qu dao de tempo ding yi de
        '''
        cong i_beat de ji suan guo cheng ke yi kan chu: shi pin jie zou su du jiu shi an zhao ti qu de tempo ji suan jie pai
        zhe ge tempo shi ti qu dao de tempo, ke yi zuo wei kong zhi de tempo jin xing yin yue sheng cheng
        er yi ge xiao jie de shi jian ke yi ren gong gai, zai optical_flow.py li
        '''
        if vbeat_dict['tick'] % args.resolution == 0:  # only select vbeat that lands on the xth tick
            vbeats_list.append(vbeat_dict)
    print('%d / %d vbeats selected' % (len(vbeats_list), len(vbeats)))

    npz = np.load(flow_dir + video.replace('.mp4', '.npz'), allow_pickle=True)
    print(npz.keys())
    flow_magnitude_list = npz['flow']
    fps = round(vlog.n_frames() / float(vlog.getDuration()))
    fpb = int(round(fps * 4 * 60 / tempo))  # frame per bar: mei xiao jie zhen shu

    fmpb = []  # flow magnitude per bar
    temp = np.zeros((len(flow_magnitude_list)))
    for i in range(0, len(flow_magnitude_list), fpb):    # yi mei xiao jie zhen shu wei bu chang
        mean_flow = np.mean(flow_magnitude_list[i: min(i + fpb, len(flow_magnitude_list))])    # mei xiao jie zhen de ping jun guang liu qiang du
        fmpb.append(float(mean_flow))    # xiao jie de ping jun guang liu qiang du
        temp[i: min(i + fpb, len(flow_magnitude_list))] = mean_flow

    if args.visualize:
        makedirs('image')

        height = vlog.getFrame(0).shape[0]
        thumbnails = [vlog.getFrameFromTime(t)[:, :int(height * 2.5 / 10), :] for t in list(frange(25, 35, 1))]
        thumbnails = np.concatenate(thumbnails, axis=1)
        cv2.cvtColor(thumbnails, cv2.COLOR_RGB2BGR)
        cv2.imwrite(osp.join('image', video + '_thumbnails_1' + '.png'), thumbnails)

        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=figsize, dpi=dpi)
        plt.subplots_adjust(bottom=0.15)

        x2_time = [float(item) / fps for item in list(range(len(flow_magnitude_list)))]    # mei yi zhen dui ying de miao shu
        plt.plot(x2_time[::3], flow_magnitude_list[::3], '-', color='#fff056', alpha=0.75, label="Per Frame")
        for i, fm in enumerate(fmpb):
            x_frame = [i * fpb, (i + 1) * fpb - 1]
            x_time = [x / fps for x in x_frame]    # mei xiao jie de shi jian (miao: s)
            y_fm = [fm, fm]
            if i == 0:
                plt.plot(x_time, y_fm, 'r-', label='Per Bar', lw=3)
            else:
                plt.plot(x_time, y_fm, 'r-', lw=3)
        if xrange is not None:
            plt.xlim(xrange)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlabel('Time (s)')
        plt.ylabel('Optical Flow Magnitude')
        plt.legend(loc="upper left")
        plt.savefig(osp.join('image', video + '_flow' + '.eps'), format='eps', transparent=True)
        plt.savefig(osp.join('image', video + '_flow' + '.png'), format='png', transparent=True)

        vlog.printVisualBeatSequences(figsize=figsize, save_path=osp.join('image', video + '_visbeat' + '.eps'),
                                      xrange=xrange, x_major_locator=x_major_locator)

    return {
        'duration': vlog.getDuration(),
        'tempo': tempo,    # shi pin kuai man, ji bei jing yin yue su du kuai man
        'vbeats': vbeats_list,
        'flow_magnitude_per_bar': fmpb,
    }
'''
mu qian zui zhong de yin yue jie zou (kuai man), dou shi you shi pin visbeat ti qu dao de jie zou jue ding de
'''


if __name__ == '__main__':
    vb.SetAssetsDir('.' + os.sep + 'VisBeatAssets' + os.sep)

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='/mnt/guided/plan2data/video2npz/videos/Nemo_50.mp4')
    parser.add_argument('--dictionary', type=str, default='/mnt/guided/plan2data/corpus-time/dictionary_py2.pkl')
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--resolution', type=int, default=1)
    parser.add_argument('--meta_data', default="/mnt/guided/plan2data/video2npz/metadata/Nemo_50.json")
    parser.add_argument('--is_tempo', type=int, default=0)
    parser.add_argument('--my_tempo', type=int, default=101)
    parser.add_argument("--is_path", required=True)
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
        video = args.video
        try:
            metadata = process_video(video, args)
            mfile = args.meta_data
            with open(mfile, "w") as f:
                json.dump(metadata, f)
            print("saved to", mfile)
        except:
            print("wrong: ", video)
