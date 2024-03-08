import argparse
import os
import pickle
import torch
from pre_video2npz import process_video, metadata2numpy
from models import TransformerModel, network_paras
import numpy as np
from collections import OrderedDict
import time
from utils import write_midi, get_random_string
from midiSynth.synth import MidiSynth
midi_synth = MidiSynth()
import moviepy.editor as mp
from pydub import AudioSegment
import json


def midi_to_mp3(midi_file, mp3_file):
    sound = AudioSegment.from_file(midi_file, format="mid")
    sound.export(mp3_file, format="mp3")


def generate(video, my_video, emotion_tag, out_dir, instrument):

    # path
    # path_ckpt = info_load_model[0] # path to ckpt dir
    # loss = info_load_model[1] # loss
    # name = 'loss_' + str(loss)
    # path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')
    path_saved_ckpt = './loss_8_params.pt'
    path_dictionary = './dictionary.pkl'
    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # video
    # video_npz = np.load(video)['input']
    video_npz = video
    # print(video_npz)
    for i in range(len(video_npz)):
        video_npz[i][0] = event2word['tempo']['Tempo_' + str(video_npz[i][0])]
        # jiang tempo zhuan hua wei dui ying de bian ma

    # outdir
    os.makedirs(out_dir, exist_ok=True)

    # config
    n_class = []   # num of classes for each token
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    n_token = len(n_class)

    # init model
    net = TransformerModel(n_class, is_training=False)
    net.cuda()
    net.eval()

    # load model
    print('[*] load model from:', path_saved_ckpt)

    try:
        net.load_state_dict(torch.load(path_saved_ckpt))
    except:
        state_dict = torch.load(path_saved_ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)

    # gen
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    cnt_tokens_all = 0 
    sidx = 0
    while sidx < 1:
        # try:
        start_time = time.time()
        print('current idx:', sidx)
        # print(video_npz)

        if n_token == 11:
            path_outfile = os.path.join(out_dir, 'emo_{}_{}'.format(str(emotion_tag), get_random_string(10)))        
            res, _ = net.inference_from_scratch(dictionary, emotion_tag, n_token, video_npz=video_npz)

        if res is None:
            continue
        np.save(path_outfile + '.npy', res)
        write_midi(res, path_outfile + '.mid', word2event, instrument)  # NOTE
        # mp3_file = midi_to_mp3(path_outfile + '.mid', path_outfile + '.mp3')
        midi_file = path_outfile + '.mid'
        wav_file = path_outfile + '.wav'
        midi_synth.play_midi(midi_file)
        midi_synth.midi2audio(midi_file, wav_file)
        print("success to mp3!")

        audio_file = AudioSegment.from_file(wav_file, format="wav")
        audio_length = audio_file.duration_seconds
        print(f"audio length: {audio_length:.3f}")

        # 获取test.mp4的播放时间
        # mp4_file = video.replace('.npz', '.mp4')
        mp4_file = my_video   #NOTE
        video_file = mp.VideoFileClip(mp4_file)
        t = video_file.duration
        print(f"mp4 time!!!!: {t:.3f}")
        # if audio_length >= t:
        import math
        if audio_length > t:
            # 裁剪test.mp3文件
            audio_cut = mp.AudioFileClip(wav_file)
            audio_cut = audio_cut.subclip(0, t)

            # 将test.mp3替换test.mp4中的声音
            result = video_file.set_audio(audio_cut)
            result.write_videofile(os.path.join("./inference", os.path.basename(mp4_file)), codec='libx264', audio_codec="aac")    # NOTE
            import shutil
            shutil.copy(os.path.join("./inference", os.path.basename(mp4_file)), os.path.join("./inference_our", os.path.basename(mp4_file)))      # NOTE
            print("success!!!!!!!!!!!!!!!!\n")
        else:
            continue

        song_time = time.time() - start_time
        word_len = len(res)
        print(f'song time:{song_time:.3f}')
        print(f'word_len:{word_len:.3f}')
        words_len_list.append(word_len)
        song_time_list.append(song_time)

        sidx += 1

    print(f'ave token time:{sum(words_len_list) / sum(song_time_list):.3f}')
    print(f'ave song time:{np.mean(song_time_list):.3f}')
    import shutil
    shutil.rmtree('./VisBeatAssets')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--emotion', type=int, required=True)  # NOTE: need
    parser.add_argument('--emotion', type=int, default=4)  # NOTE: need
    parser.add_argument('--video', type=str,
                        default='./0715.mp4')  # NOTE: need
    # parser.add_argument('--video_npz', type=str, default='/home/bing/CODE/EMOPIA-mytime/workspace/transformer/inference_npz/2_Nemo-50.npz')
    parser.add_argument('--out_dir', default="./inference")
    parser.add_argument('--our_out_dir', default="./inference_our")
    parser.add_argument('--density_threshold', type=float, default=0.2)  # NOTE: need
    parser.add_argument('--instrument', type=int, default=48)  # NOTE: need
    parser.add_argument('--is_tempo', type=int,
                        default=1)  # NOTE: need. 1 shi yong hu zhi ding tempo, 0 shi cai yong shi pin tempo
    parser.add_argument('--my_tempo', type=int, default=101)  # NOTE: need.

    args = parser.parse_args()
    # video_npz = args.video_npz
    out_dir = args.out_dir
    emotion_tag = args.emotion

    metadata = process_video(args.video, args)
    video_name = os.path.basename(args.video)
    # target_path = os.path.join(args.out_dir, video_name.replace('.mp4', '.npz'))
    # print('processing to save to %s' % target_path)
    input_numpy = metadata2numpy(metadata, args)
    # np.savez(target_path, input=input_numpy)
    # print("saved to " + str(target_path))
    generate(input_numpy, input_video, out_dir, args)
