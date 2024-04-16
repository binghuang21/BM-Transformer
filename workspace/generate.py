import argparse
import os
import pickle
import torch
from utils import write_midi
from models import TransformerModel, network_paras
import numpy as np
from collections import OrderedDict
import time
from utils import write_midi, get_random_string
from midiSynth.synth import MidiSynth
midi_synth = MidiSynth()
import moviepy.editor as mp
from pydub import AudioSegment


def midi_to_mp3(midi_file, mp3_file):
            sound = AudioSegment.from_file(midi_file, format="mid")
            sound.export(mp3_file, format="mp3")

def generate(video, out_dir):

    path_saved_ckpt =  '/home/bing/CODE/EMOPIA-mytime/workspace/transformer/loss_8_params.pt'
    path_dictionary = '/home/bing/CODE/EMOPIA-mytime/workspace/transformer/dictionary.pkl'
    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # video
    video_npz = np.load(video)['input']
    print(video_npz)
    for i in range(len(video_npz)):
        video_npz[i][0] = event2word['tempo']['Tempo_' + str(video_npz[i][0])]

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
    print('[*] load model from:',  path_saved_ckpt)
    
    
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
        print(video_npz)

        if n_token == 11:
            path_outfile = os.path.join(out_dir, 'emo_{}_{}'.format( str(emotion_tag), get_random_string(10)))        
            res, _ = net.inference_from_scratch(dictionary, emotion_tag, n_token, video_npz=video_npz)
        

        if res is None:
            continue
        np.save(path_outfile + '.npy', res)
        write_midi(res, path_outfile + '.mid', word2event)
        midi_file = path_outfile + '.mid'
        wav_file = path_outfile + '.wav'
        midi_synth.play_midi(midi_file)
        midi_synth.midi2audio(midi_file, wav_file)
        print("success to mp3!")

        audio_file = AudioSegment.from_file(wav_file, format="wav")
        audio_length = audio_file.duration_seconds
        print("audio length: ", audio_length)

        # 获取test.mp4的播放时间
        mp4_file = video.replace('.npz', '.mp4')
        video_file = mp.VideoFileClip(mp4_file)
        t = video_file.duration
        print("mp4 time!!!!: ", t)
        if audio_length >= t:
            # 裁剪test.mp3文件
            audio_cut = mp.AudioFileClip(wav_file)
            audio_cut = audio_cut.subclip(0, t)

            # 将test.mp3替换test.mp4中的声音
            result = video_file.set_audio(audio_cut)
            result.write_videofile(mp4_file, codec='libx264', audio_codec="aac")
            print("success!!!!!!!!!!!!!!!!")
        else:
             continue
        
        song_time = time.time() - start_time
        word_len = len(res)
        print('song time:', song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)

        sidx += 1

    
    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion', type=int, default=2)  # NOTE: need
    parser.add_argument('--video_npz', type=str, default='/home/bing/CODE/EMOPIA-mytime/workspace/transformer/inference_npz/2_Nemo-50.npz')
    parser.add_argument('--out_dir', default="/home/bing/CODE/EMOPIA-mytime/workspace/transformer/inference")

    args = parser.parse_args()
    video_npz = args.video_npz
    out_dir = args.out_dir
    emotion_tag = args.emotion
    generate(video_npz, out_dir)
