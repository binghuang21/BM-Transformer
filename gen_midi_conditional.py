import sys
import os
from pydub import AudioSegment
import time
import glob
import numpy as np
import random
import string
import torch
import argparse
from pre_video import process_video, metadata2numpy
import moviepy.editor as mp
from midiSynth.synth import MidiSynth
midi_synth = MidiSynth()
# sys.path.append("../dataset/")

from numpy2midi_mix import numpy2midi
from model import CMT

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def cal_control_error(err_note_number_list, err_beat_number_list):
    print("err_note_number_list", err_note_number_list)
    print("err_beat_number_list", err_beat_number_list)
    print("strength control error", np.mean(err_note_number_list) / 1.83)
    print("density control error", np.mean(err_beat_number_list) / 10.90)


def generate(input_numpy, args):
    
    num_songs = int(args.num_songs)

    if args.gpus is not None:
        if not args.gpus.isnumeric():
            raise RuntimeError('Only 1 GPU is needed for inference')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    path_saved_ckpt = args.ckpt


    # change this if using another training set (see the output of decoder_n_class in train.py)
    decoder_n_class = [18, 3, 18, 129, 18, 6, 20, 102, 5025] 
    init_n_token = [7, 1, 6]


    # init model
    net = torch.nn.DataParallel(CMT(decoder_n_class, init_n_token))

    # load model
    print('[*] load model from:', path_saved_ckpt)
    if torch.cuda.is_available():
        net.cuda()
        net.eval()
        net.load_state_dict(torch.load(path_saved_ckpt))
    else:
        net.eval()
        net.load_state_dict(torch.load(path_saved_ckpt, map_location=torch.device('cpu')))

    # gen
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    sidx = 0
    # vlog_npz = np.load(file_name)['input']
    vlog_npz = input_numpy
    vlog_npz = vlog_npz[vlog_npz[:, 2] != 1]
    print(vlog_npz)

    while sidx < num_songs:
        try:
            print("new song")
            start_time = time.time()

            res, err_note_number_list, err_beat_number_list = net(is_train=False, vlog=vlog_npz, C=0.7)

            cal_control_error(err_note_number_list, err_beat_number_list)
            path_outfile = os.path.join(out_dir, 'cmt_{}'.format(get_random_string(10)))

            numpy2midi(path_outfile, res[:, [1, 0, 2, 3, 4, 5, 6]].astype(np.int32))


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
            mp4_file = args.video
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
                result.write_videofile(os.path.join(args.out_dir, os.path.basename(mp4_file)), codec='libx264', audio_codec="aac")
                print("success!!!!!!!!!!!!!!!!\n")
            else:
                continue


            song_time = time.time() - start_time
            word_len = len(res)
            print('song time:', song_time)
            print('word_len:', word_len)
            words_len_list.append(word_len)
            song_time_list.append(song_time)

            sidx += 1
        except KeyboardInterrupt:
            raise ValueError(' [x] terminated.')
    
    import shutil
    shutil.rmtree('./VisBeatAssets')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Args for generating background music")
    parser.add_argument('-c', '--ckpt', default="./loss_8_params.pt", help="Model checkpoint to be loaded")
    parser.add_argument('--video', type=str,
                        default='./0814.mp4')  # NOTE: need
    # parser.add_argument('-f', '--files', required=True, help="Input npz file of a video")
    # parser.add_argument('-f', '--files', default="./0814.npz", help="Input npz file of a video")
    parser.add_argument('-g', '--gpus', help="Id of gpu. Only ONE gpu is needed")
    parser.add_argument('-n', '--num_songs', default=1, help="Number of generated songs")
    parser.add_argument('--out_dir', default="./inference_cmt")
    args = parser.parse_args()

    # video_npz = args.video_npz
    out_dir = args.out_dir
    

    metadata = process_video(args.video, args)
    video_name = os.path.basename(args.video)
    input_numpy = metadata2numpy(metadata)


    # metadata = process_video(args.video, args)
    # video_name = os.path.basename(args.video)
    # # target_path = os.path.join(args.out_dir, video_name.replace('.mp4', '.npz'))
    # # print('processing to save to %s' % target_path)
    # input_numpy = metadata2numpy(metadata, args)
    # # np.savez(target_path, input=input_numpy)
    # # print("saved to " + str(target_path))
    # generate(input_numpy, out_dir, args)



    print("inference")
    generate(input_numpy, args)
