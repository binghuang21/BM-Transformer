import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# config
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


# utilities
def plot_hist(data, path_outfile):
    print('[Fig] >> {}'.format(path_outfile))
    data_mean = np.mean(data)
    data_std = np.std(data)

    print('mean:', data_mean)
    print(' std:', data_std)

    plt.figure(dpi=100)
    plt.hist(data, bins=50)
    plt.title('mean: {:.3f}_std: {:.3f}'.format(data_mean, data_std))
    plt.savefig(path_outfile)
    plt.close()

def traverse_dir(
        root_dir,
        extension=('mid', 'MID'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir):] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                
                
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list

# ---- define event ---- # #todo
''' 8 kinds:
     tempo: 0:   IGN     
            1:   no change
            int: tempo
     chord: 0:   IGN
            1:   no change
            str: chord types
  bar-beat: 0:   IGN     
            1: bar (bar)
            int: beat position (1...16)
      type: 0:   eos    
            1:   metrical -> emotion
            2:   note -> bar(Metrical)
            3:   emotion -> note
            4:   Rhythm
  duration: 0:   IGN
            int: length
     pitch: 0:   IGN
            int: pitch
  velocity: 0:   IGN    
            int: velocity
  strength: 0:   IGN    
            int: mo ni yin fu qiang du (mei yi ge he xian zhong bao han san ge yin fu, mei yi ge mo ni yin fu zhong bao han ji ge yin fu)
  density: 0:   IGN    
            int: mo ni yin fu mi du (mei yi ge bar li you ji ge mo ni yin fu he he xian)
  emotion:  0:   IGN
            1:   Q1
            2:   Q2
            3:   Q3
            4:   Q4
'''

# emotion map
emo_map = {
    'Q1': 1,
    'Q2': 2,
    'Q3': 3,
    'Q4': 4,

}


# event template
compound_event = {
    'tempo': 0, 
    'chord': 0,
    'bar-beat': 0, 
    'type': 0,
    'pitch': 0,
    'duration': 0,
    'velocity': 0,
    'strength': 0,
    'density': 0,
    'p_time' :0,
    'emotion': 0
}

def create_emo_event(emo_tag):
    emo_event = compound_event.copy()
    emo_event['emotion'] = emo_tag
    emo_event['type'] = 'Emotion'
    return emo_event

def create_bar_event(p_time):
    meter_event = compound_event.copy()
    meter_event['bar-beat'] = 'Bar'
    meter_event['type'] = 'Metrical'
    meter_event['p_time'] = p_time
    return meter_event


def create_rhythm_strength_event(strength, p_time):
    meter_event = compound_event.copy()
    meter_event['strength'] = strength
    # meter_event['chord'] = chord
    #todo
    meter_event['type'] = 'Rhythm'
    meter_event['p_time'] = p_time
    return meter_event


def create_rhythm_density_event(density, p_time):
    meter_event = compound_event.copy()
    meter_event['density'] = density
    #todo
    meter_event['type'] = 'Rhythm'
    meter_event['p_time'] = p_time
    return meter_event


def create_piano_metrical_event(tempo, chord, pos, p_time):
    meter_event = compound_event.copy()
    meter_event['tempo'] = tempo
    meter_event['chord'] = chord
    meter_event['bar-beat'] = pos
    #todo
    meter_event['type'] = 'Metrical'
    meter_event['p_time'] = p_time
    return meter_event


def create_piano_note_event(pitch, duration, velocity, p_time):
    note_event = compound_event.copy()
    note_event['pitch'] = pitch
    note_event['duration'] = duration
    note_event['velocity'] = velocity
    note_event['type'] = 'Note'
    note_event['p_time'] = p_time
    return note_event


def create_eos_event():
    eos_event = compound_event.copy()
    eos_event['type'] = 'EOS'
    return eos_event


# ----------------------------------------------- #
# core functions
def corpus2event_cp(path_infile, path_outfile):
    '''
    task: 2 track 
        1: piano      (note + tempo)
    ---
    remove duplicate position tokens
    '''
    
    data = pickle.load(open(path_infile, 'rb'))


    # global tag
    global_end = data['metadata']['last_bar'] * BAR_RESOL
    emo_tag = emo_map[data['metadata']['emotion']]
    
    # process
    final_sequence = []
    final_sequence.append(create_emo_event(emo_tag))    # 'emotion' type
    cur_abs_time = 0
    tempo_time_list = []
    tempo_tick_list = []
    tempo_bpm_list = []
    tempo_time_list.append(cur_abs_time)
    tempo_time_list.append(cur_abs_time)

    cur_timing = 0
    for k in range(0, global_end, TICK_RESOL):
        if data['tempos'][k] != []:
            cur_timing = data['tempos'][k][-1].time
            if cur_timing == 0:
                j = k
                tempo_tick_list.append(j)
                tempo_tick_list.append(j)
                tempo_bpm_list.append(data['tempos'][j][-1].tempo)
                tempo_bpm_list.append(data['tempos'][j][-1].tempo)
                continue
            cur_bpm = data['tempos'][j][-1].tempo
            cur_abs_time += ((cur_timing - data['tempos'][j][-1].time) / (8 * cur_bpm))
            tempo_time_list.append(cur_abs_time)
            j = k
            tempo_tick_list.append(j)
            tempo_bpm_list.append(data['tempos'][j][-1].tempo)
    cur_abs_time += ((global_end - cur_timing) / (8 * (data['tempos'][cur_timing][-1].tempo)))
    tempo_time_list.append(cur_abs_time)

    whole_abs_time = cur_abs_time
    cur_abs_time = 0
    tempo_change_k = 0

    for bar_step in range(0, global_end, BAR_RESOL):
        cur_abs_time = tempo_time_list[tempo_change_k] + (bar_step - tempo_tick_list[tempo_change_k]) / (
                    8 * tempo_bpm_list[tempo_change_k])
        bar_time = int((cur_abs_time * 100) // whole_abs_time)

        final_sequence.append(create_bar_event(bar_time))
        pos_density = 0 
        bar_sequence = []

        # --- piano track --- #
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            pos_on = False
            pos_events = []
            pos_text = 'Beat_' + str((timing-bar_step)//TICK_RESOL)
            cur_abs_time = tempo_time_list[tempo_change_k] + (timing - tempo_tick_list[tempo_change_k]) / (8 * tempo_bpm_list[tempo_change_k])
            p_time = int((cur_abs_time * 100) // whole_abs_time)

            # unpack
            t_tempos = data['tempos'][timing]
            t_chords = data['chords'][timing]
            t_notes = data['notes'][0][timing] # piano track
            if len(t_chords) and len(t_notes):
                pos_density += 2
            elif (len(t_chords) == 0) and (len(t_notes) == 0):
                pos_density += 0
            else:
                pos_density += 1

            # metrical
            #todo
            if len(t_tempos) or len(t_chords):
                # chord
                pos_strength = 0
                # tempo
                if len(t_tempos):
                    tempo_text = 'Tempo_' + str(
                        t_tempos[-1].tempo)
                    tempo_change_k += 1
                else:
                    tempo_text = 'CONTI'


                if len(t_chords):
             
                    root, quality, bass = t_chords[-1].text.split('_')
                    chord_text = root+'_'+quality
                    pos_strength += 3
                    pos_events.append(
                        create_rhythm_strength_event(pos_strength, p_time))
                else:
                    chord_text = 'CONTI'

                # create
                pos_events.append(
                    create_piano_metrical_event(
                        tempo_text, chord_text, pos_text, p_time))    # 'Metrical' type
                pos_on = True


            # note 
            if len(t_notes):
                if not pos_on:
                    pos_events.append(
                        create_piano_metrical_event(
                            'CONTI', 'CONTI', pos_text, p_time))
                pos_events.append(
                    create_rhythm_strength_event(len(t_notes), p_time))

                for note in t_notes:
                    note_pitch_text = 'Note_Pitch_' + str(note.pitch)
                    note_duration_text = 'Note_Duration_' + str(note.duration)
                    note_velocity_text = 'Note_Velocity_' + str(note.velocity)
                    
                    pos_events.append(
                        create_piano_note_event(
                            note_pitch_text, 
                            note_duration_text, 
                            note_velocity_text, p_time))    # 'Note' type
                    
            # collect & beat
            if len(pos_events):
                bar_sequence.extend(pos_events)
        if len(bar_sequence):
            final_sequence.append(
                create_rhythm_density_event(pos_density, bar_time))
            final_sequence.extend(bar_sequence)

    # BAR ending
    p_time = int((cur_abs_time * 100) // whole_abs_time)
    final_sequence.append(create_bar_event(p_time))   # 'Metrical' type, but have bar

    # EOS
    final_sequence.append(create_eos_event())    # 'EOS' type

    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
    pickle.dump(final_sequence, open(path_outfile, 'wb'))

    return len(final_sequence)


if __name__ == '__main__':
    path_root = '/mnt/guided/plan2data/corpus-time-my'
    path_indir = '/mnt/guided/plan2data/corpus-time-my/fixed/'
    path_outdir = '/mnt/guided/plan2data/corpus-time-my/events'
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(
        path_indir,
        extension=('pkl'),
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num files:', n_files)

    # run all
    len_list = []
    paths = []
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx+1, n_files))

        # paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, path_midi)
       
        # proc
        num_tokens = corpus2event_cp(path_infile, path_outfile)
        print(' > num_token:', num_tokens)
        len_list.append(num_tokens)
        paths.append(path_midi)
        


    plot_hist(
       len_list, 
       os.path.join(path_root, 'num_tokens.png')
    )

    # build dic
    d = {'filename': paths, 'num_tokens': len_list}
    df = pd.DataFrame(data=d)
    df.to_csv('len_token.csv', index=False)