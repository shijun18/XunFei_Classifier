import os
import numpy as np
import pandas as pd
import librosa
import random


def add_noise(wav,factor):
    wav_n = wav + factor*np.random.normal(0,1,len(wav))
    return wav_n

def time_shifting(wav,sr,factor):
    wav_roll = np.roll(wav,int(sr/factor))
    return wav_roll

def time_stretch(wav,factor):
    wav_stch = librosa.effects.time_stretch(wav,factor)
    return wav_stch

def pitch_shifting(wav,sr,n_steps=-5):
    wav_pitch_sf = librosa.effects.pitch_shift(wav,sr,n_steps=n_steps)
    return wav_pitch_sf


def random_aug(wav,sr):
    aug_list = ['add_noise','time_shifting','time_stretch','pitch_shifting',None]
    aug_op = random.choice(aug_list)
    if aug_op == 'add_noise':
        factor = random.choice(range(8,12,2))
        factor *= 0.001
        aug_wav = add_noise(wav,factor)
    elif aug_op == 'time_shifting':
        factor = random.choice(range(5,20,5))
        aug_wav = time_shifting(wav,sr,factor)
    elif aug_op == 'time_stretch':
        factor = random.choice(range(5,20,5))
        factor *= 0.1
        aug_wav = time_stretch(wav,factor)
    elif aug_op == 'pitch_shifting':
        factor = random.choice(range(-10,10,5))
        aug_wav = pitch_shifting(wav,sr,n_steps=factor)
    else:
        aug_wav = wav
    return aug_wav


def extract_audio_fea_aug(input_path,save_path,aug_times=None,label_flag=False):
    assert aug_times is not None
    if label_flag:
        info = {
            'id':[],
            'label':[]
        }
    else:
        info = {
            'id':[]
        }
    for item in os.scandir(input_path): 
        y,sr=librosa.load(item.path,sr=44100,mono=True,duration=min(30,librosa.get_duration(filename=item.path))) 

        for aug in range(aug_times):
            info['id'].append(f'{str(aug)}_{item.name}')
            if label_flag:
                info['label'].append(os.path.basename(input_path))
            y = random_aug(y,sr)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,n_fft=2048, hop_length=1024,fmax=16000) 

            fea_list = {
                'mfcc':librosa.feature.mfcc(y, sr, S=librosa.power_to_db(S),n_mfcc=40),# mfcc 128*T
                'energy':np.sum(np.square(abs(librosa.stft(y,n_fft=2048,hop_length=1024))),0), # 1*T
                'chroma_stft':librosa.feature.chroma_stft(y=y, sr=sr,n_fft=2048, hop_length=1024), # 12*T
                'spec_cent':librosa.feature.spectral_centroid(y=y, sr=sr,n_fft=2048, hop_length=1024), # 1*T
                'spec_bw':librosa.feature.spectral_bandwidth(y=y, sr=sr,n_fft=2048, hop_length=1024), # 1*T
                'rolloff':librosa.feature.spectral_rolloff(y=y, sr=sr,n_fft=2048, hop_length=1024),# 1*T
                'zcr':librosa.feature.zero_crossing_rate(y,hop_length=1024)
            }

            for fea in fea_list.keys():
                fea_val = fea_list[fea]
                if len(fea_val.shape) == 1:
                    fea_val = np.expand_dims(fea_val,axis=0)
                avg_val = np.mean(fea_val,axis=1)
                max_val = np.max(fea_val,axis=1)
                for i in range(len(avg_val)):
                    if f'{fea}_avg_{str(i)}' not in info.keys():
                        info[f'{fea}_avg_{str(i)}'] = []
                    if f'{fea}_max_{str(i)}' not in info.keys():
                        info[f'{fea}_max_{str(i)}'] = []
                    info[f'{fea}_avg_{str(i)}'].append(avg_val[i])
                    info[f'{fea}_max_{str(i)}'].append(max_val[i])

    csv_file = pd.DataFrame(data=info)
    csv_file.to_csv(save_path,index=False)


def extract_audio_fea(input_path,save_path,label_flag=False):
    if label_flag:
        info = {
            'id':[],
            'label':[]
        }
    else:
        info = {
            'id':[]
        }
    for item in os.scandir(input_path): 
        y,sr=librosa.load(item.path,sr=44100,mono=True,duration=min(30,librosa.get_duration(filename=item.path))) 
        
        info['id'].append(item.name)
        if label_flag:
            info['label'].append(os.path.basename(input_path))

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,n_fft=2048, hop_length=1024,fmax=16000) 
        fea_list = {
            'mfcc':librosa.feature.mfcc(y, sr, S=librosa.power_to_db(S),n_mfcc=40),# mfcc 40*T
            'energy':np.sum(np.square(abs(librosa.stft(y,n_fft=2048,hop_length=1024))),0), # 1*T
            'chroma_stft':librosa.feature.chroma_stft(y=y, sr=sr,n_fft=2048, hop_length=1024), # 12*T
            'spec_cent':librosa.feature.spectral_centroid(y=y, sr=sr,n_fft=2048, hop_length=1024), # 1*T
            'spec_bw':librosa.feature.spectral_bandwidth(y=y, sr=sr,n_fft=2048, hop_length=1024), # 1*T
            'rolloff':librosa.feature.spectral_rolloff(y=y, sr=sr,n_fft=2048, hop_length=1024),# 1*T
            'zcr':librosa.feature.zero_crossing_rate(y,hop_length=1024)
        }

        for fea in fea_list.keys():
            fea_val = fea_list[fea]
            if len(fea_val.shape) == 1:
                fea_val = np.expand_dims(fea_val,axis=0)
            avg_val = np.mean(fea_val,axis=1)
            max_val = np.max(fea_val,axis=1)
            for i in range(len(avg_val)):
                if f'{fea}_avg_{str(i)}' not in info.keys():
                    info[f'{fea}_avg_{str(i)}'] = []
                if f'{fea}_max_{str(i)}' not in info.keys():
                    info[f'{fea}_max_{str(i)}'] = []
                info[f'{fea}_avg_{str(i)}'].append(avg_val[i])
                info[f'{fea}_max_{str(i)}'].append(max_val[i])

    csv_file = pd.DataFrame(data=info)
    csv_file.to_csv(save_path,index=False)
  
if __name__ == '__main__':
    # data_path = '../dataset/Covid19/audio/train/cough/Positive'
    # save_path = '../dataset/Covid19/train_positive_aug.csv'
    # extract_audio_fea_aug(data_path,save_path,aug_times=5,label_flag=True)

    data_path = '../dataset/Covid19/audio/train/cough/Positive'
    save_path = '../dataset/Covid19/train_positive.csv'
    extract_audio_fea(data_path,save_path,label_flag=True)

    # data_path = '../dataset/Covid19/audio/train/cough/Negative'
    # save_path = '../dataset/Covid19/train_negative.csv'
    # extract_audio_fea(data_path,save_path,label_flag=True)

    # data_path = '../dataset/Covid19/audio/test'
    # save_path = '../dataset/Covid19/test.csv'
    # extract_audio_fea(data_path,save_path,False)
    
    # csv_p = '../dataset/Covid19/train_positive_aug.csv'
    csv_p = '../dataset/Covid19/train_positive.csv'
    df_p = pd.read_csv(csv_p)
    csv_n = '../dataset/Covid19/train_negative.csv'
    df_n = pd.read_csv(csv_n)
    cat_data = pd.concat([df_p, df_n], axis=0, ignore_index=True)
    save_path = '../dataset/Covid19/train_raw.csv'
    cat_data.to_csv(save_path,index=False)
