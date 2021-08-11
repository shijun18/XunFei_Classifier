from genericpath import exists
from math import log
import os
from PIL import Image
import numpy as np
import librosa
import noisereduce as nr
from PIL import ImageFilter
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


def stretch(x, factor, nfft=2048):
    '''
    stretch an audio sequence by a factor using FFT of size nfft converting to frequency domain
    :param x: np.ndarray, audio array in PCM float32 format
    :param factor: float, stretching or shrinking factor, depending on if its > or < 1 respectively
    :return: np.ndarray, time stretched audio
    '''
    stft = librosa.core.stft(x, n_fft=nfft).transpose()  # i prefer time-major fashion, so transpose
    stft_rows = stft.shape[0]
    stft_cols = stft.shape[1]

    times = np.arange(0, stft.shape[0], factor)  # times at which new FFT to be calculated
    hop = nfft/4                                 # frame shift
    stft_new = np.zeros((len(times), stft_cols), dtype=np.complex_)
    phase_adv = (2 * np.pi * hop * np.arange(0, stft_cols))/ nfft
    phase = np.angle(stft[0])

    stft = np.concatenate( (stft, np.zeros((1, stft_cols))), axis=0)

    for i, time in enumerate(times):
        left_frame = int(np.floor(time))
        local_frames = stft[[left_frame, left_frame + 1], :]
        right_wt = time - np.floor(time)                        # weight on right frame out of 2
        local_mag = (1 - right_wt) * np.absolute(local_frames[0, :]) + right_wt * np.absolute(local_frames[1, :])
        local_dphi = np.angle(local_frames[1, :]) - np.angle(local_frames[0, :]) - phase_adv
        local_dphi = local_dphi - 2 * np.pi * np.floor(local_dphi/(2 * np.pi))
        stft_new[i, :] =  local_mag * np.exp(phase*1j)
        phase += local_dphi + phase_adv

    return librosa.core.istft(stft_new.transpose())


def preprocess(input_path,save_path):
    save_path = f'{save_path}/{os.path.basename(input_path)}'
    img = Image.open(input_path).convert('RGB')
    # print(img.size)

    w,h = img.size
    if w <= h: 
        if w < h//3:
            paste_time = 2*(h//3) // w
            new_image = Image.new(size=(paste_time*w,h),mode='RGB')
            for i in range(paste_time):
                new_image.paste(img,(i*w,0))
            img = new_image
    else:
        if h < w//3:
            paste_time = 3*(w//2) // h
            new_image = Image.new(size=(w,paste_time*h),mode='RGB')
            for i in range(paste_time):
                new_image.paste(img,(0,i*h))
            img = new_image

    img.save(save_path)


def make_data(input_path,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for entry in os.scandir(input_path):
        print(entry.path)
        if entry.is_dir():
            tmp_save_path = os.path.join(save_path,entry.name)
            make_data(entry.path,tmp_save_path)
        else:
            preprocess(entry.path,save_path)


def extract_frame(input_path,save_path):
    import cv2
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for video in os.scandir(input_path):
        tmp_save_path = save_path + '/' + os.path.splitext(video.name)[0]
        video_capture = cv2.VideoCapture(video.path)

        success, frame = video_capture.read()
        i = 0
        fps = 30
        while success:
            i += 1
            if (i%fps == 0):
                address = tmp_save_path + f'_{str(i)}.jpg'
                cv2.imwrite(address,frame)
                print('save image:%d'%i)
            success, frame = video_capture.read()



def voice_to_png(input_path,denoise=False,palette=None):
    x,sr = librosa.load(input_path,sr=44100,duration=min(56,librosa.get_duration(filename=input_path)))
    if denoise:
        x = nr.reduce_noise(y=x, sr=sr)
    melspec = librosa.feature.melspectrogram(y=x, sr=sr,n_fft=2048,hop_length=1024,n_mels=128)
    log_melspec = librosa.power_to_db(melspec,ref=np.max)
    # scale to [0,1]
    log_melspec = (log_melspec - np.min(log_melspec))/(np.max(log_melspec)-np.min(log_melspec))
    
    # extract
    mel_sum = np.sum(log_melspec,axis=0)
    log_melspec = log_melspec[:,mel_sum > 0.01*log_melspec.shape[0]]
    mel_sum = np.sum(log_melspec,axis=1)
    log_melspec = log_melspec[mel_sum > 0.01*log_melspec.shape[1]]

    voice_array = np.asarray(log_melspec*255,dtype=np.uint8)
    img = Image.fromarray(voice_array).convert('RGB')
    img = img.filter(ImageFilter.MedianFilter(3))
    img = img.convert('P')
    if palette is not None:
        img.putpalette(palette) 
    return img


def random_aug(wav,sr):
    aug_list = ['add_noise','time_shifting','time_stretch','pitch_shifting']
    aug_op = random.choice(aug_list)
    if aug_op == 'add_noise':
        factor = random.choice(range(8,12))
        factor *= 0.001
        aug_wav = add_noise(wav,factor)
    elif aug_op == 'time_shifting':
        factor = random.choice(range(5,15,2))
        aug_wav = time_shifting(wav,sr,factor)
    elif aug_op == 'time_stretch':
        factor = random.choice(range(5,15,2))
        factor *= 0.1
        aug_wav = time_stretch(wav,factor)
    elif aug_op == 'pitch_shifting':
        factor = random.choice(range(-5,5))
        aug_wav = pitch_shifting(wav,sr,n_steps=factor)
    return aug_wav


def voice_to_png_aug(input_path,denoise=False,palette=None):
    x,sr = librosa.load(input_path,sr=44100,duration=min(56,librosa.get_duration(filename=input_path)))
    if denoise:
        x = nr.reduce_noise(y=x, sr=sr)
    x = random_aug(x,sr)
    melspec = librosa.feature.melspectrogram(y=x, sr=sr,n_fft=2048,hop_length=1024,n_mels=128)
    log_melspec = librosa.power_to_db(melspec,ref=np.max)
    # scale to [0,1]
    log_melspec = (log_melspec - np.min(log_melspec))/(np.max(log_melspec)-np.min(log_melspec))
    
    # extract
    mel_sum = np.sum(log_melspec,axis=0)
    log_melspec = log_melspec[:,mel_sum > 0.01*log_melspec.shape[0]]
    mel_sum = np.sum(log_melspec,axis=1)
    log_melspec = log_melspec[mel_sum > 0.01*log_melspec.shape[1]]

    voice_array = np.asarray(log_melspec*255,dtype=np.uint8)
    img = Image.fromarray(voice_array).convert('RGB')
    img = img.filter(ImageFilter.MedianFilter(3))
    img = img.convert('P')
    if palette is not None:
        img.putpalette(palette) 
    return img


def voice_to_spectrogram(input_path,save_path,denoise=False,palette=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for item in os.scandir(input_path):
        if item.is_dir():
            tmp_save_path = os.path.join(save_path,item.name)
            voice_to_spectrogram(item.path,tmp_save_path,denoise,palette)
        else:
            # print(item.name)
            tmp_save_path = os.path.join(save_path,f'{os.path.splitext(item.name)[0]}.png')
            try:
                img = voice_to_png(item.path,denoise,palette)
                img.save(tmp_save_path)
                print('save as:',os.path.basename(tmp_save_path))
            except:
                print(item.name)
                continue


def voice_to_spectrogram_aug(input_path,save_path,denoise=False,palette=None,aug_times=10):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for item in os.scandir(input_path):
        if item.is_dir():
            tmp_save_path = os.path.join(save_path,item.name)
            voice_to_spectrogram(item.path,tmp_save_path,denoise,palette,aug_times)
        else:
            for i in range(aug_times):
                tmp_save_path = os.path.join(save_path,f'{os.path.splitext(item.name)[0]}_{str(i)}.png')
                print('save as:',os.path.basename(tmp_save_path))
                try:
                    img = voice_to_png_aug(item.path,denoise,palette)
                    img.save(tmp_save_path)
                except:
                    continue



if __name__ == '__main__':
    # input_path = '../dataset/Adver_Material/pre_train'
    # save_path = '../dataset/Adver_Material/train'
    # make_data(input_path,save_path)

    # input_path = '../dataset/Adver_Material/pre_test'
    # save_path = '../dataset/Adver_Material/test'
    # make_data(input_path,save_path)   

    # for i in range(4):
    #     input_path = '../dataset/Farmer_Work/video_train/' + str(i)
    #     save_path = '../dataset/Farmer_Work/train/' + str(i)
    #     extract_frame(input_path,save_path)
    
    
    palette = Image.open('../dataset/Temp_Freq/train/0/120_0.png').convert('P').getpalette()

    # input_path = '../dataset/Bird_Voice/train_data'
    # save_path = '../dataset/Bird_Voice/train'
    # voice_to_spectrogram(input_path,save_path,False,palette)

    # input_path = '../dataset/Bird_Voice/dev_data'
    # save_path = '../dataset/Bird_Voice/dev'
    # voice_to_spectrogram(input_path,save_path,False,palette)

    # input_path = '../dataset/Bird_Voice/test_data'
    # save_path = '../dataset/Bird_Voice/test'
    # voice_to_spectrogram(input_path,save_path,False,palette)

    # input_path = '../dataset/Family_Env/audio/train'
    # save_path = '../dataset/Family_Env/train'
    # voice_to_spectrogram(input_path,save_path,False,palette)

    # input_path = '../dataset/Family_Env/audio/test'
    # save_path = '../dataset/Family_Env/test'
    # voice_to_spectrogram(input_path,save_path,False,palette)

    
    input_path = '../dataset/Covid19/audio/train/cough/Negative'
    save_path = '../dataset/Covid19/train/0'
    voice_to_spectrogram(input_path,save_path,False,palette)

    input_path = '../dataset/Covid19/audio/train/cough/Positive'
    save_path = '../dataset/Covid19/train/1'
    voice_to_spectrogram_aug(input_path,save_path,False,palette,aug_times=5)

    input_path = '../dataset/Covid19/audio/test'
    save_path = '../dataset/Covid19/test'
    voice_to_spectrogram(input_path,save_path,False,palette)