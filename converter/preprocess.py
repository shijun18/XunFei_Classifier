import os
from PIL import Image
import numpy as np
import librosa
import noisereduce as nr

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



def voice_to_png(input_path,denoise=False):
    x,sr = librosa.load(input_path)
    if denoise:
        x = nr.reduce_noise(y=x, sr=sr)
    x = librosa.stft(x)
    xdb = librosa.amplitude_to_db(abs(x))
    # scale to [0,1]
    xdb = (xdb - np.min(xdb))/(np.max(xdb)-np.min(xdb))
    _,w = xdb.shape
    xdb = xdb[:,:w//2]
    voice_array = np.asarray(xdb*255,dtype=np.uint8)
    img = Image.fromarray(voice_array).convert('RGB')
    return img


def voice_to_spectrogram(input_path,save_path,denoise=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for item in os.scandir(input_path):
        if item.is_dir():
            tmp_save_path = os.path.join(save_path,item.name)
            voice_to_spectrogram(item.path,tmp_save_path,denoise)
        else:
            tmp_save_path = os.path.join(save_path,f'{os.path.splitext(item.name)[0]}.png')
            print('save as:',os.path.basename(tmp_save_path))
            img = voice_to_png(item.path,denoise)
            img.save(tmp_save_path)



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
    input_path = '../dataset/Bird_Voice/dev_data'
    save_path = '../dataset/Bird_Voice/dev'
    voice_to_spectrogram(input_path,save_path,False)