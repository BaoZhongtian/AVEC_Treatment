import os
import tqdm
import numpy
import librosa
from scipy import signal

if __name__ == '__main__':
    load_path = 'C:/ProjectData/AVEC2017_Treatment/Audio_Step1_CutSegment/'
    save_path = 'C:/ProjectData/AVEC2017_Treatment/Audio_Step2_SpectrumGeneration/'
    m_bands = 40
    s_rate = 16000
    win_length = int(0.025 * s_rate)  # Window length 15ms, 25ms, 50ms, 100ms, 200ms
    hop_length = int(0.010 * s_rate)  # Window shift  10ms
    n_fft = win_length

    for part_name in os.listdir(load_path):
        for fold_name in tqdm.tqdm(os.listdir(os.path.join(load_path, part_name))):
            if not os.path.exists(os.path.join(save_path, part_name, fold_name)):
                os.makedirs(os.path.join(save_path, part_name, fold_name))
            else:
                continue

            for file_name in os.listdir(os.path.join(load_path, part_name, fold_name)):
                y, sr = librosa.load(path=os.path.join(load_path, part_name, fold_name, file_name), sr=s_rate)
                try:
                    D = numpy.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                               window=signal.hamming, center=False)) ** 2
                except:
                    continue
                S = librosa.feature.melspectrogram(S=D, n_mels=m_bands)
                gram = librosa.power_to_db(S, ref=numpy.max)
                gram = numpy.transpose(gram, (1, 0))

                with open(os.path.join(save_path, part_name, fold_name, file_name.replace('wav', 'csv')), 'w') as file:
                    for indexX in range(len(gram)):
                        for indexY in range(len(gram[indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(gram[indexX][indexY]))
                        file.write('\n')

            # exit()
