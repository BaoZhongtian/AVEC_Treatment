import os
import tqdm
import numpy
from pydub import AudioSegment

if __name__ == '__main__':
    load_path = 'C:/ProjectData/AVEC2017_Treatment/Step1_CopyFiles/'
    save_path = 'C:/ProjectData/AVEC2017_Treatment/Audio_Step1_CutSegment/'
    for part_name in os.listdir(load_path):
        for fold_name in tqdm.tqdm(os.listdir(os.path.join(load_path, part_name))):
            if not os.path.exists(os.path.join(save_path, part_name, fold_name)):
                os.makedirs(os.path.join(save_path, part_name, fold_name))

            song = AudioSegment.from_wav(os.path.join(load_path, part_name, fold_name, fold_name[0:-1] + 'AUDIO.wav'))
            transcript = numpy.genfromtxt(
                fname=os.path.join(load_path, part_name, fold_name, fold_name[0:-1] + 'TRANSCRIPT.csv'), dtype=str,
                delimiter='\t')
            for index in range(1, numpy.shape(transcript)[0]):
                if transcript[index][2] == 'Ellie': continue
                song[int(float(transcript[index][0]) * 1000):int(float(transcript[index][1]) * 1000)].export(
                    os.path.join(save_path, part_name, fold_name, 'Segment_%04d.wav' % index), format='wav')
            # exit()
