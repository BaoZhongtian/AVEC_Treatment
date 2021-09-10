import os
import json
import tqdm
import numpy

if __name__ == '__main__':
    load_path = 'C:/ProjectData/AVEC2017_Treatment/Audio_Step3_NormalizationGeneration_ALL/'
    save_path = 'C:/ProjectData/AVEC2017_Treatment/TreatedData/AudioData-ALL/'

    for part_name in os.listdir(load_path):
        if not os.path.exists(os.path.join(save_path, part_name)):
            os.makedirs(os.path.join(save_path, part_name))
        for fold_name in tqdm.tqdm(os.listdir(os.path.join(load_path, part_name))[2::3]):
            patient_data = []
            for file_name in os.listdir(os.path.join(load_path, part_name, fold_name)):
                current_data = numpy.genfromtxt(fname=os.path.join(load_path, part_name, fold_name, file_name),
                                                dtype=float, delimiter=',').tolist()
                patient_data.append(current_data)
            json.dump(patient_data, open(os.path.join(save_path, part_name, fold_name + '.json'), 'w'))
