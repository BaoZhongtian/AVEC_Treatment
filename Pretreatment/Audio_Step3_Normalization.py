import os
import tqdm
import numpy
from Tools import search_fold
from sklearn.preprocessing import scale

if __name__ == '__main__':
    load_path = 'C:/ProjectData/AVEC2017_Treatment/Audio_Step2_SpectrumGeneration_All/'
    save_path = 'C:/ProjectData/AVEC2017_Treatment/Audio_Step3_NormalizationGeneration_All/'
    total_path = search_fold(load_path)

    total_data = []
    for filename in tqdm.tqdm(total_path):
        data = numpy.genfromtxt(fname=filename, dtype=float, delimiter=',')
        total_data.extend(data)
    print(numpy.shape(total_data))

    start_position = 0
    total_data = scale(total_data)

    for filename in tqdm.tqdm(total_path):
        data = numpy.genfromtxt(fname=filename, dtype=float, delimiter=',')
        part_data = total_data[start_position:start_position + numpy.shape(data)[0]]
        start_position += numpy.shape(data)[0]

        fold_path = filename[:-filename[::-1].find('\\')].replace(load_path, save_path)
        if not os.path.exists(fold_path): os.makedirs(fold_path)

        with open(filename.replace(load_path, save_path), 'w') as file:
            for indexX in range(len(part_data)):
                for indexY in range(len(part_data[indexX])):
                    if indexY != 0: file.write(',')
                    file.write(str(part_data[indexX][indexY]))
                file.write('\n')
        # exit()
