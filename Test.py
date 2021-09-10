import os
import numpy
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == '__main__':
    load_path = 'C:/ProjectData/AVECResult-Self/Text-Average-Self-BLSTM/'
    mae_list, rmse_list = [], []
    for index in range(100):
        data = numpy.genfromtxt(fname=load_path + '%04d-Eval.csv' % index, dtype=float, delimiter=',')
        mae_list.append(mean_absolute_error(data[:, 0], data[:, 1]))
        rmse_list.append(numpy.sqrt(mean_squared_error(data[:, 0], data[:, 1])))

    print(numpy.argmin(mae_list))
    print(min(mae_list), rmse_list[numpy.argmin(mae_list)])
    print()
