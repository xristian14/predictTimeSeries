import numpy as np
import tensorflow as tf
import features
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

files = ["E:/Моя папка/data/binance/BTCUSDT-1h-2020-01 - 2023-01 — копия.csv"]
sequence_length = 100
test_split = 0.05
X_learn, Y_learn, X_test, Y_test = features.csv_files_to_learn_test_data(files, features.normalize_min_max_scaler, sequence_length, test_split)
print(X_test)
print(Y_test)



#перевод источников данных [path, path, ...] в последовательности: learning(X, Y), testing(X, Y) (указывается метод нормалзизации данных)
#рассчет среднего отклонения от истинных данных для учебных и тестовых данных, при прогнозировании на n шагов вперед
#создание файлов визуализации прогнозирования на n шагов вперед (указывается длина последовательных данных, и вероятность визуализировать последовательность, чтобы можно было увидеть как различается прогнозирование последовательных данных)
#сохранение обученной сети в файл, а так же информацию об настройках при обучении
#функция расчета и записи в файл данных, спрогнозированных на n шагов вперед для всех данных и для всех файлов, учавствующих при обучении модели
#возможность пропустить шаг обучения модели, используя загруженную с диска модель
