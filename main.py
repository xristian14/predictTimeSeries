import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import features
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


app_files_folder_name = "app_data"
if not os.path.isdir(app_files_folder_name):
    os.makedirs(app_files_folder_name)
folder_id = 0
while os.path.isdir(f"{app_files_folder_name}/{str(folder_id).rjust(4, '0')}"):
    folder_id += 1
save_folder_path = f"{app_files_folder_name}/{str(folder_id).rjust(4, '0')}"
os.makedirs(save_folder_path)
os.makedirs(f"{save_folder_path}/images")


features.save_folder_path = save_folder_path
features.data_split_sequence_length = 20 #данные в обучающую, валидационную и тестовую выборки будут добавляться последовательностями данной длины
features.sequence_length = 200 #длина последовательных данных для которой делается прогноз следующего значения
features.predict_length = 50 #количество шагов на которое будут спрогнозированы данные
features.validation_split = 0.05 #размер данных для валидации относительно всех данных
features.test_split = 0.2 #размер тестовых данных относительно всех данных
features.part_learn_predict = 0.2 #часть от учебных данных для которых будет выполнено прогнозирование на predict_length шагов вперед
features.part_test_predict = 0.5 #часть от тестовых данных для которых будет выполнено прогнозирование на predict_length шагов вперед
features.part_learn_predict_visualize = 0.5 #часть от спрогнозированных данных, которые нужно визуализировать в файлы
features.part_test_predict_visualize = 0.5 #часть от спрогнозированных данных, которые нужно визуализировать в файлы
features.is_visualize_prediction = True #визуализировать спрогнозированные последовательности, и сохранить в файлы
features.is_save_prediction_data = False #сохранять ли спрогнозированные данные. Когда True, part_learn_predict и part_test_predict не будут иметь значения, т.к. выполнится прогнозирование для всех данных, включая валидационные. part_learn_predict_visualize будет иметь значение, и будет составлять часть от всех обучающих данных, то же самое для тестовых
features.data_sources_paths = [ #источникики данных, состоящие из файлов в формате csv
    ["E:/Моя папка/data/binance/BTCUSDT-1h-2020-01 - 2023-01_lim_0-600.csv", "E:/Моя папка/data/binance/BTCUSDT-1h-2020-01 - 2023-01_lim_400-1000.csv"],
    ["E:/Моя папка/data/binance/ETHUSDT-1h-2020-01 - 2023-01_lim_0-600.csv", "E:/Моя папка/data/binance/ETHUSDT-1h-2020-01 - 2023-01_lim_400-1000.csv"]
]
is_load_model = False #загружать модель нейронной сети с файла
model_path = "" #путь к файлу модели нейронной сети
features.normalize_method = features.normalize_min_max_scaler #метод нормализации данных


X_learn, Y_learn, X_valid, Y_valid, X_test, Y_test = features.csv_files_to_learn_test_data(data_sources, features.normalize_min_max_scaler, sequence_length, data_split_sequence_length, validation_split, test_split)
x_learn_np, y_learn_np, x_valid_np, y_valid_np, x_test_np, y_test_np = np.array(X_learn), np.array(Y_learn), np.array(X_valid), np.array(Y_valid), np.array(X_test), np.array(Y_test)

model = Sequential()
model.add(Input((features.sequence_length, len(X_learn[0][0]))))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(X_learn[0][0])))
model.summary()

model.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()], optimizer='adam')
#"binary_crossentropy"
history = model.fit(x_learn_np, y_learn_np, batch_size=32, epochs=300, validation_data=(x_valid_np, y_valid_np))

plt.figure(figsize=(6,5))
plt.plot(history.history['loss'][3:])
plt.plot(history.history['val_loss'][3:])
plt.show()

print(f"длина обучающей выборки: {len(X_learn)}")
print(f"длина выборки валидации: {len(X_valid)}")
print(f"длина тестовой выборки: {len(X_test)}")

features.predict_data(model, sequence_length, predict_length, part_learn_predict, part_test_predict, True, save_folder_path)

#перевод источников данных [path, path, ...] в последовательности: learning(X, Y), testing(X, Y) (указывается метод нормалзизации данных)
#рассчет среднего отклонения от истинных данных для учебных и тестовых данных, при прогнозировании на n шагов вперед
#создание файлов визуализации прогнозирования на n шагов вперед (указывается длина последовательных данных, и вероятность визуализировать последовательность, чтобы можно было увидеть как различается прогнозирование последовательных данных)
#сохранение обученной сети в файл, а так же информацию об настройках при обучении
#функция расчета и записи в файл данных, спрогнозированных на n шагов вперед для всех данных и для всех файлов, учавствующих при обучении модели
#возможность пропустить шаг обучения модели, используя загруженную с диска модель
#расчет отклонения спрогнозированных данных от истинных делается: отдельно для каждого файла, и суммарно для всех файлов