import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import normalizers
import features
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# x_hour = [
#     [[0.5],[0.4],[0.5],[0.4]],
#     [[0.4],[0.5],[0.4],[0.5]],
#     [[0.5],[0.4],[0.5],[0.4]],
#     [[0.4],[0.5],[0.4],[0.5]],
#     [[0.5],[0.4],[0.5],[0.4]],
#     [[0.4],[0.5],[0.4],[0.5]]
# ]
# y_hour = [
#     [0.1],
#     [1],
#     [0.9],
#     [0],
#     [0.9],
#     [0]
# ]
# x_day = [
#     [[0.9],[1]],
#     [[0.2],[0.1]],
#     [[0.1],[0]],
#     [[0.8],[1]],
#     [[0.1],[0.2]],
#     [[0.9],[0.8]]
# ]
# x_day = [
#     [[0.1],[0.1]],
#     [[0.2],[0.2]],
#     [[1],[1]],
#     [[0.5],[0.5]],
#     [[0.6],[0.6]],
#     [[1],[1]]
# ]

# main_input = Input(shape=(len(x_hour[0]),len(x_hour[0][0])), name='main_input')
# lstm_out = LSTM(8)(main_input)
# day_input = Input(shape=(len(x_day[0]),len(x_day[0][0])), name='day_input')
# lstm_day_out = LSTM(8)(day_input)
# x = tf.keras.layers.concatenate([lstm_out, lstm_day_out])
# main_output = Dense(1, activation='sigmoid', name='main_output')(x)
# model = tf.keras.models.Model(inputs=[main_input, day_input], outputs=[main_output])
# model.compile(optimizer='adam', loss={'main_output': 'binary_crossentropy'})
# history = model.fit({'main_input': np.array(x_hour), 'day_input': np.array(x_day)}, {'main_output': np.array(y_hour)}, epochs=1000, batch_size=1)
# plt.figure(figsize=(12,5))
# plt.plot(history.history['loss'])
# plt.show()
#
# for i in range(len(x_hour)):
#     x_np_hour = np.array(x_hour[i])
#     inp_hour = x_np_hour.reshape(1, len(x_hour[i]), len(x_hour[i][0]))
#     x_np_day = np.array(x_day[i])
#     inp_day = x_np_day.reshape(1, len(x_day[i]), len(x_day[i][0]))
#     pred = model.predict([inp_hour, inp_day], verbose=0)
#     pred_list = pred[0].tolist()
#     print(f"x_hour[{i}] = {x_hour[i]},   x_day[{i}] = {x_day[i]},   pred_list = {pred_list}")




#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_split_sequence_length = 7 # данные в обучающую, валидационную и тестовую выборки будут добавляться последовательностями данной длины
validation_split = 0.05 # размер данных для валидации относительно всех данных
test_split = 0.25 # размер тестовых данных относительно всех данных
sequence_length = 600 # длина последовательных данных для которых делается прогноз следующего значения
predict_length = 72 # количество шагов на которое будут спрогнозированы данные
part_learn_predict = 0.01 # часть от учебных данных для которых будет выполнено прогнозирование на predict_length шагов вперед
part_test_predict = 0.03 # часть от тестовых данных для которых будет выполнено прогнозирование на predict_length шагов вперед
part_learn_predict_visualize = (True, 15) # (False, 0.01) - вероятность визуализировать в файл спрогнозированные данные, которые начинаются обучающих данных. (True, 20) - количество визуализаций для которого вероятность подберется автоматически
part_test_predict_visualize = (True, 15) # (False, 0.01) - вероятность визуализировать в файл спрогнозированные данные, которые начинаются на тестовых данных. (True, 20) - количество визуализаций для которого вероятность подберется автоматически
is_visualize_prediction_union = True # визуализировать спрогнозированные последовательности, и сохранить в файлы. Все источники данных будут на одном изображении
is_visualize_prediction_single = True # визуализировать спрогнозированные последовательности, и сохранить в файлы. Каждый источник данных будет на собственном изображении.
visualize_prediction_cut = 400 # до какой длины обрезать визуализируемые данные. Чтобы если длина последовательности и длина предсказания большие, можно было понять как предсказание корелирует с истинными данными. Независимо от данного значения, визуализированы будут все данные предсказания.
is_save_predict_data = False # сохранять ли спрогнозированные данные. Когда True, part_learn_predict и part_test_predict не будут иметь значения, т.к. выполнится прогнозирование для всех данных, включая валидационные. part_learn_predict_visualize будет иметь значение, и будет составлять часть от всех обучающих данных, то же самое для тестовых
first_file_offset = 10 # отступ от начала данных первого файла в источниках данных
over_rate = 0 # подставляю это значение в параметр нормализаторов, определяет насколько больше будет диапазон нормализации относительно формата: вплотную, 0.1 - на 10% больше
data_sources_meta = [
    features.DataSourceMeta(files=[
            "E:/Моя папка/data/binance/BTCUSDT-1h-2020-01 - 2023-01 lim_0_3000.csv"
        ], date_index = 0, data_indexes = [1,2,3,4,5],
        normalizers=[
            normalizers.RelativeMinMaxScaler(data_indexes=[1,2,3,4], is_range_part=True, is_high_part=True, is_low_part=True, over_rate=over_rate),
            normalizers.RelativeMinMaxScaler(data_indexes=[5], is_range_part=True, is_high_part=True, is_low_part=True, over_rate=over_rate)
        ], visualize=[("candle", [1,2,3,4]), ("line", [5])], is_visualize=True, visualize_ratio=[3,1], visualize_name=["price", "volume"]),
] # data_indexes - индексы данных в файле. Индексы данных для визуализации в visualize это индексы данных от 1 до количества элементов в data_indexes, то есть данные, полученные из файла по таким индексам: data_indexes=[2,3,5,6] отображаются с использование таких индексов: visualize=[("candle", [1,2,3,4]), т.к. в visualize указываются не индексы данных в файле, а индексы уже считанных данных, которые нумеруются от 1 до колчества индексов в data_indexes

loaded_models = []
is_load_models = True
if is_load_models:
    loaded_models = [
        tf.keras.models.load_model("path_0"),
        tf.keras.models.load_model("path_1")
    ]

available_best_model_criteria = {"learn": "learn", "validation": "validation", "test": "test"} # возможные критерии оценки выбора лучшей модели
best_model_criteria = available_best_model_criteria["test"] # критерий оценки выбора лучшей модели

# генератор создает периоды, начало последующего периода сдвинуто от начала предыдущего на длительность теста предыдущего периода
is_generate_periods = True # генерировать периоды, или использовать указанные в списке periods
periods_generator_start = features.DateTime(year=2020, month=2, day=1)
periods_generator_learning_duration = features.Duration(years=0, months=0, days=377)
periods_generator_testing_duration = features.Duration(years=0, months=0, days=377)
periods_generator_model_learn_count = 1 # сколько раз нужно обучать модель с новой начальной инициализацией, будет выбрана модель с наименьшей ошибкой
periods_generator_model_desired_loss = 0 # желаемая ошибка для best_model_criteria, если ошибка модели будет меньше или равна данному значению, дополнительные обучения проводиться не будут
periods_generator_count = 5
periods_generator_is_load_models = False # использовать в периодах загруженные модели в loaded_models, будут подставляться модели с индексами от 0 до periods_generator_count

periods = []
if is_generate_periods:
    last_period_date_time_start = periods_generator_start.date_time
    for i in range(periods_generator_count):
        period_date_time_start = features.DateTime(date_time=last_period_date_time_start)
        if i > 0:
            period_date_time_start.add_duration(periods_generator_testing_duration)
        last_period_date_time_start = period_date_time_start.date_time
        if not periods_generator_is_load_models:
            periods.append(features.Period(period_date_time_start, periods_generator_learning_duration, periods_generator_testing_duration, periods_generator_model_learn_count, periods_generator_model_desired_loss, best_model_criteria))
        else:
            periods.append(features.Period(period_date_time_start, periods_generator_learning_duration, periods_generator_testing_duration, periods_generator_model_learn_count, periods_generator_model_desired_loss, best_model_criteria, loaded_models[i]))
else:
    periods = [
        features.Period(features.DateTime(year=2020, month=2, day=1), features.Duration(years=1, months=0, days=0), features.Duration(years=0, months=1, days=0), 1, 0, best_model_criteria, loaded_models[0]),
        features.Period(features.DateTime(year=2020, month=3, day=1), features.Duration(years=1, months=0, days=0), features.Duration(years=0, months=1, days=0), 1, 0, best_model_criteria, loaded_models[1])
    ]

data_manager = features.DataManager(data_sources_meta, first_file_offset, sequence_length, data_split_sequence_length, validation_split, test_split)

model = Sequential()
model.add(Input((sequence_length, len(data_manager.x_learn[0][0]))))
#model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(data_manager.y_learn[0]), activation='sigmoid'))
model.summary()

model.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()], optimizer='adam')
#"binary_crossentropy"

history = model.fit(np.array(data_manager.x_learn), np.array(data_manager.y_learn), batch_size=32, epochs=200, validation_data=(np.array(data_manager.x_valid), np.array(data_manager.y_valid)))

plt.figure(figsize=(6,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
if len(history.history['loss']) > 12:
    plt.plot(history.history['loss'][10:])
    plt.plot(history.history['val_loss'][10:])
    plt.show()

print(f"длина обучающей выборки: {len(data_manager.x_learn)}")
print(f"длина выборки валидации: {len(data_manager.x_valid)}")
print(f"длина тестовой выборки: {len(data_manager.x_test)}")

probability_learn_predict_visualize = part_learn_predict_visualize[1] / (len(data_manager.x_learn) * part_learn_predict) if part_learn_predict_visualize[0] else part_learn_predict_visualize[1]
probability_test_predict_visualize = part_test_predict_visualize[1] / (len(data_manager.x_test) * part_test_predict) if part_test_predict_visualize[0] else part_test_predict_visualize[1]
print(f"probability_learn_predict_visualize={probability_learn_predict_visualize}, probability_test_predict_visualize={probability_test_predict_visualize}")

data_manager.predict_data(model, predict_length, is_save_predict_data, part_learn_predict, part_test_predict, probability_learn_predict_visualize, probability_test_predict_visualize, is_visualize_prediction_union, is_visualize_prediction_single, visualize_prediction_cut)



#перевод источников данных [path, path, ...] в последовательности: learning(X, Y), testing(X, Y) (указывается метод нормалзизации данных)
#рассчет среднего отклонения от истинных данных для учебных и тестовых данных, при прогнозировании на n шагов вперед
#создание файлов визуализации прогнозирования на n шагов вперед (указывается длина последовательных данных, и вероятность визуализировать последовательность, чтобы можно было увидеть как различается прогнозирование последовательных данных)
#сохранение обученной сети в файл, а так же информацию об настройках при обучении
#функция расчета и записи в файл данных, спрогнозированных на n шагов вперед для всех данных и для всех файлов, учавствующих при обучении модели
#возможность пропустить шаг обучения модели, используя загруженную с диска модель
#расчет отклонения спрогнозированных данных от истинных делается: отдельно для каждого файла, и суммарно для всех файлов