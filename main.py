import os
import sys
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

validation_split = 0.05 # размер данных для валидации относительно всех данных
sequence_length = 500 # длина последовательных данных для которых делается прогноз следующего значения
predict_length = 60 # количество шагов на которое будут спрогнозированы данные
part_learn_predict = 0.00833 # часть от учебных данных для которых будет выполнено прогнозирование на predict_length шагов вперед
part_test_predict = 0.1 # часть от тестовых данных для которых будет выполнено прогнозирование на predict_length шагов вперед
part_learn_predict_visualize = (True, 20) # первое логическое значение - False: считать второе значение (число) вероятностью визуализировать спрогнозированные на несколько шагов значения, True: считать второе значение (число) максимальным количеством, случайно выбранных, визуализаций
part_test_predict_visualize = (True, 20) # первое логическое значение - False: считать второе значение (число) вероятностью визуализировать спрогнозированные на несколько шагов значения, True: считать второе значение (число) максимальным количеством, случайно выбранных, визуализаций
learn_predict_visualize_one_step_limit = 999 # максимальное количество визуализаций прогнозирования на один шаг вперед для учебного периода
test_predict_visualize_one_step_limit = 999 # максимальное количество визуализаций прогнозирования на один шаг вперед для тестового периода
is_visualize_prediction_union = True # визуализировать спрогнозированные последовательности, и сохранить в файлы. Все источники данных будут на одном изображении. Данная визуализация будет выполнена только если количество визуализируемых источников данных будет больше 1.
is_visualize_prediction_single = False # визуализировать спрогнозированные последовательности, и сохранить в файлы. Каждый источник данных будет на собственном изображении. Если is_visualize_prediction_union=True и количество визуализируемых источников данных будет равно 1, а is_visualize_prediction_single=False, визуализация все равно будет происходить как если is_visualize_prediction_single=True
visualize_prediction_cut = 510 # до какой длины обрезать визуализируемые данные. Чтобы если длина последовательности и длина предсказания большие, можно было понять как предсказание корелирует с истинными данными. Независимо от данного значения, визуализированы будут все данные предсказания.
is_save_predict_data = False # сохранять ли спрогнозированные данные. Когда True, part_learn_predict и part_test_predict не будут иметь значения, т.к. выполнится прогнозирование для всех данных. part_learn_predict_visualize будет иметь значение, и будет составлять часть от всех обучающих данных, то же самое для тестовых
over_rate = 0.2 # подставляю это значение в параметр нормализаторов, определяет насколько больше будет диапазон нормализации относительно формата: вплотную, 0.1 - на 10% больше
# data_indexes_in_file в инициализаторе DataSourceMeta - это индексы, по которым будут браться данные из файла. data_indexes в инициализаторе нормализаторов - это индексы считанных данных из файла, то есть такой код: DataSourceMeta(data_indexes_in_file = [1,11]) считает 2 значения из файла, по индексам 1 и 11, чтобы использовать далее эти занчения в номрализаторе следует написать: normalizers.DynamicAbsoluteMinMaxScaler(data_indexes=[1,2]). По 0 индексу располагается дата данных, поэтому его следует указывать лишь для специальных нормализаторов, например: normalizers.DateTimeOneHotVector(data_indexes=[0])
data_sources_meta = [
    features.DataSourceMeta(files=[
            "C:/Users/Христиан/PycharmProjects/fileProcessing/fill_files 2017-10-10 16h 00m - 2023-03-31 23h 00m/BINANCE SPOT BTCUSDT 1h 2017-08-17 04h 00m - 2023-03-31 23h 00m.csv"
        ], short_name="BTC", date_index = 0, data_indexes_in_file = [1,2,3,4,5,11], losses_data_indexes=[1,2,3,4], is_save_data=True,
        output_inserts=[
            normalizers.InsertionFixedValue(insert_index=6, value=0)
        ],
        normalizers=[
            normalizers.DateTimeOneHotVector(data_indexes=[0], is_input_denormalize=False, input_denormalize_weight=1, is_output_denormalize=False, output_denormalize_weight=1, is_month=False, is_day_of_week=True, is_day=True, is_hour=True),
            normalizers.DynamicAbsoluteMinMaxScaler(data_indexes=[1,2,3,4], is_input_denormalize=True, input_denormalize_weight=1, is_output_denormalize=True, output_denormalize_weight=1, over_rate_low=over_rate, over_rate_high=over_rate, add_values=[2416], is_auto_over_rate_low=True, auto_over_rate_low_multipy=1.5, auto_over_rate_low_min=0.1, is_auto_over_rate_high=True, auto_over_rate_high_multipy=1.5, auto_over_rate_high_min=0.1),
            normalizers.DynamicAbsoluteMinMaxScaler(data_indexes=[5], is_input_denormalize=True, input_denormalize_weight=1, is_output_denormalize=True, output_denormalize_weight=1, over_rate_low=0.05, over_rate_high=over_rate, add_values=[-100], is_auto_over_rate_high=True, auto_over_rate_high_multipy=1.5, auto_over_rate_high_min=0.1),
            normalizers.SameValuesNoOutput(data_indexes=[6], is_input_denormalize=False, input_denormalize_weight=1, is_output_denormalize=False, output_denormalize_weight=1)
        ], visualize=[("candle", [1,2,3,4]), ("line", [5])], is_visualize=True, visualize_ratio=[3,1], visualize_name=["price", "volume"])
    # features.DataSourceMeta(files=[
    #         "C:/Users/Христиан/PycharmProjects/fileProcessing/fill_files 2017-10-10 16h 00m - 2023-03-31 23h 00m/BINANCE SPOT ETHUSDT 1h 2017-08-17 04h 00m - 2023-03-31 23h 00m.csv"
    #     ], short_name="ETH", date_index = 0, data_indexes_in_file = [1,2,3,4,5,11], losses_data_indexes=[1,2,3,4], is_save_data=True,
    #     output_inserts=[
    #         normalizers.InsertionFixedValue(insert_index=6, value=0)
    #     ],
    #     normalizers=[
    #         normalizers.DynamicAbsoluteMinMaxScaler(data_indexes=[1,2,3,4], is_input_denormalize=True, input_denormalize_weight=1, is_output_denormalize=True, output_denormalize_weight=1, over_rate_low=over_rate, over_rate_high=over_rate, add_values=[2416], is_auto_over_rate_low=True, auto_over_rate_low_multipy=1.5, auto_over_rate_low_min=0.1, is_auto_over_rate_high=True, auto_over_rate_high_multipy=1.5, auto_over_rate_high_min=0.1),
    #         normalizers.DynamicAbsoluteMinMaxScaler(data_indexes=[5], is_input_denormalize=True, input_denormalize_weight=1, is_output_denormalize=True, output_denormalize_weight=1, over_rate_low=0.05, over_rate_high=over_rate, add_values=[-100], is_auto_over_rate_high=True, auto_over_rate_high_multipy=1.5, auto_over_rate_high_min=0.1),
    #         normalizers.SameValuesNoOutput(data_indexes=[6], is_input_denormalize=False, input_denormalize_weight=1, is_output_denormalize=False, output_denormalize_weight=1)
    #     ], visualize=[("candle", [1,2,3,4]), ("line", [5])], is_visualize=True, visualize_ratio=[3,1], visualize_name=["price", "volume"])
    # features.DataSourceMeta(files=[
    #         "C:/Users/Христиан/PycharmProjects/fileProcessing/fill_files 2017-10-10 16h 00m - 2023-03-31 23h 00m/TIINGO SPY 1Hour 2017-10-10 16h 00m - 2023-03-31 19h 00m (SNP500 ETF).csv"
    #     ], short_name="SPY", date_index = 0, data_indexes_in_file = [1,2,3,4,6], losses_data_indexes=[1,2,3,4], is_save_data=True,
    #     output_inserts=[
    #         normalizers.InsertionAmericanIsFiller(insert_index=5)
    #     ],
    #     normalizers=[
    #         normalizers.DynamicAbsoluteMinMaxScaler(data_indexes=[1, 2, 3, 4], is_input_denormalize=True, input_denormalize_weight=1, is_output_denormalize=True, output_denormalize_weight=1, over_rate_low=over_rate, over_rate_high=over_rate, add_values=[154], is_auto_over_rate_low=True, auto_over_rate_low_multipy=1.5, auto_over_rate_low_min=0.1, is_auto_over_rate_high=True, auto_over_rate_high_multipy=1.5, auto_over_rate_high_min=0.1),
    #         normalizers.SameValuesNoOutput(data_indexes=[5], is_input_denormalize=False, input_denormalize_weight=1, is_output_denormalize=False, output_denormalize_weight=1)
    #     ], visualize=[("candle", [1,2,3,4])], is_visualize=True, visualize_ratio=[3], visualize_name=["price"])
]

loaded_models = []
is_load_models = False
if is_load_models:
    loaded_models = [
        tf.keras.models.load_model("E:/Моя папка/predictTimeSeries/app_data/0007/2018-01-01 00h 00m - 2019-02-01 00h 00m/save_model")
    ]

# генератор создает периоды, начало последующего периода сдвинуто от начала предыдущего на длительность теста предыдущего периода
is_generate_periods = True # генерировать периоды, или использовать указанные в списке periods
# periods_generator_is_load_models = False # использовать в периодах загруженные модели в loaded_models, будут подставляться модели с индексами от 0 до periods_generator_count
periods_generator_start = features.DateTime(year=2018, month=1, day=1)
periods_generator_learning_duration = features.Duration(years=0, months=0, days=5)
periods_generator_testing_duration = features.Duration(years=0, months=0, days=5)
periods_generator_model_learn_count = 1 # сколько раз нужно обучать модель с новой начальной инициализацией, будет выбрана модель с наименьшей ошибкой
periods_generator_model_desired_loss = 0 # желаемая ошибка для best_model_criteria, если ошибка модели будет меньше или равна данному значению, дополнительные обучения проводиться не будут
periods_generator_count = 1 # количество периодов

if is_generate_periods:
    periods = []
    last_period_date_time_start = periods_generator_start.date_time
    for i in range(periods_generator_count):
        period_date_time_start = features.DateTime(date_time=last_period_date_time_start)
        if i > 0:
            period_date_time_start.add_duration(periods_generator_testing_duration)
        last_period_date_time_start = period_date_time_start.date_time
        if not is_load_models:
            periods.append(features.Period(period_date_time_start, periods_generator_learning_duration, periods_generator_testing_duration, periods_generator_model_learn_count, periods_generator_model_desired_loss))
        else:
            periods.append(features.Period(period_date_time_start, periods_generator_learning_duration, periods_generator_testing_duration, periods_generator_model_learn_count, periods_generator_model_desired_loss, loaded_models[i]))
else:
    periods = [
        features.Period(features.DateTime(year=2020, month=2, day=1), features.Duration(years=1, months=0, days=0), features.Duration(years=0, months=1, days=0), 1, 0, loaded_models[0]),
        features.Period(features.DateTime(year=2020, month=3, day=1), features.Duration(years=1, months=0, days=0), features.Duration(years=0, months=1, days=0), 1, 0, loaded_models[1])
    ]

data_manager = features.DataManager(data_sources_meta, validation_split, sequence_length, predict_length, part_learn_predict, part_test_predict, part_learn_predict_visualize, part_test_predict_visualize, learn_predict_visualize_one_step_limit, test_predict_visualize_one_step_limit, is_visualize_prediction_union, is_visualize_prediction_single, visualize_prediction_cut, is_save_predict_data, periods)
