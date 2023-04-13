import sys
import time
import random
import os
import numpy as np
import datetime
import calendar
import mplfinance as mpf
import pandas as pd
import copy
import normalizers
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# функция возвращает дату в формате UTC+00:00
def timestamp_to_utc_datetime(date):
    return datetime.datetime.utcfromtimestamp(date / 1000).replace(tzinfo=datetime.timezone.utc)
# функция возвращает наивную дату
def timestamp_to_local_datetime(date):
    return datetime.datetime.fromtimestamp(date / 1000)
# функция принимает дату в формате UTC+00:00
def utc_datetime_to_timestamp(date):
    return int(calendar.timegm(date.timetuple()) * 1000)
# функция принимает наивную дату
def local_datetime_to_timestamp(date):
    return int(time.mktime(date.timetuple()) * 1000)

def timedelta_to_milliseconds(timedelta):
    return timedelta.total_seconds() * 1000

def float_to_str_format(value, digits=0):
    return f"{value:.{digits}f}"

class DateTime:
    MonthLengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    @classmethod
    def is_leap_year(cls, year):
        return True if year % 4 == 0 else False

    @classmethod
    def month_length(cls, year, month):
        add_day = 1 if cls.is_leap_year(year) and month == 2 else 0
        return cls.MonthLengths[month - 1] + add_day

    # предполагается передавать либо date_time либо year, month и day
    def __init__(self, date_time=None, year=None, month=None, day=None):
        if date_time != None:
            self.date_time = datetime.datetime(date_time.year, date_time.month, date_time.day, tzinfo=datetime.timezone.utc)
        else:
            self.date_time = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)

    def add_years(self, years):
        self.date_time = datetime.datetime(self.date_time.year + years, self.date_time.month, self.date_time.day, tzinfo=datetime.timezone.utc)

    def add_months(self, months):
        new_month = self.date_time.month + months - 1
        years = new_month // 12
        months_remaind = new_month % 12 + 1
        if years > 0:
            self.add_years(years)
        self.date_time = datetime.datetime(self.date_time.year, months_remaind, min(self.date_time.day, self.month_length(self.date_time.year, months_remaind)), tzinfo=datetime.timezone.utc)

    def add_days(self, days):
        days_remaind = days
        while days_remaind > 0:
            days_to_next_month = self.month_length(self.date_time.year, self.date_time.month) - self.date_time.day + 1
            if days_remaind >= days_to_next_month:
                days_remaind -= days_to_next_month
                self.date_time = datetime.datetime(self.date_time.year, self.date_time.month, 1, tzinfo=datetime.timezone.utc)
                self.add_months(1)
            else:
                self.date_time = datetime.datetime(self.date_time.year, self.date_time.month, self.date_time.day + days_remaind, tzinfo=datetime.timezone.utc)
                days_remaind -= days_remaind

    def add_duration(self, duration):
        self.add_years(duration.years)
        self.add_months(duration.months)
        self.add_days(duration.days)

class Duration:
    def __init__(self, years, months, days):
        self.years = years
        self.months = months
        self.days = days

class Period:
    # model_learn_count - сколько раз нужно обучать модель с новой начальной инициализацией, будет выбрана модель с наименьшей ошибкой
    # model_desired_loss - желаемая ошибка, если ошибка модели будет меньше или равна данному значению, дополнительные обучения проводиться не будут
    # model - если есть обученная модель, и не нужно проводить обучение, её следует передать в этот параметр
    def __init__(self, date_time_start, learning_duration, testing_duration, model_learn_count, model_desired_loss, model=None):
        self.learning_start = DateTime(date_time=date_time_start.date_time)
        self.learning_end = DateTime(date_time=date_time_start.date_time)
        self.learning_end.add_duration(learning_duration)

        self.testing_start = DateTime(date_time=self.learning_end.date_time)
        self.testing_end = DateTime(date_time=self.learning_end.date_time)
        self.testing_end.add_duration(testing_duration)

        self.model_learn_count = model_learn_count
        if model_learn_count < 1:
            raise ValueError(f"model_learn_count должно быть не менее 1")
        self.model_desired_loss = model_desired_loss
        self.model = model

class DataSourceMeta:
    def __init__(self, files, date_index, data_indexes, is_save_data, normalizers, visualize, is_visualize, visualize_ratio, visualize_name, visualize_data_source_panel = 1):
        self.files = files # список с файлами
        self.date_index = date_index # индекс даты
        self.data_indexes = data_indexes # индексы данных, которые нужно считать из файла
        self.is_save_data = is_save_data # сохранять ли спрогнозированные данные для данного источника данных
        self.normalizers = normalizers # список с нормализаторами для источника данных
        self.visualize = visualize # список с панелями которые будут созданы для отображения источника данных. Элементы списка: ("type", [data_indexes]), type может быть: "candle", "line". Для "candle" может быть передано или 4 или 2 индекса данных, для "line" может быть передан только один индекс данных. Пример графиков цены и объема: [("candle", [1,2,3,4]), ("line", [5])]
        self.is_visualize = is_visualize # нужно ли визуализировать источник данных
        self.visualize_ratio = visualize_ratio  # список со значениями размера панелей visualize
        self.visualize_name = visualize_name  # список с названиями панелей visualize
        self.visualize_data_source_panel = visualize_data_source_panel  # номер панели visualize в имени которой будет название источника данных (первого файла)

class DataManager:
    @classmethod
    def read_csv_file(cls, file_path, date_index, data_indexes):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            first = True
            for line in file:
                if first:  # пропускаем шапку файла
                    first = False
                else:
                    list_line = line.split(",")
                    data_line = []
                    data_line.append(int(list_line[date_index]))
                    for i in data_indexes:
                        data_line.append(float(list_line[i]))
                    data.append(data_line)
        return data

    def create_base_folder(self):
        app_files_folder_name = "app_data"
        if not os.path.isdir(app_files_folder_name):
            os.makedirs(app_files_folder_name)
        folder_id = 0
        while os.path.isdir(f"{app_files_folder_name}/{str(folder_id).rjust(4, '0')}"):
            folder_id += 1
        self.base_folder = f"{app_files_folder_name}/{str(folder_id).rjust(4, '0')}"
        os.makedirs(self.base_folder)
        # self.folder_images_learn_predict = f"{self.folder_base}/images/learn_predict"
        # self.folder_images_test_predict = f"{self.folder_base}/images/test_predict"
        # self.folder_images_learn_result = f"{self.folder_base}/images/learn_result"
        # os.makedirs(f"{self.folder_base}/images")
        # os.makedirs(self.folder_images_learn_predict)
        # os.makedirs(self.folder_images_test_predict)
        # os.makedirs(self.folder_images_learn_result)

    def __init__(self, data_sources_meta, validation_split, sequence_length, predict_length, part_learn_predict, part_test_predict, part_learn_predict_visualize, part_test_predict_visualize, is_visualize_prediction_union, is_visualize_prediction_single, visualize_prediction_cut, is_save_predict_data, periods):
        # создаем папки для сохранения информации
        self.create_base_folder()

        self.data_sources_meta = data_sources_meta
        self.validation_split = validation_split
        self.sequence_length = sequence_length
        self.predict_length = predict_length
        self.part_learn_predict = part_learn_predict
        self.part_test_predict = part_test_predict
        self.part_learn_predict_visualize = part_learn_predict_visualize
        self.part_test_predict_visualize = part_test_predict_visualize
        self.is_visualize_prediction_union = is_visualize_prediction_union
        self.is_visualize_prediction_single = is_visualize_prediction_single
        self.visualize_prediction_cut = visualize_prediction_cut
        self.is_save_predict_data = is_save_predict_data
        self.periods = periods

        # определяем имена файлов без полных путей
        self.data_sources_file_names = []  # имена файлов источников данных без полного пути
        for i_ds in range(len(data_sources_meta)):
            data_source_file_names = []
            for i_f in range(len(data_sources_meta[i_ds].files)):
                file_name = data_sources_meta[i_ds].files[i_f].replace("\\", "/").split("/")[-1]
                data_source_file_names.append(file_name)
            self.data_sources_file_names.append(data_source_file_names)

        # считываем источники данных
        self.data_sources = []  # данные всех файлов всех источников данных
        for i_ds in range(len(data_sources_meta)):
            self.data_sources.append([self.read_csv_file(file, data_sources_meta[i_ds].date_index, data_sources_meta[i_ds].data_indexes) for file in data_sources_meta[i_ds].files])

        # проверяем, совпадает ли: количество файлов у разных источников данных, количество данных в файлах разных источников данных, даты в данных разных источников данных
        error_messages = []
        if len(self.data_sources) > 1:
            # количество файлов у разных источников данных
            for i_ds in range(1, len(self.data_sources)):
                if len(self.data_sources[i_ds]) != len(self.data_sources[0]):
                    if len(error_messages) < 25:
                        error_messages.append(f"Не совпадает количество файлов у источников данных: {self.data_sources_file_names[i_ds]} и {self.data_sources_file_names[0]}.")
            # количество данных в файлах разных источников данных
            if len(error_messages) == 0:
                for i_f in range(len(self.data_sources[0])):
                    for i_ds in range(1, len(self.data_sources)):
                        if len(self.data_sources[i_ds][i_f]) != len(self.data_sources[0][i_f]):
                            if len(error_messages) < 25:
                                error_messages.append(f"Не совпадает количество свечек в файлах разных источников данных: {self.data_sources_file_names[i_ds][i_f]} и {self.data_sources_file_names[0][i_f]}.")
            # даты в свечках разных источников данных
            if len(error_messages) == 0:
                for i_f in range(len(self.data_sources[0])):
                    for i_c in range(len(self.data_sources[0][i_f])):
                        for i_ds in range(1, len(self.data_sources)):
                            if self.data_sources[i_ds][i_f][i_c][0] != self.data_sources[0][i_f][i_c][0]:
                                if len(error_messages) < 25:
                                    error_messages.append(f"Не совпадают даты в данных разных источников данных: (файл={self.data_sources_file_names[i_ds][i_f]}, индекс свечки={i_c}, дата={self.data_sources[i_ds][i_f][i_c][0]}) и (файл={self.data_sources_file_names[0][i_f]}, индекс свечки={i_c}, дата={self.data_sources[0][i_f][i_c][0]}).")

        if len(error_messages) == 0:
            self.interval_milliseconds = self.data_sources[0][0][1][0] - self.data_sources[0][0][0][0]
            for i_f in range(len(self.data_sources[0])):
                for i_c in range(1, len(self.data_sources[0][i_f])):
                    current_interval_milliseconds = self.data_sources[0][i_f][i_c][0] - self.data_sources[0][i_f][i_c - 1][0]
                    if current_interval_milliseconds != self.interval_milliseconds:
                        if len(error_messages) < 25:
                            error_messages.append(f"i_ds=0, i_f={i_f}, i_c={i_c}, разрыв в данных.")

        if len(error_messages) > 0:
            for message in error_messages:
                print(message)
            raise ValueError(error_messages[0])

        self.data_sources_start_timestamp = self.data_sources[0][0][0][0]
        self.data_sources_end_timestamp = None
        for i_f in range(len(self.data_sources[0])):
            for i_c in range(len(self.data_sources[0][i_f])):
                self.data_sources_end_timestamp = self.data_sources[0][i_f][i_c][0]

        # выполняем подготовку нормализаторов
        for i_ds in range(len(data_sources_meta)):
            for i_n in range(len(data_sources_meta[i_ds].normalizers)):
                data_sources_meta[i_ds].normalizers[i_n].summary(self.data_sources[i_ds], self.sequence_length)

        self.periods_process()

    def get_data_sources_input_sequence(self, i_f, i_last_input_candle):
        data_sources_inp_seq = []
        for i_ds in range(len(self.data_sources)):
            data_source_inp_seq = []
            for i_seq in range(i_last_input_candle + 1 - self.sequence_length, i_last_input_candle + 1):
                data_source_inp_seq.append(copy.deepcopy(self.data_sources[i_ds][i_f][i_seq]))
            data_sources_inp_seq.append(data_source_inp_seq)
        return data_sources_inp_seq

    def get_data_sources_output(self, i_f, i_output_candle):
        data_sources_output = []
        for i_ds in range(len(self.data_sources)):
            data_sources_output.append(copy.deepcopy(self.data_sources[i_ds][i_f][i_output_candle]))
        return data_sources_output

    # создает входные последовательности, соответствующие им выходные значения и настройки для денормализации, для дат: от начала до окончания периодов
    def create_data_sources_periods_x_y(self):
        total_count = round((self.periods_end_timestamp - self.periods_start_timestamp) / self.interval_milliseconds)
        number = 0
        print(f"Создание входных последовательностей: {number}%")
        number += 5
        # создаю список, заполненный None, размерностью self.data_sources[0]
        self.data_sources_periods_x_y = [[None] * len(self.data_sources[0][i_f]) for i_f in range(len(self.data_sources[0]))]
        # доходим до первой свечки начала периодов
        i_f = 0
        i_c = 0
        while self.data_sources[0][i_f][i_c][0] < self.periods_start_timestamp:
            i_c += 1
            if i_c >= len(self.data_sources[0][i_f]):
                i_c = 0
                i_f += 1
        current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]
        while current_datetime_timestamp <= self.periods_end_timestamp:
            data_sources_inp_seq = self.get_data_sources_input_sequence(i_f, i_c - 1)
            data_sources_output = self.get_data_sources_output(i_f, i_c)
            x, y, data_sources_normalizers_settings = self.normalize_data_sources(i_f, i_c - 1, data_sources_inp_seq, data_sources_output)
            self.data_sources_periods_x_y[i_f][i_c] = (x, y, data_sources_normalizers_settings)

            if ((current_datetime_timestamp - self.periods_start_timestamp) / self.interval_milliseconds) / total_count >= number / 100:
                print(f"Создание входных последовательностей: {number}%")
                number += 5

            while self.data_sources[0][i_f][i_c][0] <= current_datetime_timestamp:
                i_c += 1
                if i_c >= len(self.data_sources[0][i_f]):
                    i_c = 0
                    i_f += 1
            current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]

    def get_learning_valid_data(self, datetime_start_timestamp, datetime_end_timestamp):
        # доходим до первой свечки начала периода
        i_f = 0
        i_c = 0
        while self.data_sources[0][i_f][i_c][0] < datetime_start_timestamp:
            i_c += 1
            if i_c >= len(self.data_sources[0][i_f]):
                i_c = 0
                i_f += 1
        current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]

        x_learn = []
        y_learn = []
        x_valid = []
        y_valid = []

        learn_count = 0
        valid_count = 0
        data_types = [0, 1] # 0 - учебные, 1 - валидационные
        while current_datetime_timestamp <= datetime_end_timestamp:
            learn_part = learn_count / (learn_count + valid_count)
            valid_part = valid_count / (learn_count + valid_count)
            if learn_part < 1 - self.validation_split and valid_part < self.validation_split:
                data_type = random.choice(data_types)
            elif learn_part < 1 - self.validation_split:
                data_type = data_types[0]
            else:
                data_type = data_types[1]

            x, y, data_sources_normalizers_settings = self.data_sources_periods[i_f][i_c]

            if data_type == data_types[0]:
                x_learn.append(x)
                y_learn.append(y)
                learn_count += 1
            else:
                x_valid.append(x)
                y_valid.append(y)
                valid_count += 1

            while self.data_sources[0][i_f][i_c][0] <= current_datetime_timestamp:
                i_c += 1
                if i_c >= len(self.data_sources[0][i_f]):
                    i_c = 0
                    i_f += 1
            current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]
        return (x_learn, y_learn, x_valid, y_valid)

    def handle_period(self, period):
        learning_start_timestamp = utc_datetime_to_timestamp(period.learning_start.date_time)
        learning_end_timestamp = utc_datetime_to_timestamp(period.learning_end.date_time)
        x_learn, y_learn, x_valid, y_valid = self.get_learning_valid_data(learning_start_timestamp, learning_end_timestamp)
        print(f"длина обучающей выборки={len(x_learn)}")
        print(f"длина выборки валидации={len(x_valid)}")
        if period.model is None:
            models = []
            models_losses = []
            i = 0
            while True:
                model = Sequential()
                model.add(Input((self.sequence_length, len(x_learn[0][0]))))
                # model.add(LSTM(128, return_sequences=True))
                model.add(LSTM(128))
                model.add(Dense(len(y_learn[0]), activation='sigmoid'))
                model.summary()

                model.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()], optimizer='adam')
                # "binary_crossentropy" tf.keras.losses.MeanAbsolutePercentageError()

                history = model.fit(np.array(x_learn), np.array(y_learn), batch_size=32, epochs=200, validation_data=(np.array(x_valid), np.array(y_valid)))

                fig_w = 6
                fig_h = 5
                plt.figure(figsize=(fig_w, fig_h))
                plt.title(f"{period.learning_start.date_time.strftime('%Y-%m-%d %Hh %Mm')} - {period.testing_end.date_time.strftime('%Y-%m-%d %Hh %Mm')} [:] _{i + 1}")
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.show()
                if len(history.history['loss']) > 12:
                    plt.figure(figsize=(fig_w, fig_h))
                    plt.title(f"{period.learning_start.date_time.strftime('%Y-%m-%d %Hh %Mm')} - {period.testing_end.date_time.strftime('%Y-%m-%d %Hh %Mm')} [7:] _{i + 1}")
                    plt.plot(history.history['loss'][7:])
                    plt.plot(history.history['val_loss'][7:])
                    plt.show()
                models.append(model)
                models_losses.append(history.history['loss'][-1])

                i += 1
                if history.history['loss'][-1] <= period.model_desired_loss or i >= period.model_learn_count:
                    break
            val, idx = min((val, idx) for (idx, val) in enumerate(models_losses))
            period.model = models[idx]



    def periods_process(self):
        error_messages = []
        is_increase = True
        for i in range(1, len(self.periods)):
            if utc_datetime_to_timestamp(self.periods[i - 1].learning_start.date_time) >= utc_datetime_to_timestamp(self.periods[i].learning_start.date_time):
                is_increase = False
            if utc_datetime_to_timestamp(self.periods[i - 1].learning_end.date_time) >= utc_datetime_to_timestamp(self.periods[i].learning_end.date_time):
                is_increase = False
            if utc_datetime_to_timestamp(self.periods[i - 1].testing_start.date_time) >= utc_datetime_to_timestamp(self.periods[i].testing_start.date_time):
                is_increase = False
            if utc_datetime_to_timestamp(self.periods[i - 1].testing_end.date_time) >= utc_datetime_to_timestamp(self.periods[i].testing_end.date_time):
                is_increase = False
        if not is_increase:
            error_messages.append(f"Не все даты в периодах идут по возрастанию от предыдущего к последующему.")

        if len(error_messages) == 0:
            if utc_datetime_to_timestamp(self.periods[0].learning_start.date_time) <= self.data_sources_start_timestamp:
                error_messages.append(f"Дата начала периодов раньше первой даты в источниках данных.")
            if utc_datetime_to_timestamp(self.periods[-1].testing_end.date_time) > self.data_sources_end_timestamp:
                error_messages.append(f"Дата окончания периодов позже последней даты в источниках данных.")

        if len(error_messages) == 0:
            for i in range(len(self.periods)):
                period_start_timestamp = utc_datetime_to_timestamp(self.periods[i].learning_start.date_time)
                current_datetime_timestamp = None
                for i_f in range(len(self.data_sources[0])):
                    is_first_next_date = True
                    for i_c in range(len(self.data_sources[0][i_f])):
                        if current_datetime_timestamp is None:
                            current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]
                        else:
                            if self.data_sources[0][i_f][i_c][0] > current_datetime_timestamp:
                                current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]
                                if current_datetime_timestamp >= period_start_timestamp:
                                    if is_first_next_date:
                                        is_first_next_date = False
                                        if i_c < self.sequence_length - 1:
                                            error_messages.append(f"i_f={i_f}, количество свечек перед датой периода меньше sequence_length.")
        if len(error_messages) > 0:
            for message in error_messages:
                print(message)
            raise ValueError(error_messages[0])

        self.periods_start_timestamp = utc_datetime_to_timestamp(self.periods[0].learning_start.date_time)
        self.periods_end_timestamp = utc_datetime_to_timestamp(self.periods[-1].testing_end.date_time)

        self.create_data_sources_periods_x_y()

        for i in range(len(self.periods)):
            print(f"Обрабатывается период {i + 1}/{len(self.periods)}")
            self.handle_period(self.periods[i])



    # Нормализует входные последовательности для всех источников данных, а так же выходное значение если оно указано
    # data_sources_inp_seq - последовательности входных данных для всех источников данных
    # возвращает последовательность входных векторов, выходной вектор, настройки для денормализации. настройки для денормализации вида: settings[i_ds][i_normalize].
    def normalize_data_sources(self, i_f, i_c, data_sources_inp_seq, data_sources_output=None):
        data_sources_normalizers_inp_seq = []
        data_sources_normalizers_out = []
        data_sources_normalizers_settings = []

        for i_ds in range(len(self.data_sources)):
            normalizers_inp_seq = []
            normalizers_out = []
            normalizers_settings = []
            for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                x, y, n_setting = self.data_sources_meta[i_ds].normalizers[i_n].normalize(data_sources_inp_seq[i_ds], data_sources_output[i_ds], i_f=i_f, i_c=i_c)
                normalizers_inp_seq.append(x)
                normalizers_out.append(y)
                normalizers_settings.append(n_setting)

            data_sources_normalizers_inp_seq.append(normalizers_inp_seq)
            data_sources_normalizers_out.append(normalizers_out)
            data_sources_normalizers_settings.append(normalizers_settings)

        finally_inp_seq = []
        for i_candle in range(len(data_sources_normalizers_inp_seq[0][0])):
            one_data = []
            for i_ds in range(len(data_sources_normalizers_inp_seq)):
                for i_n in range(len(data_sources_normalizers_inp_seq[i_ds])):
                    one_data.extend(data_sources_normalizers_inp_seq[i_ds][i_n][i_candle])
            finally_inp_seq.append(one_data)

        finally_out_seq = []
        if data_sources_output != None:
            for i_ds in range(len(data_sources_normalizers_out)):
                for i_n in range(len(data_sources_normalizers_out[i_ds])):
                    finally_out_seq.extend(data_sources_normalizers_out[i_ds][i_n])

        self.denormalize_input_vectors_sequence_output_vector(data_sources_normalizers_settings, finally_inp_seq, finally_out_seq)

        return finally_inp_seq, finally_out_seq, data_sources_normalizers_settings

    # Денормализует входные последовательности для всех нормализаторов, всех источников данных, а так же выходное значение если оно указано
    # input_vectors_sequence - последовательности входных данных для всех источников данных, для всех нормализаторов
    # возвращает входную последовательность для всех источников данных и выходное значение для всех источников данных
    def denormalize_input_vectors_sequence_output_vector(self, data_sources_normalizers_settings, input_vectors_sequence=None, output_vector=None):
        data_sources_normalizers_inp_seq = self.input_vectors_sequence_to_data_sources_normalizers(input_vectors_sequence) if input_vectors_sequence != None else None
        data_sources_normalizers_out = self.output_vector_to_data_sources_normalizers(output_vector) if output_vector != None else None

        data_sources_inp_seq_denorm = []
        data_sources_out_denorm = []

        for i_ds in range(len(self.data_sources)):
            data_source_inp_seq_denorm = []
            data_source_out_denorm = []
            for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                x, y = self.data_sources_meta[i_ds].normalizers[i_n].denormalize(data_sources_normalizers_settings[i_ds][i_n], normalized_inp_sequence=data_sources_normalizers_inp_seq[i_ds][i_n] if data_sources_normalizers_inp_seq != None else None, normalized_output = data_sources_normalizers_out[i_ds][i_n] if data_sources_normalizers_out != None else None)
                data_source_inp_seq_denorm.extend(x)
                data_source_out_denorm.extend(y)

            data_sources_inp_seq_denorm.append(data_source_inp_seq_denorm)
            data_sources_out_denorm.append(data_source_out_denorm)

        # finally_inp_seq = []
        # if inp_seq_data_sources != None:
        #     for i_candle in range(len(data_sources_normalizers_inp_seq_denorm[0][0])):
        #         one_data = []
        #         for i_ds in range(len(data_sources_normalizers_inp_seq_denorm)):
        #             for i_n in range(len(data_sources_normalizers_inp_seq_denorm[i_ds])):
        #                 one_data.extend(data_sources_normalizers_inp_seq_denorm[i_ds][i_n][i_candle])
        #         finally_inp_seq.append(one_data)
        #
        # finally_out_seq = []
        # if output_data_sources != None:
        #     for i_ds in range(len(data_sources_normalizers_out_denorm)):
        #         for i_n in range(len(data_sources_normalizers_out_denorm[i_ds])):
        #             finally_out_seq.extend(data_sources_normalizers_out_denorm[i_ds][i_n])

        return data_sources_inp_seq_denorm, data_sources_out_denorm

    def output_vector_to_data_sources_normalizers(self, output):
        index = 0
        output_data_sources_normalizers = []
        for i_ds in range(len(self.data_sources_meta)):
            data_source_normalizers = []
            for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                data_source_normalizer = []
                for i in range(index, index + self.data_sources_meta[i_ds].normalizers[i_n].normalize_output_length):
                    data_source_normalizer.append(copy.deepcopy(output[i]))
                index += self.data_sources_meta[i_ds].normalizers[i_n].normalize_output_length
                data_source_normalizers.append(data_source_normalizer)
            output_data_sources_normalizers.append(data_source_normalizers)
        return output_data_sources_normalizers

    def input_vectors_sequence_to_data_sources_normalizers(self, input):
        index = 0
        input_data_sources_normalizers = []
        for i_ds in range(len(self.data_sources_meta)):
            data_source_normalizers = []
            for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                data_source_normalizer = []
                for i in range(index, index + self.data_sources_meta[i_ds].normalizers[i_n].normalize_output_length):
                    data_source_normalizer.append(copy.deepcopy(input[i]))
                index += self.data_sources_meta[i_ds].normalizers[i_n].normalize_input_length
                data_source_normalizers.append(data_source_normalizer)
            input_data_sources_normalizers.append(data_source_normalizers)
        return input_data_sources_normalizers

    def data_to_mplfinance_candle(self, data):  # data format: [date (1675087200000, милисекунды), open, high, low, close]
        reformatted_data = dict()
        reformatted_data['Date'] = []
        reformatted_data['Open'] = []
        reformatted_data['High'] = []
        reformatted_data['Low'] = []
        reformatted_data['Close'] = []
        for item in data:
            if len(item) == 1:
                reformatted_data['Date'].append(timestamp_to_utc_datetime(int(item[0])))
                reformatted_data['Open'].append(None)
                reformatted_data['High'].append(None)
                reformatted_data['Low'].append(None)
                reformatted_data['Close'].append(None)
            else:
                reformatted_data['Date'].append(timestamp_to_utc_datetime(int(item[0])))
                reformatted_data['Open'].append(item[1])
                reformatted_data['High'].append(item[2])
                reformatted_data['Low'].append(item[3])
                reformatted_data['Close'].append(item[4])
        return reformatted_data

    def data_to_mplfinance_line(self, data):  # data format: [date (1675087200000, милисекунды), value]
        reformatted_data = dict()
        reformatted_data['Date'] = []
        reformatted_data['Open'] = []
        reformatted_data['High'] = []
        reformatted_data['Low'] = []
        reformatted_data['Close'] = []
        for item in data:
            if len(item) == 1:
                reformatted_data['Date'].append(timestamp_to_utc_datetime(int(item[0])))
                reformatted_data['Open'].append(None)
                reformatted_data['High'].append(None)
                reformatted_data['Low'].append(None)
                reformatted_data['Close'].append(None)
            else:
                reformatted_data['Date'].append(timestamp_to_utc_datetime(int(item[0])))
                reformatted_data['Open'].append(item[1])
                reformatted_data['High'].append(item[1])
                reformatted_data['Low'].append(item[1])
                reformatted_data['Close'].append(item[1])
        return reformatted_data

    # data_sources_true[0] и data_sources_predict[0] должны иметь одинаковое количество дат
    def visualize_predict_to_file(self, data_sources_true, data_sources_predict, data_type, save_file_path):
        add_plot = []
        where_values_none = []
        where_values_learn = []
        where_values_validate = []
        where_values_test = []
        where_values_end_inp_seq = []
        y_over_rate = 0.02 # отступ от верхнего и нижнего края
        data_type_rate = 0.008 # толщина линии типа данных
        for i in range(len(data_type)):
            where_values_none.append(True if data_type[i] == -1 else False)
            where_values_learn.append(True if data_type[i] == 0 else False)
            where_values_validate.append(True if data_type[i] == 1 else False)
            where_values_test.append(True if data_type[i] == 2 else False)
            where_values_end_inp_seq.append(False)
        for i in range(len(data_sources_predict[0])):
            if len(data_sources_predict[0][i]) > 1:
                where_values_end_inp_seq[i - 1] = True
                break
        for i_ds in range(len(data_sources_true) - 1, -1, -1):
            for i_p in range(len(self.data_sources_meta[i_ds].visualize) - 1, -1, -1):
                type_chart, data_indexes = self.data_sources_meta[i_ds].visualize[i_p]
                if type_chart == "candle":
                    if len(data_indexes) != 2 and len(data_indexes) != 4:
                        raise ValueError("Количество индексов в типе визуализации candle должно быть 4 или 2.")
                    # формируем список с данными формата: [date, open, high, low, close]
                    data_true = [[data_sources_true[i_ds][i_c][dat_ind] for dat_ind in data_indexes] for i_c in range(len(data_sources_true[i_ds]))]
                    data_predict = []
                    i_candle = 0
                    while len(data_sources_predict[i_ds][i_candle]) == 1:
                        data_predict.append([])
                        i_candle += 1
                    data_predict.extend([[data_sources_predict[i_ds][i_c][dat_ind] for dat_ind in data_indexes] for i_c in range(i_candle, len(data_sources_predict[i_ds]))])
                    if len(data_indexes) == 2:
                        for i_d in range(len(data_true)):
                            data_true[i_d].append(data_true[i_d][1])
                            data_true[i_d].insert(0, data_true[i_d][0])

                            if i_d >= i_candle:
                                data_predict[i_d].append(data_predict[i_d][1])
                                data_predict[i_d].insert(0, data_predict[i_d][0])

                    # добавляем даты
                    for i_d in range(len(data_true)):
                        data_true[i_d].insert(0, data_sources_true[i_ds][i_d][0])
                        data_predict[i_d].insert(0, data_sources_predict[i_ds][i_d][0])

                    high_canal = [item[2] for item in data_true]
                    low_canal = [item[3] for item in data_true]

                    mplfinance_data_true = self.data_to_mplfinance_candle(data_true)
                    mplfinance_data_predict = self.data_to_mplfinance_candle(data_predict)

                    p_data_true = pd.DataFrame.from_dict(mplfinance_data_true)
                    p_data_true.set_index('Date', inplace=True)
                    p_data_predict = pd.DataFrame.from_dict(mplfinance_data_predict)
                    p_data_predict.set_index('Date', inplace=True)

                    ymin = min(min([min(item[1:5]) for item in data_true]), min([min(item[1:5]) for item in data_predict[i_candle:]]))
                    ymax = max(max([max(item[1:5]) for item in data_true]), max([max(item[1:5]) for item in data_predict[i_candle:]]))
                    data_type_width = (ymax - ymin) * data_type_rate
                    y_over = (ymax - ymin) * y_over_rate
                    ymin -= y_over + data_type_width
                    ymax += y_over

                    y_label = self.data_sources_meta[i_ds].visualize_name[i_p]
                    if i_p == 0:
                        y_label = f"{y_label} \"{self.data_sources_file_names[i_ds][0][:len(self.data_sources_file_names[i_ds][0]) - 4].replace('.', '_')}\""
                    panel_num = i_ds * len(self.data_sources_meta[i_ds].visualize) + i_p

                    dict_none = dict(y1=ymin, y2=ymin + data_type_width, where=where_values_none, alpha=0.55, color='red')
                    dict_learn = dict(y1=ymin, y2=ymin + data_type_width, where=where_values_learn, alpha=0.6, color='darkorange')
                    dict_validate = dict(y1=ymin, y2=ymin + data_type_width, where=where_values_validate, alpha=0.5, color='blueviolet')
                    dict_test = dict(y1=ymin, y2=ymin + data_type_width, where=where_values_test, alpha=0.9, color='deepskyblue')
                    dict_end_inp_seq = dict(y1=ymin, y2=ymax, where=where_values_end_inp_seq, alpha=0.55, color='red')

                    if i_ds == 0 and i_p == 0:
                        add_plot.append(mpf.make_addplot(high_canal, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="black"))
                        add_plot.append(mpf.make_addplot(low_canal, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="black"))
                        add_plot.append(mpf.make_addplot(p_data_predict, type='candle', ylim=(ymin,ymax)))
                    else:
                        add_plot.append(mpf.make_addplot(high_canal, type='line', panel=panel_num, fill_between=[dict_none,dict_learn,dict_validate,dict_test,dict_end_inp_seq], ylabel=y_label, ylim=(ymin, ymax), linewidths=1, alpha=1, color="black"))
                        add_plot.append(mpf.make_addplot(low_canal, type='line', panel=panel_num, ylim=(ymin, ymax), linewidths=1, alpha=1, color="black"))
                        add_plot.append(mpf.make_addplot(p_data_true, type='candle', panel=panel_num, ylim=(ymin, ymax)))
                        add_plot.append(mpf.make_addplot(p_data_predict, type='candle', panel=panel_num, ylim=(ymin, ymax)))

                elif type_chart == "line":
                    if len(data_indexes) != 1:
                        raise ValueError("Количество индексов в типе визуализации line должно быть 1.")
                    data_true = [[data_sources_true[i_ds][i_c][dat_ind] for dat_ind in data_indexes] for i_c in range(len(data_sources_true[i_ds]))]
                    data_predict = []
                    i_candle = 0
                    while len(data_sources_predict[i_ds][i_candle]) == 1:
                        data_predict.append([])
                        i_candle += 1
                    data_predict.extend([[data_sources_predict[i_ds][i_c][dat_ind] for dat_ind in data_indexes] for i_c in range(i_candle, len(data_sources_predict[i_ds]))])

                    # добавляем даты
                    for i_d in range(len(data_true)):
                        data_true[i_d].insert(0, data_sources_true[i_ds][i_d][0])
                        data_predict[i_d].insert(0, data_sources_predict[i_ds][i_d][0])

                    mplfinance_data_true = self.data_to_mplfinance_line(data_true)
                    mplfinance_data_predict = self.data_to_mplfinance_line(data_predict)

                    p_data_true = pd.DataFrame.from_dict(mplfinance_data_true)
                    p_data_true.set_index('Date', inplace=True)
                    p_data_predict = pd.DataFrame.from_dict(mplfinance_data_predict)
                    p_data_predict.set_index('Date', inplace=True)

                    ymin = min(min([item[1] for item in data_true]), min([item[1] for item in data_predict[i_candle:]]))
                    ymax = max(max([item[1] for item in data_true]), max([item[1] for item in data_predict[i_candle:]]))
                    data_type_width = (ymax - ymin) * data_type_rate
                    y_over = (ymax - ymin) * y_over_rate
                    ymin -= y_over + data_type_width
                    ymax += y_over

                    y_label = self.data_sources_meta[i_ds].visualize_name[i_p]
                    if i_p == 0:
                        y_label = f"{y_label} \"{self.data_sources_file_names[i_ds][0][:len(self.data_sources_file_names[i_ds][0]) - 4].replace('.', '_')}\""
                    panel_num = i_ds * len(self.data_sources_meta[i_ds].visualize) + i_p

                    dict_none = dict(y1=ymin, y2=ymin + data_type_width, where=where_values_none, alpha=0.55, color='red')
                    dict_learn = dict(y1=ymin, y2=ymin + data_type_width, where=where_values_learn, alpha=0.6, color='darkorange')
                    dict_validate = dict(y1=ymin, y2=ymin + data_type_width, where=where_values_validate, alpha=0.5, color='blueviolet')
                    dict_test = dict(y1=ymin, y2=ymin + data_type_width, where=where_values_test, alpha=0.9, color='deepskyblue')
                    dict_end_inp_seq = dict(y1=ymin, y2=ymax, where=where_values_end_inp_seq, alpha=0.55, color='red')

                    if i_ds == 0 and i_p == 0:
                        add_plot.append(mpf.make_addplot(p_data_true, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="black"))
                        add_plot.append(mpf.make_addplot(p_data_predict, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="springgreen"))
                    else:
                        add_plot.append(mpf.make_addplot(p_data_true, type='line', panel=panel_num, fill_between=[dict_none,dict_learn,dict_validate,dict_test,dict_end_inp_seq], ylabel=y_label, ylim=(ymin, ymax), linewidths=1, alpha=1, color="black"))
                        add_plot.append(mpf.make_addplot(p_data_predict, type='line', panel=panel_num, ylim=(ymin, ymax), linewidths=1, alpha=1, color="springgreen"))

        myrcparams = {'axes.labelsize': 'small'}
        my_style = mpf.make_mpf_style(base_mpf_style='yahoo', facecolor='white', y_on_right=False, rc=myrcparams)
        panel_ratios = ()
        for i_ds in range(len(data_sources_true)):
            for i_p in range(len(self.data_sources_meta[i_ds].visualize_ratio)):
                panel_ratios += (self.data_sources_meta[i_ds].visualize_ratio[i_p],)
        mpf.plot(p_data_true, type='candle' if type_chart == "candle" else 'line', style=my_style, ylabel=y_label, ylim=(ymin,ymax), addplot=add_plot, fill_between=[dict_none,dict_learn,dict_validate,dict_test,dict_end_inp_seq], panel_ratios=panel_ratios, figsize=(18,9), datetime_format="%Y-%b-%d", tight_layout=True, savefig=save_file_path)
        #, fill_between = [dict1, dict2, dict3]  image_path = f"{save_folder_path}/{str(i).rjust(2, '0')}_{str(candle_index).rjust(7, '0')}", columns=None if type_chart == "candle" else ['Close']
        print(f"save image {save_file_path}")

    # выполняет прогнозирование, и записывает спрогнозированные данные в self.data_sources_predict. Данные, находящиеся в self.data_sources_predict[i_ds][i_f][10], соответствуют реальным данным, идущим за 10 индексом, то есть дата первого спрогнозированного значения в self.data_sources_predict[i_ds][i_f][10] соответствует дате self.data_sources[i_ds][i_f][11]
    # выполняет визуализацию спрогнозированных последовательностей. Для свечки self.data_sources[i_ds][i_f][11] будет взята спрогнозированная последовательность self.data_sources_predict[i_ds][i_f][10], т.к. разделение на учебные и тестовые прогнозы означает что первое спрогнозированное значение должно быть либо учебным либо тестовым
    def predict_data(self, model, predict_length, is_save_predict_data, part_learn_predict, part_test_predict, part_learn_predict_visualize, part_test_predict_visualize, is_visualize_prediction_union, is_visualize_prediction_single, visualize_prediction_cut):
        self.model = model
        self.predict_length = predict_length
        self.is_save_predict_data = is_save_predict_data
        self.part_learn_predict = part_learn_predict
        self.part_test_predict = part_test_predict
        self.part_learn_predict_visualize = part_learn_predict_visualize
        self.part_test_predict_visualize = part_test_predict_visualize
        self.is_visualize_prediction_union = is_visualize_prediction_union
        self.is_visualize_prediction_single = is_visualize_prediction_single
        self.visualize_prediction_cut = visualize_prediction_cut

        self.data_sources_predict = [[[None] * len(self.data_sources[0][i_f]) for i_f in range(len(self.data_sources[0]))] for i_ds in range(len(self.data_sources))]

        # выполняем прогнозирование
        pred_num = 0
        for i_f in range(len(self.data_sources[0])):
            for i_c in range(len(self.data_sources[0][i_f])):
                if self.data_sources_data_type[i_f][i_c] != -1:
                    is_let_in = is_save_predict_data
                    if not is_let_in and self.data_sources_data_type[i_f][i_c] != 1:
                        probability = part_learn_predict
                        if self.data_sources_data_type[i_f][i_c] == 2:
                            probability = part_test_predict
                        rand = random.random()
                        if rand <= probability:
                            is_let_in = True

                    if is_let_in:
                        predict_data_sources = []
                        for i_ds_pr in range(len(self.data_sources)):
                            predict_data_sources.append([])

                        elapsed_create_inp_seq = 0
                        elapsed_normalize = 0
                        elapsed_predict = 0

                        for i_p in range(self.predict_length):
                            # формируем входную последовательность для всех источников данных
                            time_create_inp_seq_start = time.process_time()
                            data_sources_inp_seq = []
                            true_data_length = max(self.sequence_length - i_p, 0) # количество данных которые нужно взять из источников данных
                            predict_data_length = self.sequence_length - true_data_length # количество данных которые нужно взять из спрогнозированных данных
                            for i_ds in range(len(self.data_sources)):
                                data_source_inp_seq = []
                                for i_seq in range(i_c - self.sequence_length + i_p + 1, i_c - self.sequence_length + i_p + 1 + true_data_length):
                                    data_source_inp_seq.append(copy.deepcopy(self.data_sources[i_ds][i_f][i_seq]))
                                for i_seq in range(i_p - predict_data_length, i_p):
                                    data_source_inp_seq.append(copy.deepcopy(predict_data_sources[i_ds][i_seq]))
                                data_sources_inp_seq.append(data_source_inp_seq)

                            elapsed_create_inp_seq += time.process_time() - time_create_inp_seq_start
                            time_normalize_start = time.process_time()
                            x, y, data_sources_normalizers_settings = self.normalize_data_sources(data_sources_inp_seq)
                            elapsed_normalize += time.process_time() - time_normalize_start
                            x_np = np.array(x)
                            inp = x_np.reshape(1, self.sequence_length, len(x[0]))
                            time_predict_start = time.process_time()
                            pred = model.predict(inp, verbose=0)
                            elapsed_predict += time.process_time() - time_predict_start
                            pred_list = pred[0].tolist()

                            data_sources_inp_seq_denorm, data_sources_out_denorm = self.denormalize_input_vectors_sequence_output_vector(data_sources_normalizers_settings, output_vector=pred_list)
                            # определяем дату спрогнозированной свечки
                            if len(predict_data_sources[0]) > 0:
                                next_date = copy.deepcopy(predict_data_sources[0][-1][0]) + self.data_interval
                            else:
                                next_date = copy.deepcopy(self.data_sources[i_ds][i_f][i_c][0]) + self.data_interval
                            # добавляем выходное значение для каждого источника данных
                            for i_ds_pr in range(len(self.data_sources)):
                                data_sources_out_denorm[i_ds_pr].insert(0, next_date)
                                predict_data_sources[i_ds_pr].append(data_sources_out_denorm[i_ds_pr])

                        for i_ds in range(len(self.data_sources)):
                            self.data_sources_predict[i_ds][i_f][i_c] = predict_data_sources[i_ds]

                        pred_num += 1
                        print(f"pred_num={pred_num}, elapsed_create_inp_seq={elapsed_create_inp_seq}, elapsed_normalize={elapsed_normalize}, elapsed_predict={elapsed_predict}")

        # визуализируем прогнозирование
        if self.is_visualize_prediction_union or self.is_visualize_prediction_single:
            approved_sequence_length = 1 if self.predict_length >= self.visualize_prediction_cut else min(self.sequence_length, visualize_prediction_cut - self.predict_length) #длина входной последовательности которую будем визуализировать, это нужно чтобы при большой длине входных значений была возможность не показывать часть или всю входную последовательность и сосредоточить внимание на сравнении спрогнозированных значений с истинными.
            for i_f in range(len(self.data_sources_predict[0])):
                for i_c in range(len(self.data_sources_predict[0][i_f]) - 1): # -1 т.к. мы последний элемент не проверяем, т.к. следующий за ним индекс следующей свечки (для определения типа данных) не существует
                    if self.data_sources_predict[0][i_f][i_c] != None:
                        i_c_dt = i_c + 1 # index candle data type
                        if self.data_sources_data_type[i_f][i_c_dt] != -1 and self.data_sources_data_type[i_f][i_c_dt] != 1 and len(self.data_sources_data_type[i_f]) - i_c_dt >= self.predict_length:
                            probability = self.part_learn_predict_visualize
                            if self.data_sources_data_type[i_f][i_c_dt] == 2:
                                probability = self.part_test_predict_visualize
                            rand = random.random()
                            if rand <= probability:
                                # создаем данные для всех источников данных
                                data_sources_true = []
                                data_sources_predict = []
                                for i_ds in range(len(self.data_sources)):
                                    data_source_true = [copy.deepcopy(self.data_sources[i_ds][i_f][i_candle]) for i_candle in range(i_c_dt - approved_sequence_length, i_c_dt + self.predict_length)]
                                    # в спрогнозированных данных, на месте истинных данных присутствуют только дата без данных для этой даты, такая дата не будет иметь данных на графике
                                    data_source_predict = [copy.deepcopy(self.data_sources[i_ds][i_f][i_candle][0:1]) for i_candle in range(i_c_dt - approved_sequence_length, i_c_dt)]
                                    #добавляем спрогнозированные данные
                                    data_source_predict.extend(self.data_sources_predict[i_ds][i_f][i_c])
                                    data_sources_true.append(data_source_true)
                                    data_sources_predict.append(data_source_predict)

                                data_type = [self.data_sources_data_type[i_f][i_candle] for i_candle in range(i_c_dt - approved_sequence_length, i_c_dt + self.predict_length)]


                                if self.is_visualize_prediction_union and len(self.data_sources) > 1:
                                    save_file_path = f"{self.folder_images_learn_predict if self.data_sources_data_type[i_f][i_c_dt] == 0 else self.folder_images_test_predict}/union_{str(i_f).rjust(3, '0')}_{str(i_c_dt).rjust(7, '0')}"
                                    self.visualize_predict_to_file(data_sources_true, data_sources_predict, data_type, save_file_path)

                                if self.is_visualize_prediction_single:
                                    for i_ds in range(len(self.data_sources)):
                                        save_file_path = f"{self.folder_images_learn_predict if self.data_sources_data_type[i_f][i_c_dt] == 0 else self.folder_images_test_predict}/single_{self.data_sources_file_names[i_ds][0][:len(self.data_sources_file_names[i_ds][0]) - 4].replace('.', '_')}_{str(i_f).rjust(3, '0')}_{str(i_c_dt).rjust(7, '0')}"
                                        self.visualize_predict_to_file(data_sources_true[i_ds:i_ds + 1], data_sources_predict[i_ds:i_ds + 1], data_type, save_file_path)

