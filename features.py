import sys
import time
import random
import os
import warnings
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


warnings.filterwarnings("ignore", module = "mplfinance\..*" )
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
    def __init__(self, files, short_name, date_index, data_indexes_in_file, losses_data_indexes, is_save_data, output_inserts, normalizers, visualize, is_visualize, visualize_ratio, visualize_name, visualize_data_source_panel = 1):
        self.files = files # список с файлами
        self.short_name = short_name
        self.date_index = date_index # индекс даты
        self.data_indexes_in_file = data_indexes_in_file # индексы данных, которые нужно считать из файла
        self.losses_data_indexes = losses_data_indexes # индексы считанных данных, для которых будет вычисляться ошибка для данного источника данных
        self.is_save_data = is_save_data # сохранять ли спрогнозированные данные для данного источника данных
        self.output_inserts = output_inserts # список с вставками в денормализованный список значений данного источника данных. Нужен чтобы добавлять значения, которые не были спрогнозированы нейронной сетью, и поэтому отсутствуют в денормализованном списке, но имеются во входном списке значений данного источника данных
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

    def __init__(self, data_sources_meta, validation_split, sequence_length, predict_length, part_learn_predict, part_test_predict, part_learn_predict_visualize, part_test_predict_visualize, learn_predict_visualize_one_step_limit, test_predict_visualize_one_step_limit, is_visualize_prediction_union, is_visualize_prediction_single, visualize_prediction_cut, is_save_predict_data, periods):
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
        self.learn_predict_visualize_one_step_limit = learn_predict_visualize_one_step_limit
        self.test_predict_visualize_one_step_limit = test_predict_visualize_one_step_limit
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
            self.data_sources.append([self.read_csv_file(file, data_sources_meta[i_ds].date_index, data_sources_meta[i_ds].data_indexes_in_file) for file in data_sources_meta[i_ds].files])

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

        # выполняем подготовку вставок в выходные данные
        for i_ds in range(len(data_sources_meta)):
            for i_oi in range(len(data_sources_meta[i_ds].output_inserts)):
                data_sources_meta[i_ds].output_inserts[i_oi].summary()

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
        number += 10
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
            x, y, data_sources_normalizers_settings = self.normalize_data_sources(output_i_f=i_f, output_i_c=i_c, data_sources_inp_seq=data_sources_inp_seq, data_sources_output=data_sources_output)
            self.data_sources_periods_x_y[i_f][i_c] = (x, y, data_sources_normalizers_settings)

            if (current_datetime_timestamp - self.periods_start_timestamp) / (self.periods_end_timestamp - self.periods_start_timestamp) >= number / 100:
                print(f"Создание входных последовательностей: {number}%")
                number += 10

            while self.data_sources[0][i_f][i_c][0] <= current_datetime_timestamp:
                i_c += 1
                if i_c >= len(self.data_sources[0][i_f]):
                    i_c = 0
                    i_f += 1
            current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]

    def get_learning_valid_x_y(self, datetime_start_timestamp, datetime_end_timestamp):
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
            learn_part = learn_count / (learn_count + valid_count) if learn_count + valid_count > 0 else 0
            valid_part = valid_count / (learn_count + valid_count) if learn_count + valid_count > 0 else 0
            if learn_part < 1 - self.validation_split and valid_part < self.validation_split:
                data_type = random.choice(data_types)
            elif learn_part < 1 - self.validation_split:
                data_type = data_types[0]
            else:
                data_type = data_types[1]

            x, y, data_sources_normalizers_settings = self.data_sources_periods_x_y[i_f][i_c]

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

    def get_period_x_y(self, datetime_start_timestamp, datetime_end_timestamp):
        # доходим до первой свечки начала периода
        i_f = 0
        i_c = 0
        while self.data_sources[0][i_f][i_c][0] < datetime_start_timestamp:
            i_c += 1
            if i_c >= len(self.data_sources[0][i_f]):
                i_c = 0
                i_f += 1
        current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]

        x_period = []
        y_period = []
        data_sources_normalizers_settings_period = []
        file_candle_indexes_period = []

        while current_datetime_timestamp <= datetime_end_timestamp:
            x, y, data_sources_normalizers_settings = self.data_sources_periods_x_y[i_f][i_c]

            x_period.append(x)
            y_period.append(y)
            data_sources_normalizers_settings_period.append(data_sources_normalizers_settings)
            file_candle_indexes_period.append((i_f, i_c))

            while self.data_sources[0][i_f][i_c][0] <= current_datetime_timestamp:
                i_c += 1
                if i_c >= len(self.data_sources[0][i_f]):
                    i_c = 0
                    i_f += 1
            current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]
        return (x_period, y_period, data_sources_normalizers_settings_period, file_candle_indexes_period)

    def save_predict_data(self, data_source_y_predict_multi_step, new_file_path):
        with open(f"{new_file_path}", "w", encoding='utf-8') as new_file:
            for i in range(len(data_source_y_predict_multi_step)):
                new_line = ""
                for i_predict in range(len(data_source_y_predict_multi_step[i])):
                    new_line += f"{str(data_source_y_predict_multi_step[i][i_predict][0])},"
                    new_line += ",".join([float_to_str_format(data_source_y_predict_multi_step[i][i_predict][i_dat], 8) for i_dat in range(1, len(data_source_y_predict_multi_step[i][i_predict]))]) + "|"
                new_line = f"{new_line[:-1]}\n"
                new_file.write(new_line)

    def create_period_folders(self, period):
        period.base_folder = f"{self.base_folder}/{period.learning_start.date_time.strftime('%Y-%m-%d %Hh %Mm')} - {period.testing_end.date_time.strftime('%Y-%m-%d %Hh %Mm')}"
        os.makedirs(period.base_folder)
        period.save_model_folder = f"{period.base_folder}/save_model"
        os.makedirs(period.save_model_folder)
        period.learning_folder = f"{period.base_folder}/learning"
        os.makedirs(period.learning_folder)
        period.learning_data_folder = f"{period.learning_folder}/data"
        os.makedirs(period.learning_data_folder)
        period.learning_images_folder = f"{period.learning_folder}/images"
        os.makedirs(period.learning_images_folder)
        period.learning_images_predict_folder = f"{period.learning_images_folder}/predict"
        os.makedirs(period.learning_images_predict_folder)

        period.testing_folder = f"{period.base_folder}/testing"
        os.makedirs(period.testing_folder)
        period.testing_data_folder = f"{period.testing_folder}/data"
        os.makedirs(period.testing_data_folder)
        period.testing_images_folder = f"{period.testing_folder}/images"
        os.makedirs(period.testing_images_folder)
        period.testing_images_predict_folder = f"{period.testing_images_folder}/predict"
        os.makedirs(period.testing_images_predict_folder)

    def predict_one_step(self, period, datetime_start_timestamp, datetime_end_timestamp, progress_label):
        x_period, y_period, data_sources_normalizers_settings_period, file_candle_indexes_period = self.get_period_x_y(datetime_start_timestamp, datetime_end_timestamp)
        data_sources_y_predict_period = []
        number = 0
        print(f"{progress_label} {number}%")
        number += 20
        for i in range(len(x_period)):
            x_np = np.array(x_period[i])
            inp = x_np.reshape(1, self.sequence_length, len(x_period[i][0]))
            pred = period.model.predict(inp, verbose=0)
            output_vector = pred[0].tolist()
            output_datetime_timestamp = self.data_sources[0][file_candle_indexes_period[i][0]][file_candle_indexes_period[i][1]][0]
            data_sources_input_sequence_denorm, data_sources_output_denorm = self.denormalize_insert_input_vectors_sequence_output_vector(data_sources_normalizers_settings_period[i], input_vectors_sequence=None, output_vector=output_vector, output_datetime_timestamp=output_datetime_timestamp)
            data_sources_y_predict_period.append(data_sources_output_denorm)
            if i / (len(x_period) - 1) >= number / 100:
                print(f"{progress_label} {number}%")
                number += 20

        return (data_sources_y_predict_period, file_candle_indexes_period)

    def loss_one_step(self, data_sources_y_predict, file_candle_indexes):
        mean_absolute_percentage_error_one_step_sum_single = [0] * len(self.data_sources)
        for i in range(len(data_sources_y_predict)):
            i_f, i_c = file_candle_indexes[i]
            for i_ds in range(len(self.data_sources)):
                mean_absolute_percentage_error = sum([abs(self.data_sources[i_ds][i_f][i_c][data_index] - data_sources_y_predict[i][i_ds][data_index]) / (self.data_sources[i_ds][i_f][i_c][data_index] + self.mean_absolute_percentage_error_epsilon) for data_index in self.data_sources_meta[i_ds].losses_data_indexes]) / len(self.data_sources_meta[i_ds].losses_data_indexes) * 100
                mean_absolute_percentage_error_one_step_sum_single[i_ds] += mean_absolute_percentage_error

        mean_absolute_percentage_error_one_step_single = [0] * len(self.data_sources)
        for i_ds in range(len(self.data_sources)):
            mean_absolute_percentage_error_one_step_single[i_ds] = mean_absolute_percentage_error_one_step_sum_single[i_ds] / len(data_sources_y_predict)
        mean_absolute_percentage_error_one_step_join = sum(mean_absolute_percentage_error_one_step_single) / len(self.data_sources)
        return mean_absolute_percentage_error_one_step_single, mean_absolute_percentage_error_one_step_join

    def visualize_one_step(self, save_image_folder, data_sources_y_predict, file_candle_indexes, visualize_one_step_limit):
        data_sources_true = []
        data_sources_predict = []
        visualize_data_sources_indexes = []
        for i_ds in range(len(self.data_sources_meta)):
            if self.data_sources_meta[i_ds].is_visualize:
                data_sources_true.append([])
                data_sources_predict.append([])
                visualize_data_sources_indexes.append(i_ds)
        visualize_start_timestamp = None
        visualize_count = 0
        i = 0
        while True:
            if visualize_start_timestamp is None:
                visualize_start_timestamp = data_sources_y_predict[i][0][0]
            visualize_end_timestamp = data_sources_y_predict[i][0][0]

            data_sources_add_labels = [""] * len(visualize_data_sources_indexes)

            i_ds = 0
            for i_visual_ds in visualize_data_sources_indexes:
                i_f, i_c = file_candle_indexes[i]
                data_sources_true[i_ds].append(self.data_sources[i_visual_ds][i_f][i_c])
                data_sources_predict[i_ds].append(data_sources_y_predict[i][i_visual_ds])
                i_ds += 1

            if len(data_sources_true[0]) >= self.visualize_prediction_cut or i == len(data_sources_y_predict) - 1:
                if self.is_visualize_prediction_union and len(visualize_data_sources_indexes) > 1:
                    self.visualize_predict_to_file(data_sources_true, data_sources_predict, visualize_data_sources_indexes, data_sources_add_labels, f"{save_image_folder}/union {timestamp_to_utc_datetime(visualize_start_timestamp).strftime('%Y-%m-%d %Hh %Mm')} - {timestamp_to_utc_datetime(visualize_end_timestamp).strftime('%Y-%m-%d %Hh %Mm')}.png")

                if self.is_visualize_prediction_single or (not self.is_visualize_prediction_single and self.is_visualize_prediction_union and len(visualize_data_sources_indexes) == 1):
                    i_ds = 0
                    for i_visual_ds in visualize_data_sources_indexes:
                        self.visualize_predict_to_file([data_sources_true[i_ds]], [data_sources_predict[i_ds]], [i_visual_ds], [data_sources_add_labels[i_ds]], f"{save_image_folder}/single {self.data_sources_meta[i_visual_ds].short_name} {timestamp_to_utc_datetime(visualize_start_timestamp).strftime('%Y-%m-%d %Hh %Mm')} - {timestamp_to_utc_datetime(visualize_end_timestamp).strftime('%Y-%m-%d %Hh %Mm')}.png")
                        i_ds += 1

                for i_ds2 in range(len(data_sources_true)):
                    del data_sources_true[i_ds2][:]
                    del data_sources_predict[i_ds2][:]
                visualize_start_timestamp = None
                visualize_count += 1
                if visualize_count >= visualize_one_step_limit:
                    break
            if i >= len(data_sources_y_predict) - 1:
                break
            i += 1

    # mean_absolute_percentage_error_y_predict_multi_step_single - значения ошибок для каждого прогнозирования
    def visualize_multi_step(self, save_image_folder, data_sources_y_predict_multi_step, file_candle_indexes_multi_step, mean_absolute_percentage_error_y_predict_multi_step_single, part_predict_visualize):
        approved_sequence_length = 1 if self.predict_length >= self.visualize_prediction_cut else min(self.sequence_length, self.visualize_prediction_cut - self.predict_length)  # длина входной последовательности которую будем визуализировать
        data_sources_true = []
        data_sources_predict = []
        visualize_data_sources_indexes = []
        for i_ds in range(len(self.data_sources_meta)):
            if self.data_sources_meta[i_ds].is_visualize:
                data_sources_true.append([])
                data_sources_predict.append([])
                visualize_data_sources_indexes.append(i_ds)

        if len(data_sources_y_predict_multi_step) > 0:
            available_predict_indexes = list(range(len(data_sources_y_predict_multi_step[0]))) # список с доступными для выбора индексами, среди которых случайно будет выбрано для визуализации
            visualize_count = 0
            i = 0
            while True:
                # определяем индекс визуализируемого прогнозирования
                is_let_in = True
                i_predict = i
                if part_predict_visualize[0]:
                    rand_index = random.randrange(len(available_predict_indexes))
                    i_predict = available_predict_indexes[rand_index]
                    available_predict_indexes.pop(rand_index)
                else:
                    is_let_in = False
                    rand = random.random()
                    if rand <= part_predict_visualize[1]:
                        is_let_in = True

                if is_let_in:
                    i_f, i_c = file_candle_indexes_multi_step[i_predict]
                    if i_c + self.predict_length < len(self.data_sources[0][i_f]):
                        data_sources_add_labels = []
                        i_ds = 0
                        for i_visual_ds in visualize_data_sources_indexes:
                            data_sources_true[i_ds].extend([self.data_sources[i_visual_ds][i_f][i_candle] for i_candle in range(i_c - approved_sequence_length, i_c + self.predict_length)])
                            data_sources_predict[i_ds].extend([self.data_sources[i_visual_ds][i_f][i_candle][0:1] for i_candle in range(i_c - approved_sequence_length, i_c)])
                            data_sources_predict[i_ds].extend(data_sources_y_predict_multi_step[i_visual_ds][i_predict])

                            data_sources_add_labels.append(f"  loss= {float_to_str_format(mean_absolute_percentage_error_y_predict_multi_step_single[i_predict][i_visual_ds], 2)}%")
                            i_ds += 1

                        visualize_start_timestamp = data_sources_true[0][0][0]
                        visualize_end_timestamp = data_sources_true[0][-1][0]

                        if self.is_visualize_prediction_union and len(visualize_data_sources_indexes) > 1:
                            self.visualize_predict_to_file(data_sources_true, data_sources_predict, visualize_data_sources_indexes, data_sources_add_labels, f"{save_image_folder}/union {timestamp_to_utc_datetime(visualize_start_timestamp).strftime('%Y-%m-%d %Hh %Mm')} - {timestamp_to_utc_datetime(visualize_end_timestamp).strftime('%Y-%m-%d %Hh %Mm')}.png")

                        if self.is_visualize_prediction_single or (not self.is_visualize_prediction_single and self.is_visualize_prediction_union and len(visualize_data_sources_indexes) == 1):
                            i_ds = 0
                            for i_visual_ds in visualize_data_sources_indexes:
                                self.visualize_predict_to_file([data_sources_true[i_ds]], [data_sources_predict[i_ds]], [i_visual_ds], [data_sources_add_labels[i_ds]], f"{save_image_folder}/single {self.data_sources_meta[i_visual_ds].short_name} {timestamp_to_utc_datetime(visualize_start_timestamp).strftime('%Y-%m-%d %Hh %Mm')} - {timestamp_to_utc_datetime(visualize_end_timestamp).strftime('%Y-%m-%d %Hh %Mm')}.png")
                                i_ds += 1

                        for i_ds2 in range(len(data_sources_true)):
                            del data_sources_true[i_ds2][:]
                            del data_sources_predict[i_ds2][:]
                        visualize_count += 1
                        if part_predict_visualize[0] and visualize_count >= part_predict_visualize[1]:
                            break
                if len(available_predict_indexes) == 0 or i >= len(data_sources_y_predict_multi_step[0]):
                    break
                i += 1


    def predict_multi_step(self, period, datetime_start_timestamp, datetime_end_timestamp, probability_predict, progress_label):
        # доходим до первой свечки начала периода
        i_f = 0
        i_c = 0
        while self.data_sources[0][i_f][i_c][0] < datetime_start_timestamp:
            i_c += 1
            if i_c >= len(self.data_sources[0][i_f]):
                i_c = 0
                i_f += 1
        current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]

        data_sources_y_predict = []
        for i_ds in range(len(self.data_sources)):
            data_sources_y_predict.append([])
        file_candle_indexes = [] # содержит индекс выходной свечки для первой входной последовательности
        number = 0
        print(f"{progress_label} {number}%")
        number += 10

        while current_datetime_timestamp <= datetime_end_timestamp:
            is_let_in = False
            if self.is_save_predict_data:
                is_let_in = True
            else:
                rand = random.random()
                if rand <= probability_predict:
                    is_let_in = True

            if is_let_in:
                data_sources_y_predict_current = []
                for i_ds_pr in range(len(self.data_sources)):
                    data_sources_y_predict_current.append([])

                elapsed_create_inp_seq = 0
                elapsed_normalize = 0
                elapsed_predict = 0
                elapsed_denormalize = 0

                for i_p in range(self.predict_length):
                    # формируем входную последовательность для всех источников данных
                    time_create_inp_seq_start = time.process_time()
                    data_sources_inp_seq = []
                    true_data_length = max(self.sequence_length - i_p, 0)  # количество данных которые нужно взять из источников данных
                    predict_data_length = self.sequence_length - true_data_length  # количество данных которые нужно взять из спрогнозированных данных
                    for i_ds in range(len(self.data_sources)):
                        data_source_inp_seq = []
                        for i_seq in range(i_c - self.sequence_length + i_p + 1, i_c - self.sequence_length + i_p + 1 + true_data_length):
                            data_source_inp_seq.append(copy.deepcopy(self.data_sources[i_ds][i_f][i_seq]))
                        for i_seq in range(i_p - predict_data_length, i_p):
                            data_source_inp_seq.append(copy.deepcopy(data_sources_y_predict_current[i_ds][i_seq]))
                        data_sources_inp_seq.append(data_source_inp_seq)

                    elapsed_create_inp_seq += time.process_time() - time_create_inp_seq_start
                    time_normalize_start = time.process_time()
                    x, y, data_sources_normalizers_settings = self.normalize_data_sources(output_i_f=i_f, output_i_c=i_c, data_sources_inp_seq=data_sources_inp_seq, data_sources_add_candles=data_sources_y_predict_current, data_sources_output=None)
                    elapsed_normalize += time.process_time() - time_normalize_start
                    x_np = np.array(x)
                    inp = x_np.reshape(1, self.sequence_length, len(x[0]))
                    time_predict_start = time.process_time()
                    pred = period.model.predict(inp, verbose=0)
                    elapsed_predict += time.process_time() - time_predict_start
                    output_vector = pred[0].tolist()

                    if predict_data_length > 0:
                        output_datetime_timestamp = data_sources_y_predict_current[i_ds][-1][0] + self.interval_milliseconds
                    else:
                        output_datetime_timestamp = self.data_sources[i_ds][i_f][i_c][0] + self.interval_milliseconds
                    time_denormalize_start = time.process_time()
                    data_sources_input_sequence_denorm, data_sources_output_denorm = self.denormalize_insert_input_vectors_sequence_output_vector(data_sources_normalizers_settings, input_vectors_sequence=None, output_vector=output_vector, output_datetime_timestamp=output_datetime_timestamp)
                    elapsed_denormalize += time.process_time() - time_denormalize_start

                    for i_ds_pr in range(len(self.data_sources)):
                        data_sources_y_predict_current[i_ds_pr].append(data_sources_output_denorm[i_ds_pr])

                # print(f"i_f={i_f} i_c={i_c}")
                # print(f"elapsed_create_inp_seq={elapsed_create_inp_seq}")
                # print(f"elapsed_normalize=     {elapsed_normalize}")
                # print(f"elapsed_predict=       {elapsed_predict}")
                # print(f"elapsed_denormalize=   {elapsed_denormalize}")

                for i_ds in range(len(self.data_sources)):
                    data_sources_y_predict[i_ds].append(data_sources_y_predict_current[i_ds])
                file_candle_indexes.append((i_f, i_c + 1))

            if (current_datetime_timestamp - datetime_start_timestamp) / (datetime_end_timestamp - datetime_start_timestamp) >= number / 100:
                print(f"{progress_label} {number}%")
                number += 10

            while self.data_sources[0][i_f][i_c][0] <= current_datetime_timestamp:
                i_c += 1
                if i_c >= len(self.data_sources[0][i_f]):
                    i_c = 0
                    i_f += 1
            current_datetime_timestamp = self.data_sources[0][i_f][i_c][0]

        return data_sources_y_predict, file_candle_indexes

    def loss_multi_step(self, data_sources_y_predict_multi_step, file_candle_indexes_multi_step):
        mean_absolute_percentage_error_y_predict_multi_step_single = [] # значения ошибок для каждого прогнозирования
        mean_absolute_percentage_error_multi_step_sum_single = [0] * len(self.data_sources)
        multi_step_count = 0  # количество подсчитанных спрогнозированных последовательностей. Если истинных данных нет (например прогноз для последних данных) то расчет ошибки для этого прогноза делаться не будет.
        if len(data_sources_y_predict_multi_step) > 0:
            for i in range(len(data_sources_y_predict_multi_step[0])):
                mean_absolute_percentage_error_y_predict_multi_step_single.append([0.0] * len(self.data_sources))
                i_f, i_c = file_candle_indexes_multi_step[i]
                if i_c + self.predict_length < len(self.data_sources[0][i_f]):
                    for i_pred in range(self.predict_length):
                        i_c_curr = i_c + i_pred
                        for i_ds in range(len(self.data_sources)):
                            mean_absolute_percentage_error = sum([abs(self.data_sources[i_ds][i_f][i_c_curr][data_index] - data_sources_y_predict_multi_step[i_ds][i][i_pred][data_index]) / (self.data_sources[i_ds][i_f][i_c_curr][data_index] + self.mean_absolute_percentage_error_epsilon) for data_index in self.data_sources_meta[i_ds].losses_data_indexes]) / len(self.data_sources_meta[i_ds].losses_data_indexes) * 100
                            mean_absolute_percentage_error_multi_step_sum_single[i_ds] += mean_absolute_percentage_error
                            mean_absolute_percentage_error_y_predict_multi_step_single[-1][i_ds] += mean_absolute_percentage_error
                    for i_ds in range(len(self.data_sources)):
                        mean_absolute_percentage_error_y_predict_multi_step_single[-1][i_ds] /= self.predict_length
                    multi_step_count += 1

        mean_absolute_percentage_error_multi_step_single = None
        mean_absolute_percentage_error_multi_step_join = None
        if multi_step_count > 0:
            is_mean_absolute_percentage_error_multi_step = True
            mean_absolute_percentage_error_multi_step_single = [0.0] * len(self.data_sources)
            for i_ds in range(len(self.data_sources)):
                mean_absolute_percentage_error_multi_step_single[i_ds] = mean_absolute_percentage_error_multi_step_sum_single[i_ds] / multi_step_count / self.predict_length
            mean_absolute_percentage_error_multi_step_join = sum(mean_absolute_percentage_error_multi_step_single) / len(self.data_sources)
        else:
            is_mean_absolute_percentage_error_multi_step = False
        return is_mean_absolute_percentage_error_multi_step, mean_absolute_percentage_error_multi_step_single, mean_absolute_percentage_error_multi_step_join, mean_absolute_percentage_error_y_predict_multi_step_single

    def handle_period(self, period):
        self.create_period_folders(period)

        learning_start_timestamp = utc_datetime_to_timestamp(period.learning_start.date_time)
        learning_end_timestamp = utc_datetime_to_timestamp(period.learning_end.date_time)
        testing_start_timestamp = utc_datetime_to_timestamp(period.testing_start.date_time)
        testing_end_timestamp = utc_datetime_to_timestamp(period.testing_end.date_time)
        x_learn, y_learn, x_valid, y_valid = self.get_learning_valid_x_y(learning_start_timestamp, learning_end_timestamp)
        print(f"длина обучающей выборки={len(x_learn)}")
        print(f"длина выборки валидации={len(x_valid)}")
        if period.model is None:
            models = []
            models_losses = []
            i = 0
            while True:
                print(f"Input Shape: {(self.sequence_length, len(x_learn[0][0]))}")
                model = Sequential()
                model.add(Input((self.sequence_length, len(x_learn[0][0]))))
                # model.add(LSTM(128, return_sequences=True))
                model.add(LSTM(64))
                model.add(Dense(len(y_learn[0]), activation='sigmoid'))
                model.summary()

                model.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()], optimizer='adam')
                # "binary_crossentropy" tf.keras.losses.MeanAbsolutePercentageError()

                history = model.fit(np.array(x_learn), np.array(y_learn), batch_size=32, epochs=10, validation_data=(np.array(x_valid), np.array(y_valid)))

                fig_w = 6
                fig_h = 5
                plt.figure(figsize=(fig_w, fig_h))
                plt.title(f"{period.learning_start.date_time.strftime('%Y-%m-%d %Hh %Mm')} - {period.learning_end.date_time.strftime('%Y-%m-%d %Hh %Mm')} [:] _{i + 1}")
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.show()
                if len(history.history['loss']) > 9:
                    plt.figure(figsize=(fig_w, fig_h))
                    plt.title(f"{period.learning_start.date_time.strftime('%Y-%m-%d %Hh %Mm')} - {period.learning_end.date_time.strftime('%Y-%m-%d %Hh %Mm')} [7:] _{i + 1}")
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

        error_messages = []
        if period.model.input_shape[1] != self.sequence_length:
            error_messages.append(f"Длина последовательности входных данных не совпадает с формой входного тензора модели нейронной сети.")
        if period.model.input_shape[2] != len(x_learn[0][0]):
            error_messages.append(f"Длина входного вектора не совпадает с формой входного тензора модели нейронной сети.")
        if period.model.output_shape[1] != len(y_learn[0]):
            error_messages.append(f"Длина выходного вектора не совпадает с формой входного тензора модели нейронной сети.")

        if len(error_messages) > 0:
            for message in error_messages:
                print(message)
            raise ValueError(error_messages[0])

        x_np = np.array(x_learn[0])
        inp = x_np.reshape(1, self.sequence_length, len(x_learn[0][0]))
        pred = period.model.predict(inp, verbose=0)
        single_elapsed_start = time.time()
        for i in range(4000):
            pred = period.model.predict(inp, verbose=0)
        print(f"single_elapsed= {time.time() - single_elapsed_start}")
        output_vector = pred[0].tolist()
        print(f"single output_vector={output_vector}")

        count1 = 100
        x_np = np.array([x_learn[0]] * count1)
        inp = x_np.reshape(count1, self.sequence_length, len(x_learn[0][0]))
        combo_elapsed_start = time.time()
        for i in range(40):
            pred = period.model.predict(inp, verbose=0)
        print(f"combo_elapsed= {time.time() - combo_elapsed_start}")
        output_vector1 = pred[0].tolist()
        output_vector2 = pred[1].tolist()
        print(f"combo output_vector1={output_vector1}")
        print(f"combo output_vector2={output_vector2}")

        count2 = 1000
        x_np = np.array([x_learn[0]] * count2)
        inp = x_np.reshape(count2, self.sequence_length, len(x_learn[0][0]))
        super_combo_elapsed_start = time.time()
        for i in range(4):
            pred = period.model.predict(inp, verbose=0)
        print(f"super_combo_elapsed= {time.time() - super_combo_elapsed_start}")
        output_vector3 = pred[0].tolist()
        output_vector4 = pred[1].tolist()
        print(f"super combo output_vector3={output_vector3}")
        print(f"super combo output_vector4={output_vector4}")


        sys.exit()

        period.model.save(period.save_model_folder) # сохраняем модель

        # прогнозирование на несколько шагов для учебных и тестовых данных и вычисление ошибки
        data_sources_y_predict_multi_step_learn, file_candle_indexes_multi_step_learn = self.predict_multi_step(period, learning_start_timestamp, learning_end_timestamp, self.part_learn_predict, "1/4 Прогнозирование на несколько шагов вперед для учебных данных:")
        is_mean_absolute_percentage_error_multi_step_learn, mean_absolute_percentage_error_multi_step_learn_single, mean_absolute_percentage_error_multi_step_learn_join, mean_absolute_percentage_error_y_predict_multi_step_learn_single = self.loss_multi_step(data_sources_y_predict_multi_step_learn, file_candle_indexes_multi_step_learn)
        period.is_mean_absolute_percentage_error_multi_step_learn = is_mean_absolute_percentage_error_multi_step_learn
        if is_mean_absolute_percentage_error_multi_step_learn:
            period.mean_absolute_percentage_error_multi_step_learn_single = mean_absolute_percentage_error_multi_step_learn_single
            period.mean_absolute_percentage_error_multi_step_learn_join = mean_absolute_percentage_error_multi_step_learn_join

        data_sources_y_predict_multi_step_test, file_candle_indexes_multi_step_test = self.predict_multi_step(period, testing_start_timestamp, testing_end_timestamp, self.part_test_predict, "2/4 Прогнозирование на несколько шагов вперед для тестовых данных:")
        is_mean_absolute_percentage_error_multi_step_test, mean_absolute_percentage_error_multi_step_test_single, mean_absolute_percentage_error_multi_step_test_join, mean_absolute_percentage_error_y_predict_multi_step_test_single = self.loss_multi_step(data_sources_y_predict_multi_step_test, file_candle_indexes_multi_step_test)
        period.is_mean_absolute_percentage_error_multi_step_test = is_mean_absolute_percentage_error_multi_step_test
        if is_mean_absolute_percentage_error_multi_step_test:
            period.mean_absolute_percentage_error_multi_step_test_single = mean_absolute_percentage_error_multi_step_test_single
            period.mean_absolute_percentage_error_multi_step_test_join = mean_absolute_percentage_error_multi_step_test_join

        if self.is_save_predict_data:
            for i_ds in range(len(self.data_sources)):
                self.save_predict_data(data_sources_y_predict_multi_step_learn[i_ds], f"{period.learning_data_folder}/{self.data_sources_file_names[i_ds][0]}")
            for i_ds in range(len(self.data_sources)):
                self.save_predict_data(data_sources_y_predict_multi_step_test[i_ds], f"{period.testing_data_folder}/{self.data_sources_file_names[i_ds][0]}")
        else:
            print("Сохранение спрогнозированных данных отключено.")

        # прогнозирование на один шаг для учебных и тестовых данных и вычисление ошибки
        data_sources_y_predict_one_step_learn, file_candle_indexes_one_step_learn = self.predict_one_step(period, learning_start_timestamp, learning_end_timestamp, "3/4 Прогнозирование на один шаг вперед для учебных данных:")
        mean_absolute_percentage_error_one_step_learn_single, mean_absolute_percentage_error_one_step_learn_join = self.loss_one_step(data_sources_y_predict_one_step_learn, file_candle_indexes_one_step_learn)
        period.mean_absolute_percentage_error_one_step_learn_single = mean_absolute_percentage_error_one_step_learn_single
        period.mean_absolute_percentage_error_one_step_learn_join = mean_absolute_percentage_error_one_step_learn_join

        data_sources_y_predict_one_step_test, file_candle_indexes_one_step_test = self.predict_one_step(period, testing_start_timestamp, testing_end_timestamp, "4/4 Прогнозирование на один шаг вперед для тестовых данных:")
        mean_absolute_percentage_error_one_step_test_single, mean_absolute_percentage_error_one_step_test_join = self.loss_one_step(data_sources_y_predict_one_step_test, file_candle_indexes_one_step_test)
        period.mean_absolute_percentage_error_one_step_test_single = mean_absolute_percentage_error_one_step_test_single
        period.mean_absolute_percentage_error_one_step_test_join = mean_absolute_percentage_error_one_step_test_join

        print(f"period.is_mean_absolute_percentage_error_multi_step_learn={period.is_mean_absolute_percentage_error_multi_step_learn}")
        if period.is_mean_absolute_percentage_error_multi_step_learn:
            print(f"period.mean_absolute_percentage_error_multi_step_learn_single={period.mean_absolute_percentage_error_multi_step_learn_single}")
            print(f"period.mean_absolute_percentage_error_multi_step_learn_join={period.mean_absolute_percentage_error_multi_step_learn_join}")

        print(f"period.is_mean_absolute_percentage_error_multi_step_test={period.is_mean_absolute_percentage_error_multi_step_test}")
        if period.is_mean_absolute_percentage_error_multi_step_test:
            print(f"period.mean_absolute_percentage_error_multi_step_test_single={period.mean_absolute_percentage_error_multi_step_test_single}")
            print(f"period.mean_absolute_percentage_error_multi_step_test_join={period.mean_absolute_percentage_error_multi_step_test_join}")

        print(f"period.mean_absolute_percentage_error_one_step_learn_single={period.mean_absolute_percentage_error_one_step_learn_single}")
        print(f"period.mean_absolute_percentage_error_one_step_learn_join={period.mean_absolute_percentage_error_one_step_learn_join}")

        print(f"period.mean_absolute_percentage_error_one_step_test_single={period.mean_absolute_percentage_error_one_step_test_single}")
        print(f"period.mean_absolute_percentage_error_one_step_test_join={period.mean_absolute_percentage_error_one_step_test_join}")

        # визуализируем прогнозирования
        is_at_least_one_visualize = False
        for i_ds_meta in range(len(self.data_sources_meta)):
            if self.data_sources_meta[i_ds_meta].is_visualize:
                is_at_least_one_visualize = True
        if is_at_least_one_visualize:
            self.visualize_one_step(period.learning_images_folder, data_sources_y_predict_one_step_learn, file_candle_indexes_one_step_learn, self.learn_predict_visualize_one_step_limit)
            self.visualize_one_step(period.testing_images_folder, data_sources_y_predict_one_step_test, file_candle_indexes_one_step_test, self.test_predict_visualize_one_step_limit)
            self.visualize_multi_step(period.learning_images_predict_folder, data_sources_y_predict_multi_step_learn, file_candle_indexes_multi_step_learn, mean_absolute_percentage_error_y_predict_multi_step_learn_single, self.part_learn_predict_visualize)
            self.visualize_multi_step(period.testing_images_predict_folder, data_sources_y_predict_multi_step_test, file_candle_indexes_multi_step_test, mean_absolute_percentage_error_y_predict_multi_step_test_single, self.part_test_predict_visualize)
        else:
            print("Ни один источник данных не настроен на визуализацию.")

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

        self.mean_absolute_percentage_error_epsilon = 0.3 # устанавливаю значение, прибавляемое к истинным данным при рассчете ошибки, чтобы не было деления на ноль когда истинные данные ноль

        for i in range(len(self.periods)):
            print(f"Обрабатывается период {i + 1}/{len(self.periods)}")
            self.handle_period(self.periods[i])

        with open(f"{self.base_folder}/info.txt", "w", encoding='utf-8') as new_file:
            new_file.write("Periods:\n")
            new_file.write("------------------------------------------------------------\n")
            for i in range(len(self.periods)):
                new_file.write(f"#{i} {self.periods[i].learning_start.date_time.strftime('%Y-%m-%d %Hh %Mm')} - {self.periods[i].learning_end.date_time.strftime('%Y-%m-%d %Hh %Mm')} => {self.periods[i].testing_start.date_time.strftime('%Y-%m-%d %Hh %Mm')} - {self.periods[i].testing_end.date_time.strftime('%Y-%m-%d %Hh %Mm')}\n")

                if self.periods[i].is_mean_absolute_percentage_error_multi_step_learn and self.periods[i].is_mean_absolute_percentage_error_multi_step_test:
                    new_file.write(f"loss multi step Learn Single: {', '.join([float_to_str_format(self.periods[i].mean_absolute_percentage_error_multi_step_learn_single[k], 2)  for k in range(len(self.periods[i].mean_absolute_percentage_error_multi_step_learn_single))])}\n")
                    new_file.write(f"loss multi step Test  Single: {', '.join([float_to_str_format(self.periods[i].mean_absolute_percentage_error_multi_step_test_single[k], 2)  for k in range(len(self.periods[i].mean_absolute_percentage_error_multi_step_test_single))])}\n")
                    new_file.write(f"loss multi step Learn Join:   {float_to_str_format(self.periods[i].mean_absolute_percentage_error_multi_step_learn_join, 2)}\n")
                    new_file.write(f"loss multi step Test  Join:   {float_to_str_format(self.periods[i].mean_absolute_percentage_error_multi_step_test_join, 2)}\n")
                else:
                    new_file.write(f"is there multi step learn and test loss: {self.periods[i].is_mean_absolute_percentage_error_multi_step_learn and self.periods[i].is_mean_absolute_percentage_error_multi_step_test}\n")

                new_file.write(f"loss one step Learn Single: {', '.join([float_to_str_format(self.periods[i].mean_absolute_percentage_error_one_step_learn_single[k], 2) for k in range(len(self.periods[i].mean_absolute_percentage_error_one_step_learn_single))])}\n")
                new_file.write(f"loss one step Test  Single: {', '.join([float_to_str_format(self.periods[i].mean_absolute_percentage_error_one_step_test_single[k], 2) for k in range(len(self.periods[i].mean_absolute_percentage_error_one_step_test_single))])}\n")
                new_file.write(f"loss one step Learn Join:   {float_to_str_format(self.periods[i].mean_absolute_percentage_error_one_step_learn_join, 2)}\n")
                new_file.write(f"loss one step Test  Join:   {float_to_str_format(self.periods[i].mean_absolute_percentage_error_one_step_test_join, 2)}\n")
                new_file.write("------------------------------------------------------------\n")
            new_file.write(f"sequence length: {self.sequence_length}\n")
            new_file.write(f"predict length: {self.predict_length}\n")
            new_file.write(f"is save predict data: {self.is_save_predict_data}\n")
            new_file.write(f"Data sources:\n")
            new_file.write("------------------------------------------------------------\n")
            for i_ds in range(len(self.data_sources_meta)):
                new_file.write(f"#{i_ds} {self.data_sources_meta[i_ds].short_name}\n")
                new_file.write(f"files:\n")
                for i_f in range(len(self.data_sources_file_names[i_ds])):
                    new_file.write(f"    {self.data_sources_file_names[i_ds][i_f]}\n")
                new_file.write(f"normalizers:\n")
                for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                    new_file.write(f"    {self.data_sources_meta[i_ds].normalizers[i_n].__class__.__name__}\n")
                new_file.write("------------------------------------------------------------\n")

    # нормализует последовательность свечек для всех источников данных, и выходные свечки для всех источников данных, если указаны. Возвращает последовательность входных векторов нейронной сети, и выходной вектор нейронной сети, если указан, и настройки нормализации вида settings[i_ds][i_normalize]. output_i_f, output_i_c - индексы
    def normalize_data_sources(self, output_i_f, output_i_c, data_sources_inp_seq, data_sources_add_candles = [], data_sources_output=None):
        data_sources_normalizers_inp_seq = []
        data_sources_normalizers_out = []
        data_sources_normalizers_settings = []

        for i_ds in range(len(self.data_sources)):
            normalizers_inp_seq = []
            normalizers_out = []
            normalizers_settings = []
            for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                x, y, n_setting = self.data_sources_meta[i_ds].normalizers[i_n].normalize(inp_sequence=data_sources_inp_seq[i_ds], output=data_sources_output[i_ds] if not data_sources_output is None else None, output_i_f=output_i_f, output_i_c=output_i_c, add_candles=data_sources_add_candles[i_ds] if len(data_sources_add_candles) > 0 else [])
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

        finally_output = []
        if data_sources_output != None:
            for i_ds in range(len(data_sources_normalizers_out)):
                for i_n in range(len(data_sources_normalizers_out[i_ds])):
                    finally_output.extend(data_sources_normalizers_out[i_ds][i_n])

        return finally_inp_seq, finally_output, data_sources_normalizers_settings

    # денормализует последовательность входных векторов нейронной сети, если указана, и/или выходной вектор нейронной сети, если указан. Также выполняет вставки в денормализованные данные. Возвращает данные в формате: input[i_ds][i_c], output[i_ds]
    def denormalize_insert_input_vectors_sequence_output_vector(self, data_sources_normalizers_settings, input_vectors_sequence=None, output_vector=None, output_datetime_timestamp=None):
        data_sources_normalizers_inp_seq = self.input_vectors_sequence_to_data_sources_normalizers(input_vectors_sequence) if not input_vectors_sequence is None else None
        data_sources_normalizers_out = self.output_vector_to_data_sources_normalizers(output_vector) if not output_vector is None else None

        data_sources_normalizers_input_sequence_denorm = []
        data_sources_normalizers_output_denorm = []

        for i_ds in range(len(self.data_sources)):
            normalizers_input_sequence_denorm = []
            normalizers_output_denorm = []
            for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                x, y = self.data_sources_meta[i_ds].normalizers[i_n].denormalize(data_sources_normalizers_settings[i_ds][i_n], normalized_inp_sequence_by_data_indexes=data_sources_normalizers_inp_seq[i_ds][i_n] if not data_sources_normalizers_inp_seq is None else None, normalized_output_by_data_indexes=data_sources_normalizers_out[i_ds][i_n] if not data_sources_normalizers_out is None else None)
                normalizers_input_sequence_denorm.append(x)
                normalizers_output_denorm.append(y)

            data_sources_normalizers_input_sequence_denorm.append(normalizers_input_sequence_denorm)
            data_sources_normalizers_output_denorm.append(normalizers_output_denorm)

        # определяю все индексы данных, используемые в нормализаторах источниках данных
        data_sources_normalizers_data_indexes = []
        for i_ds in range(len(self.data_sources_meta)):
            normalizers_data_indexes = []
            for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                for data_index in self.data_sources_meta[i_ds].normalizers[i_n].data_indexes:
                    if not data_index in normalizers_data_indexes:
                        normalizers_data_indexes.append(data_index)
            data_sources_normalizers_data_indexes.append(sorted(normalizers_data_indexes))

        data_sources_input_sequence_denorm = [] # перевод из фомата data_sources_normalizers_input_sequence_denorm[i_ds][i_n][i_c] в формат data_sources_input_sequence_denorm[i_ds][i_c]
        for i_ds in range(len(data_sources_normalizers_input_sequence_denorm)):
            input_sequence_denorm = []
            for i_seq in range(max([len(data_sources_normalizers_input_sequence_denorm[i_ds][n]) for n in range(len(data_sources_normalizers_input_sequence_denorm[i_ds]))])):
                input_denorm = []
                # проходим по индексам данных у нормализаторов данного источника данных
                for data_index in data_sources_normalizers_data_indexes[i_ds]:
                    # находим индексы всех нормализаторов, у которых используется данный индекс данных, и у которых указано что нужно денормализовывать входные данные
                    data_index_normalizers_indexes = []
                    for i_n in range(len(data_sources_normalizers_input_sequence_denorm[i_ds])):
                        if data_index in self.data_sources_meta[i_ds].normalizers[i_n].data_indexes and self.data_sources_meta[i_ds].normalizers[i_n].is_input_denormalize:
                            data_index_normalizers_indexes.append(i_n)
                    # если есть нормализаторы, у которых указано что нужно денормализовывать входные данные, формируем для них пары: денормализованное значение и вес входной денормализации
                    data_index_values_weights = []
                    if len(data_index_normalizers_indexes) > 0:
                        for i_n in data_index_normalizers_indexes:
                            data_index_values_weights.append((data_sources_normalizers_input_sequence_denorm[i_ds][i_n][i_seq][self.data_sources_meta[i_ds].normalizers[i_n].data_indexes_offsets[data_index]], self.data_sources_meta[i_ds].normalizers[i_n].input_denormalize_weight))
                    if len(data_index_values_weights) > 0:
                        weighted_values_sum = sum([data_index_values_weights[k][0] * data_index_values_weights[k][1] for k in range(len(data_index_values_weights))])
                        weights_sum = sum([data_index_values_weights[k][1] for k in range(len(data_index_values_weights))])
                        if weights_sum > 0:
                            input_denorm.append(weighted_values_sum / weights_sum)
                        else:
                            input_denorm.append(0)
                input_sequence_denorm.append(input_denorm)
            data_sources_input_sequence_denorm.append(input_sequence_denorm)

        data_sources_output_denorm_insert = [] # перевод из фомата data_sources_normalizers_output_denorm[i_ds][i_n] в формат data_sources_output_denorm_insert[i_ds]
        for i_ds in range(len(data_sources_normalizers_output_denorm)):
            output_denorm = []
            # проходим по индексам данных у нормализаторов данного источника данных
            for data_index in data_sources_normalizers_data_indexes[i_ds]:
                # находим индексы всех нормализаторов, у которых используется данный индекс данных, и у которых указано что нужно денормализовывать выходные данные
                data_index_normalizers_indexes = []
                for i_n in range(len(data_sources_normalizers_output_denorm[i_ds])):
                    if data_index in self.data_sources_meta[i_ds].normalizers[i_n].data_indexes and self.data_sources_meta[i_ds].normalizers[i_n].is_output_denormalize:
                        data_index_normalizers_indexes.append(i_n)
                # если есть нормализаторы, у которых указано что нужно денормализовывать выходные данные, формируем для них пары: денормализованное значение и вес выходной денормализации
                data_index_values_weights = []
                if len(data_index_normalizers_indexes) > 0:
                    for i_n in data_index_normalizers_indexes:
                        data_index_values_weights.append((data_sources_normalizers_output_denorm[i_ds][i_n][self.data_sources_meta[i_ds].normalizers[i_n].data_indexes_offsets[data_index]], self.data_sources_meta[i_ds].normalizers[i_n].output_denormalize_weight))
                if len(data_index_values_weights) > 0:
                    weighted_values_sum = sum([data_index_values_weights[k][0] * data_index_values_weights[k][1] for k in range(len(data_index_values_weights))])
                    weights_sum = sum([data_index_values_weights[k][1] for k in range(len(data_index_values_weights))])
                    if weights_sum > 0:
                        output_denorm.append(weighted_values_sum / weights_sum)
                    else:
                        output_denorm.append(0)
            data_sources_output_denorm_insert.append(output_denorm)

        if len(data_sources_output_denorm_insert) > 0:
            self.make_output_inserts(data_sources_output_denorm_insert, output_datetime_timestamp)

        return data_sources_input_sequence_denorm, data_sources_output_denorm_insert

    def make_output_inserts(self, data_sources_output, output_datetime_timestamp):
        for i_ds in range(len(self.data_sources_meta)):
            data_sources_output[i_ds].insert(0, output_datetime_timestamp) # вставляем дату в 0-ю позицию
            for i_oi in range(len(self.data_sources_meta[i_ds].output_inserts)):
                self.data_sources_meta[i_ds].output_inserts[i_oi].make_insert(target_list=data_sources_output[i_ds], datetime_timestamp=output_datetime_timestamp, interval_milliseconds=self.interval_milliseconds)

    def input_vectors_sequence_to_data_sources_normalizers(self, input):
        index = 0
        data_sources_normalizers = []
        for i_ds in range(len(self.data_sources_meta)):
            data_source_normalizers = []
            for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                normalizer_input_sequence = []
                for i_input in range(len(input)):
                    input_normalizer = input[i_input][index:index + self.data_sources_meta[i_ds].normalizers[i_n].normalize_input_length]
                    normalizer_input_sequence.append(input_normalizer)
                index += self.data_sources_meta[i_ds].normalizers[i_n].normalize_input_length
                data_source_normalizers.append(normalizer_input_sequence)
            data_sources_normalizers.append(data_source_normalizers)
        return data_sources_normalizers

    def output_vector_to_data_sources_normalizers(self, output):
        index = 0
        data_sources_normalizers = []
        for i_ds in range(len(self.data_sources_meta)):
            data_source_normalizers = []
            for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                output_normalizer = output[index:index + self.data_sources_meta[i_ds].normalizers[ i_n].normalize_output_length]
                index += self.data_sources_meta[i_ds].normalizers[i_n].normalize_output_length
                data_source_normalizers.append(output_normalizer)
            data_sources_normalizers.append(data_source_normalizers)
        return data_sources_normalizers

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

    # data_sources_true[0] и data_sources_predict[0] должны иметь одинаковую длину
    def visualize_predict_to_file(self, data_sources_true, data_sources_predict, visualize_data_sources_indexes, data_sources_add_labels, save_image_path):
        add_plot = []
        where_values_end_inp_seq = [False] * len(data_sources_predict[0])
        y_over_rate = 0.02 # отступ от верхнего и нижнего края
        for i in range(len(data_sources_predict[0])):
            if len(data_sources_predict[0][i]) > 1:
                if i - 1 > 0:
                    where_values_end_inp_seq[i - 1] = True
                break
        for i_ds in range(len(data_sources_true) - 1, -1, -1):
            i_visual_ds = visualize_data_sources_indexes[i_ds]
            for i_panel in range(len(self.data_sources_meta[i_visual_ds].visualize) - 1, -1, -1):
                type_chart, data_indexes = self.data_sources_meta[i_visual_ds].visualize[i_panel]
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
                    y_over = (ymax - ymin) * y_over_rate
                    ymin -= y_over
                    ymax += y_over

                    y_label = self.data_sources_meta[i_visual_ds].visualize_name[i_panel]
                    if i_panel == 0:
                        y_label = f"{y_label} <{self.data_sources_meta[i_visual_ds].short_name}>{data_sources_add_labels[i_ds]}"
                    panel_num = sum([len(self.data_sources_meta[visualize_data_sources_indexes[i_i_ds]].visualize) for i_i_ds in range(i_ds, -1, -1)]) - (len(self.data_sources_meta[visualize_data_sources_indexes[i_ds]].visualize) - i_panel)

                    dict_end_inp_seq = dict(y1=ymin, y2=ymax, where=where_values_end_inp_seq, alpha=0.55, color='red')

                    if i_ds == 0 and i_panel == 0:
                        add_plot.append(mpf.make_addplot(high_canal, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="black"))
                        add_plot.append(mpf.make_addplot(low_canal, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="black"))
                        add_plot.append(mpf.make_addplot(p_data_predict, type='candle', ylim=(ymin,ymax)))
                    else:
                        add_plot.append(mpf.make_addplot(high_canal, type='line', panel=panel_num, fill_between=[dict_end_inp_seq], ylabel=y_label, ylim=(ymin, ymax), linewidths=1, alpha=1, color="black"))
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
                    y_over = (ymax - ymin) * y_over_rate
                    ymin -= y_over
                    ymax += y_over

                    y_label = self.data_sources_meta[i_visual_ds].visualize_name[i_panel]
                    if i_panel == 0:
                        y_label = f"{y_label} <{self.data_sources_meta[i_visual_ds].short_name}>{data_sources_add_labels[i_ds]}"
                    panel_num = sum([len(self.data_sources_meta[visualize_data_sources_indexes[i_i_ds]].visualize) for i_i_ds in range(i_ds, -1, -1)]) - (len(self.data_sources_meta[visualize_data_sources_indexes[i_ds]].visualize) - i_panel)

                    dict_end_inp_seq = dict(y1=ymin, y2=ymax, where=where_values_end_inp_seq, alpha=0.55, color='red')

                    if i_ds == 0 and i_panel == 0:
                        add_plot.append(mpf.make_addplot(p_data_true, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="black"))
                        add_plot.append(mpf.make_addplot(p_data_predict, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="springgreen"))
                    else:
                        add_plot.append(mpf.make_addplot(p_data_true, type='line', panel=panel_num, fill_between=[dict_end_inp_seq], ylabel=y_label, ylim=(ymin, ymax), linewidths=1, alpha=1, color="black"))
                        add_plot.append(mpf.make_addplot(p_data_predict, type='line', panel=panel_num, ylim=(ymin, ymax), linewidths=1, alpha=1, color="springgreen"))

        myrcparams = {'axes.labelsize': 'small'}
        my_style = mpf.make_mpf_style(base_mpf_style='yahoo', facecolor='white', y_on_right=False, rc=myrcparams)
        panel_ratios = ()
        for i_ds in range(len(data_sources_true)):
            i_visual_ds = visualize_data_sources_indexes[i_ds]
            for i_panel in range(len(self.data_sources_meta[i_visual_ds].visualize_ratio)):
                panel_ratios += (self.data_sources_meta[i_visual_ds].visualize_ratio[i_panel],)
        mpf.plot(p_data_true, type='candle' if type_chart == "candle" else 'line', style=my_style, ylabel=y_label, ylim=(ymin,ymax), addplot=add_plot, fill_between=[dict_end_inp_seq], panel_ratios=panel_ratios, figsize=(18,9), datetime_format="%Y-%b-%d", tight_layout=True, savefig=save_image_path)
        #, fill_between = [dict1, dict2, dict3]  image_path = f"{save_folder_path}/{str(i).rjust(2, '0')}_{str(candle_index).rjust(7, '0')}", columns=None if type_chart == "candle" else ['Close']
        print(f"save image {save_image_path}")
