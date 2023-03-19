import time
import random
import os
import numpy as np
import datetime
import mplfinance as mpf
import pandas as pd
import copy


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
            self.date_time = datetime.datetime(date_time.year, date_time.month, date_time.day)
        else:
            self.date_time = datetime.datetime(year, month, day)

    def add_years(self, years):
        self.date_time = datetime.datetime(self.date_time.year + years, self.date_time.month, self.date_time.day)

    def add_months(self, months):
        new_month = self.date_time.month + months - 1
        years = new_month // 12
        months_remaind = new_month % 12 + 1
        if years > 0:
            self.add_years(years)
        self.date_time = datetime.datetime(self.date_time.year, months_remaind, min(self.date_time.day, self.month_length(self.date_time.year, months_remaind)))

    def add_days(self, days):
        days_remaind = days
        while days_remaind > 0:
            days_to_next_month = self.month_length(self.date_time.year, self.date_time.month) - self.date_time.day + 1
            if days_remaind >= days_to_next_month:
                days_remaind -= days_to_next_month
                self.date_time = datetime.datetime(self.date_time.year, self.date_time.month, 1)
                self.add_months(1)
            else:
                self.date_time = datetime.datetime(self.date_time.year, self.date_time.month, self.date_time.day + days_remaind)
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
    def __init__(self, date_time_start, learning_duration, testing_duration):
        self.learning_start = DateTime(date_time=date_time_start.date_time)
        self.learning_end = DateTime(date_time=date_time_start.date_time)
        self.learning_end.add_years(learning_duration.years)
        self.learning_end.add_months(learning_duration.months)
        self.learning_end.add_days(learning_duration.days)

        self.testing_start = DateTime(date_time=self.learning_end.date_time)
        self.testing_end = DateTime(date_time=self.learning_end.date_time)
        self.testing_end.add_years(testing_duration.years)
        self.testing_end.add_months(testing_duration.months)
        self.testing_end.add_days(testing_duration.days)

class DataSourceMeta:
    def __init__(self, files, date_index, data_indexes, normalizers, visualize, is_visualize, visualize_ratio, visualize_name, visualize_data_source_panel = 1):
        self.files = files # список с файлами
        self.date_index = date_index # индекс даты
        self.data_indexes = data_indexes # индексы данных, которые нужно считать из файла
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

    def create_save_folders(self):
        app_files_folder_name = "app_data"
        if not os.path.isdir(app_files_folder_name):
            os.makedirs(app_files_folder_name)
        folder_id = 0
        while os.path.isdir(f"{app_files_folder_name}/{str(folder_id).rjust(4, '0')}"):
            folder_id += 1
        self.folder_base = f"{app_files_folder_name}/{str(folder_id).rjust(4, '0')}"
        self.folder_images_learn_predict = f"{self.folder_base}/images/learn_predict"
        self.folder_images_test_predict = f"{self.folder_base}/images/test_predict"
        self.folder_images_learn_result = f"{self.folder_base}/images/learn_result"
        os.makedirs(self.folder_base)
        os.makedirs(f"{self.folder_base}/images")
        os.makedirs(self.folder_images_learn_predict)
        os.makedirs(self.folder_images_test_predict)
        os.makedirs(self.folder_images_learn_result)

    def __init__(self, data_sources_meta, first_file_offset, sequence_length, data_split_sequence_length, validation_split, test_split):
        # создаем папки для сохранения информации
        self.create_save_folders()

        self.data_sources_meta = data_sources_meta
        self.first_file_offset = first_file_offset # отступ в еденицах данных (свечках) от начала первого файла. (на случай если это экспирируемые фьючерсные контракты, и файлы с данными имеют запас перед главными торговыми датами данного контракта, чтобы обучение и тестирование выполнялось в главные торговые даты данного контракта.) (впоследствии, если это экспирируемые фьючерсные контракты, данные для следующих файлов будут выбираться как следующая дата за последней датой предыдущего файла.)
        self.sequence_length = sequence_length
        self.data_split_sequence_length = data_split_sequence_length
        self.validation_split = validation_split
        self.test_split = test_split

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
            # data_source_files = []
            # for i_f in range(len(data_sources_meta[i_ds].files)):
            #     data_source_files.append(self.read_csv_file(data_sources_meta[i_ds].files[i_f], data_sources_meta[i_ds].date_index, data_sources_meta[i_ds].data_indexes))
            # self.data_sources.append(data_source_files)

        # проверяем, совпадает ли: количество файлов у разных источников данных, количество данных в файлах разных источников данных, даты в данных разных источников данных
        is_files_fit = True
        error_messages = []
        if len(self.data_sources) > 1:
            # количество файлов у разных источников данных
            for i_ds in range(1, len(self.data_sources)):
                if len(self.data_sources[i_ds]) != len(self.data_sources[0]):
                    is_files_fit = False
                    error_messages.append(f"Не совпадает количество файлов у источников данных: {self.data_sources_file_names[i_ds]} и {self.data_sources_file_names[0]}.")
            # количество данных в файлах разных источников данных
            if is_files_fit:
                for i_f in range(len(self.data_sources[0])):
                    for i_ds in range(1, len(self.data_sources)):
                        if len(self.data_sources[i_ds][i_f]) != len(self.data_sources[0][i_f]):
                            is_files_fit = False
                            error_messages.append(f"Не совпадает количество свечек в файлах разных источников данных: {self.data_sources_file_names[i_ds][i_f]} и {self.data_sources_file_names[0][i_f]}.")
            # даты в свечках разных источников данных
            if is_files_fit:
                for i_f in range(len(self.data_sources[0])):
                    for i_c in range(len(self.data_sources[0][i_f])):
                        for i_ds in range(1, len(self.data_sources)):
                            if self.data_sources[i_ds][i_f][i_c][0] != self.data_sources[0][i_f][i_c][0]:
                                is_files_fit = False
                                error_messages.append(f"Не совпадают даты в данных разных источников данных: (файл={self.data_sources_file_names[i_ds][i_f]}, индекс свечки={i_c}, дата={self.data_sources[i_ds][i_f][i_c][0]}) и (файл={self.data_sources_file_names[0][i_f]}, индекс свечки={i_c}, дата={self.data_sources[0][i_f][i_c][0]}).")

        # выводим ошибки
        for message in error_messages[:25]:
            print(message)

        # если не было ошибок, опредлеяем типы данных для всех свечек, и формируем: обучающие, валидационные и тестовые данные
        if is_files_fit:
            self.data_interval = self.data_sources[0][0][1][0] - self.data_sources[0][0][0][0]
            if len(self.data_sources[0][0]) > 9:
                for i in range(9):
                    if self.data_sources[0][0][i + 1][0] - self.data_sources[0][0][i][0] != self.data_interval:
                        raise ValueError("При определении временного интервала данных, в первых 10 данных обнаружены разные временные интервалы.")
            else:
                raise ValueError("Количество данных в файле должно быть не менее 10.")
            # опредлеяем типы данных для всех свечек
            self.data_sources_data_type = []  # тип данных для всех данных для всех файлов: [0(обучающие), 1(валидационные), 2(тестовые), -1(не участвует в выборках)]. Нет разделения на источники данных, т.к. тип данных относится ко всем источникам данных
            learn_count = 0
            valid_count = 0
            test_count = 0
            sequence_number = 0
            data_types = [0, 1, 2]
            data_type = data_types[random.randint(0, 2)]
            last_file_date = None
            for i_f in range(len(self.data_sources[0])):
                file_data_types = []
                for i_c in range(len(self.data_sources[0][i_f])):
                    is_next_date = False
                    if last_file_date != None:
                        if self.data_sources[0][i_f][i_c][0] > last_file_date:
                            is_next_date = True
                            last_file_date = self.data_sources[0][i_f][i_c][0]
                    else:
                        last_file_date = self.data_sources[0][i_f][i_c][0]
                        is_next_date = True

                    if sequence_number >= data_split_sequence_length:
                        sequence_number = 0
                        # выбираем случайный тип среди тех, которые составляют от всех данных меньшую часть чем указано для них, если ни один из типов не является меньше указанного, выбираем случайный тип
                        data_types_less_than_split = []  # типы данных, количество которых меньше чем их указанное в настройках количество
                        if learn_count / (learn_count + valid_count + test_count) < 1 - (validation_split + test_split):
                            data_types_less_than_split.append(data_types[0])
                        if valid_count / (learn_count + valid_count + test_count) < validation_split:
                            data_types_less_than_split.append(data_types[1])
                        if test_count / (learn_count + valid_count + test_count) < test_split:
                            data_types_less_than_split.append(data_types[2])

                        if len(data_types_less_than_split) > 0:
                            data_type = random.choice(data_types_less_than_split)
                        else:
                            data_type = data_types[random.randint(0, 2)]

                    if i_c >= sequence_length and (i_c >= first_file_offset if i_f == 0 else True) and is_next_date:
                        file_data_types.append(data_type)
                        if data_type == data_types[0]:
                            learn_count += 1
                        elif data_type == data_types[1]:
                            valid_count += 1
                        else:
                            test_count += 1
                        sequence_number += 1
                    else:
                        file_data_types.append(-1)  # если перед свечкой нет последовательности длиной sequence_length или это первый файл и мы не отошли от начала на first_file_offset или текущая свечка не является следующей за последней датой, отмечаем что она не относится ни к одной выборке
                self.data_sources_data_type.append(file_data_types)

            # выполняем подготовку нормализаторов
            for i_ds in range(len(data_sources_meta)):
                for i_n in range(len(data_sources_meta[i_ds].normalizers)):
                    data_sources_meta[i_ds].normalizers[i_n].summary(self.data_sources[i_ds], self.data_sources_data_type, self.sequence_length)

            # формируем обучающие, валидационные и тестовые данные
            self.x_learn = []
            self.y_learn = []
            self.x_valid = []
            self.y_valid = []
            self.x_test = []
            self.y_test = []
            for i_f in range(len(self.data_sources_data_type)):
                for i_c in range(len(self.data_sources_data_type[i_f])):
                    if self.data_sources_data_type[i_f][i_c] != -1:

                        data_sources_inp_seq = []
                        for i_ds in range(len(self.data_sources)):
                            data_source_inp_seq = []
                            for i_seq in range(i_c - self.sequence_length, i_c):  # проходим по всем свечкам входной последовательности
                                data_source_inp_seq.append(copy.deepcopy(self.data_sources[i_ds][i_f][i_seq]))
                            data_sources_inp_seq.append(data_source_inp_seq)

                        data_sources_output = []
                        for i_ds in range(len(self.data_sources)):
                            data_sources_output.append(copy.deepcopy(self.data_sources[i_ds][i_f][i_c]))

                        x, y, data_sources_normalizers_settings = self.normalize_data_sources(data_sources_inp_seq, data_sources_output)
                        if self.data_sources_data_type[i_f][i_c] == 0:
                            self.x_learn.append(x)
                            self.y_learn.append(y)
                        elif self.data_sources_data_type[i_f][i_c] == 1:
                            self.x_valid.append(x)
                            self.y_valid.append(y)
                        elif self.data_sources_data_type[i_f][i_c] == 2:
                            self.x_test.append(x)
                            self.y_test.append(y)
        else:
            raise ValueError(error_messages[0])


    """нормализует входные последовательности для всех источников данных, а так же выходное значение если оно указано
    data_sources_inp_seq - последовательности входных данных для всех источников данных
    возвращает последовательность входных векторов, выходной вектор, настройки для денормализации. настройки для денормализации вида: settings[i_ds][i_normalize]."""
    def normalize_data_sources(self, data_sources_inp_seq, data_sources_output=None):
        data_sources_normalizers_inp_seq = []
        data_sources_normalizers_out = []
        data_sources_normalizers_settings = []

        for i_ds in range(len(self.data_sources)):
            normalizers_inp_seq = []
            normalizers_out = []
            normalizers_settings = []
            for i_n in range(len(self.data_sources_meta[i_ds].normalizers)):
                x, y, n_setting = self.data_sources_meta[i_ds].normalizers[i_n].normalize(data_sources_inp_seq[i_ds], data_sources_output[i_ds] if data_sources_output != None else None)
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

        return finally_inp_seq, finally_out_seq, data_sources_normalizers_settings

    """денормализует входные последовательности для всех нормализаторов, всех источников данных, а так же выходное значение если оно указано
        input_vectors_sequence - последовательности входных данных для всех источников данных, для всех нормализаторов
        возвращает входную последовательность для всех источников данных и выходное значение для всех источников данных"""
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
                for i in range(index, index + self.data_sources_meta[i_ds].normalizers[i_n].out_norm_data_length):
                    data_source_normalizer.append(copy.deepcopy(output[i]))
                index += self.data_sources_meta[i_ds].normalizers[i_n].out_norm_data_length
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
                for i in range(index, index + self.data_sources_meta[i_ds].normalizers[i_n].out_norm_data_length):
                    data_source_normalizer.append(copy.deepcopy(input[i]))
                index += self.data_sources_meta[i_ds].normalizers[i_n].inp_norm_data_length
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
                reformatted_data['Date'].append(datetime.datetime.fromtimestamp(int(item[0]) / 1000))
                reformatted_data['Open'].append(None)
                reformatted_data['High'].append(None)
                reformatted_data['Low'].append(None)
                reformatted_data['Close'].append(None)
            else:
                reformatted_data['Date'].append(datetime.datetime.fromtimestamp(int(item[0]) / 1000))
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
                reformatted_data['Date'].append(datetime.datetime.fromtimestamp(int(item[0]) / 1000))
                reformatted_data['Open'].append(None)
                reformatted_data['High'].append(None)
                reformatted_data['Low'].append(None)
                reformatted_data['Close'].append(None)
            else:
                reformatted_data['Date'].append(datetime.datetime.fromtimestamp(int(item[0]) / 1000))
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

