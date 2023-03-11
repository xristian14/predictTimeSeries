import random
import os
import numpy as np
import datetime
import mplfinance as mpf
import pandas as pd
import copy

class DataSourceMeta:
    def __init__(self, files, date_index, data_indexes, normalizers, visualize, visualize_ratio, visualize_name, visualize_data_source_panel = 1):
        self.files = files # список с файлами
        self.date_index = date_index # индекс даты
        self.data_indexes = data_indexes # индексы данных, которые нужно считать из файла
        self.normalizers = normalizers # список с нормализаторами для источника данных
        self.visualize = visualize # список с панелями которые будут созданы для отображения источника данных. Элементы списка: ("type", [data_indexes]), type может быть: "candle", "line". Для "candle" может быть передано или 4 или 2 индекса данных, для "line" может быть передан только один индекс данных. Пример графиков цены и объема: [("candle", [1,2,3,4]), ("line", [5])]
        self.visualize_ratio = visualize_ratio # список со значениями размера панелей visualize
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
        reformatted_data['Close'] = []
        for item in data:
            if len(item) == 1:
                reformatted_data['Date'].append(datetime.datetime.fromtimestamp(int(item[0]) / 1000))
                reformatted_data['Close'].append(None)
            else:
                reformatted_data['Date'].append(datetime.datetime.fromtimestamp(int(item[0]) / 1000))
                reformatted_data['Close'].append(item[1])
        return reformatted_data

    # data_sources_true[0] и data_sources_predict[0] должны иметь одинаковое количество дат
    def visualize_predict_to_file(self, data_sources_true, data_sources_predict, save_file_path):
        add_plot = []
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

                    y_label = self.data_sources_meta[i_ds].visualize_name[i_p]
                    panel_num = i_ds * len(self.data_sources_meta[i_ds].visualize) + i_p

                    if i_ds == 0 and i_p == 0:
                        add_plot.append(mpf.make_addplot(high_canal, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="black"))
                        add_plot.append(mpf.make_addplot(low_canal, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="black"))
                        add_plot.append(mpf.make_addplot(p_data_predict, type='candle', ylim=(ymin,ymax)))
                    else:
                        add_plot.append(mpf.make_addplot(high_canal, type='line', panel=panel_num, ylabel=y_label, ylim=(ymin, ymax), linewidths=1, alpha=1, color="black"))
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

                    y_label = self.data_sources_meta[i_ds].visualize_name[i_p]
                    panel_num = i_ds * len(self.data_sources_meta[i_ds].visualize) + i_p

                    if i_ds == 0 and i_p == 0:
                        add_plot.append(mpf.make_addplot(p_data_true, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="black"))
                        add_plot.append(mpf.make_addplot(p_data_predict, type='line', ylim=(ymin,ymax), linewidths = 1, alpha = 1, color="springgreen"))
                    else:
                        add_plot.append(mpf.make_addplot(p_data_true, type='line', panel=panel_num, ylabel=y_label, ylim=(ymin, ymax), linewidths=1, alpha=1, color="black"))
                        add_plot.append(mpf.make_addplot(p_data_predict, type='line', panel=panel_num, ylim=(ymin, ymax), linewidths=1, alpha=1, color="springgreen"))

        myrcparams = {'axes.labelsize': 'small'}
        my_style = mpf.make_mpf_style(base_mpf_style='yahoo', facecolor='white', y_on_right=False, rc=myrcparams)
        panel_ratios = ()
        for i_ds in range(len(data_sources_true)):
            for i_p in range(len(self.data_sources_meta[i_ds].visualize_ratio)):
                panel_ratios += (self.data_sources_meta[i_ds].visualize_ratio[i_p],)
        mpf.plot(p_data_true, type='candle', style=my_style, ylabel=y_label, ylim=(ymin,ymax), addplot=add_plot, panel_ratios=panel_ratios, figsize=(18,9), datetime_format="%Y-%b-%d", tight_layout=True, savefig=save_file_path)
        #, fill_between = [dict1, dict2, dict3]  image_path = f"{save_folder_path}/{str(i).rjust(2, '0')}_{str(candle_index).rjust(7, '0')}"
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

                        for i_p in range(self.predict_length):
                            # формируем входную последовательность для всех источников данных
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

                            x, y, data_sources_normalizers_settings = self.normalize_data_sources(data_sources_inp_seq)
                            x_np = np.array(x)
                            inp = x_np.reshape(1, self.sequence_length, len(x[0]))
                            pred = model.predict(inp, verbose=0)
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

                                if self.is_visualize_prediction_union:
                                    save_file_path = f"{self.folder_images_learn_predict if self.data_sources_data_type[i_f][i_c_dt] == 2 else self.folder_images_test_predict}/union_{str(i_f).rjust(3, '0')}_{str(i_c_dt).rjust(7, '0')}"
                                    self.visualize_predict_to_file(data_sources_true, data_sources_predict, save_file_path)

                                if self.is_visualize_prediction_single:
                                    for i_ds in range(len(self.data_sources)):
                                        save_file_path = f"{self.folder_images_learn_predict if self.data_sources_data_type[i_f][i_c_dt] == 2 else self.folder_images_test_predict}/single_{self.data_sources_file_names[i_ds][0].split('.')[0]}_{str(i_f).rjust(3, '0')}_{str(i_c_dt).rjust(7, '0')}"
                                        self.visualize_predict_to_file(data_sources_true[i_ds:i_ds + 1], data_sources_predict[i_ds:i_ds + 1], save_file_path)


# --------------------------------------------------------------------------------


def min_max_scaler(min, max, val):
    return (val - min) / (max - min)

def un_min_max_scaler(min, max, val):
    return val * (max - min) + min

#считывает все свечки файла в список формата: [[date, open, high, low, close, volume], [date, open, high, low, close, volume],..]
def read_csv_file(file_path):
    candles = []
    with open(file_path, 'r', encoding='utf-8') as file:
        first = True
        for line in file:
            if first:  # пропускаем шапку файла
                first = False
            else:
                list_line = line.split(",")
                candles.append([int(list_line[0]), float(list_line[1]), float(list_line[2]), float(list_line[3]), float(list_line[4]), float(list_line[5])])
    return candles

# normalize_relativity_min_max_scaler - нормализует свечки по принципу min_max_scaler в следующем диапазоне: (минимум в последовательности:min, максимум в п.:max) -> range=max-min -> output_offset = ((максимальный множитель, на который, следующая за последовательностью свечек длины sequence_length, свечка, увеличивала диапазон последовательности) - 1) * range * 1.2 -> max += output_offset, min -= output_offset. Также добавляет во входную последовательность для каждой свечки файлов 4 значения: часть размера текущего диапазона от максимального диапазона, найденного среди всех последовательностей по данной формуле для цены и для объема, а так же часть максимума данного диапазона
normalize_relativity_min_max_scaler_data_sources_max_price_range = []
normalize_relativity_min_max_scaler_data_sources_volume_price_range = []
normalize_relativity_min_max_scaler_data_sources_output_offset = []
def normalize_relativity_min_max_scaler_init():
    global sequence_length
    # опредлеяем output_offset и масмиальные размеры диапазонов цены и объема для всех источников данных
    for i_ds in range(len(data_sources)):
        for i_f in range(len(data_sources[i_ds])):
            for i_c in range(len(data_sources[i_ds][i_f])):
                if data_sources_data_type[i_f][i_c] != -1:
                    max_price_input_sequence = max([max(data_sources[i_ds][i_f][i_c_s][1:5]) for i_c_s in range(i_c - sequence_length, i_c)])
                    min_price_input_sequence = min([min(data_sources[i_ds][i_f][i_c_s][1:5]) for i_c_s in range(i_c - sequence_length, i_c)])
                    max_volume_input_sequence = max([data_sources[i_ds][i_f][i_c_s][5] for i_c_s in range(i_c - sequence_length, i_c)])
                    min_volume_input_sequence = min([data_sources[i_ds][i_f][i_c_s][5] for i_c_s in range(i_c - sequence_length, i_c)])
    input()



# normalize_min_max_scaler - нормализует свечки по принципу min_max_scaler, offset - отступ от минимума и максимума как часть от диапазона
normalize_min_max_scaler_Min_price = [] #списки со значениями минимума и максимума в каждом файле
normalize_min_max_scaler_Max_price = []
normalize_min_max_scaler_Min_volume = []
normalize_min_max_scaler_Max_volume = []

def normalize_min_max_scaler_init(candle_files, offset = 0.0):
    global normalize_min_max_scaler_Min_price
    global normalize_min_max_scaler_Max_price
    global normalize_min_max_scaler_Min_volume
    global normalize_min_max_scaler_Max_volume
    normalize_min_max_scaler_Min_price = [file_candles[3] for file_candles in candle_files]
    normalize_min_max_scaler_Max_price = [file_candles[2] for file_candles in candle_files]
    normalize_min_max_scaler_Min_volume = [file_candles[5] for file_candles in candle_files]
    normalize_min_max_scaler_Max_volume = [file_candles[5] for file_candles in candle_files]
    for i in range(len(candle_files)):
        normalize_min_max_scaler_Min_price[i] = min([candle_files[i][k][3] for k in range(len(candle_files[i]))])
        normalize_min_max_scaler_Max_price[i] = max([candle_files[i][k][2] for k in range(len(candle_files[i]))])
        normalize_min_max_scaler_Min_volume[i] = min([candle_files[i][k][5] for k in range(len(candle_files[i]))])
        normalize_min_max_scaler_Max_volume[i] = max([candle_files[i][k][5] for k in range(len(candle_files[i]))])

    for i in range(len(candle_files)):
        price_offset = (normalize_min_max_scaler_Max_price[i] - normalize_min_max_scaler_Min_price[i]) * offset
        volume_offset = (normalize_min_max_scaler_Max_volume[i] - normalize_min_max_scaler_Min_volume[i]) * offset
        normalize_min_max_scaler_Min_price[i] -= price_offset
        normalize_min_max_scaler_Max_price[i] += price_offset
        normalize_min_max_scaler_Min_volume[i] -= volume_offset
        normalize_min_max_scaler_Max_volume[i] += volume_offset

def normalize_min_max_scaler_do(candles_sequence_files, is_replace=False): #при каждом вызове сохраняет в специальный список параметры с которыми была проведена нормализация последовательности свечек (впоследствии эти параметры используются для правильной денормализации), is_replace - определяет, нужно ли заменить последний элемент в этом списке на тот который будет создан при этом вызове. Нужно указывать is_replace=True когда последовательность денормализуется после нормазизации, и между этим не было нормализации другой последовательности, таким образом список не будет заполняться ненужными значениями.
    normalized_candles_sequence_files = []
    for i in range(len(candles_sequence_files)):
        normalized_file_candles_sequence = []
        for k in range(len(candles_sequence_files[i])):
            n_candle = []
            n_candle.append(candles_sequence_files[i][k][0])
            n_candle.append(min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], candles_sequence_files[i][k][1]))
            n_candle.append(min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], candles_sequence_files[i][k][2]))
            n_candle.append(min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], candles_sequence_files[i][k][3]))
            n_candle.append(min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], candles_sequence_files[i][k][4]))
            n_candle.append( min_max_scaler(normalize_min_max_scaler_Min_volume[i], normalize_min_max_scaler_Max_volume[i], candles_sequence_files[i][k][5]))
            normalized_file_candles_sequence.append(n_candle)
        normalized_candles_sequence_files.append(normalized_file_candles_sequence)
    return normalized_candles_sequence_files

def normalize_min_max_scaler_undo(normalized_candles_sequence_files, index=-1): #index - индекс нормализованной последовательности, необходимо указывать для правильной денормализации. index=-1 значит нужно взять параметры нормализации последней нормализованной последовательности
    global normalize_min_max_scaler_Min_price
    global normalize_min_max_scaler_Max_price
    global normalize_min_max_scaler_Min_volume
    global normalize_min_max_scaler_Max_volume

    unnormalized_candles_sequence_files = []
    for i in range(len(normalized_candles_sequence_files)):
        unnormalized_file_candles_sequence = []
        for k in range(len(normalized_candles_sequence_files[i])):
            n_candle = []
            n_candle.append(normalized_candles_sequence_files[i][k][0])
            n_candle.append(un_min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], normalized_candles_sequence_files[i][k][1]))
            n_candle.append(un_min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], normalized_candles_sequence_files[i][k][2]))
            n_candle.append(un_min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], normalized_candles_sequence_files[i][k][3]))
            n_candle.append(un_min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], normalized_candles_sequence_files[i][k][4]))
            n_candle.append(un_min_max_scaler(normalize_min_max_scaler_Min_volume[i], normalize_min_max_scaler_Max_volume[i], normalized_candles_sequence_files[i][k][5]))
            unnormalized_file_candles_sequence.append(n_candle)
        unnormalized_candles_sequence_files.append(unnormalized_file_candles_sequence)
    return unnormalized_candles_sequence_files

normalize_min_max_scaler = [normalize_min_max_scaler_init, normalize_min_max_scaler_do, normalize_min_max_scaler_undo]


save_folder_path = ""
data_split_sequence_length = 0
sequence_length = 0
predict_length = 0
validation_split = 0
test_split = 0
part_learn_predict = 0
part_test_predict = 0
part_learn_predict_visualize = 0
part_test_predict_visualize = 0
is_visualize_prediction = False
is_save_prediction_data = False
data_sources_paths = []
normalize_method = []

data_sources_file_names = [] # то же самое что и data_sources_paths, но без полного пути, только имя файла
data_sources = [] #свечки для всех файлов всех источников данных
data_sources_data_type = [] #тип данных для всех свечек для всех файлов: [0(обучающие), 1(валидационные), 2(тестовые), -1(не имеет входной последовательности)]. Нет разделения на источники данных, т.к. тип свечки относится ко всем источникам данных

#X_learn_candle_indexes_files = [] #индексы обучающих свечек, данная свечка соответствует выходу нейронной сети, а входная последовательность это предыдущие свечки длиной sequence_length. Индексы свечек для всех файлов одного источника данных, т.к. индексы относятся ко всем источникам данных. X_learn_candle_indexes_files[0][0] - индекс файла 0-й обучающей свечки, X_learn_candle_indexes_files[0][1] - индекс свечки в файле 0-й обучающей свечки
#X_test_candle_indexes_files = [] #индексы тестовых свечек, данная свечка соответствует выходу нейронной сети, а входная последовательность это предыдущие свечки длиной sequence_length. Индексы свечек для всех файлов одного источника данных, т.к. индексы относятся ко всем источникам данных. X_test_candle_indexes_files[0][0] - индекс файла 0-й тестовой свечки, X_test_candle_indexes_files[0][1] - индекс свечки в файле 0-й тестовой свечки

#нужно прогнозировать данные даже для последних свечк, для которых нет примера действительных данных, т.к. при переходе на следующий файл нужно чтобы самая последняя свечка предыдущего файла была торговой


candle_files = []
candle_files_type_data = [] #типы данных для всех свечек(без деления на файлы, т.к. индекс свечки будет относиться ко всем файлам): 0-обучающие, 1-валидационные, 2-тестовые, -1-ни к какому типу не относится. Тип свечки означает что эта свечка является прогнозируемой, то есть если эта свечка обучающая, значит последовательность предыдущих свечек длиной sequence_length будет входом, а эта свечка выходом
normalized_candle_files = [] #нормализованные свечки
candle_files_inputs = [] #последовательности входных массивов для нейросети, для всех свечек, начиная с sequence_length. Тип данных свечки с индексом [0]: candle_files_inputs[0] находится в candle_files_type_data по индексу [0 + sequence_length]: candle_files_type_data[0 + sequence_length]
X_learn_indexes = [] #индексы свечек в normalized_candle_files для обучающих данных, индекс соответствует индексу начальной свечки в последовательных данных образца
X_test_indexes = [] #индексы свечек в normalized_candle_files для тестовых данных, индекс соответствует индексу начальной свечки в последовательных данных образца
X_learn = []
Y_learn = []
X_valid = []
Y_valid = []
X_test = []
Y_test = []



def data_sources_to_input_output_data():
    global data_sources_paths
    global data_sources_file_names
    global data_sources
    global data_sources_data_type
    # устанавливаем имена файлов без полных путей
    data_sources_file_names = []
    for i_ds in range(len(data_sources_paths)):
        data_source_file_names = []
        for i_f in range(len(data_sources_paths[i_ds])):
            file_name = data_sources_paths[i_ds][i_f].replace("\\", "/").split("/")[-1]
            data_source_file_names.append(file_name)
        data_sources_file_names.append(data_source_file_names)
    # считываем источники данных
    data_sources = []
    for i_ds in range(len(data_sources_paths)):
        data_sources.append([read_csv_file(file_path) for file_path in data_sources_paths[i_ds]])
    # проверяем, совпадает ли: количество файлов у разных источников данных, количество свечек в файлах разных источников данных, даты в свечках разных источников данных
    is_files_fit = True
    error_messages = []
    if len(data_sources) > 1:
        # количество файлов у разных источников данных
        for i_ds in range(1, len(data_sources)):
            if len(data_sources[i_ds]) != len(data_sources[0]):
                is_files_fit = False
                error_messages.append(f"Не совпадает количество файлов у источниковв данных: {i_ds} и {0}.")
        # количество свечек в файлах разных источников данных
        if is_files_fit:
            for i_f in range(len(data_sources[0])):
                for i_ds in range(1, len(data_sources)):
                    if len(data_sources[i_ds][i_f]) != len(data_sources[0][i_f]):
                        is_files_fit = False
                        error_messages.append(f"Не совпадает количество свечек в файлах разных источников данных: {data_sources_file_names[i_ds][i_f]} и {data_sources_file_names[0][i_f]}.")
        # даты в свечках разных источников данных
        if is_files_fit:
            for i_f in range(len(data_sources[0])):
                for i_c in range(len(data_sources[0][i_f])):
                    for i_ds in range(1, len(data_sources)):
                        if data_sources[i_ds][i_f][i_c][0] != data_sources[0][i_f][i_c][0]:
                            is_files_fit = False
                            error_messages.append(f"Не совпадают даты в свечках разных источников данных: (файл={data_sources_file_names[i_ds][i_f]}, индекс свечки={i_c}, дата={data_sources[i_ds][i_f][i_c][0]}) и (файл={data_sources_file_names[0][i_f]}, индекс свечки={i_c}, дата={data_sources[0][i_f][i_c][0]}).")
    # выводим ошибки
    for message in error_messages:
        print(message)
    # если не было ошибок,  и формируем обучающие, валидационные и тестовые данные
    if is_files_fit:
        # опредлеяем типы данных для всех свечек
        global data_split_sequence_length
        global validation_split
        global test_split
        data_sources_data_type = []
        learn_count = 0
        valid_count = 0
        test_count = 0
        sequence_number = 0
        data_types = [0, 1, 2]
        data_type = data_types[random.randint(0, 2)]
        for i_f in range(len(data_sources[0])):
            file_data_types = []
            for i_c in range(len(data_sources[0][i_f])):
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
                if i_c >= sequence_length:
                    file_data_types.append(data_type)
                    if data_type == data_types[0]:
                        learn_count += 1
                    elif data_type == data_types[1]:
                        valid_count += 1
                    else:
                        test_count += 1
                    sequence_number += 1
                else:
                    file_data_types.append(-1)  # если перед свечкой нет последовательности длиной sequence_length, отмечаем что она не относится ни к какой выборке
            data_sources_data_type.append(file_data_types)
        #формируем обучающие, валидационные и тестовые данные
        global X_learn
        global Y_learn
        global X_valid
        global Y_valid
        global X_test
        global Y_test


#считывает файлы, нормализует их данные, и возвращает 6 списков: X_learn, Y_learn, X_valid, Y_valid, X_test, Y_test
def csv_files_to_learn_test_data(file_paths, normalize_method, sequence_length, data_split_sequence_length, validation_split, test_split):
    global candle_files
    global X_learn_indexes
    global X_test_indexes
    global X_learn
    global Y_learn
    global X_valid
    global Y_valid
    global X_test
    global Y_test
    candle_files = [read_csv_file(file_path) for file_path in file_paths]
    #проверка, совпадают ли количество свечек их даты во всех файлах
    is_files_fit = True
    if(len(candle_files) > 1):
        length = len(candle_files[0])
        for i in range(1, len(candle_files)):
            if (len(candle_files[i]) != length):
                is_files_fit = False
        if(is_files_fit):
            for i in range(len(candle_files[0])):
                date = candle_files[0][i]
                for k in range(1, len(candle_files)):
                    if(candle_files[k][i] != date):
                        is_files_fit = False
    if(is_files_fit):
        #нормализуем свечки
        global normalized_candle_files
        normalized_candle_files = normalize_method(candle_files)
        global candle_files_type_data
        candle_files_type_data = []
        global candle_files_inputs
        candle_files_inputs = []
        #определяем типы данных свечек
        total_candles = len(candle_files[0]) - sequence_length
        learn_count = 0
        valid_count = 0
        test_count = 0
        sequence_number = 0
        data_types = [0, 1, 2]
        data_type = data_types[random.randint(0, 2)]
        for i in range(len(candle_files[0])): #проходим только по 0-му файлу потому что индекс свечки действителен для всех файлов
            if sequence_number >= data_split_sequence_length:
                sequence_number = 0
                #выбираем случайный тип среди тех, которые составляют от всех данных меньшую часть чем указано для них, если ни один из типов не является меньше указанного, выбираем случайный тип
                data_types_less_than_split = [] #типы данных, количество которых меньше чем их указанное в аргументах количество
                if learn_count / (learn_count + valid_count + test_count) < 1 - (validation_split + test_split):
                    data_types_less_than_split.append(data_types[0])
                if valid_count / (learn_count + valid_count + test_count) < validation_split:
                    data_types_less_than_split.append(data_types[1])
                if test_count / (learn_count + valid_count + test_count) < test_split:
                    data_types_less_than_split.append(data_types[2])

                if len(data_types_less_than_split) > 0:
                    data_type = data_types_less_than_split[random.randint(0, len(data_types_less_than_split) - 1)]
                else:
                    data_type = data_types[random.randint(0, 2)]
            if i >= sequence_length:
                candle_files_type_data.append(data_type)
                if data_type == data_types[0]:
                    learn_count += 1
                elif data_type == data_types[1]:
                    valid_count += 1
                else:
                    test_count += 1
                sequence_number += 1
            else:
                candle_files_type_data.append(-1) #если не достигнута свечка, перед которой имеется последовательность длиной sequence_length, отмечаем что она не относится ни к какой выборке

        #формируем последовательности входных данных для всех свечек (без деления на файлы, т.к. во входном массиве данные всех файлов), а так же обучающую, валидационную и тестовую выборки
        for i in range(len(candle_files[0])):
            if(candle_files_type_data[i] != -1):
                input_sequence = []
                for u in range(-sequence_length, 0): #идем от -sequence_length до 0, т.к. наша свечка следующая за последовательностью, соответственно последовательность должна быть перед нашей свечкой
                    input_list = []
                    for k in range(len(normalized_candle_files)):
                        input_list += normalized_candle_files[k][i + u][1:]
                    input_sequence.append(input_list)
                output_list = []
                for k in range(len(normalized_candle_files)):
                    output_list += normalized_candle_files[k][i][1:]

                candle_files_inputs.append(input_sequence)
                if candle_files_type_data[i] == 0:
                    X_learn.append(input_sequence)
                    X_learn_indexes.append(i)
                    Y_learn.append(output_list)
                elif candle_files_type_data[i] == 1:
                    X_valid.append(input_sequence)
                    Y_valid.append(output_list)
                else:
                    X_test.append(input_sequence)
                    X_test_indexes.append(i)
                    Y_test.append(output_list)
    else:
        print("Ошибка! Количество свечек в файлах или их даты не совпадают.")
    return (X_learn, Y_learn, X_valid, Y_valid, X_test, Y_test)

def convert_candles_to_mplfinance_data(candles): #candles format: [[data (1675087200000, милисекунды), open, high, low, close, volume],..]
    reformatted_data = dict()
    reformatted_data['Date'] = []
    reformatted_data['Open'] = []
    reformatted_data['High'] = []
    reformatted_data['Low'] = []
    reformatted_data['Close'] = []
    reformatted_data['Volume'] = []
    for candle in candles:
        reformatted_data['Date'].append(datetime.datetime.fromtimestamp(int(candle[0]) / 1000))
        reformatted_data['Open'].append(candle[1])
        reformatted_data['High'].append(candle[2])
        reformatted_data['Low'].append(candle[3])
        reformatted_data['Close'].append(candle[4])
        reformatted_data['Volume'].append(candle[5])
    return reformatted_data

def visualize_learning_model(visualize_length):
    print("")


def visualize_predict_to_file(candle_index, sequence_length, predict_length, predict_sequence, save_folder_path):
    global normalized_candle_files
    global candle_files
    normalize_open = []
    data_true = []
    # print(f"len(candle_files)={len(candle_files)}")
    for i in range(len(candle_files)):
        data_true.append(copy.deepcopy(candle_files[i][candle_index - sequence_length:candle_index + predict_length]))
        # print(f"candle_index - sequence_length={candle_index - sequence_length}, candle_index + predict_length={candle_index + predict_length}")
        normalize_open.append(normalized_candle_files[i][candle_index - sequence_length][1])
    data_predict = []
    for i in range(len(candle_files)):
        data_predict.append(copy.deepcopy(candle_files[i][candle_index - sequence_length:candle_index + predict_length]))
    predict_sequence_files = []
    for i in range(len(candle_files)):
        predict_sequence_file = []
        for k in range(len(predict_sequence)):
            predict_sequence_file.append(predict_sequence[k][i * 5:(i + 1) * 5][0])
        predict_sequence_files.append(predict_sequence_file)
    #print(f"predict_sequence_files={predict_sequence_files}")
    for i in range(len(candle_files)):
        for k in range(sequence_length):
            # print(f"data_true={data_true}")
            # print(f"data_predict={data_predict}")
            # print(f"data_predict[{i}]={data_predict[i]}")
            # print(f"k={k}")
            # print(f"data_predict[{i}][{k}]={data_predict[i][k]}")
            # input()
            data_predict[i][k][1] = copy.deepcopy(normalize_open[i])
            data_predict[i][k][2] = copy.deepcopy(normalize_open[i])
            data_predict[i][k][3] = copy.deepcopy(normalize_open[i])
            data_predict[i][k][4] = copy.deepcopy(normalize_open[i])
            data_predict[i][k][5] = copy.deepcopy(normalize_open[i])
        #print(f"sequence_length={sequence_length}, len(data_predict[i])={len(data_predict[i])}")
        for k in range(sequence_length, len(data_predict[i])):
            #print(f"i={i}, k={k}")
            #print(f"data_predict[i][k][1]={data_predict[i][k][1]}")
            #print(f"predict_sequence_files[i][k - sequence_length][0]={predict_sequence_files[i][k - sequence_length][0]}")
            data_predict[i][k][1] = predict_sequence_files[i][k - sequence_length][0]
            data_predict[i][k][2] = predict_sequence_files[i][k - sequence_length][1]
            data_predict[i][k][3] = predict_sequence_files[i][k - sequence_length][2]
            data_predict[i][k][4] = predict_sequence_files[i][k - sequence_length][3]
            data_predict[i][k][5] = predict_sequence_files[i][k - sequence_length][4]
    #print(f"data_predict before unnormalize={data_predict}")
    data_predict = un_normalize_min_max_scaler(data_predict)
    #print(f"data_predict after unnormalize={data_predict}")

    for i in range(len(candle_files)):
        high_canal = [data_true[i][k][2] for k in range(len(data_true[i]))]
        low_canal = [data_true[i][k][3] for k in range(len(data_true[i]))]
        #print(f"data_true   ={data_true}")
        #print(f"data_predict={data_predict}")
        #input()
        ymin = min(min([data_true[i][k][3] for k in range(len(data_true[i]))]), min([min(data_predict[i][k][1:5]) for k in range(len(data_predict[i]))]))
        ymax = max(max([data_true[i][k][2] for k in range(len(data_true[i]))]), max([max(data_predict[i][k][1:5]) for k in range(len(data_predict[i]))]))
        yrange = ymax - ymin
        ymin -= yrange * 0.02
        ymax += yrange * 0.02

        data_true_reformatted = convert_candles_to_mplfinance_data(data_true[i])
        data_predict_reformatted = convert_candles_to_mplfinance_data(data_predict[i])
        pdata_true = pd.DataFrame.from_dict(data_true_reformatted)
        pdata_true.set_index('Date', inplace=True)
        pdata_predict = pd.DataFrame.from_dict(data_predict_reformatted)
        pdata_predict.set_index('Date', inplace=True)
        image_path = f"{save_folder_path}/{str(i).rjust(2, '0')}_{str(candle_index).rjust(7, '0')}"
        add_plot = [mpf.make_addplot(high_canal, type='line', linewidths=1, alpha=1, color="black", ylim=(ymin,ymax)),
                    mpf.make_addplot(low_canal, type='line', linewidths=1, alpha=1, color="black", ylim=(ymin,ymax)),
                    mpf.make_addplot(pdata_predict, type='candle', ylim=(ymin,ymax))]
        mpf.plot(pdata_true, type='candle', style='yahoo', addplot=add_plot, ylim=(ymin,ymax), figsize=(16, 9), ylabel='price', datetime_format="%Y-%b-%d", tight_layout=True, savefig=image_path)
        print(f"save image {image_path}")

def predict_data(model, sequence_length, predict_length, part_learn_predict, part_test_predict, is_visualize_prediction, save_folder_path):
    global normalized_candle_files
    #прогнозирование для обучающих данных
    learn_images_folder_path = f"{save_folder_path}/images/learn"
    test_images_folder_path = f"{save_folder_path}/images/test"
    os.makedirs(learn_images_folder_path)
    os.makedirs(test_images_folder_path)
    learn_predict = dict() #словарь: ключ - индекс свечки начала последовательности для которой сделан прогноз, значение - список с спрогнозированными на predict_length шагов вперед значениями
    test_predict = dict()  #словарь: ключ - индекс свечки начала последовательности для которой сделан прогноз, значение - список с спрогнозированными на predict_length шагов вперед значениями
    for i in range(len(X_learn_indexes) + len(X_test_indexes)):
        if i < len(X_learn_indexes):
            is_learn = True
            candle_index = X_learn_indexes[i]
        else:
            is_learn = False
            candle_index = X_test_indexes[i - len(X_learn_indexes)]
        if candle_index <= len(normalized_candle_files[0]) - predict_length: #если от последней свечки есть отступ в predict_length
            rand = random.random()
            if (is_learn and rand <= part_learn_predict) or (not is_learn and rand <= part_test_predict):
                predict_sequence = []
                for k in range(predict_length):
                    input_sequence = []
                    s_index = k
                    e_index = s_index + sequence_length
                    for u in range(s_index, min(e_index, sequence_length)):
                        input_list = []
                        for m in range(len(normalized_candle_files)):
                            input_list += normalized_candle_files[m][candle_index + u - sequence_length][1:]
                        input_sequence.append(input_list)
                    for u in range(max(s_index - sequence_length, 0), e_index - sequence_length):
                        input_list = []
                        for m in range(len(normalized_candle_files)):
                            input_list += list(predict_sequence[u][0][m * 5:(m + 1) * 5])
                        input_sequence.append(input_list)
                    x = np.array(input_sequence)
                    inp = x.reshape(1, sequence_length, len(input_sequence[0]))
                    pred = model.predict(inp, verbose=0)
                    predict_sequence.append(pred)
                    #print(f"k={k}, s_index={s_index}, e_index={e_index}, input_sequence={input_sequence}")
                if is_learn:
                    learn_predict[i] = predict_sequence
                else:
                    test_predict[i] = predict_sequence
                #print(f"candle_files[{0}][{i}:{i + sequence_length + predict_length}]={candle_files[0][i:i + sequence_length + predict_length]}")
                #print(f"normalized_candle_files[{0}][{i}:{i + sequence_length + predict_length}]={normalized_candle_files[0][i:i + sequence_length + predict_length]}")
                #print(f"predict_sequence={predict_sequence}")
                #input()
                #визуализируем и сохраняем в файл
                if is_visualize_prediction:
                    folder_path = learn_images_folder_path if is_learn else test_images_folder_path
                    visualize_predict_to_file(candle_index, sequence_length, predict_length, predict_sequence, folder_path)
