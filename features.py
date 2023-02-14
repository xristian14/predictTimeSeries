import random
import os
import numpy as np
import datetime
import mplfinance as mpf
import pandas as pd
import copy


def min_max_scaler(min, max, val):
    return (val - min) / (max - min)

def un_min_max_scaler(min, max, val):
    return val * (max - min) + min

#считывает все свечки файла в список формата: [[date, open, high, low, close, volume], [date, open, high, low, close, volume],..]
def read_csv_file(file_path):
    file_candles = []
    with open(file_path, 'r', encoding='utf-8') as file:
        first = True
        for line in file:
            if first:  # пропускаем шапку файла
                first = False
            else:
                list_line = line.split(",")
                file_candles.append([int(list_line[0]), float(list_line[1]), float(list_line[2]), float(list_line[3]), float(list_line[4]), float(list_line[5])])
    return file_candles

normalized_candle_files = [] #нормализованные свечки
normalize_min_max_scaler_Min_price = [] #списки со значениями минимума и максимума в каждом файле
normalize_min_max_scaler_Max_price = []
normalize_min_max_scaler_Min_volume = []
normalize_min_max_scaler_Max_volume = []
#нормализует свечки по принципу min_max_scaler, margin - отступ от минимума и максимума как часть от диапазона
def normalize_min_max_scaler(candle_files, offset = 0.0):
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

    normalized_candle_files = []
    for i in range(len(candle_files)):
        normalized_file_candles = []
        for k in range(len(candle_files[i])):
            n_candle = []
            n_candle.append(candle_files[i][k][0])
            n_candle.append(min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], candle_files[i][k][1]))
            n_candle.append(min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], candle_files[i][k][2]))
            n_candle.append(min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], candle_files[i][k][3]))
            n_candle.append(min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], candle_files[i][k][4]))
            n_candle.append(min_max_scaler(normalize_min_max_scaler_Min_volume[i], normalize_min_max_scaler_Max_volume[i], candle_files[i][k][5]))
            normalized_file_candles.append(n_candle)
        normalized_candle_files.append(normalized_file_candles)
    return normalized_candle_files

def un_normalize_min_max_scaler(normalized_candle_files):
    global normalize_min_max_scaler_Min_price
    global normalize_min_max_scaler_Max_price
    global normalize_min_max_scaler_Min_volume
    global normalize_min_max_scaler_Max_volume

    un_normalized_candle_files = []
    for i in range(len(normalized_candle_files)):
        un_normalized_file_candles = []
        for k in range(len(normalized_candle_files[i])):
            n_candle = []
            n_candle.append(normalized_candle_files[i][k][0])
            n_candle.append(un_min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], normalized_candle_files[i][k][1]))
            n_candle.append(un_min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], normalized_candle_files[i][k][2]))
            n_candle.append(un_min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], normalized_candle_files[i][k][3]))
            n_candle.append(un_min_max_scaler(normalize_min_max_scaler_Min_price[i], normalize_min_max_scaler_Max_price[i], normalized_candle_files[i][k][4]))
            n_candle.append(un_min_max_scaler(normalize_min_max_scaler_Min_volume[i], normalize_min_max_scaler_Max_volume[i], normalized_candle_files[i][k][5]))
            un_normalized_file_candles.append(n_candle)
        un_normalized_candle_files.append(un_normalized_file_candles)
    return un_normalized_candle_files

candle_files = []
X_learn_indexes = [] #индексы свечек в normalized_candle_files для обучающих данных, индекс соответствует индексу начальной свечки в последовательных данных образца
X_test_indexes = [] #индексы свечек в normalized_candle_files для тестовых данных, индекс соответствует индексу начальной свечки в последовательных данных образца
X_learn = []
Y_learn = []
X_valid = []
Y_valid = []
X_test = []
Y_test = []
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
        #формируем обучающую, валидационную и тестовую выборки
        sequence_number = data_split_sequence_length
        data_types = ["learning", "validation", "testing"]
        data_type = data_types[0]
        for i in range(len(normalized_candle_files[0]) - sequence_length):
            if sequence_number >= data_split_sequence_length:
                sequence_number = 0
                rand = random.random()
                if rand <= validation_split:
                    data_type = data_types[1]
                elif rand <= validation_split + test_split:
                    data_type = data_types[2]
                else:
                    data_type = data_types[0]

            input_sequence = []
            for u in range(sequence_length):
                input_list = []
                for k in range(len(normalized_candle_files)):
                    input_list += normalized_candle_files[k][i + u][1:]
                input_sequence.append(input_list)
            output_list = []
            for k in range(len(normalized_candle_files)):
                output_list += normalized_candle_files[k][i + sequence_length][1:]

            if data_type == data_types[0]:
                X_learn.append(input_sequence)
                X_learn_indexes.append(i)
                Y_learn.append(output_list)
            elif data_type == data_types[1]:
                X_valid.append(input_sequence)
                Y_valid.append(output_list)
            else:
                X_test.append(input_sequence)
                X_test_indexes.append(i)
                Y_test.append(output_list)
            sequence_number += 1
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

def visualize_to_file(candle_index, sequence_length, predict_length, predict_sequence, save_folder_path):
    global normalized_candle_files
    global candle_files
    normalize_open = []
    data_true = []
    for i in range(len(candle_files)):
        data_true.append(copy.deepcopy(candle_files[i][candle_index:candle_index + sequence_length + predict_length]))
        normalize_open.append(normalized_candle_files[i][candle_index][1])
    data_predict = []
    for i in range(len(candle_files)):
        data_predict.append(copy.deepcopy(candle_files[i][candle_index:candle_index + sequence_length + predict_length]))
    predict_sequence_files = []
    for i in range(len(candle_files)):
        predict_sequence_file = []
        for k in range(len(predict_sequence)):
            predict_sequence_file.append(predict_sequence[k][i * 5:(i + 1) * 5][0])
        predict_sequence_files.append(predict_sequence_file)
    #print(f"predict_sequence_files={predict_sequence_files}")
    for i in range(len(candle_files)):
        for k in range(sequence_length):
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

        data_true_reformatted = convert_candles_to_mplfinance_data(data_true[i])
        data_predict_reformatted = convert_candles_to_mplfinance_data(data_predict[i])
        pdata_true = pd.DataFrame.from_dict(data_true_reformatted)
        pdata_true.set_index('Date', inplace=True)
        pdata_predict = pd.DataFrame.from_dict(data_predict_reformatted)
        pdata_predict.set_index('Date', inplace=True)
        image_path = f"{save_folder_path}/{str(i).rjust(2, '0')}_{str(candle_index).rjust(7, '0')}"
        add_plot = [mpf.make_addplot(high_canal, type='line', linewidths=1, alpha=1, color="black"),
                    mpf.make_addplot(low_canal, type='line', linewidths=1, alpha=1, color="black"),
                    mpf.make_addplot(pdata_predict, type='candle')]
        mpf.plot(pdata_true, type='candle', style='yahoo', addplot=add_plot, figsize=(16, 9), ylabel='price', ylabel_lower='volume', tight_layout=True, savefig=image_path)

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
        #global candle_files #&&&
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
                        input_list += normalized_candle_files[m][candle_index + u][1:]
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
                visualize_to_file(i, sequence_length, predict_length, predict_sequence, folder_path)
    """#-----------------------saved
    for i in X_learn_indexes:
        #global candle_files #&&&
        rand = random.random()
        if rand <= part_learn_predict:
            learn_predict_sequence = []
            for k in range(predict_length):
                input_sequence = []
                s_index = k
                e_index = s_index + sequence_length
                for u in range(s_index, min(e_index, sequence_length)):
                    input_list = []
                    for m in range(len(normalized_candle_files)):
                        input_list += normalized_candle_files[m][i + u][1:]
                    input_sequence.append(input_list)
                for u in range(max(s_index - sequence_length, 0), e_index - sequence_length):
                    input_list = []
                    for m in range(len(normalized_candle_files)):
                        input_list += list(learn_predict_sequence[u][0][m * 5:(m + 1) * 5])
                    input_sequence.append(input_list)
                x = np.array(input_sequence)
                inp = x.reshape(1, sequence_length, len(input_sequence[0]))
                pred = model.predict(inp)
                learn_predict_sequence.append(pred)
                #print(f"k={k}, s_index={s_index}, e_index={e_index}, input_sequence={input_sequence}")
            learn_predict[i] = learn_predict_sequence
            #print(f"candle_files[{0}][{i}:{i + sequence_length + predict_length}]={candle_files[0][i:i + sequence_length + predict_length]}")
            #print(f"normalized_candle_files[{0}][{i}:{i + sequence_length + predict_length}]={normalized_candle_files[0][i:i + sequence_length + predict_length]}")
            #print(f"learn_predict_sequence={learn_predict_sequence}")
            #input()
            #визуализируем и сохраняем в файл
            if is_visualize_prediction:
                visualize_to_file(i, sequence_length, predict_length, learn_predict_sequence, f"{save_folder_path}/images/learn")
    #-----------------------saved
    # прогнозирование для тестовых данных
    os.makedirs(f"{save_folder_path}/images/test")
    test_predict = dict()  # словарь: ключ - индекс свечки начала последовательности для которой сделан прогноз, значение - список с спрогнозированными на predict_length шагов вперед значениями
    for i in X_test_indexes:
        rand = random.random()
        if rand <= part_test_predict:
            test_predict_sequence = []
            for k in range(predict_length):
                input_sequence = []
                s_index = len(input_sequence)
                e_index = s_index + sequence_length
                for u in range(s_index, min(e_index, sequence_length)):
                    input_list = []
                    for k in range(len(normalized_candle_files)):
                        input_list += normalized_candle_files[k][i + u][1:]
                    input_sequence.append(input_list)
                for u in range(max(s_index - sequence_length, 0), e_index - sequence_length):
                    input_list = []
                    for k in range(len(normalized_candle_files)):
                        input_list += list(test_predict_sequence[u][0][k * 5:(k + 1) * 5])
                    input_sequence.append(input_list)

                x = np.array(input_sequence)
                inp = x.reshape(1, sequence_length, len(input_sequence[0]))
                pred = model.predict(inp)
                test_predict_sequence.append(pred)
            test_predict[i] = test_predict_sequence
            # визуализируем и сохраняем в файл
            if is_visualize_prediction:
                visualize_to_file(i, sequence_length, predict_length, test_predict_sequence, f"{save_folder_path}/images/test")"""
