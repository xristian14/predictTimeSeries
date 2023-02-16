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

# normalize_relativity_min_max_scaler - нормализует свечки по принципу min_max_scaler в следующем диапазоне: (минимум в последовательности:min, максимум в п.:max) -> range=max-min -> output_offset = ((максимальный множитель, на который, следующая за последовательностью свечек длины sequence_length, свечка, увеличивала диапазон последовательности) - 1) * range * 2 -> max += output_offset, min -= output_offset. Также добавляет во входную последовательность для каждой свечки файлов 4 значения: часть размера текущего диапазона от максимального диапазона, найденного среди всех последовательностей по данной формуле для цены и для объема, а так же часть максимума данного диапазона
# сделать прогноз для нескольких последовательных файлов (btcusdt 2013-2020, btcusdt 2020-2023) candles_files[i_ds][i_f][i_c][1]

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

data_sources = [] #свечки для всех файлов всех источников данных
X_learn_candle_indexes_files = [] #индексы обучающих свечек, данная свечка соответствует выходу нейронной сети, а входная последовательность это предыдущие свечки длиной sequence_length. Индексы свечек для всех файлов одного источника данных, т.к. индексы относятся ко всем источникам данных. X_learn_candle_indexes_files[0][0] - индекс файла 0-й обучающей свечки, X_learn_candle_indexes_files[0][1] - индекс свечки в файле 0-й обучающей свечки
X_test_candle_indexes_files = [] #индексы тестовых свечек, данная свечка соответствует выходу нейронной сети, а входная последовательность это предыдущие свечки длиной sequence_length. Индексы свечек для всех файлов одного источника данных, т.к. индексы относятся ко всем источникам данных. X_test_candle_indexes_files[0][0] - индекс файла 0-й тестовой свечки, X_test_candle_indexes_files[0][1] - индекс свечки в файле 0-й тестовой свечки

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
    global data_sources

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
