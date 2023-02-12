import random

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

data_split_sequence_length = 20 #длина последовательности, которые будут формировать обучающие и тестовые данные
X_learn = []
Y_learn = []
X_test = []
Y_test = []
#считывает файлы, нормализует их данные, и возвращает 4 списка: X_learn, Y_learn, X_test, Y_test
def csv_files_to_learn_test_data(file_paths, normalize_method, sequence_length, test_split):
    global data_split_sequence_length
    global X_learn
    global Y_learn
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
        normalized_candle_files = normalize_method(candle_files)
        #формируем обучающую и тестовую выборки
        sequence_number = data_split_sequence_length
        is_learn_data = True
        for i in range(len(normalized_candle_files[0]) - sequence_length):
            if(sequence_number >= data_split_sequence_length):
                sequence_number = 0
                is_learn_data = False if random.random() <= test_split else True

            input_sequence = []
            for u in range(sequence_length):
                input_list = []
                for k in range(len(normalized_candle_files)):
                    input_list += normalized_candle_files[k][i + u][1:]
                input_sequence.append(input_list)
            output_list = []
            for k in range(len(normalized_candle_files)):
                output_list += normalized_candle_files[k][i + sequence_length][1:]

            if(is_learn_data):
                X_learn.append(input_sequence)
                Y_learn.append(output_list)
            else:
                X_test.append(input_sequence)
                Y_test.append(output_list)
            sequence_number += 1
    else:
        print("Ошибка! Количество свечек в файлах или их даты не совпадают.")
    return (X_learn, Y_learn, X_test, Y_test)