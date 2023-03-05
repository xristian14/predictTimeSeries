class NormalizerBase:
    pass

class MinMaxScalerBase:
    @classmethod
    def min_max_scaler(min_val, max_val, val):
        return (val - min_val) / (max_val - min_val)

    @classmethod
    def un_min_max_scaler(min_val, max_val, val):
        return val * (max_val - min_val) + min_val

"""InpSeqMinMaxScaler - нормализует данные по следующему правилу: при инициализации вычисляет множитель, на который выходное значение увеличивает диапазон значений входной последовательности, mult_high - для роста, и mult_low - для падения; при нормализации вычисляет диапазон значений во входной последовательности (max, min, range = max - min), прибавляет к max: range * mult_high, а из min вычитает: range * mult_low, затем нормализует данные в диапазоне от min до max"""
class InpSeqMinMaxScaler(NormalizerBase, MinMaxScalerBase):
    # data_indexes - индексы значений в data_source, для которых будет выполняться нормализация
    # ds_files_indexes - список с кортежами: (начало, конец) на каждый файл источника данных, указывающих для каких данных файла будет выполняться нормализация
    def __init__(self, data_source, data_indexes, ds_files_indexes, sequence_length):
        pass

