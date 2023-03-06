class NormalizerBase:
    def summary(self):
        raise NotImplementedError(f"Не переопределен метод summary класса {self.__class__.__name__}")

class MinMaxScalerBase:
    @classmethod
    def min_max_scaler(min_val, max_val, val):
        return (val - min_val) / (max_val - min_val)

    @classmethod
    def un_min_max_scaler(min_val, max_val, val):
        return val * (max_val - min_val) + min_val

"""InpSeqMinMaxScaler - нормализует данные по следующему правилу: при инициализации вычисляет множитель, на который выходное значение увеличивает диапазон значений входной последовательности, mult_high - для роста, и mult_low - для падения; при нормализации вычисляет диапазон значений во входной последовательности (max, min, range = max - min), прибавляет к max: range * mult_high, а из min вычитает: range * mult_low, затем нормализует данные в диапазоне от min до max"""
class InpSeqMinMaxScaler(NormalizerBase, MinMaxScalerBase):
    def __init__(self, data_indexes):
        self.__data_indexes = data_indexes

    # data_indexes - индексы значений в data_source, для которых будет выполняться нормализация
    # data_sources_data_type - тип данных для всех данных для всех файлов: [0(обучающие), 1(валидационные), 2(тестовые), -1(не участвует в выборках)]. Нет разделения на источники данных, т.к. тип данных относится ко всем источникам данных
    def summary(self, data_source, data_sources_data_type, sequence_length):
        pass