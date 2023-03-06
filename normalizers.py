class NormalizeSetting:
    pass

class NormalizerBase:
    def summary(self):
        raise NotImplementedError(f"Не переопределен метод summary класса {self.__class__.__name__}")
    def normalize(self):
        raise NotImplementedError(f"Не переопределен метод normalize класса {self.__class__.__name__}")
    def un_normalize(self):
        raise NotImplementedError(f"Не переопределен метод un_normalize класса {self.__class__.__name__}")

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
    def __init__(self, data_indexes):
        self.data_indexes = data_indexes

    # data_sources_data_type - тип данных для всех данных для всех файлов: [0(обучающие), 1(валидационные), 2(тестовые), -1(не участвует в выборках)]. Нет разделения на источники данных, т.к. тип данных относится ко всем источникам данных
    def summary(self, data_source, data_sources_data_type, sequence_length):
        self.sequence_length = sequence_length
        self.mult_high = 0
        self.mult_low = 0
        for i_f in range(len(data_source)):
            for i_c in range(len([data_source[i_f]])):
                if data_sources_data_type[i_f][i_c] != -1:
                    max_inp_seq = None
                    min_inp_seq = None
                    for i in range(i_c - self.sequence_length, i_c):
                        curr_max = max([data_source[i_f][i][data_index] for data_index in self.data_indexes])
                        curr_min = min([data_source[i_f][i][data_index] for data_index in self.data_indexes])
                        if max_inp_seq == None:
                            max_inp_seq = curr_max
                            min_inp_seq = curr_min
                        else:
                            if curr_max > max_inp_seq:
                                max_inp_seq = curr_max
                            if curr_min < min_inp_seq:
                                min_inp_seq = curr_min
                    range_inp_seq = max_inp_seq - min_inp_seq

                    output_max = max([data_source[i_f][i_c][data_index] for data_index in self.data_indexes])
                    output_min = min([data_source[i_f][i_c][data_index] for data_index in self.data_indexes])

                    # определяем, во сколько максимум и минимум выхода расширяет диапазон входной последовательности
                    if output_max > max_inp_seq:
                        curr_mult_high = 1 - (output_max - min_inp_seq) / range_inp_seq
                        if curr_mult_high > self.mult_high:
                            self.mult_high = curr_mult_high

                    if output_max > max_inp_seq:
                        curr_mult_low = 1 - (max_inp_seq - output_min) / range_inp_seq
                        if curr_mult_low > self.mult_low:
                            self.mult_low = curr_mult_low

    def normalize(self, inp_sequence, output = None):
        raise NotImplementedError(f"Не переопределен метод normalize класса {self.__class__.__name__}")