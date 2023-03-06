class NormalizeSetting:
    pass

class NormalizerBase:
    def summary(self):
        raise NotImplementedError(f"Не переопределен метод summary класса {self.__class__.__name__}")
    def normalize(self):
        raise NotImplementedError(f"Не переопределен метод normalize класса {self.__class__.__name__}")
    def denormalize(self):
        raise NotImplementedError(f"Не переопределен метод un_normalize класса {self.__class__.__name__}")

    """вычисляет минимум и максимум в данных, с начального индекса по конечный, конечный индекс не включается, data_indexes - индексы значений в data_source, для которых будет выполняться нормализация"""
    @classmethod
    def min_max_data(cls, data, start_index, end_index, data_indexes):
        min_data = None
        max_data = None
        for i in range(start_index, end_index):
            curr_min = min([data[i][data_index] for data_index in data_indexes])
            curr_max = max([data[i][data_index] for data_index in data_indexes])
            if min_data == None:
                min_data = curr_min
                max_data = curr_max
            else:
                if curr_min < min_data:
                    min_data = curr_min
                if curr_max > max_data:
                    max_data = curr_max
        return min_data, max_data

class MinMaxScalerBase:
    @classmethod
    def min_max_scaler(cls, min_val, max_val, val):
        return (val - min_val) / (max_val - min_val)
    @classmethod
    def un_min_max_scaler(cls, min_val, max_val, val):
        return val * (max_val - min_val) + min_val

"""InpSeqMinMaxScaler - нормализует данные по следующему правилу: при инициализации вычисляет множитель, на который выходное значение увеличивает диапазон значений входной последовательности, mult_high - для роста, и mult_low - для падения; при нормализации вычисляет диапазон значений во входной последовательности (max, min, range = max - min), прибавляет к max: range * mult_high, а из min вычитает: range * mult_low, затем нормализует данные в диапазоне от min до max"""
class InpSeqMinMaxScaler(NormalizerBase, MinMaxScalerBase):
    # data_indexes - индексы значений в data_source, для которых будет выполняться нормализация
    # over_rate - часть от увеличения диапазона под выходное значение, на которую диапазон для нормализации будет больше. 0.1 - увеличение диапазона будет на 10% больше
    def __init__(self, data_indexes, over_rate):
        self.data_indexes = data_indexes
        self.over_rate = over_rate

    # data_sources_data_type - тип данных для всех данных для всех файлов: [0(обучающие), 1(валидационные), 2(тестовые), -1(не участвует в выборках)]. Нет разделения на источники данных, т.к. тип данных относится ко всем источникам данных
    def summary(self, data_source, data_sources_data_type, sequence_length):
        self.sequence_length = sequence_length
        self.mult_high = 0
        self.mult_low = 0
        for i_f in range(len(data_source)):
            for i_c in range(len(data_source[i_f])):
                if data_sources_data_type[i_f][i_c] != -1:
                    min_inp_seq, max_inp_seq = self.min_max_data(data_source[i_f], i_c - self.sequence_length, i_c, self.data_indexes)
                    range_inp_seq = max_inp_seq - min_inp_seq

                    output_max = max([data_source[i_f][i_c][data_index] for data_index in self.data_indexes])
                    output_min = min([data_source[i_f][i_c][data_index] for data_index in self.data_indexes])

                    # определяем, во сколько минимум и максимум выхода расширяет диапазон входной последовательности
                    if output_min < min_inp_seq:
                        curr_mult_low = (max_inp_seq - output_min) / range_inp_seq - 1
                        if curr_mult_low > self.mult_low:
                            self.mult_low = curr_mult_low

                    if output_max > max_inp_seq:
                        curr_mult_high = (output_max - min_inp_seq) / range_inp_seq - 1
                        if curr_mult_high > self.mult_high:
                            self.mult_high = curr_mult_high

    def normalize(self, inp_sequence, output = None):
        min_inp_seq, max_inp_seq = self.min_max_data(inp_sequence, 0, len(inp_sequence), self.data_indexes)
        range_inp_seq = max_inp_seq - min_inp_seq
        norm_min = min_inp_seq - range_inp_seq * (self.mult_low + self.mult_low * self.over_rate)
        norm_max = max_inp_seq + range_inp_seq * (self.mult_high + self.mult_high * self.over_rate)

        normalized_inp_sequence = []
        for i in range(len(inp_sequence)):
            input = []
            for dat_ind in self.data_indexes:
                input.append(self.min_max_scaler(norm_min, norm_max, inp_sequence[i][dat_ind]))
            normalized_inp_sequence.append(input)

        normalized_output = []
        if output != None:
            for dat_ind in self.data_indexes:
                normalized_output.append(self.min_max_scaler(norm_min, norm_max, output[dat_ind]))

        n_setting = NormalizeSetting()
        n_setting.norm_min = norm_min
        n_setting.norm_max = norm_max

        return normalized_inp_sequence, normalized_output, n_setting

    def denormalize(self, norm_setting, normalized_inp_sequence = None, normalized_output = None):
        norm_min = norm_setting.norm_min
        norm_max = norm_setting.norm_max

        denormalized_inp_sequence = []
        if normalized_inp_sequence != None:
            for i in range(len(normalized_inp_sequence)):
                input = []
                for dat_ind in self.data_indexes:
                    input.append(self.un_min_max_scaler(norm_min, norm_max, normalized_inp_sequence[i][dat_ind]))
                denormalized_inp_sequence.append(input)

        denormalized_output = []
        if normalized_output != None:
            for dat_ind in self.data_indexes:
                denormalized_output.append(self.un_min_max_scaler(norm_min, norm_max, normalized_output[dat_ind]))

        return denormalized_inp_sequence, denormalized_output