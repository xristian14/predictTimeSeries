class NormalizeSetting:
    pass

class NormalizerBase:
    def summary(self):
        raise NotImplementedError(f"Не переопределен метод summary класса {self.__class__.__name__}")
    def normalize(self):
        raise NotImplementedError(f"Не переопределен метод normalize класса {self.__class__.__name__}")
    def denormalize(self):
        raise NotImplementedError(f"Не переопределен метод un_normalize класса {self.__class__.__name__}")

    # вычисляет минимум и максимум в данных, с начального индекса по конечный, конечный индекс не включается, data_indexes - индексы значений в data_source, для которых будет выполняться нормализация
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

# нормализует данные в диапазоне от минимума до максимума, которые были в источнике данных, до данных нормализации
class DynamicAbsoluteMinMaxScaler(NormalizerBase, MinMaxScalerBase):
    # add_values - значения, которые будут с самого начала учитываться в диапазоне нормализации
    def __init__(self, data_indexes, output_weight, over_rate_low, over_rate_high, **kwargs):
        self.data_indexes = data_indexes
        self.output_weight = output_weight
        self.over_rate_low = over_rate_low
        self.over_rate_high = over_rate_high
        if "add_values" in kwargs:
            if len(kwargs["add_values"]) > 0:
                self.add_values = kwargs["add_values"]

    def summary(self, data_source, sequence_length):
        self.sequence_length = sequence_length
        self.data_source_settings = [[None] * len(data_source[i_f]) for i_f in range(len(data_source))]
        range_min = min([data_source[0][0][data_index] for data_index in self.data_indexes])
        range_max = max([data_source[0][0][data_index] for data_index in self.data_indexes])
        if hasattr(self, "add_values"):
            range_min = min(range_min, min(self.add_values))
            range_max = max(range_max, max(self.add_values))
        for i_f in range(len(data_source)):
            for i_c in range(len(data_source[i_f])):
                range_min = min(range_min, min([data_source[i_f][i_c][data_index] for data_index in self.data_indexes]))
                range_max = max(range_max, max([data_source[i_f][i_c][data_index] for data_index in self.data_indexes]))
                n_setting = NormalizeSetting()
                n_setting.range_min = range_min
                n_setting.range_max = range_max
                self.data_source_settings[i_f][i_c] = n_setting

    def normalize(self, inp_sequence, output=None, **kwargs):
        range_min = self.data_source_settings[kwargs["i_f"]][kwargs["i_c"]].range_min
        range_max = self.data_source_settings[kwargs["i_f"]][kwargs["i_c"]].range_max
        if "add_candles" in kwargs:
            add_range_min = min([kwargs["add_candles"][0][data_index] for data_index in self.data_indexes])
            add_range_max = max([kwargs["add_candles"][0][data_index] for data_index in self.data_indexes])
            for i in range(len(kwargs["add_candles"])):
                add_range_min = min(add_range_min, min([kwargs["add_candles"][i][data_index] for data_index in self.data_indexes]))
                add_range_max = max(add_range_max, max([kwargs["add_candles"][i][data_index] for data_index in self.data_indexes]))
            range_min = min(range_min, add_range_min)
            range_max = max(range_max, add_range_max)
        range_min_max = range_max - range_min
        range_min_over_rated = range_min - range_min_max * self.over_rate_low
        range_max_over_rated = range_max + range_min_max * self.over_rate_high

        normalized_inp_sequence = []
        for i in range(len(inp_sequence)):
            input = []
            for dat_ind in self.data_indexes:
                input.append(self.min_max_scaler(range_min_over_rated, range_max_over_rated, inp_sequence[i][dat_ind]))
            normalized_inp_sequence.append(input)

        normalized_output = []
        if output != None:
            for dat_ind in self.data_indexes:
                normalized_output.append(self.min_max_scaler(range_min_over_rated, range_max_over_rated, output[dat_ind]))

        n_setting = NormalizeSetting()
        n_setting.range_min_over_rated = range_min_over_rated
        n_setting.range_max_over_rated = range_max_over_rated

        return normalized_inp_sequence, normalized_output, n_setting

    def denormalize(self, norm_setting, normalized_inp_sequence = None, normalized_output = None, **kwargs):
        denormalized_inp_sequence = []
        if normalized_inp_sequence != None:
            for i in range(len(normalized_inp_sequence)):
                input = []
                for index in range(len(self.data_indexes)):
                    input.append(self.un_min_max_scaler(norm_setting.range_min_over_rated, norm_setting.range_max_over_rated, normalized_inp_sequence[i][index]))
                denormalized_inp_sequence.append(input)

        denormalized_output = []
        if normalized_output != None:
            for index in range(len(self.data_indexes)):
                denormalized_output.append(self.un_min_max_scaler(norm_setting.range_min_over_rated, norm_setting.range_max_over_rated, normalized_output[index]))

        return denormalized_inp_sequence, denormalized_output

# RelativeMinMaxScaler - нормализует данные по следующему правилу: при инициализации вычисляет множитель, на который выходное значение увеличивает диапазон значений входной последовательности, mult_high - для роста, и mult_low - для падения; при нормализации вычисляет диапазон значений во входной последовательности (max, min, range = max - min), прибавляет к max: range * mult_high, а из min вычитает: range * mult_low, затем нормализует данные в диапазоне от min до max
class RelativeMinMaxScaler(NormalizerBase, MinMaxScalerBase):
    # data_indexes - индексы значений в data_source, для которых будет выполняться нормализация
    # over_rate - часть от увеличения диапазона под выходное значение, на которую диапазон для нормализации будет больше. 0.1 - увеличение диапазона будет на 10% больше
    # is_range_part - добавлять ли во входную последовательность значение, указывающее какой размер составляет диапазон значений текущей входной последовательности относительно максимального диапазона
    # is_high_part - добавлять ли во входную последовательность значение, указывающее какую часть от максимально возможного значения составляет максимальное значение входной последовательности
    # is_low_part - добавлять ли во входную последовательность значение, указывающее какую часть от максимально возможного значения составляет минимальное значение входной последовательности
    def __init__(self, data_indexes, is_range_part, is_high_part, is_low_part, over_rate):
        self.data_indexes = data_indexes
        self.is_range_part = is_range_part
        self.is_high_part = is_high_part
        self.is_low_part = is_low_part
        self.over_rate = over_rate
        self.inp_norm_data_length = len(data_indexes)
        if is_range_part:
            self.inp_norm_data_length += 1
        if is_high_part:
            self.inp_norm_data_length += 1
        if is_low_part:
            self.inp_norm_data_length += 1
        self.out_norm_data_length = len(data_indexes)

    # data_sources_data_type - тип данных для всех данных для всех файлов: [0(обучающие), 1(валидационные), 2(тестовые), -1(не участвует в выборках)]. Нет разделения на источники данных, т.к. тип данных относится ко всем источникам данных
    def summary(self, data_source, data_sources_data_type, sequence_length):
        self.sequence_length = sequence_length
        self.mult_low = 0
        self.mult_high = 0
        self.max_range = 0
        self.max_value = 0
        for i_f in range(len(data_source)):
            for i_c in range(len(data_source[i_f])):
                if data_sources_data_type[i_f][i_c] != -1:
                    min_inp_seq, max_inp_seq = self.min_max_data(data_source[i_f], i_c - self.sequence_length, i_c, self.data_indexes)
                    range_inp_seq = max_inp_seq - min_inp_seq

                    if range_inp_seq > self.max_range:
                        self.max_range = range_inp_seq

                    if max_inp_seq > self.max_value:
                        self.max_value = max_inp_seq

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
            if self.is_range_part:
                input.append(range_inp_seq / self.max_range)
            if self.is_high_part:
                input.append(max_inp_seq / self.max_value)
            if self.is_low_part:
                input.append(min_inp_seq / self.max_value)
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
                for index in range(len(self.data_indexes)):
                    input.append(self.un_min_max_scaler(norm_min, norm_max, normalized_inp_sequence[i][index]))
                denormalized_inp_sequence.append(input)

        denormalized_output = []
        if normalized_output != None:
            for index in range(len(self.data_indexes)):
                denormalized_output.append(self.un_min_max_scaler(norm_min, norm_max, normalized_output[index]))

        return denormalized_inp_sequence, denormalized_output