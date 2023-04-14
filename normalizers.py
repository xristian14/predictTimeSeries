import features


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

class DateTimeOneHotVector(NormalizerBase):
    def __init__(self, data_indexes, is_input_denormalize, input_denormalize_weight, is_output_denormalize, output_denormalize_weight, is_month, is_day_of_week, is_day, is_hour):
        self.data_indexes = data_indexes
        self.is_input_denormalize = is_input_denormalize
        self.input_denormalize_weight = input_denormalize_weight
        self.is_output_denormalize = is_output_denormalize
        self.output_denormalize_weight = output_denormalize_weight
        self.is_month = is_month
        self.is_day_of_week = is_day_of_week
        self.is_day = is_day
        self.is_hour = is_hour
        if not is_month and not is_day_of_week and not is_day and not is_hour:
            raise ValueError("В нормализаторе DateTimeOneHotVector не указан ни месяц, ни день недели, ни день, ни час.")
        self.normalize_input_length = is_month * 12 + is_day_of_week * 7 + is_day * 31 + is_hour * 24
        self.denormalize_input_length = 0
        self.normalize_output_length = 0
        self.denormalize_output_length = 0
        self.data_indexes_offsets = {}  # смещение во входных нормализованных данных для каждого индекса данных. С помощью этого списка можно понять по какому смещению получить значение определенного индекса данных из нормализованных данных
        for i in range(len(data_indexes)):
            self.data_indexes_offsets[data_indexes[i]] = i

    def summary(self, data_source, sequence_length):
        self.sequence_length = sequence_length
        self.months_list = [1,2,3,4,5,6,7,8,9,10,11,12]
        self.days_of_week_list = [0,1,2,3,4,5,6]
        self.days_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
        self.hours_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    def normalize(self, inp_sequence, output=None, **kwargs):
        normalized_inp_sequence_by_data_indexes = []
        for i in range(len(inp_sequence)):
            input = []
            for dat_ind in self.data_indexes:
                date = features.timestamp_to_utc_datetime(inp_sequence[i][dat_ind])
                if self.is_month:
                    month_index = self.months_list.index(date.month)
                    input.extend([0] * month_index)
                    input.extend([1])
                    input.extend([0] * (len(self.months_list) - (month_index + 1)))
                if self.is_day_of_week:
                    day_of_week_index = self.days_of_week_list.index(date.weekday())
                    input.extend([0] * day_of_week_index)
                    input.extend([1])
                    input.extend([0] * (len(self.days_of_week_list) - (day_of_week_index + 1)))
                if self.is_day:
                    day_index = self.days_list.index(date.day)
                    input.extend([0] * day_index)
                    input.extend([1])
                    input.extend([0] * (len(self.days_list) - (day_index + 1)))
                if self.is_hour:
                    hour_index = self.hours_list.index(date.hour)
                    input.extend([0] * hour_index)
                    input.extend([1])
                    input.extend([0] * (len(self.hours_list) - (hour_index + 1)))
            normalized_inp_sequence_by_data_indexes.append(input)

        normalized_output_by_data_indexes = []

        n_setting = NormalizeSetting()

        return normalized_inp_sequence_by_data_indexes, normalized_output_by_data_indexes, n_setting

    def denormalize(self, norm_setting, normalized_inp_sequence_by_data_indexes = None, normalized_output_by_data_indexes = None, **kwargs):
        denormalized_inp_sequence_by_data_indexes = []
        denormalized_output_by_data_indexes = []

        return denormalized_inp_sequence_by_data_indexes, denormalized_output_by_data_indexes

# добавляет во входной вектор значения в изначальном виде, и не добавляет их в выходной вектор
class SameValuesNoOutput(NormalizerBase):
    def __init__(self, data_indexes, is_input_denormalize, input_denormalize_weight, is_output_denormalize, output_denormalize_weight):
        self.data_indexes = data_indexes
        self.is_input_denormalize = is_input_denormalize
        self.input_denormalize_weight = input_denormalize_weight
        self.is_output_denormalize = is_output_denormalize
        self.output_denormalize_weight = output_denormalize_weight
        self.normalize_input_length = len(data_indexes)
        self.denormalize_input_length = 0
        self.normalize_output_length = 0
        self.denormalize_output_length = 0
        self.data_indexes_offsets = {}  # смещение во входных нормализованных данных для каждого индекса данных. С помощью этого списка можно понять по какому смещению получить значение определенного индекса данных из нормализованных данных
        for i in range(len(data_indexes)):
            self.data_indexes_offsets[data_indexes[i]] = i

    def summary(self, data_source, sequence_length):
        self.sequence_length = sequence_length

    def normalize(self, inp_sequence, output=None, **kwargs):
        normalized_inp_sequence_by_data_indexes = []
        for i in range(len(inp_sequence)):
            input = []
            for dat_ind in self.data_indexes:
                input.append(inp_sequence[i][dat_ind])
            normalized_inp_sequence_by_data_indexes.append(input)
        normalized_output_by_data_indexes = []
        n_setting = NormalizeSetting()
        return normalized_inp_sequence_by_data_indexes, normalized_output_by_data_indexes, n_setting

    def denormalize(self, norm_setting, normalized_inp_sequence_by_data_indexes = None, normalized_output_by_data_indexes = None, **kwargs):
        denormalized_inp_sequence_by_data_indexes = []
        denormalized_output_by_data_indexes = []
        return denormalized_inp_sequence_by_data_indexes, denormalized_output_by_data_indexes

# нормализует данные в диапазоне от минимума до максимума, которые были в источнике данных, до данных нормализации
class DynamicAbsoluteMinMaxScaler(NormalizerBase, MinMaxScalerBase):
    # add_values - значения, которые будут с самого начала учитываться в диапазоне нормализации
    def __init__(self, data_indexes, is_input_denormalize, input_denormalize_weight, is_output_denormalize, output_denormalize_weight, over_rate_low, over_rate_high, add_values, is_auto_over_rate_low=False, auto_over_rate_low_multipy=1.5, auto_over_rate_low_min=0.1, is_auto_over_rate_high=False, auto_over_rate_high_multipy=1.5, auto_over_rate_high_min=0.1):
        self.data_indexes = data_indexes
        self.is_input_denormalize = is_input_denormalize
        self.input_denormalize_weight = input_denormalize_weight
        self.is_output_denormalize = is_output_denormalize
        self.output_denormalize_weight = output_denormalize_weight
        self.over_rate_low = over_rate_low
        self.over_rate_high = over_rate_high
        self.add_values = add_values
        self.is_auto_over_rate_low = is_auto_over_rate_low
        self.auto_over_rate_low_multipy = auto_over_rate_low_multipy
        self.auto_over_rate_low_min = auto_over_rate_low_min # минимальное значение для over_rate_low при автоматическом определении
        self.is_auto_over_rate_high = is_auto_over_rate_high
        self.auto_over_rate_high_multipy = auto_over_rate_high_multipy
        self.auto_over_rate_high_min = auto_over_rate_high_min # минимальное значение для over_rate_high при автоматическом определении
        self.normalize_input_length = len(data_indexes)
        self.denormalize_input_length = len(data_indexes)
        self.normalize_output_length = len(data_indexes)
        self.denormalize_output_length = len(data_indexes)
        self.data_indexes_offsets = {} # смещение во входных нормализованных данных для каждого индекса данных. С помощью этого списка можно понять по какому смещению получить значение определенного индекса данных из нормализованных данных
        for i in range(len(data_indexes)):
            self.data_indexes_offsets[data_indexes[i]] = i

    def summary(self, data_source, sequence_length):
        # автоматически определяем over_rate_low и over_rate_high
        if self.is_auto_over_rate_low or self.is_auto_over_rate_high:
            over_rate_low = 0
            over_rate_high = 0
            range_min = min([data_source[0][0][data_index] for data_index in self.data_indexes])
            range_max = max([data_source[0][0][data_index] for data_index in self.data_indexes])
            if len(self.add_values) > 0:
                range_min = min(range_min, min(self.add_values))
                range_max = max(range_max, max(self.add_values))
            for i_f in range(len(data_source)):
                for i_c in range(len(data_source[i_f])):
                    current_min = min(range_min, min([data_source[i_f][i_c][data_index] for data_index in self.data_indexes]))
                    current_max = max(range_max, max([data_source[i_f][i_c][data_index] for data_index in self.data_indexes]))
                    range_min_max = range_max - range_min
                    if current_min < range_min:
                        current_over_rate_low = (range_min - current_min) / range_min_max
                        if current_over_rate_low > over_rate_low:
                            over_rate_low = current_over_rate_low
                        range_min = current_min
                    if current_max > range_max:
                        current_over_rate_high = (current_max - range_max) / range_min_max
                        if current_over_rate_high > over_rate_high:
                            over_rate_high = current_over_rate_high
                        range_max = current_max

            over_rate_low_multiplied = over_rate_low * self.auto_over_rate_low_multipy
            over_rate_high_multiplied = over_rate_high * self.auto_over_rate_high_multipy

            if self.is_auto_over_rate_low:
                self.over_rate_low = max(self.auto_over_rate_low_min, over_rate_low_multiplied)
            if self.is_auto_over_rate_high:
                self.over_rate_high = max(self.auto_over_rate_high_min, over_rate_high_multiplied)

        self.sequence_length = sequence_length
        self.data_source_settings = [[None] * len(data_source[i_f]) for i_f in range(len(data_source))]
        range_min = min([data_source[0][0][data_index] for data_index in self.data_indexes])
        range_max = max([data_source[0][0][data_index] for data_index in self.data_indexes])
        if len(self.add_values) > 0:
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

        normalized_inp_sequence_by_data_indexes = []
        for i in range(len(inp_sequence)):
            input = []
            for dat_ind in self.data_indexes:
                input.append(self.min_max_scaler(range_min_over_rated, range_max_over_rated, inp_sequence[i][dat_ind]))
            normalized_inp_sequence_by_data_indexes.append(input)

        normalized_output_by_data_indexes = []
        if output != None:
            for dat_ind in self.data_indexes:
                normalized_output_by_data_indexes.append(self.min_max_scaler(range_min_over_rated, range_max_over_rated, output[dat_ind]))

        n_setting = NormalizeSetting()
        n_setting.range_min_over_rated = range_min_over_rated
        n_setting.range_max_over_rated = range_max_over_rated

        return normalized_inp_sequence_by_data_indexes, normalized_output_by_data_indexes, n_setting

    def denormalize(self, norm_setting, normalized_inp_sequence_by_data_indexes = None, normalized_output_by_data_indexes = None, **kwargs):
        denormalized_inp_sequence_by_data_indexes = []
        if normalized_inp_sequence_by_data_indexes != None:
            for i in range(len(normalized_inp_sequence_by_data_indexes)):
                input = []
                for index in range(len(normalized_inp_sequence_by_data_indexes[0])):
                    input.append(self.un_min_max_scaler(norm_setting.range_min_over_rated, norm_setting.range_max_over_rated, normalized_inp_sequence_by_data_indexes[i][index]))
                denormalized_inp_sequence_by_data_indexes.append(input)

        denormalized_output_by_data_indexes = []
        if normalized_output_by_data_indexes != None:
            for index in range(len(normalized_output_by_data_indexes)):
                denormalized_output_by_data_indexes.append(self.un_min_max_scaler(norm_setting.range_min_over_rated, norm_setting.range_max_over_rated, normalized_output_by_data_indexes[index]))

        return denormalized_inp_sequence_by_data_indexes, denormalized_output_by_data_indexes

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