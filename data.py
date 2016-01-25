import csv
import random
import collections


class DataReader(object):
    def __init__(self, delimiter=';'):
        self.delimiter = delimiter

    def read_csv(self, filename):
        with open(filename, 'rb') as csv_file:
            reader = csv.reader(csv_file, delimiter=self.delimiter)
            attributes = []
            data = []
            first_row = True
            for row in reader:
                if first_row:
                    attributes = row
                    first_row = False
                else:
                    data.append(self.line_to_data_row(attributes, row))
            return DataSet(attributes, data)

    @staticmethod
    def line_to_data_row(attributes, line):
        return dict(zip(attributes, [datum.strip() for datum in line]))


class DataSet(object):

    def __init__(self, attributes, data):
        self.attributes = attributes
        self.data = data
        self.target_attr = attributes[-1]

    def split(self, from_index, to_index):
        """
        :type from_index: int
        :type to_index: int
        :rtype: DataSet
        """
        inner_data = self.data[from_index:to_index]
        outer_data = self.data[:from_index] + self.data[to_index:]
        return DataSet(self.attributes, inner_data), DataSet(self.attributes, outer_data)

    def get_random_samples(self, amount, with_replacement=True):
        """
        :type amount: int
        :type with_replacement: bool
        :rtype: DataSet
        """
        if with_replacement:
            data = [random.choice(self.data) for _ in xrange(amount)]
        else:
            data = random.sample(self.data, amount)
        return DataSet(self.attributes, data)

    def get_all_with_given_attribute(self, attribute, value):
        """
        :type attribute: string
        :type value: string
        :rtype: DataSet
        """
        data = filter(lambda record: record[attribute] == value, self.data)
        attributes = [attr for attr in self.attributes if attr != attribute]
        return DataSet(attributes, data)

    def get_unique_values_of_attribute(self, attribute):
        """
        :type attribute: string
        :return: set
        """
        return set(record[attribute] for record in self.data)

    def get_values_frequency(self, attribute):
        """
        :return: dict[str:str]
        """
        counter = collections.defaultdict(lambda: 0)
        for record in self.data:
            counter[record[attribute]] += 1
        return counter

    def get_most_common_class(self):
        """
        :return: str
        """
        counter = self.get_values_frequency(self.target_attr)
        return max(counter.iterkeys(), key=lambda k: counter[k])

    def get_best_attribute(self, fitness_func):
        """
        :param fitness_func: (DataSet, str) => float
        :return: str
        """
        return max(
            self.attributes, key=lambda attr:
            0.0 if attr == self.target_attr else fitness_func(self, attr))

    def __len__(self):
        return len(self.data)
