import csv


class DataReader(object):
    def __init__(self, delimiter=','):
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

    def line_to_data_row(self, attributes, line):
        return dict(zip(attributes, [datum.strip() for datum in line]))

#
# class SetCreator(object):
#     def __init__(self):


class DataSet(object):
    def __init__(self, attributes, data):
        self.attributes = attributes
        self.data = data
