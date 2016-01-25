import csv


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


class CrossValidation(object):
    def __init__(self, random_forest_creator, folds, verbosity_level=0):
        self.random_forest_creator = random_forest_creator
        self.folds = folds
        self.verbosity_level = verbosity_level

    def train(self, training_set, attributes, target_attr):
        forest = self.random_forest_creator.create_random_forest(training_set, attributes, target_attr)
        if self.verbosity_level >= 3:
            print forest
        return forest

    def test(self, test_set, target_attr, forest):
        classification = forest.classify_many(test_set)
        valid_counter = 0
        for i, res in enumerate(classification):
            expected = test_set[i][target_attr]
            valid_counter += int(expected == res)
            if self.verbosity_level >= 2:
                print 'result: {0:10}, expected {1:10} - {2}'.format(res, expected, expected == res)
        return valid_counter / float(len(test_set))

    def validate(self, data_set):
        subset_size = len(data_set.data) / self.folds
        results = []
        for i in range(self.folds):
            testing_this_round = data_set.data[i * subset_size:(i + 1) * subset_size]
            training_this_round = data_set.data[:i * subset_size] + data_set.data[(i + 1) * subset_size:]
            forest = self.train(training_this_round, data_set.attributes, data_set.target_attr)
            results.append(self.test(testing_this_round, data_set.target_attr, forest))
            if self.verbosity_level >= 1:
                print 'Accuracy for fold no {0}: {1}'.format(i + 1, results[-1])

        return sum(results) / self.folds


class DataSet(object):
    def __init__(self, attributes, data):
        self.attributes = attributes
        self.data = data
        self.target_attr = attributes[-1]
