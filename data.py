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

    def line_to_data_row(self, attributes, line):
        return dict(zip(attributes, [datum.strip() for datum in line]))


class CrossValidation(object):
    def __init__(self, random_forest_creator, folds, verbose=False):
        self.random_forest_creator = random_forest_creator
        self.folds = folds
        self.verbose = verbose

    def train(self, training_set, attributes, target_attr):
        forest = self.random_forest_creator.create_random_forest(training_set, attributes, target_attr)
        if self.verbose:
            print forest
        return forest

    def test(self, test_set, target_attr, forest):
        classification = forest.classify_many(test_set)
        valid = 0
        i = 0
        for res in classification:
            if self.verbose:
                print res + ' ?? ' + test_set[i][target_attr]
            valid += 1 if test_set[i][target_attr] == res else 0
            i += 1
        return valid / float(len(test_set))

    def validate(self, data_set):
        subset_size = len(data_set.data) / self.folds
        results = []
        for i in range(self.folds):
            testing_this_round = data_set.data[i * subset_size:(i + 1) * subset_size]
            training_this_round = data_set.data[:i * subset_size] + data_set.data[(i + 1) * subset_size:]
            forest = self.train(training_this_round, data_set.attributes, data_set.target_attr)
            results.append(self.test(testing_this_round, data_set.target_attr, forest))
            if self.verbose:
                print 'Result for fold no {}: {}'.format(i, results[-1])

        return sum(results) / self.folds


class DataSet(object):
    def __init__(self, attributes, data):
        self.attributes = attributes
        self.data = data
        self.target_attr = attributes[-1]
