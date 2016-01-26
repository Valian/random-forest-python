import argparse
from data import DataReader
from core import CrossValidation, DecisionTreeCreator, RandomForestCreator
from entropy import gain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='path to csv file with data to test')
    parser.add_argument('--max-depth', '-d', dest='depth', default=10, type=int, help='maximum depth')
    parser.add_argument('--trees', '-t', dest='trees', default=10, type=int, help='number of trees in a forest')
    parser.add_argument('--with-replacement', '-r', dest='replacement', default=True, action='store_true',
                        help='use same record many times when creating sample data for a decision tree?')
    parser.add_argument('--sample-size', '-s', dest='sample', default=0.7, type=float,
                        help='ratio of training data to validation data in set')
    parser.add_argument('--folds', '-f', dest='folds', default=4, type=int, help='cross validation folds number')
    parser.add_argument('--verbosity', '-v', dest='verbosity', default=0, type=int, choices=[0, 1, 2, 3],
                        help='detailed output')
    parser.add_argument('--discretization', '-c', dest='discretization', default=None, type=int,
                        help='amount of values to perform discretization')
    return parser.parse_args()


def test(filename, depth, trees, replacement, sample, folds, verbosity, discretization):
    reader = DataReader(discretization_level=discretization)
    data_set = reader.read_csv(filename)
    tree_creator = DecisionTreeCreator(gain, max_depth=depth)
    forest_creator = RandomForestCreator(tree_creator, trees, with_replacement=replacement, sample_size=sample)
    validator = CrossValidation(forest_creator, folds, verbosity_level=verbosity)
    return validator.validate(data_set)


if __name__ == "__main__":
    args = parse_args()
    accuracy = test(**vars(args))

    print "Combined accuracy: " + str(accuracy)
