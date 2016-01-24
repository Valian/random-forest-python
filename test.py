from __future__ import with_statement
import sys
import os.path
from data import DataReader, CrossValidation
from decision_tree import DecisionTreeCreator, RandomForestCreator
from id3 import gain


def get_file_names():
    if len(sys.argv) < 2:
        data_file_name = raw_input("Data Filename: ")
    else:
        data_file_name = sys.argv[1]

    def file_exists(filename):
        if os.path.isfile(filename):
            return True
        else:
            print "Error: The file '%s' does not exist." % filename
            return False

    if not file_exists(data_file_name):
        sys.exit(0)

    return data_filename


if __name__ == "__main__":
    data_filename = 'sample_data/basic_data_binary.csv'
    reader = DataReader()
    data_set = reader.read_csv(data_filename)
    tree_creator = DecisionTreeCreator(gain, max_depth=3)
    forest_creator = RandomForestCreator(tree_creator, 10, with_replacement=True, sample_size=0.7)
    validator = CrossValidation(forest_creator, 4, verbose=True)
    accuracy = validator.validate(data_set)

    print "------------------------\n"
    print "--   Accuracy: " + str(accuracy) + "   --\n"
    print "------------------------\n"
