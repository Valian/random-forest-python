from __future__ import with_statement
from data import DataReader, CrossValidation
from decision_tree import DecisionTreeCreator, RandomForestCreator
from id3 import gain


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
