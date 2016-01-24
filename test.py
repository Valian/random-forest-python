from __future__ import with_statement
import sys
import os.path
from data import DataReader, CrossValidation


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
    data_filename = 'data'
    reader = DataReader(',')
    data_set = reader.read_csv(data_filename)
    validator = CrossValidation(4, True)
    accuracy = validator.validate(data_set)

    print "------------------------\n"
    print "--   Accuracy: " + str(accuracy) + "   --\n"
    print "------------------------\n"
