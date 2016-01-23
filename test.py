from __future__ import with_statement
from decision_tree import DecisionTreeCreator, RandomForestCreator
from id3 import *
import sys
import os.path
from data import DataReader, DataSet

def get_filenames():
    """
    Tries to extract the training and test data filenames from the command
    line.  If none are present, it prompts the user for both filenames and
    and makes sure that both exist in the system before returning the names
    back to the calling function.
    """
    # If no filenames were given at the command line, ask for them
    if len(sys.argv) < 3:
        training_filename = raw_input("Training Filename: ")
        test_filename = raw_input("Test Filename: ")
    # otherwise, read the filenames from the command line
    else:
        training_filename = sys.argv[1]
        test_filename = sys.argv[2]

    # This is a local function that takes a filename and returns true or false
    # depending on whether or not the file exists in the system.
    def file_exists(filename):
        if os.path.isfile(filename):
            return True
        else:
            print "Error: The file '%s' does not exist." % filename
            return False

    # Make sure both files exist, otherwise print an error and exit execution
    if ((not file_exists(training_filename)) or
        (not file_exists(test_filename))):
        sys.exit(0)

    # Return the filenames of the training and test data files
    return training_filename, test_filename

def get_attributes(filename):
    """
    Parses the attribute names from the header line of the given file.
    """
    # Create a list of all the lines in the training file
    with open(filename, 'r') as fin:
        header = fin.readline().strip()

    # Parse the attributes from the header
    attributes = [attr.strip() for attr in header.split(",")]

    return attributes

def get_data(filename, attributes):
    """
    This function takes a file and list of attributes and returns a list of
    dict objects that represent each record in the file.
    """
    # Create a list of all the lines in the training file
    with open(filename) as fin:
        lines = [line.strip() for line in fin.readlines()]

    # Remove the attributes line from the list of lines
    del lines[0]

    # Parse all of the individual data records from the given file
    data = []
    for line in lines:
        data.append(dict(zip(attributes,
                             [datum.strip() for datum in line.split(",")])))
    
    return data


if __name__ == "__main__":
    # Get the training and test data filenames from the user
    training_filename, test_filename = 'data', 'test_data'
    #training_filename, test_filename = 'wines.csv', 'wines_test.csv'
    # Get the training and test data from the given files
    reader = DataReader()
    #reader = DataReader(';')
    training_data_set = reader.read_csv(training_filename)
    training_data = training_data_set.data
    test_data_set = reader.read_csv(test_filename)
    test_data = test_data_set.data

    attributes = training_data_set.attributes
    target_attr = attributes[-1]
    
    # Create the decision tree
    tree_creator = DecisionTreeCreator(gain, max_depth=3)
    forest_creator = RandomForestCreator(tree_creator)
    forest = forest_creator.create_random_forest(10, training_data, attributes, target_attr)

    # Classify the records in the test data
    classification = forest.classify_many(test_data)

    # Print the results of the test
    print "------------------------\n"
    print "--   Classification   --\n"
    print "------------------------\n"
    print "\n"    
    for item in classification: print item
    
    # Print the contents of the decision tree
    print "\n"
    print "------------------------\n"
    print "--   Decision Tree    --\n"
    print "------------------------\n"
    print "\n"
    print forest
