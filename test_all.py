import os
from test import test

depth_values = [8, 12]
trees_values = [8, 12]
replacement_values = [True, False]
sample_size_values = [0.5, 0.7]
folds_values = [4, 8]
discretization_values = [None, 4]
sample_data_dir = './sample_data'
result_filename = 'results.csv'


def run_for_many(file_list):
    with open(result_filename, 'w') as result_file:
        result_file.write('filename, depth, trees, repl, sam_size, fold, dis, result\n')
        for filename in file_list:
            run_for_values_comb(filename, result_file)


def run_for_values_comb(filename, result_file):
    print 'Data set: {}\n'.format(filename)
    for depth, trees, repl, sam_size, fold, dis in [(depth, trees, repl, sam_size, fold, dis)
                                                    for depth in depth_values
                                                    for trees in trees_values
                                                    for repl in replacement_values
                                                    for sam_size in sample_size_values
                                                    for fold in folds_values
                                                    for dis in discretization_values]:
        result = test(os.path.join(sample_data_dir, filename), depth, trees, repl, sam_size, fold, 0, dis)
        print '{} for: Depth: {}, tree no: {}, replacement: {}, sample size: {}, folds no: {}, discretization: {}' \
            .format(result, depth, trees, repl, sam_size, fold, dis)
        result_file.write('{},{},{},{},{},{},{},{}\n'
                   .format(filename, depth, trees, repl, sam_size, fold, dis, result))


if __name__ == "__main__":
    run_for_many(os.listdir(sample_data_dir))
