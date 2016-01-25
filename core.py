import collections


class RandomForestCreator(object):

    def __init__(self, tree_creator, num_of_trees=10, with_replacement=True, sample_size=0.6):
        """
        :type tree_creator: DecisionTreeCreator
        :type num_of_trees: int
        :type with_replacement: bool
        :type sample_size: float
        """
        self.num_of_trees = num_of_trees
        self.sample_size = sample_size
        self.with_replacement = with_replacement
        self.tree_creator = tree_creator

    def create_random_forest(self, data):
        """
        :type data: data.DataSet
        :return: RandomForest
        """
        forest = RandomForest()
        sample_size = int(len(data) * self.sample_size + 1)
        for i in xrange(self.num_of_trees):
            sample_data = data.get_random_samples(sample_size, self.with_replacement)
            tree = self.tree_creator.create_decision_tree(sample_data)
            forest.add_tree(tree)
        return forest


class RandomForest(object):

    def __init__(self):
        self.trees = []

    def add_tree(self, tree):
        """
        :type tree: DecisionTreeNode
        """
        self.trees.append(tree)

    def classify(self, record):
        """
        :type record: dict[str:str]
        :rtype: str
        """
        results = collections.defaultdict(lambda: 0)
        for tree in self.trees:
            single_result = tree.classify(record)
            results[single_result] += 1
        return max(results.keys(), key=lambda key: results[key])

    def classify_many(self, data):
        for record in data:
            yield self.classify(record)

    def __str__(self):
        return '\n'.join(['Tree number {0}: \n{1}\n'.format(i, tree) for i, tree in enumerate(self.trees)])


class DecisionTreeCreator(object):

    def __init__(self, fitness_func, max_depth=None):
        self.max_depth = max_depth
        self.fitness_func = fitness_func

    def create_decision_tree(self, data, _actual_depth=0):
        """
        Returns a new decision tree based on the examples given.
        :type data: data.DataSet
        :type _actual_depth: int
        """
        most_common_class = data.get_most_common_class()
        reached_max_depth = self.max_depth is not None and _actual_depth >= self.max_depth
        if not data or (len(data.attributes) - 1) <= 0 or reached_max_depth:
            return DecisionTreeNode(data.target_attr, value=most_common_class)
        elif len(data.get_unique_values_of_attribute(data.target_attr)) == 1:
            return DecisionTreeNode(data.target_attr, value=data.data[0][data.target_attr])
        else:
            best = data.get_best_attribute(self.fitness_func)
            tree = DecisionTreeNode(best, most_common_class)
            unique_values = data.get_unique_values_of_attribute(best)
            for val in unique_values:
                examples = data.get_all_with_given_attribute(best, val)
                subtree = self.create_decision_tree(examples, _actual_depth + 1)
                tree.add_child(subtree, val)
        return tree



class DecisionTreeNode(object):

    def __init__(self, attr_name, default=None, value=None):
        self.default = default
        self.attr_name = attr_name
        self.value = value
        self._children = {}

    def add_child(self, node, attr_value):
        """
        :type node: DecisionTreeNode
        :type attr_value: Any
        """
        assert node.value not in self._children, "Value {0} already in subtree".format(node.value)
        self._children[attr_value] = node

    def is_leaf(self):
        return len(self._children) == 0

    def classify(self, record):
        """
        :type record: dict[str:str]
        :rtype: str
        """
        if self.is_leaf():
            return self.value
        attr_value = record.get(self.attr_name)
        assert attr_value is not None
        child = self._children.get(attr_value)
        if child is None:
            return self.default
        return child.classify(record)

    def classify_many(self, data):
        return map(self.classify, data)

    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        children = ['--> {0}:\n\t{1}'.format(val, str(child).replace('\n', '\n\t'))
                    for val, child in self._children.iteritems()]
        return 'Attr: {0}\n{1}'.format(self.attr_name, '\n'.join(children))


class CrossValidation(object):
    def __init__(self, random_forest_creator, folds, verbosity_level=0):
        self.random_forest_creator = random_forest_creator
        self.folds = folds
        self.verbosity_level = verbosity_level

    def validate(self, data_set):
        """
        :param data_set: Data to perform cross validation
        :type data_set: data.DataSet
        :return: Result of using cross validation on given set
        :rtype: float
        """
        subset_size = len(data_set.data) / self.folds
        results = []
        for i in range(self.folds):
            test_set, train_set = data_set.split(i * subset_size, (i + 1) * subset_size)
            forest = self._train(train_set)
            results.append(self._test(test_set, forest))
            if self.verbosity_level >= 1:
                print 'Accuracy for fold no {0}: {1}'.format(i + 1, results[-1])
        return sum(results) / self.folds

    def _train(self, training_set):
        forest = self.random_forest_creator.create_random_forest(training_set)
        if self.verbosity_level >= 3:
            print forest
        return forest

    def _test(self, test_set, forest):
        classification = forest.classify_many(test_set.data)
        valid_counter = 0
        for i, res in enumerate(classification):
            expected = test_set.data[i][test_set.target_attr]
            valid_counter += int(expected == res)
            if self.verbosity_level >= 2:
                print 'result: {0:10}, expected {1:10} - {2}'.format(res, expected, expected == res)
        return valid_counter / float(len(test_set))
