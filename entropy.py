import math


def entropy(data):
    """ Calculates the entropy of the given data set for the target attribute.
    :type data: data.DataSet
    :rtype: float
    """
    val_freq = data.get_values_frequency(data.target_attr)

    # Calculate and return entropy of the data for the target attribute
    data_entropy = 0.0
    for freq in val_freq.values():
        data_entropy += (-freq * 1.0/len(data)) * math.log(freq * 1.0/len(data), 2)
    return data_entropy


def gain(data, attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    :type data: data.DataSet
    :rtype: float
    """
    subset_entropy = 0.0
    val_freq = data.get_values_frequency(attr)

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] * 1.0 / len(data)
        data_subset = data.get_all_with_given_attribute(attr, val)
        subset_entropy += val_prob * entropy(data_subset)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return entropy(data) - subset_entropy

