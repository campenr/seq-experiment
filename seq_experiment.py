

class SeqExp(object):
    """
    Main sequence experiment object.

    Container for the separate feature_table, classification_table, and sample_data_table records.

    """

    def __init__(self, feature_table, classification_table=None, sample_data_table=None):

        self.set_feature_table(feature_table)
        self.set_classification_table(classification_table)
        self.set_sample_data_table(sample_data_table)

    def get_feature_table(self):
        return self._feature_table

    def set_feature_table(self, value):
        self._feature_table = value

    def get_classification_table(self):
        return self._classification_table

    def set_classification_table(self, value):
        self._classification_table = value


    feature_table = property(fget=get_feature_table, fset=set_feature_table)
    classification_table = property(fget=get_classification_table, fset=set_classification_table)


class FeatureTable(object):
    """
    Feature table object.

    A feature table consists of counts of features per class. An example of this is the `sample x species` OTU table
    used to describe abundances across different samples.

    This object should not be manipulated directly, but rather as part of a SeqExp object.

    """


    def __init__(self):
        pass


class ClassificationTable(object):
    """
    Classification table object.

    A classification table contains tabulated classifications of the features within a FeatureTable object. An example
    of this is taxanomic classifications of the OTUs in an OTU table.

    ..note:: Classifications can contain multiple ranks.

    """

    def __init__(self):
        pass

class SampleDataTable(object):
    """
    Sample data table.

    A table of data that accompanies the classes (columns) in a FeatureTable object. For example, if the classes in the
    FeatureTable are different samples, the sample data may include different locations, geophysical parameters, etc
    for each sample.

    """

    def __init__(self):
        pass

