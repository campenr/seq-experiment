import numpy as np
import pandas as pd

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
        # enforce correct type for feature_table
        if isinstance(value, pd.DataFrame):
            self._feature_table = value
        else:
            raise(TypeError('feature_table should be of type pd.DataFrame'))

    def get_classification_table(self):
        return self._classification_table

    def set_classification_table(self, value):
        # enforce correct type for classification_table
        if isinstance(value, pd.DataFrame):

            # check that classification_table matches the feature_table
            if value.index.tolist() != self._feature_table.index.tolist():
                raise IndexError('classification_table index does not match the feature_table index.')
            else:
                self._classification_table = value

        elif value is None:
            self._classification_table = None
        else:
            raise(TypeError('classification_table should be of type pd.DataFrame or None'))

    def get_sample_data_table(self):
        return self._sample_data

    def set_sample_data_table(self, value):
        # enforce correct type for classification_table
        if isinstance(value, pd.DataFrame):

            # check that classification_table matches the feature_table
            if value.index.tolist() != self._feature_table.columns.values.tolist():
                raise IndexError('sample_data_table index does not match the feature_table columns.')
            else:
                self._sample_data_table = value

        elif value is None:
            self._sample_data_table = None
        else:
            raise (TypeError('sample_data_table should be of type pd.DataFrame or None'))

    # configure properties
    feature_table = property(fget=get_feature_table, fset=set_feature_table)
    classification_table = property(fget=get_classification_table, fset=set_classification_table)
    sample_data_table = property(fget=get_sample_data_table, fset=set_sample_data_table)

    def __str__(self):
        """."""

        feature_summary = 'feature_table:\t{features} features x {classes} classes'.format(
            features=len(self._feature_table.index),
            classes=len(self._feature_table.columns)
        )

        if self._classification_table is not None:
            classification_summary = 'classification_table:\t{features} features x {ranks} classification ranks'.format(
                features=len(self._classification_table.index),
                ranks=len(self._classification_table.columns)
            )
        else:
            classification_summary = None

        if self._sample_data_table is not None:
            sample_data_summary = 'sample_data_table:\t{classes} classes x {sample_data} sample data'.format(
                classes=len(self._sample_data_table.index),
                sample_data=len(self._sample_data_table.columns)
            )
        else:
            sample_data_summary = None

        outputs = [feature_summary]
        for i in [classification_summary, sample_data_summary]:
            if i is not None:
                outputs.append(i)

        return '\n'.join(outputs)


class FeatureTable(pd.DataFrame):
    """
    Feature table object.

    A feature table consists of counts of features per class. An example of this is the `sample x species` OTU table
    used to describe abundances across different samples.

    This object should not be manipulated directly, but rather as part of a SeqExp object.

    """

    @property
    def _constructor(self):
        return FeatureTable

    def __init__(self, x=True, *args, **kwargs):
        super(FeatureTable, self).__init__(*args, **kwargs)
        print(x)



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

