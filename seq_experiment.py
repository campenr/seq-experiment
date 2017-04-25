import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap

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
        if isinstance(value, FeatureTable):
            self._feature_table = value
        else:
            raise(TypeError('feature_table should be of type FeatureTable'))

    def get_classification_table(self):
        return self._classification_table

    def set_classification_table(self, value):
        # enforce correct type for classification_table
        if isinstance(value, ClassificationTable):

            # check that classification_table matches the feature_table
            if value.index.tolist() != self._feature_table.index.tolist():
                raise IndexError('classification_table index does not match the feature_table index.')
            else:
                self._classification_table = value

        elif value is None:
            self._classification_table = None
        else:
            raise(TypeError('classification_table should be of type ClassificationTable or None'))

    def get_sample_data_table(self):
        return self._sample_data_table

    def set_sample_data_table(self, value):
        # enforce correct type for classification_table
        if isinstance(value, SampleDataTable):

            # check that classification_table matches the feature_table
            if value.index.tolist() != self._feature_table.columns.values.tolist():
                raise IndexError('sample_data_table index does not match the feature_table columns.')
            else:
                self._sample_data_table = value

        elif value is None:
            self._sample_data_table = None
        else:
            raise (TypeError('sample_data_table should be of type SampleDataTable or None'))

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

        return '\n'.join(outputs) + '\n'

    def __getitem__(self, item):
        """Subset the data contained within the SeqExp by columns or rows."""

        # TODO: implement indexing by single column i.e. seq_exp['class_0']. Normally this returns a Series,
        # TODO: but we do not have a Series implementation so need to handle this differently.

        # if the subset item is type `slice` then we are subsetting by classes in the feature table
        # if the subset item is type `list` then we are subsetting by features in the feature table

        print('__getitem__.item: ', item)
        print('type(__getitem__.item): ', type(item))

        # feature_table is always subset
        self.feature_table = self.feature_table[item]

        # if the subset is by rows then we need to also subset the classification_table if it exists
        if isinstance(item, slice):
            if self.classification_table is not None:
                self.classification_table = self.classification_table[item]

        # if the subset is by columns then we need to also subset the sample_data_table if it exists
        if isinstance(item, list):
            if self.sample_data_table is not None:
                self.sample_data_table = self.sample_data_table.ix[item]

        return self

    def plot_bar(self, **kwargs):
        """Plots bar chart using matplotlib."""

        # create custom cmap
        paired_cmap = get_cmap('Paired')

        paired_cols = []
        for i in paired_cmap.colors:
            rgbs = []
            for j in i:
                rgbs.append(j)

            paired_cols.append(rgbs)

        cust_cmap = LinearSegmentedColormap.from_list('custom_map', paired_cols)

        # specify our custom arguments for plotting bar charts
        default_args = {
            'stacked': True,
            'cmap': cust_cmap,
            'width': 0.8,
            'linewidth': 1,
            'edgecolor': 'black'
        }

        # override kind if specified by the user
        kwargs['kind'] = 'bar'

        # allow user defined kwargs to override defaults
        for arg, value in default_args.items():
            if arg not in kwargs.keys():
                kwargs[arg] = value

        # correct table orientation for plotting
        bar_plot = self.feature_table.transpose().plot(**kwargs)

        return bar_plot



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

    def __init__(self, *args, **kwargs):
        super(FeatureTable, self).__init__(*args, **kwargs)


class ClassificationTable(pd.DataFrame):
    """
    Classification table object.

    A classification table contains tabulated classifications of the features within a FeatureTable object. An example
    of this is taxanomic classifications of the OTUs in an OTU table.

    ..note:: Classifications can contain multiple ranks.

    """

    @property
    def _constructor(self):
        return ClassificationTable

    def __init__(self, *args, **kwargs):
        super(ClassificationTable, self).__init__(*args, **kwargs)


class SampleDataTable(pd.DataFrame):
    """
    Sample data table.

    A table of data that accompanies the classes (columns) in a FeatureTable object. For example, if the classes in the
    FeatureTable are different samples, the sample data may include different locations, geophysical parameters, etc
    for each sample.

    """

    @property
    def _constructor(self):
        return SampleDataTable

    def __init__(self, *args, **kwargs):
        super(SampleDataTable, self).__init__(*args, **kwargs)

