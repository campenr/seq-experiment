import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap

from ordination import MDS, NMDS
from scipy.spatial.distance import pdist, squareform

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

            # check that sample_data_table matches the feature_table
            if value.index.tolist() != self._feature_table.columns.values.tolist():
                raise IndexError('sample_data_table index does not match the feature_table columns.')
            else:
                self._sample_data_table = value

        elif value is None:
            self._sample_data_table = None
        else:
            raise (TypeError('sample_data_table should be of type SampleDataTable or None'))

    @property
    def sample_names(self):
        return self.feature_table.columns.tolist()

    @property
    def feature_names(self):
        return self.feature_table.index.tolist()

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
        # if the subset item is type `pd.Series` then we are subsetting by either classes or features

        # print('__getitem__.item: ', item)
        # print('type(__getitem__.item): ', type(item))

        # feature_table is always subset
        new_feature_table = self.feature_table[item]

        # construct new object
        new_seq_exp = SeqExp(new_feature_table)

        # conditional subsetting of classification_table depending on the subsetting items type
        if self.classification_table is not None:
            # TODO need to also allow subsetting using a Series or DataFrame index, not just lists
            if isinstance(item, slice):
                new_seq_exp.classification_table = self.classification_table[item]
            elif isinstance(item, pd.Series):
                # only subset the classification_table if the index's match
                # TODO: this is hacky. Would be safer to overide some of the pd.DataFrame methods / operators
                if self.classification_table.index.tolist() == item.index.tolist():
                    # print('will trim classification_table')
                    new_seq_exp.classification_table = self.classification_table[item]
                else:
                    # print('wont trim classification_table')
                    new_seq_exp.classification_table = self.classification_table
            else:
                new_seq_exp.classification_table = self.classification_table

        # conditional subsetting of sample_data_table depending on the subsetting items type
        if self.sample_data_table is not None:
            # TODO need to also allow subsetting using a Series or DataFrame index, not just lists
            if isinstance(item, list):
                new_seq_exp.sample_data_table = self.sample_data_table.ix[item]
            elif isinstance(item, pd.Series):
                # only subset the sample_data_table if the index's match
                # TODO: this is hacky. Would be safer to overide some of the pd.DataFrame methods / operators
                if self.sample_data_table.index.tolist() == item.index.tolist():
                    # print('will trim sample_data_table')
                    new_seq_exp.sample_data_table = self.sample_data_table[item]
                else:
                    # print('wont trim sample_data_table')
                    new_seq_exp.sample_data_table = self.sample_data_table
            else:
                new_seq_exp.sample_data_table = self.sample_data_table

        return new_seq_exp

    def relabund(self):
        """
        Returns a new object with abundances converted to relative abundances.
        
        ..note:: This method leaves the original object intact, returning a new modified copy.
        
        """

        new_feature_table = self.feature_table
        class_sums = new_feature_table.sum(axis=0)
        new_feature_table = new_feature_table.div(class_sums)

        # return a new object
        new_seq_exp = SeqExp(
            feature_table=new_feature_table,
            classification_table=self.classification_table,
            sample_data_table=self.sample_data_table
        )

        return new_seq_exp

    def groupby(self, rank=None, ):
        """
        Returns a new object with features grouped by a classification rank or a sample data name.
        
        ..note:: This method leaves the original obejct intact, return a new modified copy.        
        
        """

        pass

    def drop(self, labels, from_='features'):
        """
        Drop either features or samples from the SeqExp object.
        
        :param labels: labels to drop from the SeqExp object
        Ltype items: str or list(str)
        :param from_: whether to drop the supplied labels from the features or samples
        :type from_: str
               
        :return: a new SeqExp object
         
        ..note:: can specify either the label of a single feature/sample to drop, or a list of labels to drop.
        ..seealso:: SeqExp.__getitem__()
                 
        """

        # check for valid `from` argument
        if from_ not in ['features', 'samples']:
            raise(ValueError('from_ must be one either \'features\' or \'samples\'.'))

        # create new SeqExp object with subset feature_table
        new_feature_table = self.feature_table
        if from_ == 'features':
            new_feature_table = new_feature_table.drop(labels, axis=0)
        elif from_ == 'samples':
            new_feature_table = new_feature_table.drop(labels, axis=1)
        new_seq_exp = SeqExp(new_feature_table)

        # use the merge function to do the subsetting on the classification and sample data tables if they exist
        if self.classification_table is not None:
            new_seq_exp = new_seq_exp.merge(self.classification_table)
        if self.sample_data_table is not None:
            new_seq_exp = new_seq_exp.merge(self.sample_data_table)

        return new_seq_exp

    def merge(self, right):
        """
        Merges this SeqExp with another SeqExp or SeqExp component.
        
        Provides similar functionality to the pandas DataFrame.merge or phyloseq's merge_phyloseq.
        
        This method takes the input SeqExp or components thereof and returns the features, classifications, and sample
        data that matches across all the supplied objects.
        
        :param right: SeqExp, FeatureTable, ClassificationTable, or SampleDataTable to merge with
        :type right: SeqExp, FeatureTable, ClassificationTable, or SampleDataTable
        
        ..note:: can accept either another SeqExp object, or other feature_table, classification_table, or 
        sample_data_table objects.
        ..seealso:: to create a SeqExp record from only the component parts using the same process use 
        `seq_experiment.concat`.
        
        :return: a new SeqExp object
        
        """

        new_feature_table = self.feature_table

        if isinstance(right, ClassificationTable):
            try:
                # get the intersection of this objects index and the supplied classification_table's index

                new_index = self.feature_table.index.intersection(right.index)

                # subset based on new index and use to return new SeqExp
                new_feature_table = new_feature_table.ix[new_index]

                # print(new_feature_table)

                new_classificaiton_table = right.ix[new_index]

                new_seq_experiment = SeqExp(new_feature_table, new_classificaiton_table, self.sample_data_table)
                return new_seq_experiment

            except Exception:
                raise

        elif isinstance(right, SampleDataTable):
            try:

                # print(self.feature_table.columns)
                # print(right.index)

                # get the intersection of this objects columns and the supplied sample_data_table's columns
                new_columns = self.feature_table.columns.intersection(right.index)
                # print('new_columsn: ', new_columns)

                # subset based on new index and use to return new SeqExp
                new_feature_table = new_feature_table[new_columns]

                # print(new_feature_table)

                new_sample_data_table = right.ix[new_columns]

                new_seq_experiment = SeqExp(new_feature_table, self.classification_table, new_sample_data_table)
                return new_seq_experiment

            except Exception:
                raise

        elif isinstance(right, type(None)):
            return self

        else:
            raise(ValueError('invalid type for \'right\' argument'))

    def distance(self):
        """      
        
        :return: 
        """

        pass


    def ordinate(self, method, distance=None, metric=None, *args, **kwargs):
        """
        Performs an ordination on the SeqExp object using.
        
        Uses either a precomputed distance/dissimilarity matrix, or computes one for the user if supplied with a valid
        mdistance metric.
        
        :param method: the ordination method to use
        :type method: str
        :param distance: distance or dissimilarity matrix to ordinate
        :type distance: (optional) 2d np.array like or None
        :param metric: (optional) metric to use for computing a distance matrix
        :type metric: str

        :return: 
        
        ..note:: User must supply either a precomputed distance matrix or a metric to compute a distance matrix with,
            but not both. If a metric is supplied it must be a valid metric for `scipy.spatial.distance.pdist`.
        
        ..see also:: scipy.spatial.distance
                
        """

        ord_methods = {
            'MDS': MDS,
            'NMDS': NMDS
        }
        dist_metrics = ['braycurtis']

        has_distance = distance is not None
        has_metric = metric is not None

        has_distance_or_metric = has_distance or has_metric
        has_distance_and_metric = has_distance and has_metric

        # enforce only distance or metric, but not both
        if not (has_distance_or_metric and not has_distance_and_metric):
            raise(ValueError('must supply either distance or metric, and not both.'))

        # check for valid method
        if method in ord_methods:

            if not has_distance:
                if metric in dist_metrics:
                    # need to transpose feature table to get it in the correct orientation for pdist
                    distance = squareform(pdist(self.feature_table.transpose(), metric=metric))
                else:
                    raise(ValueError('must supply a valid metric for calculating distances/dissimilarities.'))

                # perform ordination with specified method, passing additional arguments to the ordination functions
                ord = ord_methods[method](dissimilarity_mtx=distance, *args, *kwargs)
                return ord
        else:
            raise(ValueError('must supply a valid ordination method.'))

    def plot_bar(self, **kwargs):
        """Plots bar chart using matplotlib."""

        # # try importing seaborn to make prettier graphs
        # try:
        #     import seaborn as sns
        #
        # except ImportError:
        #     pass

        # make pretty
        plt.style.use('seaborn-white')

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
            'edgecolor': 'black',
        }

        # override kind if specified by the user
        kwargs['kind'] = 'bar'

        # TODO: allow for plotting subplots in a way that visually makes more sense

        # allow user defined kwargs to override our defaults
        for arg, value in default_args.items():
            if arg not in kwargs.keys():
                kwargs[arg] = value

        ax = self.feature_table.transpose().plot(**kwargs)

        return ax



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

    @property
    def _constructor_sliced(self):
        return FeatureSeries

    def __init__(self, *args, **kwargs):
        super(FeatureTable, self).__init__(*args, **kwargs)


class FeatureSeries(pd.Series):

    @property
    def _constructor(self):
        return FeatureSeries

    @property
    def _constructor_expanddim(self):
        return FeatureTable

    def __init__(self, *args, **kwargs):
        super(FeatureSeries, self).__init__(*args, **kwargs)


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

