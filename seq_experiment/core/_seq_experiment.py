import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap

from collections import OrderedDict

from seq_experiment.ordination import pcoa, nmds, meta_nmds
from seq_experiment.distance import DistanceMatrix
from scipy.spatial.distance import pdist, squareform

from seq_experiment.core import FeatureTable, ClassificationTable, MetadataTable

class SeqExp(object):
    """
    Main sequence experiment object.

    Container for the separate feature_table, classification_table, and metadata_table records.

    """

    def __init__(self, feature_table, classification_table=None, metadata_table=None):

        self.set_feature_table(feature_table)
        self.set_classification_table(classification_table)
        self.set_metadata_table(metadata_table)

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

    def get_metadata_table(self):
        return self._metadata_table

    def set_metadata_table(self, value):
        # enforce correct type for metadata_table
        if isinstance(value, MetadataTable):

            # check that metadata_table matches the feature_table
            if value.index.tolist() != self._feature_table.columns.values.tolist():
                raise IndexError('metadata_table index does not match the feature_table columns.')
            else:
                self._metadata_table = value

        elif value is None:
            self._metadata_table = None
        else:
            raise (TypeError('metadata_table should be of type MetadataTable or None'))

    @property
    def sample_names(self):
        return self.feature_table.columns.tolist()

    @property
    def feature_names(self):
        return self.feature_table.index.tolist()

    # configure properties
    feature_table = property(fget=get_feature_table, fset=set_feature_table)
    classification_table = property(fget=get_classification_table, fset=set_classification_table)
    metadata_table = property(fget=get_metadata_table, fset=set_metadata_table)

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

        if self._metadata_table is not None:
            metadata_summary = 'metadata_table:\t{classes} classes x {metadata} sample data'.format(
                classes=len(self._metadata_table.index),
                metadata=len(self._metadata_table.columns)
            )
        else:
            metadata_summary = None

        outputs = [feature_summary]
        for i in [classification_summary, metadata_summary]:
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

        # conditional subsetting of metadata_table depending on the subsetting items type
        if self.metadata_table is not None:
            # TODO need to also allow subsetting using a Series or DataFrame index, not just lists
            if isinstance(item, list):
                new_seq_exp.metadata_table = self.metadata_table.loc[item]
            elif isinstance(item, pd.Series):
                # only subset the metadata_table if the index's match
                # TODO: this is hacky. Would be safer to overide some of the pd.DataFrame methods / operators
                if self.metadata_table.index.tolist() == item.index.tolist():
                    # print('will trim metadata_table')
                    new_seq_exp.metadata_table = self.metadata_table[item]
                else:
                    # print('wont trim metadata_table')
                    new_seq_exp.metadata_table = self.metadata_table
            else:
                new_seq_exp.metadata_table = self.metadata_table

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
            metadata_table=self.metadata_table
        )

        return new_seq_exp

    def subset_samples(self, sample_names):

        new_feature_table = self.feature_table.loc[:, sample_names]
        new_sxp = SeqExp(feature_table=new_feature_table)
        if self.classification_table is not None:
            new_sxp = new_sxp.merge(self.classification_table)
        if self.metadata_table is not None:
            new_sxp = new_sxp.merge(self.metadata_table)

        return new_sxp

    def subset_features(self, feature_names):

        new_feature_table = self.feature_table.loc[feature_names]
        new_sxp = SeqExp(feature_table=new_feature_table)
        if self.classification_table is not None:
            new_sxp = new_sxp.merge(self.classification_table)
        if self.metadata_table is not None:
            new_sxp = new_sxp.merge(self.metadata_table)

        return new_sxp

    def drop_samples(self, sample_names):

        new_feature_table = self.feature_table.drop(sample_names, axis=1)
        new_sxp = SeqExp(feature_table=new_feature_table)
        if self.classification_table is not None:
            new_sxp = new_sxp.merge(self.classification_table)
        if self.metadata_table is not None:
            new_sxp = new_sxp.merge(self.metadata_table)

        return new_sxp

    def to_mothur_shared(self, out_file):
        """Exports FeatureTable to a Mothur shared file."""

        shared = self.feature_table
        shared = shared.transpose()
        shared = shared.reset_index()
        shared['label'] = self.label
        shared['numOtus'] = len(self.feature_table)
        new_columns = ['label', 'Group', 'numOtus', *self.feature_table.index]
        shared = shared[new_columns]

        shared.to_csv(out_file, sep='\t', header=True, index=False)

        return shared

    def groupby_classification(self, level):
        """Group the SeqExp features by a classificaiton level."""

        # create table that combines features with classification at specified level
        combined = pd.DataFrame(pd.concat([self.feature_table, self.classification_table[level]], axis=1))
        combined = combined.groupby(level).sum()
        combined.columns.name = 'Group'

        # create new SeqExp object from grouped data
        new_feature_table = FeatureTable(combined)
        new_sxp = SeqExp(new_feature_table)

        # only retain classification levels higher than the one used to group
        new_classification_table = self.classification_table.loc[:, :level]
        new_classification_table = new_classification_table.groupby(level).first()
        new_classification_table = ClassificationTable(new_classification_table)

        new_sxp = new_sxp.merge(new_classification_table)

        if self.metadata_table is not None:
            new_sxp.metadata_table = self.metadata_table

        new_sxp.label = level

        return new_sxp

    def grouby_metadata(self, data_label):
        """Groups samples according to their metadata."""

        label_set = set(self.metadata_table[data_label])

        new_feats_dict = OrderedDict()
        for label in label_set:
            classes = (self.metadata_table[self.metadata_table[data_label] == label]).index
            features = self.feature_table[classes]
            feature_means = features.mean(axis=1)

            new_feats_dict[label] = feature_means

        new_feats_df = pd.concat(new_feats_dict, axis=1)

        return new_feats_df

    # def drop(self, labels, from_='features'):
    #     """
    #     Drop either features or samples from the SeqExp object.
    #
    #     :param labels: labels to drop from the SeqExp object
    #     Ltype items: str or list(str)
    #     :param from_: whether to drop the supplied labels from the features or samples
    #     :type from_: str
    #
    #     :return: a new SeqExp object
    #
    #     ..note:: can specify either the label of a single feature/sample to drop, or a list of labels to drop.
    #     ..seealso:: SeqExp.__getitem__()
    #
    #     """
    #
    #     # check for valid `from` argument
    #     if from_ not in ['features', 'samples']:
    #         raise(ValueError('from_ must be one either \'features\' or \'samples\'.'))
    #
    #     # create new SeqExp object with subset feature_table
    #     new_feature_table = self.feature_table
    #     if from_ == 'features':
    #         new_feature_table = new_feature_table.drop(labels, axis=0)
    #     elif from_ == 'samples':
    #         new_feature_table = new_feature_table.drop(labels, axis=1)
    #     new_seq_exp = SeqExp(new_feature_table)
    #
    #     # use the merge function to do the subsetting on the classification and sample data tables if they exist
    #     if self.classification_table is not None:
    #         new_seq_exp = new_seq_exp.merge(self.classification_table)
    #     if self.metadata_table is not None:
    #         new_seq_exp = new_seq_exp.merge(self.metadata_table)
    #
    #     return new_seq_exp

    def merge(self, right):
        """
        Merges this SeqExp with another SeqExp or SeqExp component.
        
        Provides similar functionality to the pandas DataFrame.merge or phyloseq's merge_phyloseq.
        
        This method takes the input SeqExp or components thereof and returns the features, classifications, and sample
        data that matches across all the supplied objects.
        
        :param right: SeqExp, FeatureTable, ClassificationTable, or SampleDataTable to merge with
        :type right: SeqExp, FeatureTable, ClassificationTable, or SampleDataTable
        
        ..note:: can accept either another SeqExp object, or other feature_table, classification_table, or 
        metadata_table objects.
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
                new_feature_table = new_feature_table.loc[new_index]

                # print(new_feature_table)

                new_classificaiton_table = right.loc[new_index]

                new_seq_experiment = SeqExp(new_feature_table, new_classificaiton_table, self.metadata_table)
                return new_seq_experiment

            except Exception:
                raise

        elif isinstance(right, MetadataTable):
            try:

                # print(self.feature_table.columns)
                # print(right.index)

                # get the intersection of this objects columns and the supplied metadata_table's columns
                new_columns = self.feature_table.columns.intersection(right.index)
                # print('new_columsn: ', new_columns)

                # subset based on new index and use to return new SeqExp
                new_feature_table = new_feature_table[new_columns]

                # print(new_feature_table)

                new_metadata_table = right.loc[new_columns]

                new_seq_experiment = SeqExp(new_feature_table, self.classification_table, new_metadata_table)
                return new_seq_experiment

            except Exception:
                raise

        elif isinstance(right, type(None)):
            return self

        else:
            raise(ValueError('invalid type for \'right\' argument'))

    def distance(self, metric='braycurtis'):
        """      
        
        :return: 
        """

        dist_metrics = ['braycurtis']

        if metric in dist_metrics:
            distance = squareform(pdist(self.feature_table.transpose(), metric=metric))
        else:
            raise(ValueError('must supply a valid distance or dissimilarity metric.'))

        # format results in pd.DataFrame
        dist_df = pd.DataFrame(distance)
        dist_df.index = self.sample_names
        dist_df.columns = self.sample_names

        # return as DistanceMatrix object
        return DistanceMatrix(dist_df, metric=metric)


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
            'pcoa': pcoa,
            'nmds': meta_nmds
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
        if method.lower() in ord_methods:

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

    @staticmethod
    def import_mothur(mothur_shared_file, mothur_constaxonomy_file=None):
        """
        Creates SeqExp object from mothur output files.

        :param mothur_shared_file:
        :param mothur_constaxonomy_file:
        :return:
        """

        feature_data = FeatureTable.read_mothur_shared_file(mothur_shared_file)
        sxp = SeqExp(feature_data)

        if mothur_constaxonomy_file is not None:
            classification_data = ClassificationTable.read_mothur_constaxonomy_file(mothur_constaxonomy_file)
            sxp = sxp.merge(classification_data)

        return sxp

#
# class FeatureTable(pd.DataFrame):
#     """
#     Feature table object.
#
#     A feature table consists of counts of features per class. An example of this is the `sample x species` OTU table
#     used to describe abundances across different samples.
#
#     This object should not be manipulated directly, but rather as part of a SeqExp object.
#
#     """
#
#     @property
#     def _constructor(self):
#         return FeatureTable
#
#     @property
#     def _constructor_sliced(self):
#         return FeatureSeries
#
#     def __init__(self, *args, **kwargs):
#         super(FeatureTable, self).__init__(*args, **kwargs)
#
#     @staticmethod
#     def read_mothur_shared_file(shared_file):
#         """Reads in and formats a Mothur shared file."""
#
#         feature_data = pd.read_table(shared_file)
#         feature_data = feature_data.drop(['label', 'numOtus'], axis=1)
#         feature_data = feature_data.set_index('Group').transpose()
#
#         return FeatureTable(feature_data)
#
#
# class FeatureSeries(pd.Series):
#
#     @property
#     def _constructor(self):
#         return FeatureSeries
#
#     @property
#     def _constructor_expanddim(self):
#         return FeatureTable
#
#     def __init__(self, *args, **kwargs):
#         super(FeatureSeries, self).__init__(*args, **kwargs)
#
#
# class ClassificationTable(pd.DataFrame):
#     """
#     Classification table object.
#
#     A classification table contains tabulated classifications of the features within a FeatureTable object. An example
#     of this is taxanomic classifications of the OTUs in an OTU table.
#
#     ..note:: Classifications can contain multiple ranks.
#
#     """
#
#     @property
#     def _constructor(self):
#         return ClassificationTable
#
#     def __init__(self, *args, **kwargs):
#         super(ClassificationTable, self).__init__(*args, **kwargs)
#
#     @staticmethod
#     def read_mothur_constaxonomy_file(constaxonomy_file):
#         """Reads in and formats a Mother constaxonomy file."""
#
#         classification_data = pd.read_table(constaxonomy_file)
#         classifications = classification_data['Taxonomy']
#         classifications = classifications.str.split(';', expand=True).drop(6, axis=1)
#         classifications.columns = list(range(1, 7))
#         features = classification_data['OTU']
#         classification_data = pd.concat([features, classifications], axis=1)
#         classification_data = classification_data.set_index('OTU')
#
#         return ClassificationTable(classification_data)
#
#
# class SampleDataTable(pd.DataFrame):
#     """
#     Sample data table.
#
#     A table of data that accompanies the classes (columns) in a FeatureTable object. For example, if the classes in the
#     FeatureTable are different samples, the sample data may include different locations, geophysical parameters, etc
#     for each sample.
#
#     """
#
#     @property
#     def _constructor(self):
#         return SampleDataTable
#
#     def __init__(self, *args, **kwargs):
#         super(SampleDataTable, self).__init__(*args, **kwargs)
#
