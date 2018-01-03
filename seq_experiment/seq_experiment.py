"""
Copyright (c) 2018, Richard Campen
All rights reserved.
Licensed under the Modified BSD License.
For full license terms see LICENSE.txt

"""

from collections import OrderedDict
from copy import deepcopy
import functools
import pandas as pd

from seq_experiment.indexing import get_indexer_mappings, _Indexer
from seq_experiment.plotting import plot_abundance


class SeqExp(object):
    """
    Main sequence experiment object.

    Container for the separate data frames containing the matching features, classifications, and metadata records.

    Only the features table is required to create a new SeqExp object.

    """

    def __init__(self, features, classifications=None, metadata=None, seqs=None):

        self._features = None
        self._classifications = None
        self._metadata = None
        self._seqs = None

        self.features = features
        self.classifications = classifications
        self.metadata = metadata
        self.seqs = seqs

    # -------------- basic getters -------------- #

    @property
    def features(self):
        return self._features

    @property
    def classifications(self):
        return self._classifications

    @property
    def metadata(self):
        return self._metadata

    @property
    def seqs(self):
        return self._seqs

    def _get_components(self):

        return {
            'features': self.features,
            'classifications': self.classifications,
            'metadata': self.metadata,
            'seqs': self.seqs
        }

    # -------------- basic setters -------------- #

    @features.setter
    def features(self, features):
        """Checks that if a feature table already exists the index and columns remain the same."""

        if self._features is not None:
            if not self._features.index.equals(features.index) and self._features.columns.equals(features.columns):
                raise KeyError('new features index and columns must match the existing features')

        self._features = features

    @classifications.setter
    def classifications(self, classifications):
        """Checks that the classification data matches the existing feature data before setting."""

        if classifications is not None:
            if not classifications.index.equals(self.features.index):
                raise KeyError('classifications index does not match the features index.')

        self._classifications = classifications

    @metadata.setter
    def metadata(self, metadata):
        """Checks that the metadata matches the existing feature data before setting."""

        if metadata is not None:
            if not metadata.index.equals(self.features.columns):
                raise KeyError('metadata index does not match the features columns.')

        self._metadata = metadata

    @seqs.setter
    def seqs(self, seqs):
        """Checks that the metadata matches the existing feature data before setting."""

        if seqs is not None:
            if not seqs.index.equals(self.features.index):
                raise KeyError('seqs index does not match the features index.')

        self._seqs = seqs

    # -------------- convenience getters/setters -------------- #

    @property
    def feature_names(self):
        return self.features.index

    @feature_names.setter
    def feature_names(self, feature_names):

        self.features.index = feature_names

        if self.classifications is not None:
            self.classifications.index = feature_names
        if self.seqs is not None:
            self.seqs.index = feature_names

    @property
    def sample_names(self):
        return self.features.columns

    @sample_names.setter
    def sample_names(self, sample_names):

        self.features.columns = sample_names

        if self.metadata is not None:
            self.metadata.index = sample_names

    # -------------- _ -------------- #

    def __str__(self):
        """."""

        feature_summary = 'features:\t{features} features x {classes} classes'.format(
            features=len(self.features.index),
            classes=len(self.features.columns)
        )

        if self.classifications is not None:
            classification_summary = 'classifications:\t{features} features x {ranks} classification ranks'.format(
                features=len(self.classifications.index),
                ranks=len(self.classifications.columns)
            )
        else:
            classification_summary = None

        if self.metadata is not None:
            metadata_summary = 'metadata:\t{classes} classes x {metadata} sample data'.format(
                classes=len(self.metadata.index),
                metadata=len(self.metadata.columns)
            )
        else:
            metadata_summary = None

        if self.seqs is not None:
            seqs_summary = 'seqs:\t{features} features x {seqs} seqs'.format(
                features=len(self.seqs.index),
                seqs=len(self.seqs.columns)
            )
        else:
            seqs_summary = None

        outputs = [feature_summary]
        for i in [classification_summary, metadata_summary, seqs_summary]:
            if i is not None:
                outputs.append(i)

        return '\n'.join(outputs) + '\n'

    # -------------- subsetting -------------- #

    def __getitem__(self, key):
        """
        Subsets the data contained within the SeqExp by columns in the features dataframe.

        ..see also:: for more advanced subsetting based on the contents of each separate dataframe attribute, and to
            subset the features dataframe by features rather than by samples use `sxp.fx`, `sxp.cx`, `sxp.mx`, and
            `sxp.sx` for advanced subsetting by the features, classifications, metadata, and sequences dataframes
            respectively.

        """

        # features are always subset when using SeqExp.__getitem__
        new_features = self.features[key]

        # conditionally correct type of attribute if dimensionality has been reduced during subset
        # assumes attr is pd.DataFrame or pd.DataFrame like object, with _constructor_sliced method implemented
        if isinstance(new_features, type(self.features)):
            pass
        elif isinstance(new_features, self.features._constructor_sliced):
            print('here')
            new_features = pd.DataFrame(new_features)
        else:
            new_features = pd.DataFrame([new_features], index=[key[0]], columns=[key[1]])

        # may need to correct data orientation
        if not new_features.index.isin(self.features.index).all():
            new_features = new_features.transpose()

        # merge the new features back into the SeqExp object, sorting by the order of the new features dataframe
        new_sxp = self.merge(right=new_features, component='features', sort_by='right')

        return new_sxp

    @classmethod
    def _create_indexer(cls, name, indexer):
        """Create an indexer like _name in the class."""
        if getattr(cls, name, None) is None:
            _indexer = functools.partial(indexer, name)
            setattr(cls, name, property(_indexer))  # , doc=indexer.__doc__))

    def drop(self, by, items):
        """
        Drops features or samples from SeqExp.
        
        :param by: whether to drop from features or samples
        :type by: str, one of either `features` or `samples`
        :param items: list of features or samples to drop
        :type items: list
        
        :return: new SeqExp object with items dropped from either features or samples
         
        ..note:: any features whose sum across all samples becomes zero following subsetting are removed
        
        """

        # subset the feature table in the correct dimensions
        if by == 'features':
            new_features = self.features.drop(items, axis=0)
        elif by == 'samples':
            new_features = self.features.drop(items, axis=1)
        else:
            raise(ValueError('by should be one of \'features\' or \'samples\''))

        if self.classifications is not None:
            new_classifications = self.classifications.loc[new_features.index]
        else:
            new_classifications = None

        if self.metadata is not None:
            new_metadata = self.metadata.loc[new_features.columns]
        else:
            new_metadata = None

        return SeqExp(features=new_features, classifications=new_classifications, metadata=new_metadata)

    def groupby_classification(self, level):
        """Group the SeqExp features by a classificaiton."""

        # create table that combines features with classification at specified level
        combined = pd.DataFrame(pd.concat([self.features, self.classifications[level]], axis=1))
        combined = combined.groupby(level).sum()
        combined.columns.name = 'Group'

        # create new SeqExp object from grouped data
        new_feature_table = combined
        new_sxp = SeqExp(new_feature_table)

        # only retain classification levels higher than the one used to group
        new_classification_table = self.classifications.loc[:, :level]
        new_classification_table = new_classification_table.groupby(level).first()
        new_classification_table = new_classification_table

        new_sxp = new_sxp.merge(new_classification_table, component='classifications')

        if self.metadata is not None:
            new_sxp.metadata = self.metadata

        new_sxp.label = level

        return new_sxp

    def groupby_metadata(self, data_label, func):
        """Groups samples according to their metadata using the specified function."""

        label_set = set(self.metadata[data_label])

        new_feats_dict = OrderedDict()
        for label in label_set:
            classes = (self.metadata[self.metadata[data_label] == label]).index
            features = self.features[classes]
            feature_means = features.apply(func, axis=1)

            new_feats_dict[label] = feature_means

        new_feats_df = pd.concat(new_feats_dict, axis=1)

        return new_feats_df

    def merge(self, right, component=None, sort_by='left'):
        """
        Merges this SeqExp with another SeqExp or SeqExp component.

        Provides similar functionality to phyloseq's `merge_phyloseq`.

        This method takes the input SeqExp or components thereof and returns and returns a new SeqExp. When merged, the
        index and columns of the component specified by the `right` argument are used to subset any components of the
        `left` SeqExp appropriately.

        :param right: SeqExp or component data frame to merge with this SeqExp object
        :type right: SeqExp, pd.DataFrame, or pd.DataFrame like object
        :param component: What component the `argument` represents
        :type component: str, one of 'features', 'classifications', or 'metadata'

        ..seealso:: to create a SeqExp record from only the component parts using the same process use
        `seq_experiment.concat`.

        :return: a new SeqExp object

        """

        # check for merging two SeqExp object first
        if (type(right) == type(self)) and (component is not None):
            raise ValueError('can not set component if merging two SeqExp objects')
        elif (type(right) == type(self)) and (component is None):
            # recursively merge in each attribute of the SeqExp to be then return new SeqExp
            new_sxp = deepcopy(self)
            for attr_name in ['features', 'classifications', 'metadata', 'seqs']:
                attr = getattr(right, attr_name, None)
                if attr is not None:
                    new_sxp = new_sxp.merge(right=attr, component=attr_name, sort_by=sort_by)
            return new_sxp

        # merging a component into a SeqExp object
        component = component.lower()
        if component not in self._get_components():
            raise ValueError('invalid \'component\'. Must be one of `features`, `classifications`, `metadata`, or '
                             '`seqs`')
        sort_by = sort_by.lower()
        if sort_by.lower() not in ['left', 'right']:
            raise ValueError('invalid \'sort_by\'. Must be one of `left` or `right`')

        # get left side for the merge
        left = getattr(self, component)

        # get new indexes and columns that will be sorted by, as well as the indexes/columns of the existing
        # component if available
        if left is None:
            # if component is not already present then we must sort by the new component being merged in
            index, columns = right.index, right.columns
            other_index, other_columns = None, None  # to stop the IDE complaining
        elif sort_by == 'right':
            index, columns = right.index, right.columns
            other_index, other_columns = left.index, left.columns
        else:
            index, columns = left.index, left.columns
            other_index, other_columns = right.index, right.columns

        # TODO: allow other options for merging other than intersection
        # conditionally get the intersection of the new index/columns and the other index/columns
        if left is not None:
            new_index = index.intersection(other_index)
            new_columns = columns.intersection(other_columns)
        else:
            new_index, new_columns = index, columns

        if component == 'features':
            # merge in new features, subsetting other components as necessary, returning new SeqExp

            new_feature_names = new_index
            new_sample_names = new_columns

            new_features = right.loc[new_feature_names, new_sample_names]

            if self.classifications is not None:
                new_classifications = self.classifications.loc[new_feature_names]
            else:
                new_classifications = None

            if self.metadata is not None:
                new_metadata = self.metadata.loc[new_sample_names]
            else:
                new_metadata = None

            if self.seqs is not None:
                new_seqs = self.seqs.loc[new_features.index]
            else:
                new_seqs = None

            return SeqExp(new_features, new_classifications, new_metadata, new_seqs)

        elif component == 'classifications':
            # merge in new classifications, subsetting other components as necessary, returning new SeqExp

            new_feature_names = new_index
            new_classification_names = new_columns

            new_classifications = right.loc[new_feature_names, new_classification_names]
            new_features = self.features.loc[new_feature_names]

            if self.seqs is not None:
                new_seqs = self.seqs.loc[new_feature_names]
            else:
                new_seqs = None

            return SeqExp(new_features, new_classifications, self.metadata, new_seqs)

        elif component == 'metadata':
            # merge in new metadata, subsetting other components as necessary, returning new SeqExp

            if self.metadata is not None:
                new_sample_names = new_index.intersection(new_index)
                new_metadata_names = new_columns.intersection(other_columns)
            else:
                new_sample_names = new_index
                new_metadata_names = new_columns

            new_metadata = right.loc[new_sample_names, new_metadata_names]
            new_features = self.features.loc[:, new_sample_names]

            return SeqExp(new_features, self.classifications, new_metadata, self.seqs)

        elif component == 'seqs':

            new_feature_names = new_index
            new_seq_names = new_columns

            new_seqs = right.loc[new_feature_names, new_seq_names]
            new_features = self.features.loc[new_feature_names]

            if self.classifications is not None:
                new_classifications = self.classifications.loc[new_feature_names]
            else:
                new_classifications = None

            return SeqExp(new_features, new_classifications, self.metadata, new_seqs)

    def concat(self):
        """
        
        :return: 
        """
        pass

    # -------------- convenience methods -------------- #

    def plot(self, *args, **kwargs):
        """Plots the abundances within the feature table."""

        return plot_abundance(self, *args, **kwargs)

    def relabund(self, scaling_factor=1, by='samples'):
        """
        Returns a new object with abundances converted to relative abundances.

        :param scaling_factor: number to multiple relative abundance values by
        :type scaling_factor: int or float
        :param by: whether to calculate relative abundance within samples or features
        :type by: str ('samples' or 'features')

        :return: new SeqExp object with relative abundance values

        """

        axis_map = {
            'samples': (0, 1),
            'features': (1, 0)
        }

        # make copy of SeqExp
        new_sxp = deepcopy(self)

        # calculate feature abundance relative to total sample abundance
        # note that we sum by the one axis, then divide across the other
        new_sxp.features = new_sxp.features.div(new_sxp.features.sum(axis=axis_map[by][0]), axis=axis_map[by][1]).multiply(scaling_factor)

        # return the new object
        return new_sxp

# register advanced indexing methods to SeqExp object
for _name in get_indexer_mappings():
    SeqExp._create_indexer(_name, _Indexer)
