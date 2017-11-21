import pandas as pd
import numpy as np

from copy import deepcopy


class _FeatureIndexer(object):

    def __init__(self, name, sxp):

        self.sxp = deepcopy(sxp)

    def __getitem__(self, key):

        # print('key: ', key)

        # subset features using `key`
        # try label based indexing first, fallback to index based if it fails
        try:
            new_features = self.sxp.features.loc[key]
        except TypeError:
            new_features = self.sxp.features.iloc[key]

        # conditionally correct type of new_features if dimensionality has been reduced during subset
        if isinstance(new_features, type(self.sxp.features)):
            pass
        elif isinstance(new_features, self.sxp.features._constructor_sliced):
            new_features = pd.DataFrame(new_features)
        else:
            new_features = pd.DataFrame([new_features], index=[key[0]], columns=[key[1]])

        # may need to correct dataframe orientation
        if not new_features.index.isin(self.sxp.feature_names).all():
            new_features = new_features.transpose()

        self.sxp = self.sxp.merge(right=new_features, component='features')

        return self.sxp


class _MetadataIndexer(object):

    def __init__(self, name, sxp):

        self.sxp = deepcopy(sxp)

    def __getitem__(self, key):

        # subset features using `key`
        # try label based indexing first, fallback to index based if it fails
        try:
            new_metadata = self.sxp.metadata.loc[key]
        except TypeError:
            new_metadata = self.sxp.metadata.iloc[key]

        # conditionally correct type of new_features if dimensionality has been reduced during subset
        if isinstance(new_metadata, type(self.sxp.metadata)):
            pass
        elif isinstance(new_metadata, self.sxp.metadata._constructor_sliced):
            new_metadata = pd.DataFrame(new_metadata)
        else:
            new_metadata = pd.DataFrame([new_metadata], index=[key[0]], columns=[key[1]])

        # may need to correct dataframe orientation
        if not new_metadata.index.isin(self.sxp.sample_names).all():
            new_metadata = new_metadata.transpose()

        self.sxp = self.sxp.merge(right=new_metadata, component='metadata')

        return self.sxp
