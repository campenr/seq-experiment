"""
Copyright (c) 2017, Richard Campen
All rights reserved.
Licensed under the Modified BSD License.
For full license terms see LICENSE.txt

"""

from copy import deepcopy
import pandas as pd


def get_indexer_mappings():

    return {'loc', 'iloc'}


class _Indexer(object):
    """
    Provides advanced indexing of SeqExp objects.

    """

    def __init__(self, name, sxp):

        self.sxp = deepcopy(sxp)
        self.name = name

    def __getitem__(self, key):

        # subset features using either loc or iloc
        if self.name == 'loc':
            new_features = self.sxp.features.loc[key]
        elif self.name == 'iloc':
            new_features = self.sxp.features.iloc[key]

        # conditionally correct type of new_features if dimensionality has been reduced during subset
        # assumes attr is pd.DataFrame or pd.DataFrame like object, with a `_constructor_sliced` method implemented
        if isinstance(new_features, type(self.sxp.features)):
            # dimensionality preserved after subset
            pass
        elif isinstance(new_features, self.self.sxp.features._constructor_sliced):
            # dimensionality reduced by 1, need to restore to original ndim
            new_attr = pd.DataFrame(new_features)
        else:
            # single value returned, likely because subset returned a single cell, need to restore to original ndim
            new_attr = pd.DataFrame([new_features], index=[key[0]], columns=[key[1]])

        # conditionally restore correct data orientation
        # if dimensionality was changed during subset, orientation may have also been changed
        if not new_features.index.isin(self.sxp.features.index).all():
            new_features = new_features.transpose()

        # merge in the subset attribute to a copy of the SeqExp before returning it
        # this has the effect of cascading the subset to the other attributes of the SeqExp object
        return self.sxp.merge(right=new_features, component='features')
