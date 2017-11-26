"""
Copyright (c) 2017, Richard Campen
All rights reserved.
Licensed under the Modified BSD License.
For full license terms see LICENSE.txt

"""

from copy import deepcopy
import pandas as pd


def get_indexer_mappings():

    return {
        'fx': 'features',
        'cx': 'classifications',
        'mx': 'metadata',
        'sx': 'seqs'
    }


class _Indexer(object):
    """
    Provides advanced indexing of SeqExp objects.

    """

    def __init__(self, name, sxp):

        self.sxp = deepcopy(sxp)
        self.attr_name = get_indexer_mappings()[name]
        self.attr = getattr(self.sxp, self.attr_name)

    def __getitem__(self, key):

        # subset the attribute by `key` using label based indexing first, fallback to integer based if it fails
        try:
            new_attr = self.attr.loc[key]
        except TypeError:
            new_attr = self.attr.iloc[key]

        # conditionally correct type of attribute if dimensionality has been reduced during subset
        # assumes attr is pd.DataFrame or pd.DataFrame like object, with a `_constructor_sliced` method implemented
        if isinstance(new_attr, type(self.attr)):
            # dimensionality preserved after subset
            pass
        elif isinstance(new_attr, self.attr._constructor_sliced):
            # dimensionality reduced by 1, need to restore to original ndim
            new_attr = pd.DataFrame(new_attr)
        else:
            # single value returned, likely because subset returned a single cell, need to restore to original ndim
            new_attr = pd.DataFrame([new_attr], index=[key[0]], columns=[key[1]])

        # conditionally restore correct data orientation
        # if dimensionality was changed during subset, orientation may have also been changed
        if not new_attr.index.isin(self.attr.index).all():
            new_attr = new_attr.transpose()

        # merge in the subset attribute to a copy of the SeqExp before returning it
        # this has the effect of cascading the subset to the other attributes of the SeqExp object
        return self.sxp.merge(right=new_attr, component=self.attr_name)
