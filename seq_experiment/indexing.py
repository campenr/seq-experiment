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

    def __getitem__(self, key):

        # get attribute matching given indexer
        attr = getattr(self.sxp, self.attr_name)

        # subset attribute using `key` using label based indexing first, fallback to index based if it fails
        try:
            new_attr = attr.loc[key]
        except TypeError:
            new_attr = attr.iloc[key]

        # conditionally correct type of attribute if dimensionality has been reduced during subset
        # assumes attr is pd.DataFrame or pd.DataFrame like object, with _constructor_sliced method implemented
        if isinstance(new_attr, type(attr)):
            pass
        elif isinstance(new_attr, attr._constructor_sliced):
            new_attr = pd.DataFrame(new_attr)
        else:
            new_attr = pd.DataFrame([new_attr], index=[key[0]], columns=[key[1]])

        # may need to correct data orientation
        if not new_attr.index.isin(attr.index).all():
            new_attr = new_attr.transpose()

        self.sxp = self.sxp.merge(right=new_attr, component=self.attr_name)

        return self.sxp
