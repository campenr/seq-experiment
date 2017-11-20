import pandas as pd
import numpy as np

from copy import deepcopy


class _FeatureIndexer(object):

    def __init__(self, sxp):
        self.sxp = deepcopy(sxp)

    def __getitem__(self, key):

        # subset features using `key`
        new_features = self.sxp.features.loc[key]

        # need to restore new_features to a pd.DataFrame if slice returned a 1-dimensional object
        if np.ndim(new_features) == 1:
            new_features = pd.DataFrame(new_features)

        # conditionally update classifications
        if self.sxp.classifications is not None:
            new_classifications = self.sxp.classifications.loc[new_features.index]
        else:
            new_classifications = None

        # conditionally update metadata
        if self.sxp.metadata is not None:
            new_metadata = self.sxp.metadata.loc[new_features.columns]
        else:
            new_metadata = None

        if self.sxp.seqs is not None:
            new_seqs = self.sxp.seqs.loc[new_features.index]
        else:
            new_seqs = None

        self.sxp.features = new_features

        return self.sxp
