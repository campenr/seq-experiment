import pandas as pd

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

    @staticmethod
    def read_mothur_shared_file(shared_file):
        """Reads in and formats a Mothur shared file."""

        feature_data = pd.read_table(shared_file)
        feature_data = feature_data.drop(['label', 'numOtus'], axis=1)
        feature_data = feature_data.set_index('Group').transpose()

        return FeatureTable(feature_data)
