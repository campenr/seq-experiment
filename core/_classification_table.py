import pandas as pd

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

    @staticmethod
    def read_mothur_constaxonomy_file(constaxonomy_file):
        """Reads in and formats a Mother constaxonomy file."""

        classification_data = pd.read_table(constaxonomy_file)
        classifications = classification_data['Taxonomy']
        classifications = classifications.str.split(';', expand=True).drop(6, axis=1)
        classifications.columns = list(range(1, 7))
        features = classification_data['OTU']
        classification_data = pd.concat([features, classifications], axis=1)
        classification_data = classification_data.set_index('OTU')

        return ClassificationTable(classification_data)