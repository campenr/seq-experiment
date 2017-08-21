import pandas as pd

class MetadataTable(pd.DataFrame):
    """
    Sample data table.

    A table of data that accompanies the classes (columns) in a FeatureTable object. For example, if the classes in the
    FeatureTable are different samples, the sample data may include different locations, geophysical parameters, etc
    for each sample.

    """

    @property
    def _constructor(self):
        return MetadataTable

    def __init__(self, *args, **kwargs):
        super(MetadataTable, self).__init__(*args, **kwargs)
