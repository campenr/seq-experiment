# functions for importing/exporting mothur files

import pandas as pd

class MothurIO(object):

    @staticmethod
    def read_shared_file(filepath):
        """Reads in and formats mothur shared file."""

        data = pd.read_table(filepath)
        data = data.drop(['label', 'numOtus'], axis=1)
        data = data.set_index('Group').transpose()

        # format the index for better consistency
        data.index = data.index.rename(None)
        data = data.sort_index()

        return data

    @staticmethod
    def read_count_file(filepath):
        """Reads in and formats mothur count_file."""

        data = pd.read_table(filepath, index_col=0)
        data = data.drop(['total'], axis=1)

        # format the index for better consistency
        data.index = data.index.rename(None)
        data = data.sort_index()

        return data

    @staticmethod
    def read_taxonomy_file(filepath):
        """Reads in and formats mothur taxonomy file."""

        data = pd.read_table(filepath, names=['OTU', 'Taxonomy'])
        classifications = data['Taxonomy']
        classifications = classifications.str.split(';', expand=True).drop(6, axis=1)
        classifications.columns = list(range(1, 7))
        features_names = data['OTU']
        data = pd.concat([features_names, classifications], axis=1)
        data = data.set_index('OTU')

        # format the index for better consistency
        data.index = data.index.rename(None)
        data = data.sort_index()

        return data

    @staticmethod
    def read_constaxonomy_file(filepath):
        """Reads in and formats mothur cons.taxonomy file."""

        data = pd.read_table(filepath)
        classifications = data['Taxonomy']
        classifications = classifications.str.split(';', expand=True).drop(6, axis=1)
        classifications.columns = list(range(1, 7))
        features_names = data['OTU']
        data = pd.concat([features_names, classifications], axis=1)
        data = data.set_index('OTU')

        # format the index for better consistency
        data.index = data.index.rename(None)
        data = data.sort_index()

        return data

    @staticmethod
    def write_shared_file(seq_exp, filepath):
        """Writes feature table out in shared file format."""

        shared = seq_exp.features
        shared = shared.transpose()
        shared = shared.reset_index()
        shared['label'] = seq_exp.label
        shared['numOtus'] = len(seq_exp.features)
        new_columns = ['label', 'Group', 'numOtus', *seq_exp.features.index]
        shared = shared[new_columns]

        shared.to_csv(filepath, sep='\t', header=True, index=False)

        return filepath


class MothurUtils(object):

    @staticmethod
    def tidy_mothur_classifications(classification_data):
        """Renoves the reported classification percentage agreements that Mothur appends to the classificaitons."""

        new_classification_data = classification_data.copy(deep=True)
        for column in classification_data:
            new_classification_data[column] = new_classification_data[column].str.rsplit('(', 1, expand=True)[0]

        return new_classification_data
