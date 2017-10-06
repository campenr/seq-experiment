import unittest
import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform, pdist

from seq_experiment import SeqExp, FeatureTable, ClassificationTable, MetadataTable


class Test(unittest.TestCase):

    def setUp(self):

        # set random seed
        np.random.seed(123)

        # set data dimensions
        feature_count = 11
        class_count = 10
        classification_count = 5
        sample_data_count = 3

        # create data index's, columns
        features = ['feature_{}'.format(i) for i in range(feature_count)]
        classes = ['class_{}'.format(i) for i in range(class_count)]
        classification_ranks = ['rank_{}'.format(i) for i in range(classification_count)]
        sample_data_names = ['sample_data_{}'.format(i) for i in range(sample_data_count)]

        # create dummy feature table
        self.test_feature_table = pd.DataFrame(
            np.random.randint(low=0, high=1000, size=(feature_count, class_count)),
            index=features,
            columns=classes
        )

        # create dummy classification table
        self.test_classification_table = pd.DataFrame(
            np.random.randint(low=0, high=10, size=(feature_count, classification_count)),
            index=features,
            columns=classification_ranks
        )

        self.test_metadata_table = pd.DataFrame(
            np.random.randint(low=0, high=100, size=(class_count, sample_data_count)),
            index=classes,
            columns=sample_data_names
        )

    def test_create_seq_experiment_1(self):
        """Test creating SeqExp with only feature_table."""

        try:
            SeqExp(
                features=self.test_feature_table
            )
        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_2(self):
        """Test creating SeqExp with feature_table and classification_table."""

        try:
            SeqExp(
                features=self.test_feature_table,
                classifications=self.test_classification_table
            )
        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_3(self):
        """Test creating SeqExp with feature_table and sample_data_table."""

        try:
            SeqExp(
                features=self.test_feature_table,
                metadata=self.test_metadata_table
            )
        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_4(self):
        """Test creating SeqExp with feature_table, classification_table, and sample_data_table."""

        try:
            SeqExp(
                features=self.test_feature_table,
                classifications=self.test_classification_table,
                metadata=self.test_metadata_table
            )
        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    # def test_enforce_classification_table_type(self):
    #     """test that non-pd.DataFrame types raise errors"""
    #
    #     test_types = ['str', 1, 1.0, list(), dict(), set(), pd.DataFrame()]
    #     for type_ in test_types:
    #         with self.assertRaises(TypeError):
    #             SeqExp(
    #                 features=self.test_feature_table,
    #                 classifications=type_
    #             )
    #
    # def test_enforce_sample_data_table_type(self):
    #     """test that non-pd.DataFrame types raise errors"""
    #
    #     test_types = ['str', 1, 1.0, list(), dict(), set(), pd.DataFrame()]
    #     for type_ in test_types:
    #         with self.assertRaises(TypeError):
    #             SeqExp(
    #                 features=self.test_feature_table,
    #                 metadata=type_
    #             )

    def test_ordination_with_distance(self):

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            dist = squareform(pdist(seq_exp.features.transpose(), metric='braycurtis'))
            ord = seq_exp.ordinate(method='nmds', distance=dist)
        except Exception as e:
            self.fail(e)

    # --------- test indexing --------- #

    def test_indexing_1(self):
        """
        Tests indexing by single column by name.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#basics

        """

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            seq_exp['class_0']
        except Exception as e:
            self.fail(e)

    def test_indexing_2(self):
        """
        Tests setting single column by name.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#basics

        """

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            # passing feature table directly
            seq_exp['class_0'] = seq_exp['class_0'].features * 2

            # passing SeqExp directly
            new_features = seq_exp['class_0'].features * 2
            new_seq_exp = SeqExp(features=new_features)
            new_seq_exp = new_seq_exp.merge(seq_exp.classifications, 'classifications')
            new_seq_exp = new_seq_exp.merge(seq_exp.metadata, 'metadata')

            seq_exp['class_0'] = new_seq_exp['class_0']

        except Exception as e:
            self.fail(e)

    def test_indexing_3(self):
        """
        Tests indexing multiple columns by list of names.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#basics

        """

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            seq_exp[['class_0', 'class_1']]
        except Exception as e:
            self.fail(e)

    def test_indexing_4(self):
        """
        Tests setting multiple columns by list of names.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#basics

        """

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            # passing feature table directly
            seq_exp[['class_0', 'class_1']] = seq_exp[['class_0', 'class_1']].features * 2

            # passing SeqExp directly
            new_features = seq_exp[['class_0', 'class_1']].features * 2
            new_seq_exp = SeqExp(features=new_features)
            new_seq_exp = new_seq_exp.merge(seq_exp.classifications, 'classifications')
            new_seq_exp = new_seq_exp.merge(seq_exp.metadata, 'metadata')

            seq_exp[['class_0', 'class_1']] = new_seq_exp[['class_0', 'class_1']]

        except Exception as e:
            self.fail(e)

    def test_indexing_5(self):
        """
        Tests indexing by attribute.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access

        """

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            seq_exp.class_0
        except Exception as e:
            self.fail(e)

    # TODO implement this
    # def test_indexing_6(self):
    # """
    # Tests setting column by attribute.
    #
    # https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access
    #
    # """
    #
    #     seq_exp = SeqExp(
    #         features=self.test_feature_table,
    #         classifications=self.test_classification_table,
    #         metadata=self.test_metadata_table
    #     )
    #
    #     try:
    #         seq_exp.class_0 = seq_exp.class_0.feature_table * 2
    #     except Exception as e:
    #         self.fail(e)

    def test_indexing_7(self):
        """
        Tests indexing by slice range.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#slicing-ranges

        """

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            seq_exp[:1]
            seq_exp[::-1]
        except Exception as e:
            self.fail(e)

    # def test_indexing_8(self):
    #     """
    #     Tests setting rows by slice range.
    #
    #     https://pandas.pydata.org/pandas-docs/stable/indexing.html#slicing-ranges
    #
    #     """
    #
    #     seq_exp = SeqExp(
    #         features=self.test_feature_table,
    #         classifications=self.test_classification_table,
    #         metadata=self.test_metadata_table
    #     )
    #
    #     try:
    #         # passing feature table directly
    #         seq_exp[:3] = seq_exp[:3].features * 2
    #
    #         # passing SeqExp directly
    #         new_features = seq_exp[:3] = seq_exp[:3].features * 2
    #         new_seq_exp = SeqExp(features=new_features)
    #         new_seq_exp = new_seq_exp.merge(seq_exp.classifications, 'classifications')
    #         new_seq_exp = new_seq_exp.merge(seq_exp.metadata, 'metadata')
    #
    #         seq_exp[:3] = seq_exp[:3]
    #
    #     except Exception as e:
    #         self.fail(e)

    # --------- test convenience functions --------- #

    def test_make_relabund(self):
        """."""

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            seq_exp.relabund()
            seq_exp.relabund(scaling_factor=100)

        except Exception as e:
            self.fail(e)

    def test_subset_features(self):
        """."""

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            seq_exp.subset(['feature_0', 'feature_1'], by='features')

        except Exception as e:
            self.fail(e)

    def test_subset_samples(self):
        """."""

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            seq_exp.subset(['class_0', 'class_1'], by='samples')

        except Exception as e:
            self.fail(e)

    def test_drop_features(self):
        """."""

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            seq_exp.drop(['feature_0', 'feature_1'], by='features')

        except Exception as e:
            self.fail(e)

    def test_drop_samples(self):
        """."""

        seq_exp = SeqExp(
            features=self.test_feature_table,
            classifications=self.test_classification_table,
            metadata=self.test_metadata_table
        )

        try:
            seq_exp.drop(['class_0', 'class_1'], by='samples')

        except Exception as e:
            self.fail(e)

if __name__ == '__main__':
    unittest.main()