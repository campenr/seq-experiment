import unittest
import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform, pdist

from seq_experiment import SeqExp


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

        # setup container to hold test SeqExp objects. They are actually created in the first few test functions
        self.test_sxps = dict()

    def test_create_seq_experiment_1(self):
        """Test creating SeqExp with only feature_table."""

        try:
            seq_exp = SeqExp(
                features=self.test_feature_table
            )

            # save this SeqExp to the test object for use by other test functions
            self.test_sxps['sxp_f'] = seq_exp

        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_2(self):
        """Test creating SeqExp with feature_table and classification_table."""

        try:
            seq_exp = SeqExp(
                features=self.test_feature_table,
                classifications=self.test_classification_table
            )

            # save this SeqExp to the test object for use by other test functions
            self.test_sxps['sxp_fc'] = seq_exp

        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_3(self):
        """Test creating SeqExp with feature_table and sample_data_table."""

        try:
            seq_exp = SeqExp(
                features=self.test_feature_table,
                metadata=self.test_metadata_table
            )

            # save this SeqExp to the test object for use by other test functions
            self.test_sxps['sxp_fm'] = seq_exp

        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_4(self):
        """Test creating SeqExp with feature_table, classification_table, and sample_data_table."""

        try:
            seq_exp = SeqExp(
                features=self.test_feature_table,
                classifications=self.test_classification_table,
                metadata=self.test_metadata_table
            )

            # save this SeqExp to the test object for use by other test functions
            self.test_sxps['sxp_fcm'] = seq_exp

        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_change_sample_names(self):

        try:
            seq_exp = SeqExp(
                features=self.test_feature_table,
                classifications=self.test_classification_table,
                metadata=self.test_metadata_table
            )

            new_names = ['%sx' % old_name for old_name in seq_exp.sample_names]
            seq_exp.sample_names = new_names

        except ValueError:
            self.fail('Editing sample_names raised ValueError unexpectedly')

    def test_change_feature_names(self):

        try:
            seq_exp = SeqExp(
                features=self.test_feature_table,
                classifications=self.test_classification_table,
                metadata=self.test_metadata_table
            )

            new_names = ['%sx' % old_name for old_name in seq_exp.feature_names]
            seq_exp.feature_names = new_names

        except ValueError:
            self.fail('Editing feature_names raised ValueError unexpectedly')


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

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                seq_exp['class_0']
            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    def test_indexing_2(self):
        """
        Tests setting single column by name.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#basics

        """

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                # passing feature table directly
                seq_exp['class_0'] = seq_exp['class_0'].features * 2

                # passing SeqExp directly
                new_features = seq_exp['class_0'].features * 2
                new_seq_exp = SeqExp(features=new_features, classifications=seq_exp.classifications,
                                     metadata=seq_exp.metadata)

                seq_exp['class_0'] = new_seq_exp['class_0']

            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    def test_indexing_3(self):
        """
        Tests indexing multiple columns by list of names.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#basics

        """

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                seq_exp[['class_0', 'class_1']]
            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    def test_indexing_4(self):
        """
        Tests setting multiple columns by list of names.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#basics

        """

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                # passing feature table directly
                seq_exp[['class_0', 'class_1']] = seq_exp[['class_0', 'class_1']].features * 2

                # passing SeqExp directly
                new_features = seq_exp[['class_0', 'class_1']].features * 2
                new_seq_exp = SeqExp(features=new_features, classifications=seq_exp.classifications,
                                     metadata=seq_exp.metadata)

                seq_exp[['class_0', 'class_1']] = new_seq_exp[['class_0', 'class_1']]

            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    def test_indexing_5(self):
        """
        Tests indexing by attribute.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access

        """

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                seq_exp.class_0
            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    # TODO implement this
    # def test_indexing_6(self):
    # """
    # Tests setting column by attribute.
    #
    # https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access
    #
    # """
    #
    # # test on all SeqExp permutations
    # for name, seq_exp in self.test_sxps.items():    #
    #     try:
    #         seq_exp.class_0 = seq_exp.class_0.feature_table * 2
    #     except Exception as e:
    #         self.fail('Failed on %s.' % name, e)

    def test_indexing_7(self):
        """
        Tests indexing by slice range.

        https://pandas.pydata.org/pandas-docs/stable/indexing.html#slicing-ranges

        """

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                seq_exp[:1]
                seq_exp[::-1]
            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    # def test_indexing_8(self):
    #     """
    #     Tests setting rows by slice range.
    #
    #     https://pandas.pydata.org/pandas-docs/stable/indexing.html#slicing-ranges
    #
    #     """
    #
    # # test on all SeqExp permutations
    # for name, seq_exp in self.test_sxps.items():
    #     try:
    #         # passing feature table directly
    #         seq_exp[:3] = seq_exp[:3].features * 2
    #
    #         # passing SeqExp directly
    #         new_features = seq_exp[:3] = seq_exp[:3].features * 2
    #         new_seq_exp = SeqExp(features=new_features, classifications=seq_exp.classifications,
    #                              metadata=seq_exp.metadata)
    #
    #         seq_exp[:3] = seq_exp[:3]
    #
    #     except Exception as e:
    #         self.fail('Failed on %s.' % name, e)

    # --------- test convenience functions --------- #

    def test_make_relabund(self):
        """."""

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                seq_exp.relabund()
                seq_exp.relabund(scaling_factor=100)

            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    def test_subset_features(self):
        """."""

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                seq_exp.subset(['feature_0', 'feature_1'], by='features')

            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    def test_subset_samples(self):
        """."""

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                seq_exp.subset(['class_0', 'class_1'], by='samples')

            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    def test_drop_features(self):
        """."""

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                seq_exp.drop(['feature_0', 'feature_1'], by='features')

            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    def test_drop_samples(self):
        """."""

        # test on all SeqExp permutations
        for name, seq_exp in self.test_sxps.items():
            try:
                seq_exp.drop(['class_0', 'class_1'], by='samples')

            except Exception as e:
                self.fail('Failed on %s.' % name, e)

    def test_mothur_import(self):

        from seq_experiment.io import MothurIO

        pass

if __name__ == '__main__':
    unittest.main()
