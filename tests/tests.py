import unittest
import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform, pdist

from seq_experiment import SeqExp


class Test(unittest.TestCase):

    def setUp(self):

        # --------------- SETUP --------------- #

        # set random seed
        np.random.seed(123)

        # set data dimensions
        feature_count = 12  # should me multiple of 12
        class_count = 12  # should me multiple of 12
        classification_count = 4
        metadata_count = 3

        # create data index's, columns
        features = ['feature_{}'.format(i) for i in range(feature_count)]
        classes = ['class_{}'.format(i) for i in range(class_count)]
        classification_ranks = ['rank_{}'.format(i) for i in range(classification_count)]
        metadata_names = ['metadata_{}'.format(i) for i in range(metadata_count)]

        # create dummy feature table
        self.test_features = pd.DataFrame(
            np.random.randint(low=0, high=1000, size=(feature_count, class_count)),
            index=features,
            columns=classes
        )

        # create dummy classification data
        possible_ranks = 'abcdefghijklmnopqrstuvwxyz'
        rank_levels = [1, 3, 6, 12]
        reversed_classification_ranks = list(reversed(classification_ranks))

        classification_dict = dict()
        for num_rank in reversed(range(classification_count)):
            rank_name = reversed_classification_ranks[num_rank]
            rank_level = rank_levels[num_rank]
            num_items = feature_count // rank_level
            rank_items = possible_ranks[:num_items]
            rank_data = ['%s_%s' % (rank, num_rank) for rank in sorted(rank_items * rank_level)]

            classification_dict[rank_name] = rank_data

        self.test_classifications = pd.DataFrame(
            classification_dict,
            index=features,
            columns=classification_ranks
        )

        # create dummy metadata
        self.test_metadata = pd.DataFrame(
            np.random.randint(low=0, high=100, size=(class_count, metadata_count)),
            index=classes,
            columns=metadata_names,
            dtype=np.int
        )

        # create metadata column for grouping samples
        extra_metadata_name = 'metadata_%s' % metadata_count
        extra_metadatum = 'abc' * (class_count // 3)
        extra_metadatum = [i for i in sorted(extra_metadatum)]
        self.test_metadata[extra_metadata_name] = extra_metadatum

        # setup container to hold test SeqExp objects. They are actually created in the first few test functions
        self.test_sxps = dict()

        # self.test_features = None
        # self.test_classifications = None
        # self.test_metadata = None

    def test_create_seq_experiment_1(self):
        """Test creating SeqExp with only feature_table."""

        try:
            seq_exp = SeqExp(
                features=self.test_features
            )

            # save this SeqExp to the test object for use by other test functions
            self.test_sxps['sxp_f'] = seq_exp

        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_2(self):
        """Test creating SeqExp with feature_table and classification_table."""

        try:
            seq_exp = SeqExp(
                features=self.test_features,
                classifications=self.test_classifications
            )

            # save this SeqExp to the test object for use by other test functions
            self.test_sxps['sxp_fc'] = seq_exp

        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_3(self):
        """Test creating SeqExp with feature_table and sample_data_table."""

        try:
            seq_exp = SeqExp(
                features=self.test_features,
                metadata=self.test_metadata
            )

            # save this SeqExp to the test object for use by other test functions
            self.test_sxps['sxp_fm'] = seq_exp

        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_4(self):
        """Test creating SeqExp with feature_table, classification_table, and sample_data_table."""

        try:
            seq_exp = SeqExp(
                features=self.test_features,
                classifications=self.test_classifications,
                metadata=self.test_metadata
            )

            # save this SeqExp to the test object for use by other test functions
            self.test_sxps['sxp_fcm'] = seq_exp

        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_change_sample_names(self):

        try:
            seq_exp = SeqExp(
                features=self.test_features,
                classifications=self.test_classifications,
                metadata=self.test_metadata
            )

            new_names = ['%sx' % old_name for old_name in seq_exp.sample_names]
            seq_exp.sample_names = new_names

        except ValueError:
            self.fail('Editing sample_names raised ValueError unexpectedly')

    def test_change_feature_names(self):

        try:
            seq_exp = SeqExp(
                features=self.test_features,
                classifications=self.test_classifications,
                metadata=self.test_metadata
            )

            new_names = ['%sx' % old_name for old_name in seq_exp.feature_names]
            seq_exp.feature_names = new_names

        except ValueError:
            self.fail('Editing feature_names raised ValueError unexpectedly')

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
