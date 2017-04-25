import unittest
import numpy as np
import pandas as pd

from seq_experiment import SeqExp, FeatureTable, ClassificationTable, SampleDataTable


class Test(unittest.TestCase):

    def setUp(self):

        # set random seed
        np.random.seed(123)

        # set data dimensions
        feature_count = 11
        class_count = 10
        classification_count = 2
        sample_data_count = 3

        # create data index's, columns
        features = ['feature_{}'.format(i) for i in range(feature_count)]
        classes = ['class_{}'.format(i) for i in range(class_count)]
        classification_ranks = ['rank_{}'.format(i) for i in range(classification_count)]
        sample_data_names = ['sample_data_{}'.format(i) for i in range(sample_data_count)]

        # create dummy feature table
        self.test_feature_table = FeatureTable(
            pd.DataFrame(
                np.random.randint(low=0, high=1000, size=(feature_count, class_count)),
                index=features,
                columns=classes
            )
        )

        # create dummy classification table
        self.test_classification_table = ClassificationTable(
            pd.DataFrame(
                np.random.randint(low=0, high=10, size=(feature_count, classification_count)),
                index=features,
                columns=classification_ranks
            )
        )

        self.test_sample_data_table = SampleDataTable(
            pd.DataFrame(
                np.random.randint(low=0, high=100, size=(class_count, sample_data_count)),
                index=classes,
                columns=sample_data_names
            )
        )

    def test_create_seq_experiment_1(self):
        """Test creating SeqExp with only feature_table."""

        try:
            SeqExp(
                feature_table=self.test_feature_table
            )
        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')


    def test_create_seq_experiment_2(self):
        """Test creating SeqExp with feature_table and classification_table."""

        try:
            SeqExp(
                feature_table=self.test_feature_table,
                classification_table=self.test_classification_table
            )
        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_3(self):
        """Test creating SeqExp with feature_table and sample_data_table."""

        try:
            SeqExp(
                feature_table=self.test_feature_table,
                sample_data_table=self.test_sample_data_table
            )
        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_create_seq_experiment_4(self):
        """Test creating SeqExp with feature_table, classification_table, and samle_data_table."""

        try:
            SeqExp(
                feature_table=self.test_feature_table,
                classification_table=self.test_classification_table,
                sample_data_table=self.test_sample_data_table
            )
        except TypeError:
            self.fail('SeqExp creation raised TypeError unexpectedly')

    def test_enforce_feature_table_type(self):
        """test that non-pd.DataFrame types raise errors"""

        test_types = ['str', 1, 1.0, list(), dict(), set(), pd.DataFrame()]
        for type_ in test_types:
            with self.assertRaises(TypeError):
                SeqExp(feature_table=type_)

    def test_enforce_classification_table_type(self):
        """test that non-pd.DataFrame types raise errors"""

        test_types = ['str', 1, 1.0, list(), dict(), set(), pd.DataFrame()]
        for type_ in test_types:
            with self.assertRaises(TypeError):
                SeqExp(
                    feature_table=self.test_feature_table,
                    classification_table=type_
                )

    def test_enforce_sample_data_table_type(self):
        """test that non-pd.DataFrame types raise errors"""

        test_types = ['str', 1, 1.0, list(), dict(), set(), pd.DataFrame()]
        for type_ in test_types:
            with self.assertRaises(TypeError):
                SeqExp(
                    feature_table=self.test_feature_table,
                    sample_data_table=type_
                )

    def test_subset_SeqExp(self):
        """Test that we can subset the sequence experiment object."""

        try:
            seq_exp = SeqExp(
                feature_table=self.test_feature_table,
                classification_table=self.test_classification_table,
                sample_data_table=self.test_sample_data_table
            )
            seq_exp = seq_exp[['class_1', 'class_2']]

        except Exception as e:
            self.fail(e)



if __name__ == '__main__':
    unittest.main()