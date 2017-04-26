import numpy as np
import pandas as pd

from seq_experiment import SeqExp, FeatureTable, ClassificationTable, SampleDataTable

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
test_feature_table = pd.DataFrame(
np.random.randint(low=0, high=1000, size=(feature_count, class_count)),
index=features,
columns=classes
)

# create dummy classification table
test_classification_table = pd.DataFrame(
np.random.randint(low=0, high=10, size=(feature_count, classification_count)),
index=features,
columns=classification_ranks
)

test_sample_data_table = pd.DataFrame(
np.random.randint(low=0, high=100, size=(class_count, sample_data_count)),
index=classes,
columns=sample_data_names
)

# seq_exp = SeqExp(test_feature_table, test_classification_table, test_sample_data_table)
# print(seq_exp)

feature_table = FeatureTable(test_feature_table)
classification_table = ClassificationTable(test_classification_table)
sample_data_table = SampleDataTable(test_sample_data_table)
# print(feature_table)

seq_exp = SeqExp(feature_table, classification_table, sample_data_table)
# print('initial:')
# print(seq_exp)
# print()

# seq_exp = seq_exp[['class_0', 'class_1']]
# seq_exp = seq_exp[:3]
# seq_exp = seq_exp['class_0']
# print('post slice:')
# print(seq_exp)
# print(seq_exp.feature_table)
# print(seq_exp.classification_table)
# print(seq_exp.sample_data_table)

print(seq_exp)

relabund = seq_exp.relabund()

print(relabund.feature_table)
