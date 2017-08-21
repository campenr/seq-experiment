import numpy as np
import pandas as pd

from core import SeqExp, FeatureTable, ClassificationTable, SampleDataTable

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
# print('\nmaking')
# seq_exp = SeqExp(feature_table, sample_data_table=sample_data_table)
# print(seq_exp)
#
# print('\nmerging')
# seq_exp = seq_exp.merge(classification_table)
# print(seq_exp)
#
# print('\ndropping')
# seq_exp = seq_exp.drop('feature_0')
# print(seq_exp)
#
# print('\ndropping')
# seq_exp = seq_exp.drop('class_7', from_='samples')
# print(seq_exp)

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

# print(seq_exp)
#
# relabund = seq_exp.relabund()
#
# print(relabund.feature_table)

# print(seq_exp.sample_names)
# print(seq_exp.feature_names)

# from skbio import DistanceMatrix

# from skbio.diversity import beta_diversity

# dist = pd.read_csv('subice_bray_dist.csv', index_col=0)
# print(dist)

seq_exp.sample_data_table['sample_data_3'] = [
    'a',
    'a',
    'a',
    'b',
    'b',
    'b',
    'c',
    'c',
    'c',
    'c'
]

# print(seq_exp.sample_data_table)

# seq_exp_bray = seq_exp.distance(metric='braycurtis')
# # print(seq_exp_bray)
#
# seq_exp_bray_ord = seq_exp_bray.ordinate()
# # print(seq_exp_bray_ord)
#
# seq_exp_bray_ord.plot(seq_exp=seq_exp, color='sample_data_3')
