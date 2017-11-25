from seq_experiment import SeqExp
import pandas as pd
import numpy as np

# --------------- SETUP --------------- #

# set random seed
np.random.seed(123)

# set data dimensions
feature_count = 12 # should me multiple of 12
class_count = 10
classification_count = 4
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
possible_ranks = 'abcdefghijklmnopqrstuvwxyz'
rank_levels = [1,3,6,12]
reversed_classification_ranks = list(reversed(classification_ranks))

classification_dict = dict()
for num_rank in reversed(range(classification_count)):

    rank_name = reversed_classification_ranks[num_rank]
    rank_level = rank_levels[num_rank]
    num_items = feature_count // rank_level
    rank_items = possible_ranks[:num_items]
    rank_data = ['%s_%s' % (rank, num_rank) for rank in sorted(rank_items * rank_level)]

    classification_dict[rank_name] = rank_data

test_classification_table = pd.DataFrame(
    classification_dict,
    index=features,
    columns=classification_ranks
)

test_metadata_table = pd.DataFrame(
    np.random.randint(low=0, high=100, size=(class_count, sample_data_count)),
    index=classes,
    columns=sample_data_names
)


sxp = SeqExp(features=test_feature_table, classifications=test_classification_table, metadata=test_metadata_table)


# --------------- TEST1 --------------- #

print(sxp)

# # print('\n#################################\n')
# # print(sxp.fx['feature_0'])
# # print(sxp.fx['feature_0'].features)
# # print(sxp.fx['feature_0'].classifications)
# #
# # print('\n#################################\n')
# # print(sxp.fx[0])
# #
# # print('\n#################################\n')
# # print(sxp.fx['feature_1':'feature_3'])
# # print(sxp.fx['feature_1':'feature_3'].features)
# # print(sxp.fx['feature_1':'feature_3'].classifications)
# #
# # print('\n#################################\n')
# # print(sxp.fx['feature_5', 'class_2'])
# # print(sxp.fx['feature_5', 'class_2'].features)
# # print(sxp.fx['feature_5', 'class_2'].classifications)
# #
# # print('\n#################################\n')
# # print(sxp.fx[:,:])
# #
# # print('\n#################################\n')
# # print(sxp.fx[:, 1])
# #
# # print('\n#################################\n')
# # print(sxp.fx[:, sxp.metadata['sample_data_0'] < 50].metadata)
# # print(sxp.fx[:, sxp.metadata['sample_data_0'] > 50].metadata)
#
# # print(sxp.metadata)
# # print(sxp.metadata.loc['class_0'] > 50)
#
# # print('\n#################################\n')
# # print(sxp.fx[:, sxp.metadata['sample_data_0'] > 50])
# # print(sxp.mx[sxp.metadata['sample_data_0'] > sxp.metadata['sample_data_0'].mean()])
# # print(sxp.mx[:, sxp.metadata.loc['class_0'] > 50])
# #
# # print('\n#################################\n')
# # print(sxp.fx[sxp.features['class_0'] > 500])
# #
# # print('\n#################################\n')
# # print(sxp.fx[:, sxp.features.loc['feature_0'] > 500].features)
# # print(sxp.fx[:, sxp.features.loc['feature_0'] < 500].features)
#
#
# # sxp1 = sxp.mx[:, sxp.metadata.loc['class_0'] > 50]
# # sxp2 = sxp.fx['feature_1':'feature_3']
# #
# # print(sxp1)
# # print(sxp2)
# #
# # print('################################')
# #
# # sxp3 = sxp1.merge(sxp2)
# #
# # print('################################')
# #
# # print(sxp3)
#
#
# # feats_ = sxp.features
# #
# # print(feats_.loc[:'feature_2':].index)
# # print(feats_.iloc[:3].index)
#
# # print('################################')
# # print(sxp[::2])
# # print(sxp['class_0'])
#
# # sxp4 = sxp[['class_0', 'class_1']]
# # sxp4 = sxp[['class_1', 'class_0']]
# # print(sxp4.features)
