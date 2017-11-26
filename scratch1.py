from seq_experiment import SeqExp
import pandas as pd
import numpy as np

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
test_features = pd.DataFrame(
    np.random.randint(low=0, high=1000, size=(feature_count, class_count)),
    index=features,
    columns=classes
)

# create dummy classification data
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

test_classifications = pd.DataFrame(
    classification_dict,
    index=features,
    columns=classification_ranks
)

# create dummy metadata
test_metadata = pd.DataFrame(
    np.random.randint(low=0, high=100, size=(class_count, metadata_count)),
    index=classes,
    columns=metadata_names,
    dtype=np.int
)

# create metadata column for grouping samples
extra_metadata_name = 'metadata_%s' % metadata_count
extra_metadatum = 'abc' * (class_count // 3)
extra_metadatum = [i for i in sorted(extra_metadatum)]
test_metadata[extra_metadata_name] = extra_metadatum

sxp = SeqExp(features=test_features, classifications=test_classifications, metadata=test_metadata)


# --------------- TEST1 --------------- #

print(sxp)

print('\n#################################\n')
print(sxp.fx['feature_0'])
print(sxp.fx['feature_0'].features)
print(sxp.fx['feature_0'].classifications)

print('\n#################################\n')
print(sxp.fx[0])

print('\n#################################\n')
print(sxp.fx['feature_1':'feature_3'])
print(sxp.fx['feature_1':'feature_3'].features)
print(sxp.fx['feature_1':'feature_3'].classifications)

print('\n#################################\n')
print(sxp.fx['feature_5', 'class_2'])
print(sxp.fx['feature_5', 'class_2'].features)
print(sxp.fx['feature_5', 'class_2'].classifications)

print('\n#################################\n')
print(sxp.fx[:,:])

print('\n#################################\n')
print(sxp.fx[:, 1])

print('\n#################################\n')
print(sxp.fx[:, sxp.metadata['metadata_0'] < 50].metadata)
print(sxp.fx[:, sxp.metadata['metadata_0'] > 50].metadata)

print(sxp.metadata)

print('\n#################################\n')
print(sxp.fx[:, sxp.metadata['metadata_0'] > 50])
print(sxp.mx[sxp.metadata['metadata_0'] > sxp.metadata['metadata_0'].mean()])
print(sxp.mx[:, sxp.metadata.loc['class_0'] == 'a'])

print('\n#################################\n')
print(sxp.fx[sxp.features['class_0'] > 500])

print('\n#################################\n')
print(sxp.fx[:, sxp.features.loc['feature_0'] > 500].features)
print(sxp.fx[:, sxp.features.loc['feature_0'] < 500].features)


print(sxp.mx[:, sxp.metadata.loc['class_0'] == 'a'])
sxp2 = sxp.fx['feature_1':'feature_3']


print(sxp2)

feats_ = sxp.features

print(feats_.loc[:'feature_2':].index)
print(feats_.iloc[:3].index)

print('################################')
print(sxp[::2])
print(sxp['class_0'])

sxp4 = sxp[['class_0', 'class_1']]
sxp4 = sxp[['class_1', 'class_0']]
print(sxp4.features)

print(sxp.metadata)