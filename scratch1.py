from seq_experiment import SeqExp
import pandas as pd
import numpy as np

# --------------- SETUP --------------- #

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
test_feature_table = pd.DataFrame(
    np.random.randint(low=0, high=1000, size=(feature_count, class_count)),
    index=features,
    columns=classes
)

# # create dummy classification table
# test_classification_table = pd.DataFrame(
#     np.random.randint(low=0, high=10, size=(feature_count, classification_count)),
#     index=features,
#     columns=classification_ranks
# )
#
# test_metadata_table = pd.DataFrame(
#     np.random.randint(low=0, high=100, size=(class_count, sample_data_count)),
#     index=classes,
#     columns=sample_data_names
# )

sxp = SeqExp(features=test_feature_table)#, classifications=test_classification_table, metadata=test_metadata_table)


# --------------- TEST1 --------------- #

# print(sxp.fx['feature_1'])
sxp2 = sxp.fx['feature_1':'feature_3']
print(sxp.features)
print(sxp2.features)
# print(sxp.fx[:, 'class_2'])
# print(sxp.fx[:,:])
