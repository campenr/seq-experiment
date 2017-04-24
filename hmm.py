import seq_experiment

x = seq_experiment.SeqExp('1')

print(x.feature_table)
print(x.classification_table)

x.classification_table = '2'

print(x.feature_table)
print(x.classification_table)
