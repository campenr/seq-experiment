def tidy_mothur_classifications(classification_data):
    """Renoves the reported classification percentage agreements that Mothur appends to the classificaitons."""

    new_classification_data = classification_data.copy(deep=True)
    for column in classification_data:
        new_classification_data[column] = new_classification_data[column].str.rsplit('(', 1, expand=True)[0]

    return new_classification_data


def tidy_mothur_sequences(sequence_data):
    """Renoves gap characters from sequences imported from mothur."""

    new_sequence_data = sequence_data.copy(deep=True)
    for column in sequence_data:
        new_sequence_data[column] = sequence_data[column].str.replace('-', '')

    return new_sequence_data
