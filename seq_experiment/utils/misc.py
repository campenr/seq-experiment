import hashlib

def hash_seqs(sequences):
    """
    Generates hexdigest of Sha1 hash for each sequence in a list of sequences.

    This function is useful for generating sequence specific identifiers that allow for easier comparison of features
    from multiple sequencing runs or sequence processing runs.

    """

    new_sequences = list()
    for seq in sequences:
        # get sequence string and encode using UTF-8 for consistency
        seq = seq.encode('UTF-8')

        # get sha1 hash of sequence
        hash_ = hashlib.sha1()
        hash_.update(seq)
        hash_hex = hash_.hexdigest()
        new_sequences.append(hash_hex)

    return new_sequences
