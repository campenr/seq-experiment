from seq_experiment.distance import DistanceMatrix


def d_matrix_required(f):
    def decorated(d_matrix, *args, **kwargs):
        if not isinstance(d_matrix, DistanceMatrix):
            raise(ValueError('d_matrix must be of type DistanceMatrix.'))
        else:
            f(d_matrix, *args, **kwargs)
