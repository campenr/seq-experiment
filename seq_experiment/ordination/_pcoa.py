"""
Functions for performing principal coordinates analysis (PCoA). These functions are originally from the scikit-bio 
package, and are reproduced here with modifications for the following reason(s):

* scikit-bio does not play well with Windows, but seq-experiment should, so isolating this scikit-bio functionality
 allows us to make use of it without abandoning support for an entire operating system
* I do not wish to depend upon scikit-bio's various wrapper classes i.e. DistanceMatrix, OrdinationResults.

The intended input for the pcoa function is a distance matrix/dissimilarity matrix as a numpy array like object.

For original license information see `licenses/scikit-bio.txt`.

"""

import numpy as np
import pandas as pd
from warnings import warn
from scipy.linalg import eigh
from seq_experiment.ordination._ordination_result import OrdinationResult
# from ordination._utils import d_matrix_required


def e_matrix(distance_matrix):
    """Compute E matrix from a distance matrix.
    Squares and divides by -2 the input elementwise. Eq. 9.20 in
    Legendre & Legendre 1998."""
    return distance_matrix * distance_matrix / -2


def f_matrix(E_matrix):
    """Compute F matrix from E matrix.
    Centring step: for each element, the mean of the corresponding
    row and column are substracted, and the mean of the whole
    matrix is added. Eq. 9.21 in Legendre & Legendre 1998."""
    row_means = E_matrix.mean(axis=1, keepdims=True)
    col_means = E_matrix.mean(axis=0, keepdims=True)
    matrix_mean = E_matrix.mean()
    return E_matrix - row_means - col_means + matrix_mean

# - In cogent, after computing eigenvalues/vectors, the imaginary part
#   is dropped, if any. We know for a fact that the eigenvalues are
#   real, so that's not necessary, but eigenvectors can in principle
#   be complex (see for example
#   http://math.stackexchange.com/a/47807/109129 for details) and in
#   that case dropping the imaginary part means they'd no longer be
#   so, so I'm not doing that.


def pcoa(d_matrix, supress_warning=False):
    r"""Perform Principal Coordinate Analysis.
    Principal Coordinate Analysis (PCoA) is a method similar to PCA
    that works from distance matrices, and so it can be used with
    ecologically meaningful distances like UniFrac for bacteria.
    In ecology, the euclidean distance preserved by Principal
    Component Analysis (PCA) is often not a good choice because it
    deals poorly with double zeros (Species have unimodal
    distributions along environmental gradients, so if a species is
    absent from two sites at the same site, it can't be known if an
    environmental variable is too high in one of them and too low in
    the other, or too low in both, etc. On the other hand, if an
    species is present in two sites, that means that the sites are
    similar.).
    Parameters
    ----------
    distance_matrix : DistanceMatrix
        A distance matrix.
    Returns
    -------
    OrdinationResults
        Object that stores the PCoA results, including eigenvalues, the
        proportion explained by each of them, and transformed sample
        coordinates.
    See Also
    --------
    OrdinationResults
    Notes
    -----
    It is sometimes known as metric multidimensional scaling or
    classical scaling.
    .. note::
       If the distance is not euclidean (for example if it is a
       semimetric and the triangle inequality doesn't hold),
       negative eigenvalues can appear. There are different ways
       to deal with that problem (see Legendre & Legendre 1998, \S
       9.2.3), but none are currently implemented here.
       However, a warning is raised whenever negative eigenvalues
       appear, allowing the user to decide if they can be safely
       ignored.
    """

    E_matrix = e_matrix(d_matrix.values)

    # If the used distance was euclidean, pairwise distances
    # needn't be computed from the data table Y because F_matrix =
    # Y.dot(Y.T) (if Y has been centred).
    F_matrix = f_matrix(E_matrix)

    # If the eigendecomposition ever became a bottleneck, it could
    # be replaced with an iterative version that computes the
    # largest k eigenvectors.
    eigvals, eigvecs = eigh(F_matrix)

    # eigvals might not be ordered, so we order them (at least one
    # is zero). cogent makes eigenvalues positive by taking the
    # abs value, but that doesn't seem to be an approach accepted
    # by L&L to deal with negative eigenvalues. We raise a warning
    # in that case. First, we make values close to 0 equal to 0.
    negative_close_to_zero = np.isclose(eigvals, 0)
    eigvals[negative_close_to_zero] = 0
    if np.any(eigvals < 0) and not supress_warning:
        warn(
            "The result contains negative eigenvalues."
            " Please compare their magnitude with the magnitude of some"
            " of the largest positive eigenvalues. If the negative ones"
            " are smaller, it's probably safe to ignore them, but if they"
            " are large in magnitude, the results won't be useful. See the"
            " Notes section for more details. The smallest eigenvalue is"
            " {0} and the largest is {1}.".format(eigvals.min(),
                                                  eigvals.max()),
            RuntimeWarning
        )
    idxs_descending = eigvals.argsort()[::-1]
    eigvals = eigvals[idxs_descending]
    eigvecs = eigvecs[:, idxs_descending]

    # Scale eigenvalues to have lenght = sqrt(eigenvalue). This
    # works because np.linalg.eigh returns normalized
    # eigenvectors. Each row contains the coordinates of the
    # objects in the space of principal coordinates. Note that at
    # least one eigenvalue is zero because only n-1 axes are
    # needed to represent n points in an euclidean space.

    # If we return only the coordinates that make sense (i.e., that have a
    # corresponding positive eigenvalue), then Jackknifed Beta Diversity
    # won't work as it expects all the OrdinationResults to have the same
    # number of coordinates. In order to solve this issue, we return the
    # coordinates that have a negative eigenvalue as 0
    num_positive = (eigvals >= 0).sum()
    eigvecs[:, num_positive:] = np.zeros(eigvecs[:, num_positive:].shape)
    eigvals[num_positive:] = np.zeros(eigvals[num_positive:].shape)

    coordinates = eigvecs * np.sqrt(eigvals)
    proportion_explained = eigvals / eigvals.sum()

    axis_labels = ['PC%d' % i for i in range(1, eigvals.size + 1)]

    # TODO change this to return a custom ordination results object

    ord_result = OrdinationResult(
        method='PCoA',
        axes=pd.DataFrame(coordinates, columns=axis_labels, index=d_matrix.index),
        eigvals=pd.Series(eigvals, index=axis_labels),
        explained=pd.Series(proportion_explained, index=axis_labels)
    )

    return ord_result
