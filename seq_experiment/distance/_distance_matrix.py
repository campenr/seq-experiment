from seq_experiment.ordination import pcoa, nmds, meta_nmds
import pandas as pd
# from ordination import OrdinationResult

class DistanceMatrix(pd.DataFrame):
    """
    A pandas.DataFrame that stores distance matrix/dissimilarity matrix data.
    
    This object exists instead of simply storing the distance matrix/dissimilarity matrix data in a pandas.DataFrame
    to provide specific methods for downstream analysis, and so that we can ensure certain properties inherent to these 
    types of matrices are adhered to so that assumptions made byt the downstream analyses are true.   
    
    """

    @property
    def _constructor(self):
        return DistanceMatrix

    # This doesn't work
    # @property
    # def _constructor_sliced(self):
        # raise(ValueError('can\'t slice a DistanceMatrix object.'))

    def __init__(self, *args, **kwargs):

        # get distance matrix specific arguments and append to self after super
        metric = kwargs.pop('metric', None)
        super(DistanceMatrix, self).__init__(*args, **kwargs)
        self.metric = metric

    def ordinate(self, method='meta_nmds', *args, **kwargs):
        """
        Performs an ordination on the DistanceMatrix object.

        Uses the data stored in the DistanceMatrix object.

        :param method: the ordination method to use
        :type method: str

        :return: 

        """

        ord_methods = {
            'pcoa': pcoa,
            'nmds': nmds,
            'meta_nmds': meta_nmds
        }

        # check for valid method
        if method in ord_methods:
            # perform ordination with specified method, passing additional arguments to the ordination functions
            ord_result = ord_methods[method](d_matrix=self)

            # format ord_result and return
            ord_result.metric = self.metric
            return ord_result
        else:
            raise(ValueError('must supply a valid ordination method.'))
