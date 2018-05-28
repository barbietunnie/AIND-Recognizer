import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        lowest_BIC, largest_num_data_pts = None, None
        for num_data_pts in range(self.min_n_components, self.max_n_components + 1):
            try:
                logL = self.base_model(num_data_pts).score(self.X, self.lengths)
                logN = np.log(len(self.X))
                p = num_data_pts * (num_data_pts-1) + 2 * len(self.X[0]) * num_data_pts
                BIC = -2 * logL + p * logN

                # The lower the BIC value the better the model
                if lowest_BIC > BIC or lowest_BIC is None:
                    lowest_BIC = BIC
                    largest_num_data_pts = num_data_pts
            except:
                pass
            
        if largest_num_data_pts is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(largest_num_data_pts)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        highest_DIC, largest_num_data_pts = None, None
        for num_data_pts in range(self.min_n_components, self.max_n_components + 1):
            try:
                logPXi = self.base_model(log_P_X_i = self.base_model(num_components).score(self.X, self.lengths)).score(self.X, self.lengths)
                # Sum of all log(P(X)) without i
                sum_logPX = 0.
                all_words = list(self.words.keys())
                M = len(all_words)
                words.remove(self.this_word)

                for word in all_words:
                    try:
                        # All model selectors without i
                        all_model_selectors = ModelSelector(self.words, self.hwords, word, self.n_constant, self.min_n_components, self.max_n_components, self.random_state, self.verbose)

                        sum_logPX += all_model_selectors.base_model(num_data_pts).score(all_model_selectors.X, all_model_selectors.lengths)
                    except:
                        M = M - 1

                DIC = logPXi - sum_logPX / (M - 1)

                if highest_DIC < DIC or highest_DIC is None:
                    highest_DIC = DIC
                    largest_num_data_pts = num_data_pts
            except:
                pass
            
        if largest_num_data_pts is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(largest_num_data_pts)
            


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # The best number of data points selected and highest average logL value
        largest_num_data_pts, largest_avg_logL = None, None

        for num_data_pts in range(self.min_n_components, self.max_n_components + 1):
            logL_sum = 0.
            logL_count = 0

            try:
                split_algorithm = KFold(3) # Use 3-folds
                for cv_train_idx, cv_test_idx in split_algorithm.split(self.sequences):
                    X, lengths = combine_sequences(cv_train_idx,self.sequences)

                    try:
                        logL_sum += self.base_model(num_data_pts).score(X, lengths)
                        logL_count += 1
                    except:
                        pass

                if logL_count > 0:
                    avg_logL = logL_sum / logL_count
                    if largest_avg_logL < avg_logL or largest_avg_logL is None:
                        largest_avg_logL = avg_logL
                        largest_num_data_pts = num_data_pts
            except:
                pass
        
        if largest_num_data_pts is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(largest_num_data_pts)
