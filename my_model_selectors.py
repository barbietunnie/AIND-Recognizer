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
        """ Select best model for self.this_word based on BIC score
        for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        bic_scores = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states)
                log_likelihood = hmm_model.score(self.X, self.lengths)
                num_data_points = sum(self.lengths)
                num_params = ( num_states ** 2 ) + ( 2 * num_states * num_data_points ) - 1
                bic_score = (-2 * log_likelihood) + (num_params * np.log(num_data_points))
                bic_scores.append(tuple([bic_score, hmm_model]))
            except:
                pass

        if bic_scores:
            best_bic_score = min(bic_scores, key = lambda x: x[0])[1]
        else:
            best_bic_score = None
        
        return best_bic_score

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
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on DIC scores
        other_words = []
        dic_models = []
        dic_scores = []

        for word in self.words:
            if word != self.this_word:
                other_words.append(self.hwords[word])
        
        try:
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                hmm_model = self.base_model(num_states)
                initial_word_log_likelihood = hmm_model.score(self.X, self.lengths)
                dic_models.append((initial_word_log_likelihood, hmm_model))
        except:
            pass

        for _, dic_model in enumerate(dic_models):
            initial_word_log_likelihood, hmm_model = dic_model
            dic_score = initial_word_log_likelihood - np.mean([dic_model[1].score(word[0], word[1]) for word in other_words])
            dic_scores.append(tuple([dic_score, dic_model[1]]))

        if dic_scores:
            best_dic_score = max(dic_scores, key = lambda x: x[0])[1]
        else:
            best_dic_score = None

        return best_dic_score
            


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        split_method = KFold(n_splits = 3) # Use 3-folds
        log_likelihoods = []
        score_cvs = []
        
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Check if there's sufficient data
                if len(self.sequences) > 2:
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # Recombine training sequences split using KFold
                        self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)

                        # Recombine test sequences split using KFold
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                        hmm_model = self.base_model(num_states)
                        log_likelihood = hmm_model.score(X_test, lengths_test)
                else:
                    hmm_model = self.base_model(num_states)
                    log_likelihood = hmm_model.score(self.X, self.lengths)
                
                log_likelihoods.append(log_likelihood)

                # Find the average of the log Likelihood of CV fold
                score_cvs_avg = np.mean(log_likelihoods)
                score_cvs.append(tuple([score_cvs_avg, hmm_model]))
            except:
                pass

        if score_cvs:
            best_score_cv = max(score_cvs, key = lambda x: x[0])[1]
        else:
            best_score_cv = None

        return best_score_cv
