import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # TODO implement the recognizer
    for idx in range(test_set.num_items):
    # for item in test_set.get_all_sequences():
        # X, lengths = test_set.get_item_Xlengths(item)
        X, lengths = test_set.get_item_Xlengths(idx)
        scores, highest_score, best_guess = {}, None, None

        for word, model in models.items():
            try:
                model_score = model.score(X, lengths)
                scores[word] = model_score
            except:
                # scores[word] = None
                scores[word] = float("-inf") # only numbers can be ordered in the max() function below
        
        probabilities.append(scores)
        guesses.append(max(scores, key = scores.get))

    
    return probabilities, guesses
    
