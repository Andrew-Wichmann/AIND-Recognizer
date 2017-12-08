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
    # import pudb;pudb.set_trace()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for key in test_set.get_all_Xlengths():
        best_guess = None 
        best_prob = float('-inf')
        prob_for_this_word = {}
        for word in models:
            if models[word] != None:
                prob = models[word].score(test_set.get_all_Xlengths()[key][0], test_set.get_all_Xlengths()[key][1])
                prob_for_this_word[word] = prob
                if prob > best_prob:
                    best_prob = prob
                    best_guess = word
        probabilities.append(prob_for_this_word)
        guesses.append(best_guess)
    return probabilities, guesses
