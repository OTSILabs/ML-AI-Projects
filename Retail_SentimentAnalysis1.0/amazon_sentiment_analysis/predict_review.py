'''
# Authors: -, -
# License: MIT
'''
# importing re package
import re
# importing numpy package
import numpy as np
# importing NLTK package's
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# importing gensim package's
from gensim.models import Word2Vec

def data_prediction(text):
    '''
    converts the new reveiw into vectors
    '''
	# function to perform the cleaning of data using  wordCloud corpus
    def clean_text(raw_text, remove_stopwords=False, stemming=False, split_text=False):
        '''
        Convert a raw review to a cleaned review
        '''
        letters_only = re.sub("[^a-zA-Z]", " ", raw_text)  # remove non-character
        words = letters_only.lower().split() # convert to lower case

        if remove_stopwords: # remove stopword
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]

        if stemming is True: # stemming
            stemmer = SnowballStemmer('english')
            words = [stemmer.stem(w) for w in words]

        if split_text is True:  # split text
            return words

        return " ".join(words)

    # function to perform tokenization
    def parse_sent(text, tokenizer, remove_stopwords=False):
        '''
        Parse text into sentences
        '''
        raw_sentences = tokenizer.tokenize(text.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) is not 0:
                sentences.append(clean_text(raw_sentence, remove_stopwords, split_text=True))
        return sentences

	# function to obtain features from the word2vec
    def make_feature_vec(texts, model, num_features):
        '''
        Transform a review to a feature vector by averaging feature vectors of words
        appeared in that review and in the volcabulary list created
        '''
        feat_vector = np.zeros((num_features,), dtype="float32")
        nwords = 0.
        index2word_set = set(model.wv.index2word) # index2word is the volcabulary list of the
                                                  # Word2Vec model
        zero_vector = True
        for word in texts:
            if word in index2word_set:
                nwords = nwords + 1.
                feat_vector = np.add(feat_vector, model[word])
                zero_vector = False
        if zero_vector is False:
            feat_vector = np.divide(feat_vector, nwords)
        return feat_vector

	# function to obtain average features from the sentance
    def get_avg_feature_vecs(texts, model, num_features):
        '''
        Transform all reviews to feature vectors using make_feature_vec()
        '''
        counter = 0
        review_feature_vectors = np.zeros((len(texts), num_features), dtype="float32")
        for text in texts:
            review_feature_vectors[counter] = make_feature_vec(text, model, num_features)
            counter = counter + 1
        return review_feature_vectors
	# object to store the new reviews
    new_text = [text]
    review_data = []
	# loop to clean the review and append to a list
    for txt in new_text:
        review_data.append(clean_text(txt))
	# nltk.download()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	# Parse each review in the training set into sentences
    sentences = []
    print(review_data)
    for review in review_data:
        sentences += parse_sent(review, tokenizer)

    num_feature = 300  #embedding dimension

	# perform word2vec
    w2v = Word2Vec.load('w2v_300features_10minwordcounts_10context')
    x_test_cleaned = []
    for review in new_text:
        x_test_cleaned.append(clean_text(review, remove_stopwords=True, split_text=True))
    test_vector = get_avg_feature_vecs(x_test_cleaned, w2v, num_feature)
    return test_vector
