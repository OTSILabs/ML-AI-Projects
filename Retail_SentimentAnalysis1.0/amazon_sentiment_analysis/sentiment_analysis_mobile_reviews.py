'''
# Authors: -, -
# License: MIT
'''
#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis on Amazon Product Reviews of Mobile Phones

# Part 1 - Import Data

# Pandas : Pandas is a Python package providing fast, flexible, and
# expressive data structures designed to make working with “relational”
# or “labelled” data both easy and intuitive.
#
# Matplotlib & Seaborn : These packages provide a high-level interface
# for drawing attractive and informative statistical graphics.
#
# Scikit learn : sklearn is used to perform the machine learning algorithms,
# It provides a range of supervised and unsupervised learning algorithms in Python.
#
# WordCloud : WordCLoud is packed with lot's of words in different sizes which
# represents frequency of the word.
#
# BeautifulSoup : BeautifulSoup is  a Python library. It is used for parsing XML
# and HTML.
#
# Nltk : Nltk is used primarily for general NLP tasks (tokenization, POS tagging,
# parsing, etc.).
#
# Gensim : Gensim is well optimized and highly specialized library to perform WORD2VEC
#
# Pickle : Pickle is a standard Python library that handles serialization

# importing pickle package
import pickle
# importing re package
import re
# importing pandas package
import pandas as pd
# importing numpy package
import numpy as np
# importing matplotlib and seaborn packages
import matplotlib.pyplot as plt
import seaborn as sns
# importing wordCloud packages
from wordcloud import WordCloud
# importing scikit learn packages's
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
# import BeautifulSoup package
from bs4 import BeautifulSoup
# importing NLTK package's
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
# importing gensim package's
from gensim.models import Word2Vec

# Load Amazon mobile review data
DF = pd.read_csv('Amazon_Unlocked_Mobile.csv')
DF.head()


# We are using Amazon mobile review dataset

# Step 2 - Cleansing data


# function to perform the cleaning of data using  wordCloud corpus
def clean_text(raw_text, remove_stopwords=False, stemming=False, split_text=False):
    '''
    Convert a raw review to a cleaned review
    '''
    text = BeautifulSoup(raw_text, 'lxml').get_text()  #remove html
    letters_only = re.sub("[^a-zA-Z]", " ", text)  # remove non-character
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


# step 3 - Exploratory Data Analysis
#
# Let's now visualize the Amazon Mobile data


# Summary of the data
print("Summary statistics of numerical features : \n", DF.describe())
# Total number of reviews
print("\nTotal number of reviews: ", len(DF))
# Total number of unique brand
print("\nTotal number of unique products: ", len(list(set(DF['Product Name']))))
# Percentage of positive, negative and Neutral reviews
print("\nPercentage of reviews with neutral sentiment : {:.2f}%"      \
      .format(DF[DF['Rating'] == 3]["Reviews"].count()/len(DF)*100))
print("\nPercentage of reviews with positive sentiment : {:.2f}%"     \
      .format(DF[DF['Rating'] > 3]["Reviews"].count()/len(DF)*100))
print("\nPercentage of reviews with negative sentiment : {:.2f}%"     \
      .format(DF[DF['Rating'] < 3]["Reviews"].count()/len(DF)*100))

# Drawing Bar Plot on distribution of rating
plt.figure(figsize=(12, 8))
DF['Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Count')

# Drawing Bar Plot on number of reviews for top 20 brands
BRANDS = DF["Brand Name"].value_counts()
plt.figure(figsize=(12, 8))
BRANDS[:20].plot(kind='bar')
plt.title("Number of Reviews for Top 20 Brands")

# Drawing Bar Plot on number of reviews for top 50 products
PRODUCTS = DF["Product Name"].value_counts()
plt.figure(figsize=(12, 8))
PRODUCTS[:50].plot(kind='bar')
plt.title("Number of Reviews for Top 50 Products")

# Drawing Bar Plot on distribution of review length
REVIEW_LENGTH = DF["Reviews"].dropna().map(lambda x: len(x))
plt.figure(figsize=(12, 8))
REVIEW_LENGTH.loc[REVIEW_LENGTH < 1500].hist()
plt.title("Distribution of Review Length")
plt.xlabel('Review length (Number of character)')
plt.ylabel('Count')

# Step 4 - Splitting the Data into Train & Test Datasets
#
# Before we go ahead with splitting the datasets, let us
# create a copy of the original dataset and have a look
# at the different categorical variables & the values they contain.


# We have taken sample's from overall data
DF = DF.sample(frac=0.1, random_state=0)

# Drop missing values
DF.dropna(inplace=True)

# Remove any 'neutral' ratings equal to 3
DF = DF[DF['Rating'] != 3]

# Encode 4s and 5s as 1 (positive sentiment) and 1s and 2s as 0 (negative sentiment)
DF['Sentiment'] = np.where(DF['Rating'] > 3, 1, 0)
DF.head()

# Split data into training set and validation
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(DF['Reviews'], \
                                      DF['Sentiment'], test_size=0.1, random_state=0)


print('Load %d training examples and %d validation examples. \n' \
         %(X_TRAIN.shape[0], X_TEST.shape[0]))
print('Show a review in the training set : \n', X_TRAIN.iloc[52])


# performing cleansing again after splitting the data for further training


X_TRAIN_CLEANED = []
X_TEST_CLEANED = []

# final sentence after performing cleansing
for d in X_TRAIN:
    X_TRAIN_CLEANED.append(clean_text(d))
print('Show a cleaned review in the training set : \n', X_TRAIN_CLEANED[50])

for d in X_TEST:
    X_TEST_CLEANED.append(clean_text(d))

# Split review text into parsed sentences uisng NLTK's punkt tokenizer
# nltk.download()
TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')

# Parse each review in the training set into sentences
SENTENCES = []
for review in X_TRAIN_CLEANED:
    SENTENCES += parse_sent(review, TOKENIZER)

print('%d parsed sentence in the training set\n'  %len(SENTENCES))
print('Show a parsed sentence in the training set : \n', SENTENCES[10])

NUM_FEATURES = 300  #embedding dimension
MIN_WORD_COUNT = 5
NUM_WORKERS = 4
CONTEXT = 5
DOWNSAMPLING = 1e-3

print("Training Word2Vec model ...\n")
# perform word2vec
W2V = Word2Vec(SENTENCES, workers=NUM_WORKERS, size=NUM_FEATURES, \
          min_count=MIN_WORD_COUNT, window=CONTEXT, sample=DOWNSAMPLING)
W2V.init_sims(replace=True)
W2V.save("w2v_300features_10minwordcounts_10context") #save trained word2vec model

print("Number of words in the vocabulary list : %d \n" %len(W2V.wv.index2word)) #4016
print("Show first 10 words in the vocalbulary list  vocabulary list: \n", W2V.wv.index2word[0:10])

# function to obtain features from the word2vec
def make_feature_vec(text, model, num_feat):
    '''
    Transform a review to a feature vector by averaging feature vectors of words
    appeared in that review and in the volcabulary list created
    '''
    feat_vector = np.zeros((num_feat,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word) #index2word is the
                                              # volcabulary list of the Word2Vec model
    zero_vector = True
    for word in text:
        if word in index2word_set:
            nwords = nwords + 1.
            feat_vector = np.add(feat_vector, model[word])
            zero_vector = False
    if zero_vector is False:
        feat_vector = np.divide(feat_vector, nwords)
    return feat_vector

# function to obtain average features from the sentance
def get_avg_feature_vecs(texts, model, num_feat):
    '''
    Transform all reviews to feature vectors using make_feature_vec()
    '''
    counter = 0
    review_feature_vectors = np.zeros((len(texts), num_feat), dtype="float32")
    for text in texts:
        review_feature_vectors[counter] = make_feature_vec(text, model, num_feat)
        counter = counter + 1
    return review_feature_vectors

# Get feature vectors for training set
X_TRAIN_CLEANED = []
for review in X_TRAIN:
    X_TRAIN_CLEANED.append(clean_text(review, remove_stopwords=True, split_text=True))
TRAIN_VECTOR = get_avg_feature_vecs(X_TRAIN_CLEANED, W2V, NUM_FEATURES)
print("Training set : %d feature vectors with %d dimensions" %TRAIN_VECTOR.shape)


# Get feature vectors for validation set
X_TEST_CLEANED = []
for review in X_TEST:
    X_TEST_CLEANED.append(clean_text(review, remove_stopwords=True, split_text=True))
TEST_VECTOR = get_avg_feature_vecs(X_TEST_CLEANED, W2V, NUM_FEATURES)
print("Validation set : %d feature vectors with %d dimensions" %TEST_VECTOR.shape)


# Step 5 - Applying Algorithm and Train the model on training data


# Random Forest Classifier
RF = RandomForestClassifier(n_estimators=100)
RF.fit(TRAIN_VECTOR, Y_TRAIN)


# Step 6 - Scoring the model on Test Dataset

PREDICTIONS = RF.predict(TEST_VECTOR)


# Step 7 - Evaluating the model


# function to evaluate the model
def model_evaluation(predict):
    '''
    Print model evaluation to predicted result
    '''
    print("\nAccuracy on validation set: {:.4f}".format(accuracy_score(Y_TEST, predict)))
    print("\nAUC score : {:.4f}".format(roc_auc_score(Y_TEST, predict)))
    print("\nClassification report : \n", metrics.classification_report(Y_TEST, predict))
    print("\nConfusion Matrix : \n", metrics.confusion_matrix(Y_TEST, predict))

# to predict the model
model_evaluation(PREDICTIONS)

CONFUSION_MATRIX = metrics.confusion_matrix(Y_TEST, PREDICTIONS)
sns.heatmap(CONFUSION_MATRIX, annot=True, fmt='g', xticklabels=\
              ["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.ylabel('True label')
plt.xlabel('Predicted label')

# The Word Cloud represents the most used Positive words and Negative words
# from selected Brand reviews

# function to create the Word Cloud from the reviews of given brand
def create_word_cloud(brand, sentiment):
    '''
    creating a word cloud on selected Brand
    '''
    try:
        df_brand = DF.loc[DF['Brand Name'].isin([brand])]
        df_brand_sample = df_brand.sample(frac=0.1)
        word_cloud_collection = ''

        if sentiment == 1:
            df_reviews = df_brand_sample[df_brand_sample["Sentiment"] == 1]["Reviews"]

        if sentiment == 0:
            df_reviews = df_brand_sample[df_brand_sample["Sentiment"] == 0]["Reviews"]

        for val in df_reviews.str.lower():
            tokens = nltk.word_tokenize(val)
            tokens = [word for word in tokens if word not in stopwords.words('english')]
            for words in tokens:
                word_cloud_collection = word_cloud_collection + words + ' '

        wordcloud = WordCloud(max_font_size=50, width=500, height=300)\
		              .generate(word_cloud_collection)
        plt.figure(figsize=(20, 20))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
    except:
        pass

# to call the WordCloud function for displaying most used positive words from the
# reviews of choosen brand
create_word_cloud(brand='Samsung', sentiment=1)

# to call the WordCloud function for displaying most used negative words from the
# reviews of choosen brand
create_word_cloud(brand='Samsung', sentiment=0)

# Step 8 - Deploying the model

# function to save the model in picklfile
PICKLE_FILE = "pickle_model.pkl"
with open(PICKLE_FILE, 'wb') as file_code:
    pickle.dump(RF, file_code)

# We deployed the above model on Flask server.
#
# Follow the below link to predict car price for a new set of specifications:
# link : http://127.0.0.1:5000/<review>
