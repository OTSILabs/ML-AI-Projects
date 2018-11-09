'''
# Authors: -, -
# License: MIT
'''
# importing the pickle package
import pickle
# import the 'predict_review' to predict the sentiment of new review
from amazon_sentiment_analysis import predict_review
# function to predict the sentiment of new review and return the sentiment of given review
def sentiment(review):
    '''
    the sentiment prediction of the review
    '''
    with open('Path_To_Pickle_File//pickle_model.pkl', 'rb')as file_code:
        pickle_run = pickle.load(file_code)
        predict_vector = predict_review.data_prediction(review)
        result = pickle_run.predict(predict_vector)
        if str(result.tolist()[0]) == "1":
            return "Positive"
        else:
            return "Negative"
