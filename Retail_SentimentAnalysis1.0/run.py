'''
    main program to run API
'''
from flask import Flask
from amazon_sentiment_analysis import pickle_run
API = Flask(__name__)
@API.route("/<string:review>")
def amazon_review(review):
    '''
    function to run the prediction program to predict the sentiment of review
    Amazon_review(review)
    params:
        review : the review given by the user
    return sentiment string on web page
    '''
    sentiment = pickle_run.sentiment(review)
    return sentiment
if __name__ == '__main__':
    API.run(debug=True)
