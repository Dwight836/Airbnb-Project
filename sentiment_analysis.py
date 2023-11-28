from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Instantiate the SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


def vader_analysis(text):
    """Returns a sentiment analysis score given a piece of text

    Args :
        text (string) : the piece of text to be analyzed

    Returns :
        compound (float) : a float representing the overall sentiment of the text
    """

    # Analyze the sentiment and retrieves values from dictionary
    compound = analyzer.polarity_scores(text)['compound']

    return compound
