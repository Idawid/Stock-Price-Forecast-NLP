import nltk.sentiment.util
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from backend.sentiment_analysis.sentiment.SentimentAnalysisBase import SentimentAnalysisBase


class VaderSentiment (SentimentAnalysisBase):

    nltk.downloader.download('vader_lexicon')

    def __init__(self) -> None:

        self.vader = SentimentIntensityAnalyzer()
        super().__init__()

    def calc_sentiment_score(self):

        self.df['sentiment'] = self.df['Headline'].apply(
            self.vader.polarity_scores)
        self.df['sentiment_score'] = self.df['sentiment'].apply(
            lambda x: x['compound'])
        super().calc_sentiment_score()
