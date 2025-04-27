from textblob import TextBlob
import pandas as pd
import re
from collections import Counter
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer."""
        pass
    
    def clean_text(self, text):
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        



        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        text = re.sub(r'@\w+|#\w+', '', text)
        
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob.
        Returns a tuple of (polarity, subjectivity).
        Polarity: -1.0 to 1.0 (negative to positive)
        Subjectivity: 0.0 to 1.0 (objective to subjective)
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return (0, 0)
        
        analysis = TextBlob(cleaned_text)
        return (analysis.sentiment.polarity, analysis.sentiment.subjectivity)
    
    def classify_sentiment(self, polarity):
        """Classify sentiment as positive, negative, or neutral."""
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def analyze_dataframe(self, df, text_column, candidate_column=None, date_column=None):
        """
        Analyze sentiment for text in a pandas DataFrame.
        
        Parameters:
        - df: pandas DataFrame containing the data
        - text_column: name of column containing text to analyze
        - candidate_column: optional column name for candidate mentioned
        - date_column: optional column name for dates
        
        Returns:
        - DataFrame with sentiment analysis results
        """
        result_df = df.copy()
        
        sentiments = result_df[text_column].apply(self.get_sentiment)
        result_df['polarity'] = sentiments.apply(lambda x: x[0])
        result_df['subjectivity'] = sentiments.apply(lambda x: x[1])
        result_df['sentiment'] = result_df['polarity'].apply(self.classify_sentiment)
        
        return result_df
    
    def extract_keywords(self, texts, n=10):
        """Extract most common keywords from a list of texts."""
        all_words = []
        
        stopwords = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with', 
                      'that', 'this', 'it', 'be', 'as', 'are', 'was', 'an', 'by', 'at']
        
        for text in texts:
            if not isinstance(text, str):
                continue
                
            cleaned = self.clean_text(text)
            words = [w for w in cleaned.split() if w not in stopwords and len(w) > 2]
            all_words.extend(words)
        
        return Counter(all_words).most_common(n)
    
    def get_sentiment_statistics(self, df, group_by=None):
        """
        Calculate sentiment statistics, optionally grouped by a column.
        
        Parameters:
        - df: DataFrame with sentiment analysis results
        - group_by: Optional column to group by (e.g., 'candidate')
        
        Returns:
        - DataFrame with sentiment statistics
        """
        if group_by is None:
            stats = pd.DataFrame({
                'avg_polarity': [df['polarity'].mean()],
                'std_polarity': [df['polarity'].std()],
                'positive_percent': [(df['sentiment'] == 'positive').mean() * 100],
                'neutral_percent': [(df['sentiment'] == 'neutral').mean() * 100],
                'negative_percent': [(df['sentiment'] == 'negative').mean() * 100],
                'count': [len(df)]
            })
            return stats
        else:
            return df.groupby(group_by).agg(
                avg_polarity=('polarity', 'mean'),
                std_polarity=('polarity', 'std'),
                positive_percent=('sentiment', lambda x: (x == 'positive').mean() * 100),
                neutral_percent=('sentiment', lambda x: (x == 'neutral').mean() * 100),
                negative_percent=('sentiment', lambda x: (x == 'negative').mean() * 100),
                count=('polarity', 'count')
            ).reset_index() 