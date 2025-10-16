import re
from typing import List
import string
import contractions
from nltk.corpus import stopwords
import nltk

class TextCleaner:
    """Text cleaner for sentiment analysis - participants can enhance."""
    
    def __init__(self) -> None:
        """Initialize the text cleaner."""
        self.stop_words = set(stopwords.words('english'))
        self.remove_urls_flag = True
        self.remove_mentions_flag = True
        self.remove_hashtags_flag = True
        self.remove_stopwords_flag = True
        self.expand_contractions_flag = True
        pass
        
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
            
        TODO for participants:
        - Implement comprehensive text cleaning
        - Remove URLs, mentions, hashtags
        - Handle special characters and emojis
        - Remove stopwords
        - Handle contractions
        - Normalize repeated characters
        """
        if not text:
            return ""
        
        #Lowercase
        text = text.lower()

        #Remove URLs
        if self.remove_urls_flag:
            text = self.remove_urls(text)

        #Remove mentions
        if self.remove_mentions_flag:
            text = self.remove_mentions(text)

        #Remove hashtags
        if self.remove_hashtags_flag:
            text = self.remove_hashtags(text)

        #Expand contractions
        if self.expand_contractions_flag:
            text = self.handle_contractions(text)

        #Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)

        #Remove stopwords
        if self.remove_stopwords_flag:
            text = self.remove_stopwords(text)

        #Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text
        
    def clean_texts(self, texts: List[str]) -> List[str]:
        """
        Clean a list of text strings.
        
        Args:
            texts: List of texts to clean
            
        Returns:
            List of cleaned text strings
            
        TODO for participants:
        - Implement batch processing optimization
        - Add parallel processing for large datasets
        - Add progress tracking for long operations
        """
        return [self.clean_text(text) for text in texts]
        
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs removed
            
        TODO for participants:
        - Implement URL detection and removal
        - Handle different URL formats
        - Preserve text around URLs
        """
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
    def remove_mentions(self, text: str) -> str:
        """
        Remove @mentions from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with mentions removed
            
        TODO for participants:
        - Remove @username patterns
        - Handle edge cases
        """
        return re.sub(r'@\w+', '', text)
        
    def remove_hashtags(self, text: str) -> str:
        """
        Remove #hashtags from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with hashtags removed
            
        TODO for participants:
        - Remove #hashtag patterns
        - Decide whether to keep hashtag text or remove entirely
        """
        return re.sub(r'#', '', text)
        
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
            
        TODO for participants:
        - Implement stopword removal
        - Use NLTK or custom stopword lists
        - Handle different languages
        """
        tokens = text.split()
        filtered = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered)
        
    def handle_contractions(self, text: str) -> str:
        """
        Expand contractions in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with contractions expanded
            
        TODO for participants:
        - Expand contractions (don't -> do not)
        - Handle various contraction forms
        - Maintain text meaning
        """
        return contractions.fix(text)


# TODO for participants - Additional text preprocessing features:
# - Stemming and lemmatization
# - Part-of-speech tagging
# - Named entity recognition
# - Sentiment-aware preprocessing
# - Language detection
# - Text normalization
# - Spell checking and correction
# - Emoji handling and sentiment mapping
# - Slang and abbreviation expansion
# - Text augmentation techniques
