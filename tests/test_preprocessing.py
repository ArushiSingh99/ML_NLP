"""
Tests for text preprocessing utilities - participants can enhance.
"""

import pytest
from src.preprocessing.text_cleaner import TextCleaner


class TestTextCleaner:
    """Test cases for TextCleaner class."""
    
    @pytest.fixture
    def cleaner(self):
        """Create a TextCleaner instance for testing."""
        return TextCleaner()
    
    def test_initialization(self, cleaner):
        """Test TextCleaner initialization."""
        assert cleaner is not None
        assert hasattr(cleaner, "stop_words")
    
    def test_clean_text_basic(self, cleaner):
        """Test basic text cleaning."""
        text = "I LOVE this product! It's AMAZING!!!"
        cleaned = cleaner.clean_text(text)
        
        # Basic test - participants can enhance
        assert cleaned == "i love this product! it's amazing!!!"
        expected = "love product amazing"
        assert cleaned == expected
    
    def test_clean_text_empty(self, cleaner):
        """Test cleaning empty text."""
        assert cleaner.clean_text("") == ""
        assert cleaner.clean_text("   ") == ""
        assert cleaner.clean_text(None) == ""
    
    def test_clean_texts_batch(self, cleaner):
        """Test batch text cleaning."""
        texts = [
            "I LOVE this!",
            "This is TERRIBLE!",
            "It's okay, nothing special."
        ]
        cleaned = cleaner.clean_texts(texts)
        
        assert len(cleaned) == 3
        assert cleaned[0] == "i love this!"
        assert cleaned[1] == "this is terrible!"
        assert cleaned[2] == "it's okay, nothing special."

        expected = ["love", "terrible", "okay nothing special"]
        assert cleaned == expected

    #Removed Placeholder functions

    def test_remove_urls(self, cleaner):
        text = "Check out https://example.com for more info"
        result = cleaner.remove_urls(text)
        assert result == "Check out  for more info"

    def test_remove_mentions(self, cleaner):
        text = "Hey @john, what do you think?"
        result = cleaner.remove_mentions(text)
        assert result == "Hey , what do you think?"

    def test_remove_hashtags(self, cleaner):
        text = "This is #amazing and #awesome!"
        result = cleaner.remove_hashtags(text)
        assert result == "This is amazing and awesome!"

    def test_remove_stopwords(self, cleaner):
        text = "I am very happy with this product"
        result = cleaner.remove_stopwords(text)
        assert result == "happy product"

    def test_handle_contractions(self, cleaner):
        text = "I don't like this, it's terrible"
        result = cleaner.handle_contractions(text)
        assert result == "I do not like this, it is terrible"


# TODO for participants - Additional preprocessing tests to implement:
# - Test comprehensive text cleaning pipeline
# - Test edge cases (special characters, emojis, numbers)
# - Test performance with large datasets
# - Test different languages
# - Test stemming and lemmatization
# - Test part-of-speech tagging
# - Test named entity recognition
# - Test sentiment-aware preprocessing
# - Test text normalization
# - Test spell checking and correction
