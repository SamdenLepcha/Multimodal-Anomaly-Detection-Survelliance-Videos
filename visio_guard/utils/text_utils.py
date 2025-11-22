# visio_guard/utils/text_utils.py

def clean_caption(text: str):
    """
    Simple caption cleanup.
    More complex logic unnecessary since SBERT can handle punctuation.
    """
    return text.strip().lower()
