"""
validate_output.py — Output format validation for generated captions.

Standalone importable module. No I/O, no side effects.
All functions are pure and raise no exceptions for any string input.
"""


def count_body_words(text: str) -> int:
    """Count whitespace-delimited tokens that do not begin with '#'.

    Args:
        text: The full caption text (body + hashtags).

    Returns:
        Number of tokens that are not hashtags.
    """
    return sum(1 for token in text.split() if not token.startswith("#"))


def count_hashtags(text: str) -> int:
    """Count whitespace-delimited tokens that begin with '#'.

    Args:
        text: The full caption text (body + hashtags).

    Returns:
        Number of tokens that are hashtags.
    """
    return sum(1 for token in text.split() if token.startswith("#"))


def validate(text: str) -> bool:
    """Return True iff body word count is in [30, 40] and hashtag count is exactly 5.

    Args:
        text: The full caption text (body + hashtags).

    Returns:
        True if 30 <= count_body_words(text) <= 40 and count_hashtags(text) == 5,
        False otherwise.
    """
    return 30 <= count_body_words(text) <= 40 and count_hashtags(text) == 5
