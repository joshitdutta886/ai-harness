"""Small text helpers."""

SMALL_WORDS = {"a", "an", "the", "of", "and", "or", "in", "on"}


def slugify(title):
    """Turn a title into a URL slug: lowercase words joined by single hyphens."""
    out = []
    for ch in title.lower():
        out.append(ch if ch.isalnum() else "-")
    return "".join(out)


def title_case(text):
    """Capitalise words, keeping small words lowercase (except the first)."""
    return " ".join(w.capitalize() for w in text.split())
