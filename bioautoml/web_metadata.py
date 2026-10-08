"""Install public metadata without replacing Streamlit's versioned frontend assets."""

import argparse
from pathlib import Path
import re


START = '<!-- BioAutoML-FAST metadata -->'
END = '<!-- /BioAutoML-FAST metadata -->'


def apply_metadata(index_html, template):
    """Return a patched index, preserving the installed scripts, styles and root."""
    flags = re.IGNORECASE | re.DOTALL
    heads = re.findall(r'<head\b[^>]*>(.*?)</head\s*>', template, flags)
    if len(heads) != 1:
        raise ValueError('Metadata template must contain exactly one head.')
    titles = re.findall(r'<title\b[^>]*>.*?</title\s*>', heads[0], flags)
    fallback = re.findall(r'<noscript\b[^>]*>.*?</noscript\s*>', template, flags)
    if len(titles) != 1 or len(fallback) != 1:
        raise ValueError('Metadata template must contain one title and one noscript fallback.')
    # The template is metadata-only: never transplant versioned frontend assets.
    if re.search(r'<(?:script|style|base)\b|\b(?:src|on\w+)\s*=', heads[0], flags):
        raise ValueError('Executable content and assets are not allowed in the metadata template.')
    links = re.findall(r'<link\b[^>]*>', heads[0], flags)
    if any(not re.search(r'\brel\s*=\s*[\"\']canonical[\"\']', link, flags) for link in links):
        raise ValueError('Only canonical links are allowed in the metadata template.')

    result = re.sub(re.escape(START) + r'.*?' + re.escape(END) + r'\n?', '', index_html, flags=re.DOTALL)
    result, count = re.subn(r'<title\b[^>]*>.*?</title\s*>', lambda _: titles[0], result, flags=flags)
    if count != 1:
        raise ValueError('Installed Streamlit index must contain exactly one title.')
    metadata = heads[0].replace(titles[0], '').strip()
    result, count = re.subn(r'</head\s*>', lambda _: f'{START}\n{metadata}\n{END}\n</head>', result, flags=flags)
    if count != 1:
        raise ValueError('Installed Streamlit index must contain exactly one closing head.')
    result, count = re.subn(r'<noscript\b[^>]*>.*?</noscript\s*>', lambda _: fallback[0], result, flags=flags)
    if count != 1:
        raise ValueError('Installed Streamlit index must contain exactly one noscript fallback.')
    return result


def main():
    import streamlit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'App/index.html')
    args = parser.parse_args()
    index = Path(streamlit.__file__).resolve().parent / 'static/index.html'
    original = index.read_text(encoding='utf-8')
    updated = apply_metadata(original, args.template.read_text(encoding='utf-8'))
    if updated != original:
        index.write_text(updated, encoding='utf-8')
    print(f'Public metadata installed in {index}; frontend assets preserved.')


if __name__ == '__main__':
    main()
