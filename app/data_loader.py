import re
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.datasets import fetch_20newsgroups


@dataclass
class Document:
    doc_id: int
    text: str
    category: str
    original_index: int


_HEADER_RE = re.compile(
    r"^(From|Subject|Organization|Lines|Distribution|NNTP-Posting-Host|"
    r"Reply-To|Sender|X-Newsreader|Nntp-Posting-Host|Article-I\.D\.|"
    r"Keywords|Summary|Expires|Supersedes|References|Message-ID|Date|"
    r"In-Reply-To|Originator|News-Software):\s*.*$",
    re.MULTILINE | re.IGNORECASE,
)

_QUOTE_PREAMBLE_RE = re.compile(
    r"^.*(?:writes?|wrote|said)\s*:\s*$", re.MULTILINE | re.IGNORECASE
)

_QUOTED_LINE_RE = re.compile(r"^\s*>+.*$", re.MULTILINE)

_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")

_URL_RE = re.compile(r"https?://\S+|ftp://\S+|gopher://\S+", re.IGNORECASE)

_PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}")

_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

_SIG_SEP_RE = re.compile(r"^-- ?\s*$", re.MULTILINE)


def clean_text(raw: str) -> str:
    text = raw

    text = _HEADER_RE.sub("", text)

    text = _QUOTE_PREAMBLE_RE.sub("", text)
    text = _QUOTED_LINE_RE.sub("", text)

    sig_match = _SIG_SEP_RE.search(text)
    if sig_match:
        text = text[: sig_match.start()]

    text = _EMAIL_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _PATH_RE.sub("", text)

    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    return text


def load_and_clean_dataset() -> Tuple[List[Document], List[str]]:
    print("Fetching 20 Newsgroups dataset")
    dataset = fetch_20newsgroups(
        subset="all",
        remove=(),
        shuffle=False,
    )

    categories = list(dataset.target_names)
    documents: List[Document] = []
    skipped = 0

    for idx, (raw_text, label_id) in enumerate(zip(dataset.data, dataset.target)):
        cleaned = clean_text(raw_text)

        if len(cleaned) < 50:
            skipped += 1
            continue

        documents.append(Document(
            doc_id=len(documents),
            text=cleaned,
            category=categories[label_id],
            original_index=idx,
        ))

    print(f"Loaded {len(documents)} documents ({skipped} dropped as too short)")
    return documents, categories
