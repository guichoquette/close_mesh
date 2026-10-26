"""Count the words of each section of abstract.tex against the ISMRM limits.

The body (Introduction to Conclusion) is limited to 750 words, the Impact
Statement to 40 and the Synopsis to 100. LaTeX commands, comments and
``\\todo{...}`` notes are not counted.
"""

import re
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "abstract.tex"
BODY = ["Introduction", "Methods", "Results", "Discussion", "Conclusion"]
LIMITS = {"Impact Statement": 40, "Synopsis": 100}


def plain_words(text):
    """Return the words of a LaTeX fragment, without commands or comments."""
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\todo\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", "", text)
    text = re.sub(r"\$[^$]*\$", " x ", text)
    text = re.sub(r"\\(textbf|emph|textit)\{", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", text)
    text = re.sub(r"[{}~\\]", " ", text)
    return [w for w in text.split() if re.search(r"[A-Za-z0-9]", w)]


def sections(tex):
    """Split the document into {section title: content}."""
    parts = re.split(r"\\section\*\{([^}]*)\}", tex)
    return {parts[i].split(" (")[0]: parts[i + 1] for i in range(1, len(parts), 2)}


def main():
    secs = sections(TEX.read_text(encoding="utf-8"))
    ok = True
    for name, limit in LIMITS.items():
        n = len(plain_words(secs.get(name, "")))
        ok &= n <= limit
        print(f"{name:<18} {n:4d} / {limit}")
    total = 0
    for name in BODY:
        n = len(plain_words(secs.get(name, "")))
        total += n
        print(f"  {name:<16} {n:4d}")
    ok &= total <= 750
    print(f"{'Body':<18} {total:4d} / 750")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
