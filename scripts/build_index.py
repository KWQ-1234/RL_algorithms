#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECTIONS_DIR = ROOT / "_sections"
TEMPLATE_PATH = ROOT / "templates" / "index.template.html"
OUTPUT_PATH = ROOT / "index.html"


def read(path: Path) -> str:
    return path.read_text()


def main() -> None:
    template = read(TEMPLATE_PATH)
    header = read(SECTIONS_DIR / "00-article-header.html")
    footer = read(SECTIONS_DIR / "99-article-footer.html")
    section_files = sorted(
        p
        for p in SECTIONS_DIR.glob("*.html")
        if p.name not in {"00-article-header.html", "99-article-footer.html"}
    )
    sections = "".join(read(path) for path in section_files)

    html = template
    html = html.replace("<!-- CONTENT:article-header -->", header)
    html = html.replace("<!-- CONTENT:sections -->", sections)
    html = html.replace("<!-- CONTENT:article-footer -->", footer)
    OUTPUT_PATH.write_text(html)
    print(f"Wrote {OUTPUT_PATH} from {len(section_files)} sections.")


if __name__ == "__main__":
    main()
