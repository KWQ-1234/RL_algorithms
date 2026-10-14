# Section Editing Workflow

Each file in this directory is one editable fragment of the article body.

- `00-article-header.html` contains the opening title/byline block.
- `01-*.html` through `32-*.html` are the top-level `<section>` blocks, in article order.
- `99-article-footer.html` contains the article footer.

After editing any fragment, rebuild the full page from the repository root:

```bash
python3 scripts/build_index.py
```

The generated output is `index.html`.
