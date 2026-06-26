# loftusa.github.io

Personal website for Alex Loftus. Jekyll (Minimal Mistakes theme) on GitHub Pages + FastAPI chat API on Fly.io.

## Build & Serve

```bash
bundle exec jekyll serve                    # Local dev server (localhost:4000)
bundle exec jekyll serve --config _config.yml,_config.dev.yml  # With dev overrides
bundle exec jekyll build                    # Static build → _site/
```

## Chat API (experiments/)

- `experiments/chat_api.py` — FastAPI app, Cerebras `zai-glm-4.7` model, ChromaDB RAG
- Fly.io app: `llm-resume-restless-thunder-9259`
- Deploy: `cd experiments && fly deploy`
- Endpoints: `POST /chat` (streaming), `POST /reset`, `GET /logs/download`
- $2 USD cost cap per conversation

## Key Structure

```
_config.yml / _config.dev.yml   Jekyll config (prod / dev overrides)
_data/navigation.yml            Nav menu
_pages/                         Site pages (about.md = homepage)
_posts/                         Blog posts (YYYY-MM-DD-title.md)
_drafts/                        Unpublished posts
assets/js/chat.js               Chat frontend
experiments/                    Chat API, RAG, evals
perfumes/                       Perfume Atlas pipeline (/perfumes/ + /perfumes/analyses/; has own README)
```

## Interactive maps

- `/networks/` — co-authorship graph (`_pages/coauthorship.html`, `assets/js/coauthorship-network.js`)
- `/perfumes/` — perfume similarity atlas, 24k bottles by smell (`_pages/perfumes.html` +
  `_pages/perfumes-analyses.html`; renderer `assets/js/perfumes-atlas.js`; built by
  `perfumes/build_atlas.py` → `assets/data/perfumes-*.json`). See `perfumes/README.md`.
- Both use `layout: fullscreen` (scrolling pages use `layout: bare`).

## Blog Posts

- Format: `_posts/YYYY-MM-DD-title.md`
- MathJax: add `mathjax: true` to front matter
- Markdown: kramdown with GFM, rouge syntax highlighting

## Config

- `_config.yml` — production (url: https://alex-loftus.com)
- `_config.dev.yml` — local overrides (url: localhost:4000, analytics off)
- `_data/navigation.yml` — top nav links

## Package Management

- **Ruby/Jekyll**: `bundle install` / `bundle exec`
- **Python**: `uv` (use `uv run <script>.py`)
- **JS**: `npm`
