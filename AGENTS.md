# Repository Guidelines

## Project Structure & Module Organization
- Root config: `_config.yml` (production), `_config.dev.yml` (local overrides). Use `--config _config.yml,_config.dev.yml` when serving locally.
- Content: `_pages/` (site pages), `_posts/` (blog posts named `YYYY-MM-DD-title.md`), `_drafts/` (unpublished), section folders like `_talks/`, `_publications/`, `_portfolio/`.
- Theme/layout: `_layouts/`, `_includes/`, styles in `_sass/`, images in `images/`, static files in `files/`, JS/CSS and other assets in `assets/`.
- Generated output: `_site/` (do not edit or commit).

## Build, Test, and Development Commands
- Install dependencies: `bundle install` (Ruby gems), `npm install` (optional for JS minification).
- Serve locally: `bundle exec jekyll serve --livereload --config _config.yml,_config.dev.yml`.
- Build site: `bundle exec jekyll build` (outputs to `_site/`).
- JS build: `npm run build:js` to produce `assets/js/main.min.js`; `npm run watch:js` for on-change builds.
- Sanity checks: `bundle exec jekyll doctor` to detect common config/content issues.

## Coding Style & Naming Conventions
- Markdown: one H1 per page, meaningful front matter (`title`, `permalink`, `layout`).
- Posts: `YYYY-MM-DD-title.md` in `_posts/`; drafts in `_drafts/` without dates.
- Indentation: 2 spaces for YAML, HTML, Liquid, and SCSS.
- Liquid: prefer includes in `_includes/` and layouts in `_layouts/` to avoid duplication.
- Assets: place large binaries in `files/`; images under `images/` and reference with relative paths.

## Testing Guidelines
- Build verification: `bundle exec jekyll build` should complete without errors or warnings.
- Link and layout spot-check: open `_site/index.html` and a few key pages locally.
- Optional: run `jekyll doctor` before opening a PR.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (e.g., "Add publications page"); group related changes.
- Issues: reference or open an issue; for code/theme changes, link a diff or commit in a closed issue tagged `code change` per `CONTRIBUTING.md`.
- PRs: include summary, screenshots for visual changes, steps to reproduce/verify, and note any config updates.
- Exclude `_site/` and local artifacts from commits.

## Notes for Maintainers
- GitHub Pages: built via `github-pages` gem; prefer `bundle exec` to match production. Keep `Gemfile.lock` updated or remove if it causes conflicts.
