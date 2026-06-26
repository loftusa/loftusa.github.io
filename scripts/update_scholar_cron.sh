#!/bin/bash
# Cron wrapper: fetch Google Scholar pubs, commit & push if changes found.
set -euo pipefail

REPO_DIR="/Users/alex/Library/CloudStorage/Dropbox/Code/loftusa.github.io"
cd "$REPO_DIR"

# Pull latest
git pull --ff-only

# Run the fetch script
uv run scripts/fetch_scholar.py

# Check if about.md changed
if git diff --quiet _pages/about.md; then
    echo "No changes."
    exit 0
fi

# Commit and push
git add _pages/about.md
git commit -m "Auto-update publications from Google Scholar

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push
echo "Pushed new publications."
