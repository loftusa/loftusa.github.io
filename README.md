A Github Pages template for academic websites. This was forked (then detached) by [Stuart Geiger](https://github.com/staeiou) from the [Minimal Mistakes Jekyll Theme](https://mmistakes.github.io/minimal-mistakes/), which is © 2016 Michael Rose and released under the MIT License. See LICENSE.md.

I think I've got things running smoothly and fixed some major bugs, but feel free to file issues or make pull requests if you want to improve the generic template / theme.

### Note: if you are using this repo and now get a notification about a security vulnerability, delete the Gemfile.lock file. 

# Instructions

1. Register a GitHub account if you don't have one and confirm your e-mail (required!)
1. Fork [this repository](https://github.com/academicpages/academicpages.github.io) by clicking the "fork" button in the top right. 
1. Go to the repository's settings (rightmost item in the tabs that start with "Code", should be below "Unwatch"). Rename the repository "[your GitHub username].github.io", which will also be your website's URL.
1. Set site-wide configuration and create content & metadata (see below -- also see [this set of diffs](http://archive.is/3TPas) showing what files were changed to set up [an example site](https://getorg-testacct.github.io) for a user with the username "getorg-testacct")
1. Upload any files (like PDFs, .zip files, etc.) to the files/ directory. They will appear at https://[your GitHub username].github.io/files/example.pdf.  
1. Check status by going to the repository settings, in the "GitHub pages" section
1. (Optional) Use the Jupyter notebooks or python scripts in the `markdown_generator` folder to generate markdown files for publications and talks from a TSV file.

See more info at https://academicpages.github.io/

## To run locally (not on GitHub Pages, to serve on your own computer)

1. Clone the repository and made updates as detailed above
1. Make sure you have ruby-dev, bundler, and nodejs installed: `sudo apt install ruby-dev ruby-bundler nodejs`
1. Run `bundle clean` to clean up the directory (no need to run `--force`)
1. Run `bundle install` to install ruby dependencies. If you get errors, delete Gemfile.lock and try again.
1. Run `bundle exec jekyll liveserve` to generate the HTML and serve it from `localhost:4000` the local server will automatically rebuild and refresh the pages on change.

# Changelog -- bugfixes and enhancements

There is one logistical issue with a ready-to-fork template theme like academic pages that makes it a little tricky to get bug fixes and updates to the core theme. If you fork this repository, customize it, then pull again, you'll probably get merge conflicts. If you want to save your various .yml configuration files and markdown files, you can delete the repository and fork it again. Or you can manually patch. 

To support this, all changes to the underlying code appear as a closed issue with the tag 'code change' -- get the list [here](https://github.com/academicpages/academicpages.github.io/issues?q=is%3Aclosed%20is%3Aissue%20label%3A%22code%20change%22%20). Each issue thread includes a comment linking to the single commit or a diff across multiple commits, so those with forked repositories can easily identify what they need to patch.

## Resume Chatbot Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      USER'S BROWSER                           │
│                                                              │
│  _pages/chat.md (Jekyll)  +  assets/js/chat.js              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  • getOrCreateUserId() → localStorage (name or UUID)   │  │
│  │  • Optional name input: saved to localStorage,         │  │
│  │    used as user_id if set (hidden on main page after)  │  │
│  │  • On page load: GET /health (pre-warms backend)       │  │
│  │  • On send: POST /chat?logging=true {user_id, message} │  │
│  │  • Reads streaming response, renders markdown           │  │
│  │    (marked.js + DOMPurify)                             │  │
│  │  • On 429: "refresh to start new conversation"         │  │
│  │  • apiBase auto-detects localhost vs production         │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬────────────────────────────────┘
                              │ HTTPS (CORS: alex-loftus.com,
                              │         localhost:4000,
                              │         127.0.0.1:4000/3863)
                              ▼
┌──────────────────────────────────────────────────────────────┐
│           FLY.IO  (iad region, 1 shared CPU, 1GB RAM)         │
│           min_machines_running = 1 (always warm)              │
│           auto_start/auto_stop enabled for scaling             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │          FastAPI  (experiments/chat_api.py)             │  │
│  │                                                        │  │
│  │  Startup:                                              │  │
│  │    • Module-level: load_collection() opens ChromaDB,   │  │
│  │      Cerebras() client init, reads system_prompt.txt   │  │
│  │      + resume.txt                                      │  │
│  │    • @app.on_event("startup"): warmup_rag() runs a     │  │
│  │      dummy query to pre-warm ONNX embedding model      │  │
│  │    • ONNX model (all-MiniLM-L6-v2, ~80MB) is          │  │
│  │      pre-cached in Docker image at build time          │  │
│  │                                                        │  │
│  │  Endpoints:                                            │  │
│  │    POST /chat?logging={true|false}                     │  │
│  │         → RAG retrieval → LLM call → streaming tokens  │  │
│  │    POST /reset ── clear user's conversation            │  │
│  │    GET  /health ── status, api_key check, rag count    │  │
│  │    GET  /logs/download ── Bearer auth, returns JSONL   │  │
│  │                                                        │  │
│  │  In-Memory State (OrderedDict, LRU):                   │  │
│  │    conversations[user_id] = {                          │  │
│  │      messages: [{role, content}, ...],                 │  │
│  │      cost_usd: float                                   │  │
│  │    }                                                   │  │
│  │    • Max 1,000 concurrent users (LRU eviction)         │  │
│  │    • Max 50 turns per conversation (oldest trimmed)    │  │
│  │    • $2.00 cost budget per conversation → HTTP 429     │  │
│  │    • Pricing: $2.25/M input, $2.75/M output tokens     │  │
│  │                                                        │  │
│  │  Message Construction (per request):                   │  │
│  │    1. [system_prompt.txt + resume.txt]  (system msg)   │  │
│  │    2. [top-5 RAG chunks, max_distance=1.3] (system msg)│  │
│  │    3. [conversation history]            (user/asst)    │  │
│  └──┬──────────┬────────────────────────┬─────────────────┘  │
│     │          │                        │                    │
│     ▼          ▼                        ▼                    │
│  ┌─────────┐ ┌─────────────────┐ ┌──────────────────────┐   │
│  │ChromaDB │ │  Cerebras API   │ │  Persistent Volume    │   │
│  │ (RAG)   │ │  zai-glm-4.7    │ │  (Fly.io mount)       │   │
│  │         │ │                 │ │  /app/experiments/     │   │
│  │ 448     │ │  Streaming      │ │  logs/chat_logs.jsonl  │   │
│  │ chunks  │ │  completion     │ │                        │   │
│  │         │ │                 │ │  Log fields:           │   │
│  │ Sources:│ └─────────────────┘ │  {user_id, message,    │   │
│  │ thesis, │                     │   response, latency,   │   │
│  │ papers, │                     │   token_count, cost,   │   │
│  │ GitHub  │                     │   rag_chunks_count,    │   │
│  │ READMEs,│                     │   rag_sources}         │   │
│  │ talks   │                     └──────────────────────┘   │
│  │         │                                                │
│  │ Embedder│                                                │
│  │ all-Mini│                                                │
│  │ LM-L6-v2│                                                │
│  │ (ONNX)  │                                                │
│  └─────────┘                                                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   OFFLINE TOOLING (local)                      │
│                                                              │
│  sync_logs.py ─── GET /logs/download → local chat_logs.jsonl │
│  analyze_logs.py ─ latency metrics, daily active users,      │
│                    Tufte-style plots                           │
│  run_eval.py ──── fact-checking eval against eval_dataset.json│
│                                                              │
│  Inspect AI evals (experiments/evals/):                       │
│    jailbreak eval ─ 96 samples, 9 attack categories           │
│    RAG eval ─────── 34 samples, factual + hallucination       │
│    baseline eval ── same questions without RAG for comparison  │
└──────────────────────────────────────────────────────────────┘
```

**Key design decisions**: Server-side conversation storage (prevents context injection via client-supplied history), per-conversation cost budget instead of rate limiting (aligns with actual LLM costs), `min_machines_running=1` on Fly.io (eliminates cold starts; ONNX embedding model pre-cached in Docker image to keep startup under 4s if machine does restart), JSONL logging on persistent volume (simple, append-only, survives redeploys), RAG via ChromaDB with all-MiniLM-L6-v2 embeddings (448 chunks, top-5 retrieval with max_distance=1.3, ~150MB RSS for onnxruntime).

## Stocks Page (experimental)

This repo includes an experimental Next.js app directory to render a stocks page with a minimal Tufte-style chart.

- Dev: `npm run dev` then visit `http://localhost:3000/stocks`.
- API: `GET /api/stocks` returns `{ weeklyChanges: [{ ticker, change }] }`.
- Admin sync (stubbed): `curl -fsSL "http://localhost:3000/api/admin/sync?secret=$SYNC_TOKEN"`.
- Security model: only `{symbol, quantity}` intended for storage in a future sync; no PII.

Notes: The holdings list is hard-coded in `lib/holdings.ts`. Schwab sync is a no-op stub in `lib/schwabSync.ts` and protected by `middleware.ts` requiring `process.env.SYNC_TOKEN`.
