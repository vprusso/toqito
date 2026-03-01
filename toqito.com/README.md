# toqito.com homepage

This folder contains a lightweight landing page for `toqito.com` with a modern visual style inspired by package websites such as QuTiP.

Use a static server as follows:

```bash
cd toqito.com
python -m http.server 8000
```

Then open `http://localhost:8000` (or `http://localhost:8000/toqito.com/` if serving from repo root).

## What is included

- `index.html`: Root page shell.
- `assets/app.js`: Client-side rendering logic and markdown-style content rendering.
- `assets/styles.css`: Hero/CTA/cards/code preview styling.
- `assets/logo.svg`: toqito logo.
- `blog/index.xml`, `news/index.xml`, `releases/index.xml`: RSS feeds.
