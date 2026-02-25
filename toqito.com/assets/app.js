const blogMarkdown = `# toqito blog spotlight

toqito supports practical workflows for **quantum information theory**, from matrix utilities to state/channel properties.

\`\`\`python
from toqito.states import bell
from toqito.state_metrics import fidelity

rho = bell(0)
sigma = bell(1)
score = fidelity(rho, sigma)
print(f"Fidelity: {score}")
\`\`\`

This page keeps a markdown-first workflow and a modern package-site layout.`;

const releasesMarkdown = `# Release feed

Subscribe to updates via:

- \`https://toqito.com/releases/index.xml\`

Future work can automate entries from GitHub Releases.`;

const newsMarkdown = `# News and updates

- New qutip-inspired homepage design.
- Improved onboarding and external resources.
- Dedicated RSS feeds for blog, news, and releases.`;

const quickLinks = [
  ["GitHub", "https://github.com/vprusso/toqito", "Source, issues, and contributions."],
  ["Documentation", "https://toqito.readthedocs.io", "Tutorials, API, and examples."],
  ["Project paper", "https://github.com/vprusso/toqito/blob/master/paper/paper.md", "Research context and citation details."],
  ["PyPI", "https://pypi.org/project/toqito", "Install and package release history."],
];

function escapeHtml(input) {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderMarkdown(md) {
  const lines = md.split("\n");
  let html = "";
  let inList = false;
  let inCode = false;

  for (const rawLine of lines) {
    const line = rawLine;

    if (line.startsWith("```")) {
      if (!inCode) {
        inCode = true;
        if (inList) {
          html += "</ul>";
          inList = false;
        }
        html += "<pre><code>";
      } else {
        inCode = false;
        html += "</code></pre>";
      }
      continue;
    }

    if (inCode) {
      html += `${escapeHtml(line)}\n`;
      continue;
    }

    if (!line.trim()) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      continue;
    }

    if (line.startsWith("# ")) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<h1>${renderInline(line.slice(2))}</h1>`;
      continue;
    }

    if (line.startsWith("- ")) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${renderInline(line.slice(2))}</li>`;
      continue;
    }

    if (inList) {
      html += "</ul>";
      inList = false;
    }
    html += `<p>${renderInline(line)}</p>`;
  }

  if (inList) {
    html += "</ul>";
  }
  return html;
}

function markdownCard(title, markdown, rss, archive) {
  return `
    <article class="markdown-card">
      <div class="card-header">
        <h3>${title}</h3>
        <div class="chip-links">
          <a href="${rss}">RSS</a>
          <a href="${archive}">Archive</a>
        </div>
      </div>
      ${renderMarkdown(markdown)}
    </article>
  `;
}

const root = document.getElementById("root");
root.innerHTML = `
  <div class="site-shell">
    <header class="hero">
      <div class="container">
        <nav class="top-nav">
          <a href="/" class="brand">toqito</a>
          <div class="nav-links">
            <a href="https://github.com/vprusso/toqito">GitHub</a>
            <a href="https://toqito.readthedocs.io">Docs</a>
            <a href="#feeds">Feeds</a>
          </div>
        </nav>
        <section class="hero-grid">
          <div>
            <p class="eyebrow">Open-source quantum information toolkit</p>
            <h1>Build, test, and analyze quantum information workflows in Python.</h1>
            <p class="lead">toqito provides utilities for states, channels, measurements, and optimization in a clean scientific workflow.</p>
            <div class="cta-row">
              <a class="btn btn-primary" href="https://github.com/vprusso/toqito">Get started</a>
              <a class="btn" href="https://toqito.readthedocs.io">Read docs</a>
            </div>
          </div>
          <div class="logo-panel">
            <img src="assets/logo.svg" alt="toqito logo" />
            <p>Homepage style inspired by QuTiP project website.</p>
          </div>
        </section>
      </div>
    </header>

    <main class="container">
      <section class="quick-links">
        ${quickLinks
          .map(
            ([title, href, desc]) => `
          <a class="quick-card" href="${href}">
            <h3>${title}</h3>
            <p>${desc}</p>
          </a>
        `,
          )
          .join("")}
      </section>

      <section id="feeds" class="markdown-grid">
        ${markdownCard("Blog", blogMarkdown, "blog/index.xml", "blog/")}
        ${markdownCard("Releases", releasesMarkdown, "releases/index.xml", "releases/")}
        ${markdownCard("News", newsMarkdown, "news/index.xml", "news/")}
      </section>
    </main>

    <footer class="footer">
      <div class="container">
        <p>© toqito project · landing page for toqito.com</p>
      </div>
    </footer>
  </div>
`;
