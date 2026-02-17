# toqito Landing Page

Official landing page for the [toqito](https://github.com/vprusso/toqito) project - an open-source Python library for quantum information theory.

## Overview

This is a Hugo-based static site using the PaperMod theme, designed for automatic deployment to GitHub Pages. The site features a clean, responsive design with sections for blog posts, news updates, and release notes.

## Prerequisites

- **Hugo Extended v0.146.0 or higher** (the project includes a pre-downloaded Hugo binary)
- **Git** (for version control and theme submodule)

## Quick Start

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/Akri-A/toqito
cd toqito_landing_page
```

**Note**: The `--recurse-submodules` flag is important as the PaperMod theme is included as a git submodule.

### 2. Run Local Development Server

```bash
./hugo server -D
```

Visit http://localhost:1313 to see your site. The server will auto-reload when you make changes.

### 3. Build for Production

```bash
./hugo --minify
```

The built site will be in the `public/` directory.

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── deploy.yml       # GitHub Actions deployment workflow
├── archetypes/              # Content templates for new posts
├── content/                 # Site content
│   ├── blog/               # Blog posts
│   ├── news/               # News updates
│   └── releases/           # Release notes
├── layouts/                 # Custom HTML templates
│   └── index.html          # Custom homepage layout
├── static/                  # Static assets
│   ├── images/
│   │   └── logo.svg        # toqito logo
│   └── favicon.svg         # Site favicon
├── themes/                  # Hugo themes
│   └── PaperMod/           # PaperMod theme (git submodule)
├── .gitignore              # Git ignore patterns
├── hugo.toml               # Hugo configuration
├── hugo                    # Hugo binary (v0.146.0 extended)
├── Roadmap.md              # Project roadmap and implementation plan
└── README.md               # This file
```

## Deployment

### Automatic Deployment to GitHub Pages

This project is configured for automatic deployment via GitHub Actions. Every push to the `main` branch triggers a deployment.

**Setup Steps:**

1. **Push to GitHub**:
   ```bash
   git push origin main
   ```

2. **Enable GitHub Pages**:
   - Go to repository **Settings** → **Pages**
   - Under "Build and deployment", select **Source**: "GitHub Actions"
   - The workflow will automatically deploy the site

3. **Access Your Site**:
   - Your site will be available at: `https://<username>.github.io/<repository-name>/`
   - Check the **Actions** tab to monitor deployment progress

### Custom Domain (Optional)

To use a custom domain like `toqito.com`:

1. **Update `baseURL`** in `hugo.toml`:
   ```toml
   baseURL = 'https://toqito.com/'
   ```

2. **Configure GitHub Pages**:
   - Go to repository **Settings** → **Pages**
   - Under "Custom domain", enter your domain (e.g., `toqito.com`)
   - Follow GitHub's DNS configuration instructions

3. **Update DNS Records** with your domain provider:
   - Add a `CNAME` record pointing to `<username>.github.io`
   - Or configure A records as per [GitHub's documentation](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site)

## Adding Content

### Create a New Blog Post

```bash
./hugo new blog/my-post-title.md
```

Edit the generated file in `content/blog/my-post-title.md`:

```yaml
---
title: "My Post Title"
date: 2025-02-17
draft: false
description: "A brief description of the post"
tags: ["quantum", "python"]
---

Your content here in Markdown format.
```

### Create a News Update

```bash
./hugo new news/project-update.md
```

### Create a Release Note

```bash
./hugo new releases/v1.0.0.md
```

## Configuration

The main configuration file is `hugo.toml`. Key settings:

```toml
baseURL = 'https://example.org/'  # Update this for your domain
title = 'toqito'
theme = 'PaperMod'

[params]
  defaultTheme = "auto"           # Light/dark theme
  ShowReadingTime = true
  ShowCodeCopyButtons = true

[minify]
  minifyOutput = true             # Minify HTML/CSS/JS

[[menu.main]]                     # Navigation menu items
  name = "Blog"
  url = "/blog/"
  weight = 1
```

## Customization

### Update the Homepage

Edit `layouts/index.html` to customize the landing page layout, hero section, or call-to-action buttons.

### Modify Theme Settings

The PaperMod theme supports many customization options. See the [PaperMod documentation](https://github.com/adityatelange/hugo-PaperMod/wiki) for details.

To override theme files, create the same file structure under `layouts/` or `static/`.

### Change Logo or Favicon

Replace:
- `static/images/logo.svg` - Main logo
- `static/favicon.svg` - Browser favicon

## Maintenance

### Update Hugo

To update the Hugo binary:

```bash
# Download the latest version
wget https://github.com/gohugoio/hugo/releases/download/v0.XXX.X/hugo_extended_0.XXX.X_linux-amd64.tar.gz

# Extract and replace
tar -xzf hugo_extended_0.XXX.X_linux-amd64.tar.gz hugo
rm hugo_extended_0.XXX.X_linux-amd64.tar.gz

# Verify version
./hugo version
```

### Update PaperMod Theme

```bash
git submodule update --remote --merge
./hugo server  # Test locally
git commit -am "chore: update PaperMod theme"
git push
```

## Troubleshooting

### Build Fails with "Module not compatible"

Ensure you're using Hugo v0.146.0 or higher:
```bash
./hugo version
```

### Submodule Issues

If the theme is missing:
```bash
git submodule update --init --recursive
```

### Changes Not Showing

Clear Hugo's cache:
```bash
rm -rf public/ resources/
./hugo --minify
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test locally with `./hugo server -D`
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Resources

- **Hugo Documentation**: [gohugo.io/documentation](https://gohugo.io/documentation/)
- **PaperMod Theme**: [github.com/adityatelange/hugo-PaperMod](https://github.com/adityatelange/hugo-PaperMod)
- **GitHub Pages**: [docs.github.com/pages](https://docs.github.com/en/pages)
- **toqito Project**: [github.com/vprusso/toqito](https://github.com/vprusso/toqito)

## License

See the [toqito project](https://github.com/vprusso/toqito) for license information.

---

**Built with Hugo and PaperMod** • **Deployed on GitHub Pages**
