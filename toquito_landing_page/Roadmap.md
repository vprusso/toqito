# Toquito Landing Page Roadmap

This document outlines the development roadmap for the Toquito project landing page.
The goal is to create a clean, deployment-ready site using Hugo and the PaperMod theme for GitHub Pages.

---

## Phase 0: Current Status ✅ PARTIALLY COMPLETE

**What's Actually Done:**
- ✅ Git repository initialized
- ✅ Hugo v0.139.3+extended installed
- ✅ Project directories scaffolded (archetypes/, assets/, content/, data/, i18n/, layouts/, static/, themes/)
- ✅ Logo SVG created (`logo.svg` at root, needs to be moved)

**What's NOT Done (contrary to previous claims):**
- ❌ PaperMod theme NOT installed (themes/ directory is empty)
- ❌ hugo.toml only has 3 default lines (no theme configuration)
- ❌ No commits made yet (only logo.svg staged)

**Remaining Tasks:**
- [ ] Create `.gitignore` for Hugo (ignore public/, resources/, .hugo_build.lock)
- [ ] Make initial commit
- [ ] Create basic `README.md`

**Acceptance Criteria:** Initial commit made, .gitignore in place

---

## Phase 1: Get Hugo + PaperMod Working 🎯 NEXT PRIORITY

### Install Theme
- [ ] Add PaperMod as git submodule:
  ```bash
  git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
  ```

### Configure hugo.toml
- [ ] Update configuration to use PaperMod theme:
  ```toml
  baseURL = 'https://example.org/'  # Author will configure their domain
  languageCode = 'en-us'
  title = 'toqito'
  theme = 'PaperMod'

  [params]
    defaultTheme = "auto"
    ShowReadingTime = true
    ShowShareButtons = false
    ShowCodeCopyButtons = true
  ```

### Organize Assets
- [ ] Move `logo.svg` from root to `/static/images/logo.svg`
- [ ] Generate basic `favicon.ico` from logo (optional but nice to have)

### Test It Works
- [ ] Run `hugo server -D` → verify site loads at http://localhost:1313
- [ ] Run `hugo --minify` → verify clean build in public/ directory
- [ ] Commit changes

**Acceptance Criteria:** Site builds and runs locally with PaperMod theme working

---

## Phase 2: Build Landing Page Structure

### Homepage Design
- [ ] Create custom homepage layout: `/layouts/index.html`
  - Hero section with Toquito logo (|toqito⟩)
  - Tagline about quantum information toolkit
  - "Get Started" button → links to documentation
  - "View on GitHub" button → links to repository

### Navigation Menu
- [ ] Add navigation menu items to hugo.toml:
  ```toml
  [[menu.main]]
    name = "Blog"
    url = "/blog/"
    weight = 1

  [[menu.main]]
    name = "News"
    url = "/news/"
    weight = 2

  [[menu.main]]
    name = "Releases"
    url = "/releases/"
    weight = 3

  [[menu.main]]
    name = "GitHub"
    url = "https://github.com/vprusso/toqito"
    weight = 4

  [[menu.main]]
    name = "Docs"
    url = "https://toqito.readthedocs.io/"
    weight = 5
  ```

**Acceptance Criteria:** Homepage looks good, navigation works

---

## Phase 3: Add Content Sections

### Create Section Index Pages
- [ ] **Blog Section**: Create `/content/blog/_index.md` with basic title and description
- [ ] **News Section**: Create `/content/news/_index.md` with basic title and description
- [ ] **Releases Section**: Create `/content/releases/_index.md` with basic title and description

### Add Sample Content
- [ ] Create welcome blog post: `/content/blog/welcome.md`
  - Introduction to toqito project
  - What quantum computing problems it solves
  - Links to docs and GitHub

### Configure RSS Feeds
- [ ] Add RSS output configuration to hugo.toml:
  ```toml
  [outputs]
    home = ["HTML", "RSS"]
    section = ["HTML", "RSS"]
  ```

**Acceptance Criteria:** All three sections exist and are accessible with RSS feeds generated

---

## Phase 4: GitHub Pages Deployment Setup

### Create GitHub Actions Workflow
- [ ] Create `.github/workflows/deploy.yml`:
  ```yaml
  name: Deploy to GitHub Pages

  on:
    push:
      branches: [main]

  jobs:
    deploy:
      runs-on: ubuntu-latest
      permissions:
        contents: write
      steps:
        - uses: actions/checkout@v4
          with:
            submodules: true
            fetch-depth: 0

        - name: Setup Hugo
          uses: peaceiris/actions-hugo@v2
          with:
            hugo-version: '0.139.3'
            extended: true

        - name: Build
          run: hugo --minify

        - name: Deploy
          uses: peaceiris/actions-gh-pages@v3
          with:
            github_token: ${{ secrets.GITHUB_TOKEN }}
            publish_dir: ./public
  ```

### Configure Repository
- [ ] Push code to GitHub
- [ ] Enable GitHub Pages in repository settings:
  - Go to Settings → Pages
  - Set source to "Deploy from a branch"
  - Select `gh-pages` branch
  - Note the GitHub Pages URL for testing

### Optional: Custom Domain
- [ ] (For author to configure later) Add custom domain in GitHub Pages settings
- [ ] (For author to configure later) Update `baseURL` in hugo.toml to match domain

**Acceptance Criteria:** Pushing to main branch automatically deploys to GitHub Pages

---

## Phase 5: Final Polish & Handoff

### Documentation
- [ ] Create comprehensive `README.md`:
  - Project description (Toquito landing page)
  - Prerequisites (Hugo v0.139.3+extended)
  - Local development setup (`hugo server -D`)
  - Deployment info (auto-deploys via GitHub Actions)
  - Instructions for configuring custom domain

### Build Optimization
- [ ] Add minify configuration to hugo.toml:
  ```toml
  [minify]
    minifyOutput = true
  ```

### Final Checks
- [ ] Verify build works: `hugo --minify`
- [ ] Check all internal links work locally
- [ ] Ensure mobile responsive (PaperMod handles this by default)
- [ ] Verify all menu items link correctly
- [ ] Commit all final changes

### Handoff Notes for Author
- [ ] Document in README that `baseURL` should be updated if using custom domain
- [ ] Include instructions for GitHub Pages custom domain configuration
- [ ] Note that content can be added to blog/, news/, and releases/ sections

**Acceptance Criteria:** Project is complete, documented, and ready for author to deploy and customize

---

## Critical Files Created/Modified

### Phase 0-1 (Setup):
1. `.gitignore` - Hugo ignores (public/, resources/, .hugo_build.lock)
2. `hugo.toml` - Theme config, menu, basic params
3. `.gitmodules` - PaperMod submodule reference

### Phase 2 (Landing Page):
4. `layouts/index.html` - Custom homepage with hero section
5. `content/_index.md` - Homepage metadata (optional)

### Phase 3 (Content):
6. `content/blog/_index.md` - Blog section page
7. `content/news/_index.md` - News section page
8. `content/releases/_index.md` - Releases section page
9. `content/blog/welcome.md` - Sample blog post

### Phase 4-5 (Deploy & Docs):
10. `.github/workflows/deploy.yml` - GitHub Pages auto-deployment
11. `README.md` - Setup and deployment instructions

---

## Verification Commands

### After Phase 1 (Theme Installation):
```bash
# Verify theme is installed
ls themes/PaperMod

# Test local development
hugo server -D
# Visit http://localhost:1313 - should see PaperMod theme working
```

### After Phase 2-3 (Content):
```bash
# Test full build
hugo --minify

# Check all sections exist
ls public/blog public/news public/releases

# Verify RSS feeds generated
ls public/blog/index.xml public/news/index.xml public/releases/index.xml
```

### After Phase 4 (Deployment):
```bash
# Push to main branch
git push origin main

# Check GitHub Actions:
# Go to repository → Actions tab → verify "Deploy to GitHub Pages" workflow runs successfully

# Visit GitHub Pages URL to verify site is live
```

---

## Success Criteria

### Project is "Deployment Ready" when:
- ✅ PaperMod theme installed and working
- ✅ Homepage has custom layout with logo and call-to-action buttons
- ✅ Navigation menu configured with all sections
- ✅ Blog, News, and Releases sections exist
- ✅ RSS feeds are generated for all sections
- ✅ GitHub Actions workflow auto-deploys to GitHub Pages
- ✅ README documents setup and deployment process
- ✅ Site builds cleanly with `hugo --minify`
- ✅ Mobile responsive (PaperMod default)

### Left for Author to Configure:
- Domain name (update `baseURL` in hugo.toml if using custom domain)
- Custom domain configuration in GitHub Pages settings (optional)
- Additional content for blog, news, and releases sections
- Any theme customization or branding tweaks

---

## Notes

- **No SEO optimization**: Focusing on functional deployment, not search optimization
- **No analytics**: Author can add later if needed
- **No monitoring**: Basic deployment only
- **Simple and clean**: PaperMod theme handles responsive design and accessibility by default
- **Ready to extend**: Author can easily add more content sections or customize theme

This roadmap provides a practical, working Hugo site ready for GitHub Pages deployment.
