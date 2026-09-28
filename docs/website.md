# Project homepage

[Live homepage](https://tiannuo-yang.github.io/expgym/) · [← README](../README.md)

The homepage is a dependency-free static site in [`site/`](../site). GitHub Pages publishes that directory through [the Pages workflow](../.github/workflows/pages.yml). Pushes to `main` that change the site or the workflow deploy automatically; the workflow also supports manual dispatch.

## Preview locally

From the repository root:

```bash
python -m http.server 4173 --directory site
```

Open `http://localhost:4173`. No Node package installation or build step is needed.

## Edit

- `index.html`: content, links, and copyable quick-start commands.
- `styles.css` and `product.css`: responsive layout and product sections.
- `app.js`: charts, budget/model selectors, keyboard-accessible tabs, and copy buttons.
- `data.js`: PoolAct scaling values.
- `assets/`: manuscript PDF, downloadable CSV snapshots, and [data provenance](../site/assets/provenance.json).

Use relative paths for site assets so the page works under GitHub Pages' `/expgym/` project path. The page links to local research snapshots because the original manuscript source repository is private. Preserve the original revision and metric definitions in the provenance file when refreshing results.

Before pushing, check local links and run:

```bash
node --check site/app.js
node --check site/data.js
git diff --check
```

Repository settings must use **Pages → Build and deployment → Source → GitHub Actions**. The Pages job receives only read access to repository contents plus the Pages and OIDC permissions required for deployment. The existing Python test workflow remains independent.
