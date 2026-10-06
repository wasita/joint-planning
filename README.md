# Learning to jointly plan

Working notes on **joint planning and multi-agent coordination**, written as
[marimo](https://marimo.io) notebooks and published as a book with
[marimo-book](https://github.com/wasita/marimo-book). Topics span reinforcement
learning, behavioral game theory, and computational cognitive modeling.

- **Notebooks book:** <https://wasita.space/joint-planning/book/>
- **Annotated reading list:** <https://wasita.space/joint-planning/>

## What's in the book

| Section | Notebook | |
|---|---|---|
| Cognitive hierarchy | `notebooks/rl_rewrites_of_ch_eqtns.py` | Camerer, Ho & Chong (2004) CH equations, term by term, with RL rewrites |
| Cognitive hierarchy | `notebooks/simulate_staghunt.py` | **[WIP]** 2-player stag hunt: simulate CH players, score data under the model, and fit it |

The table of contents lives in `book.yml`. `book/intro.md` is the landing page.
`notebooks/resources/` holds reference notebooks and tutorials that aren't in the book.

## Working on the notebooks

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                              # install deps (incl. marimo-book)
uv run marimo edit notebooks/simulate_staghunt.py    # open a notebook in the editor
```

To add a page, put the notebook in `notebooks/` and list it under `toc:` in `book.yml`.
Pages are rendered statically by default. Set `mode: wasm` on an entry when its sliders
need live Python, so the page runs in the browser via Pyodide.

## Building the book

```bash
uv run marimo-book serve          # local preview with live reload
uv run marimo-book check          # validate book.yml and linked content
uv run marimo-book build --strict # full build into _site/, as CI does
```

Pushing to `main` deploys both sites to GitHub Pages (`.github/workflows/deploy.yml`):
the reading list (`index.html`) at the site root and the book under `/book/`.

## The reading list

A curated, annotated list of 124 papers across eight threads, each with a tier
(read in full / skim / know it exists) and a note on why it earns its place.

| | |
|---|---|
| **A** | behavioral game theory & coordination — focal points, level-k, team reasoning |
| **B** | virtual bargaining & the current synthesis frontier |
| **C** | computational theory of mind & inverse planning |
| **D** | joint action, shared agency, commitment, norms |
| **E** | resource-rational & hierarchical planning |
| **F** | multi-agent AI & cooperative AI |
| **G** | continuous-time & real-time spatial coordination |
| **H** | collective intelligence & social learning |

`data/papers.json` is the source of record: title, authors, year, venue, tier, thread,
reading-plan week, annotation, links, and flags. Regenerate the page after editing it:

```bash
python3 scripts/build_site.py data/papers.json index.html
```

Annotations are editorial judgments, not consensus positions. They reflect one reading
of the literature and are meant to be argued with. Roughly half the papers are CogSci
proceedings with no DOI, which is worth knowing before building anything that keys on one.

## Layout

- `notebooks/` — marimo notebooks (book pages), plus `resources/` for reference material
  (tutorials, CH equation references, Sutton & Barto Ch. 3 notebook, summary, and PDF)
- `book.yml`, `book/` — marimo-book config and markdown pages
- `data/papers.json` — reading list source of record
- `scripts/build_site.py` — builds the reading list page (`index.html`)
