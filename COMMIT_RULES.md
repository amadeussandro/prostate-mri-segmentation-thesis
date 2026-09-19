# Commit Rules

Git and commit workflow for this repository. Read before making any commit.

## Repository

- Repository: `amadeussandro/prostate-mri-segmentation-thesis`
- Primary branch: `main`

## Commit Identity

All normal project commits MUST use the repository owner's Git identity:

- **Name:** `Benedict Amadeus Sandro`
- **GitHub username:** `amadeussandro`
- **Email:** the email already configured for the repository owner's GitHub
  account and used throughout this repository's history
  (`criwbasct77@gmail.com`). GitHub attributes commits by email, so this exact
  address is what links a commit to the owner's account. **Do NOT invent an
  email address**; if in doubt, read it from the existing history
  (`git log --format='%an <%ae>' | sort -u`).

Configure locally with:

```bash
git config user.name  "Benedict Amadeus Sandro"
git config user.email "criwbasct77@gmail.com"
```

Do NOT author commits as:

- Claude
- Anthropic
- OpenAI
- any AI-generated identity
- any generic bot identity

...unless a future task explicitly requires a bot/automation commit.

## Commit Messages

Use conventional-commit style when appropriate:

```
feat(scope): description
fix(scope): description
docs(scope): description
refactor(scope): description
test(scope): description
chore(scope): description
```

Examples:

```
feat(rq2): add Colab reconstruction notebook
fix(rq2): correct reconstruction geometry validation
docs(thesis): update RQ2 methodology
test(rq2): add reconstruction validation tests
```

## Before Commit

Always review what is about to be committed:

```bash
git status
git diff
git diff --cached
```

Never blindly run `git add .` unless the working tree has first been reviewed
and all unrelated/generated files are known to be ignored.

## Never Commit

Do NOT commit:

- dataset files
- MRI / NIfTI files (`.nii`, `.nii.gz`)
- `.npz` prediction artifacts
- model checkpoints (`.pt`, `.pth`)
- Google Drive data
- generated experiment outputs
- secrets, API keys, credentials
- `.env` files
- temporary Colab files
- large generated visualizations

These belong in Google Drive or other appropriate artifact storage, and are
guarded by `.gitignore`.

## Thesis Experiment Rule

Source code, configuration, tests, notebooks, and reproducibility documentation
MAY be committed.

Generated experiment artifacts (reconstructed volumes, metrics CSV/JSON,
figures, checkpoints) should normally remain OUTSIDE Git unless explicitly
requested. Directory skeletons are kept via `.gitkeep`; the heavy contents are
ignored.

## Branch Rule

Development may occur on a working branch. Before merging/pushing final thesis
work:

- verify the target branch
- review the diff
- confirm no unrelated changes
- confirm no sensitive/large files
- push only the intended changes

## AI Coding Rule

Claude (or any AI assistant) may modify code and prepare commits when explicitly
authorized. However:

- commits MUST use the repository owner's Git identity (above)
- the AI MUST NOT present itself as the author
- the AI MUST NOT create unrelated commits
- the AI MUST NOT push without explicit authorization
- destructive Git actions (history rewrite, force-push) require explicit
  per-action authorization from the owner

## Commit Attribution

- Do NOT add Claude/Anthropic (or any AI) as the commit author or committer.
- Do NOT add an AI co-author trailer (e.g. `Co-Authored-By: Claude ...`) unless
  explicitly requested.
