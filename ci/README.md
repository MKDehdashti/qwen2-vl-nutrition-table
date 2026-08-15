# CI workflow (staged here, not yet active)

`ci.yml` is the GitHub Actions workflow for this repo. It is parked here instead of
`.github/workflows/` because the token used to push could not write workflow files
(GitHub rejects pushes that create or modify `.github/workflows/*` unless the token
carries the `workflow` scope).

To activate it:

```bash
mkdir -p .github/workflows
git mv ci/ci.yml .github/workflows/ci.yml
git rm ci/README.md
git commit -m "Enable CI" && git push
```

That push needs a token with `workflow` scope, or you can paste the file into the
GitHub web UI via **Actions → New workflow → set up a workflow yourself**.

It runs pytest on Python 3.10 and 3.12 plus a ruff smoke lint. Both pass locally.
