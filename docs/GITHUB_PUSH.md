# Publish to GitHub

From the directory that contains this repository:

```bash
cd quantum-ml-drug-discovery

git init
git add .
git commit -m "Initial quantum-ML drug-discovery pipeline"
git branch -M main

gh repo create quantum-ml-drug-discovery \
  --public \
  --source=. \
  --remote=origin \
  --push
```

If the GitHub repository already exists:

```bash
cd quantum-ml-drug-discovery
git init
git add .
git commit -m "Initial quantum-ML drug-discovery pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/quantum-ml-drug-discovery.git
git push -u origin main
```

Before publishing benchmark results, run the full QM/ML pipeline and commit the generated summary metrics and selected publication-quality figures, but avoid committing very large raw QM scratch directories.
