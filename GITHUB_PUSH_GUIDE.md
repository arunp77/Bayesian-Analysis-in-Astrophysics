# Pushing to GitHub

## Step 1: Create the Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `bayesian-astrophysics`
3. Description: `A complete curriculum for Bayesian analysis in astrophysics — from probability foundations to exoplanet transit fitting`
4. Set to **Public**
5. Do NOT initialise with README, .gitignore, or license (we already have these)
6. Click **Create repository**

## Step 2: Connect Local Repo to GitHub

GitHub will show you a URL like `https://github.com/YOUR_USERNAME/bayesian-astrophysics.git`.

From inside the `bayesian-astrophysics/` directory:

```bash
# Add your GitHub repo as the remote
git remote add origin https://github.com/YOUR_USERNAME/bayesian-astrophysics.git

# Rename branch to 'main' (modern convention)
git branch -M main

# Push everything
git push -u origin main
```

## Step 3: Verify

Visit `https://github.com/YOUR_USERNAME/bayesian-astrophysics` — you should see all 8 chapter folders, the project directory, docs, and the README rendered at the top.

## Step 4: Enable GitHub Pages (Optional)

To host the docs as a website:
1. Go to Settings → Pages
2. Source: Deploy from a branch → `main` → `/docs`
3. Your curriculum will be live at `https://YOUR_USERNAME.github.io/bayesian-astrophysics/`

## Subsequent Updates

As we build chapter notebooks and project code:

```bash
git add chapters/ch01_probability_foundations/notebook.ipynb
git commit -m "Add Ch01 interactive notebook"
git push
```

## Recommended Branch Strategy

- `main` — stable, reviewed content
- `dev` — work in progress
- `project/transit` — exoplanet project code

```bash
git checkout -b dev
# make changes
git push -u origin dev
# when ready, merge to main via Pull Request on GitHub
```
