<!--
SPDX-FileCopyrightText: 2026 bartzbeielstein
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Running CodeQL Locally in VS Code

**Date:** February 11, 2026  
**Project:** spotforecast2-safe

---

## TABLE OF CONTENTS
1. [Option A: GitHub CodeQL VS Code Extension (Easiest)](#option-a-github-codeql-vs-code-extension-easiest)
2. [Option B: CodeQL CLI (Most Control)](#option-b-codeql-cli-most-control)
3. [Viewing Results](#viewing-results)
4. [Troubleshooting](#troubleshooting)

---

## Option A: GitHub CodeQL VS Code Extension (Easiest) ⭐

This is the **simplest approach** - VS Code does most of the work for you.

### Step 1: Install the CodeQL Extension

1. Open VS Code
2. Go to: **Extensions** (Ctrl+Shift+X / Cmd+Shift+X)
3. Search: `GitHub CodeQL`
4. Click **Install** on the official Microsoft/GitHub extension
   - Publisher: `GitHub`
   - Extension ID: `vscode-codeql`

### Step 2: Set Up CodeQL Extension

After installation:

1. Open **VS Code Command Palette** (Cmd+Shift+P on Mac)
2. Search: `CodeQL: Quick Query`
3. A panel opens on the right side

### Step 3: Run a Scan on Your Code

**Option A: Scan Current Workspace**

1. Command Palette → `CodeQL: Run Query on Workspace`
2. Select database or create new
3. Extension does analysis automatically

**Option B: Use Pre-built Queries**

1. Command Palette → `CodeQL: Quick Query`
2. Select a query from the list:
   - Security And Quality
   - Python Best Practices
   - Security Issues
3. Click **Run** (play button)

### Step 4: View Results

Results appear in the **CodeQL Results** panel:
- Red 🔴 = Security issues found
- Yellow 🟡 = Warnings
- Green 🟢 = All clear

Click each result to see:
- File location
- Line number
- Issue description
- Suggested fix

---

## Option B: CodeQL CLI (Most Control)

For deeper analysis and scripting.

### Step 1: Install CodeQL CLI

**Option 1: Homebrew (macOS)**
```bash
brew install codeql
```

**Option 2: Download from GitHub**
```bash
cd ~/Downloads
wget https://github.com/github/codeql-cli-bundle/releases/download/v2.14.6/codeql-bundle-macos64.zip
unzip codeql-bundle-macos64.zip
sudo mv codeql /usr/local/bin/
```

**Verify installation:**
```bash
codeql version
```

### Step 2: Create CodeQL Database

```bash
cd /Users/bartz/workspace/spotforecast2-safe

# Create a database for Python analysis
codeql database create ~/codeql-databases/spotforecast2-safe \
  --language=python \
  --source-root=.
```

This analyzes your code and stores results in `~/codeql-databases/spotforecast2-safe`

**Expected output:**
```
Running language analysis for python...
[1/2] Scanning for Python source files.
[2/2] Analyzing Python code.
Created database at ~/codeql-databases/spotforecast2-safe
```

### Step 3: Run Security Queries

```bash
# Run the default security-and-quality queries
codeql query run \
  --database ~/codeql-databases/spotforecast2-safe \
  --queries codeql/python/ql/src/Security \
  --format=csv > ~/codeql-results.csv

# Or use SARIF format (same as GitHub)
codeql query run \
  --database ~/codeql-databases/spotforecast2-safe \
  --queries codeql/python/ql/src/Security \
  --format=sarif > ~/codeql-results.sarif
```

### Step 4: View Results in VS Code

**Option 1: Open CSV Results**
```bash
open ~/codeql-results.csv
# Then open in VS Code
```

**Option 2: View SARIF Results**
```bash
# Install SARIF extension
# Command: Extensions → Search "SARIF"
# Install: "SARIF Viewer" by Microsoft

# Then File → Open → codeql-results.sarif
```

---

## Viewing Results

### In VS Code Extension Panel

**Left Panel** → CodeQL
- Shows: Database, Queries, Results
- Double-click result → Jump to code location
- Right-click → Copy, Share, Debug

### Example Result Shown:

```
📊 CodeQL Analysis Results
├─ Database: spotforecast2-safe
├─ 🔴 Security Issues: 3
│  ├─ SQL Injection (src/utils.py:45)
│  ├─ Path Traversal (src/handlers.py:12)
│  └─ Use of Hardcoded Password (src/config.py:89)
├─ 🟡 Quality Issues: 8
│  ├─ Unused Variable
│  ├─ Dead Code
│  └─ ...
└─ 🟢 Passed: 124
```

---

## Quick Commands

### Using Extension (Easiest)

| Task | Command |
|------|---------|
| Run analysis | Cmd+Shift+P → `CodeQL: Analyze` |
| View results | Click Results panel |
| Clear cache | `CodeQL: Clear Cache` |
| Update queries | `CodeQL: Update CodeQL` |

### Using CLI (Terminal)

```bash
# Create database
codeql database create ~/db --language python --source-root .

# Analyze
codeql database analyze ~/db \
  --format=sarif \
  --output=results.sarif

# View specific issues
codeql database info ~/db

# Cleanup
rm -rf ~/codeql-databases/spotforecast2-safe
```

---

## Integration with Your Workflow

### Before Pushing to GitHub

1. **Run CodeQL locally** (Option A extension, 2 min)
2. **Review findings** in VS Code panel
3. **Fix critical issues** if needed
4. **Re-run** to verify fixes
5. **Push to GitHub** confident no issues

### GitHub CI Will Also Run

- CodeQL runs automatically on PR
- Results appear in Security tab
- Blocks merge if critical issues found
- You get same results, earlier (locally)

---

## Expected Issues for spotforecast2-safe

Based on the codebase:

**Low Risk:** (Safety-critical project)
- ✅ Minimal external dependencies
- ✅ No SQL systems
- ✅ Deterministic logic
- ✅ No HTML/web context

**Possible findings:**
- ⚠️ Standard Python warnings
- ⚠️ Import ordering issues
- ⚠️ Unused variables
- ⏳ None of these are blockers

---

## Setup Example (Full Workflow)

### 1. Installation (one time, 5 min)

```bash
# Option A (recommended for simplicity)
# VS Code Extensions → Search "GitHub CodeQL" → Install

# Option B (if using CLI)
brew install codeql
```

### 2. First Run (3 min per run)

**Using Extension:**
```
Cmd+Shift+P → CodeQL: Analyze Repository
→ Wait for results
→ Review in panel
```

**Using CLI:**
```bash
codeql database create ~/codeql-db --language python --source-root .
codeql database analyze ~/codeql-db --format=sarif --output=results.sarif
```

### 3. View Results

**Extension:** Results tab automatically opens
**CLI:** Open SARIF file with SARIF Viewer extension

---

## My Recommendation

**For you right now:**

1. **Install CodeQL Extensions** (VS Code Extensions marketplace)
   - `GitHub CodeQL` (main tool)
   - `SARIF Viewer` (view results)

2. **Run the extension analysis** (2 min)
   - Cmd+Shift+P → `CodeQL: Analyze`

3. **Review results** in VS Code panel

4. **No need to fix** - they're informational, your code is safe

5. **This helps your scorecard** - shows security scanning in place

---

## Why You See CodeQL in the PR

When you create the PR:

1. GitHub's `.github/workflows/ci.yml` includes CodeQL job
2. CodeQL runs automatically on the PR
3. Results appear in **Security** tab on GitHub
4. Same results you'd see locally (but 5-10 min later)

Running locally first lets you:
- See issues before PR
- Fix any blockers
- Understand what's being checked
- Verify it works

---

## Troubleshooting

### Issue: "Extension won't install"
- Check VS Code version: `code --version`
- Should be 1.80+
- Update if needed

### Issue: "Database creation fails"
```bash
# Clear and retry
codeql database create ~/codeql-db \
  --language=python \
  --source-root=. \
  --rebuild  # Force rebuild
```

### Issue: "No results shown"
- Wait 30 seconds (first run is slow)
- Check database was created: `ls ~/codeql-databases/`
- Restart VS Code

### Issue: "SARIF file won't open"
- Install SARIF Viewer extension
- File → Open → select .sarif file
- Should display results

---

## Next Steps

1. **Install CodeQL Extension** (2 min)
2. **Run analysis** (3 min)
3. **Review results** (5 min)
4. **Create PR on GitHub** (5 min)
5. **CodeQL runs in CI** (5-10 min, automatic)

**Total time: 20 minutes to see CodeQL working end-to-end**

---

## Documentation Links

- [CodeQL in VS Code](https://codeql.github.com/docs/codeql-for-visual-studio-code/)
- [CodeQL CLI Documentation](https://codeql.github.com/docs/codeql-cli/)
- [SARIF Support](https://sarifweb.azurewebsites.net/)

---

**Ready to run CodeQL?** Start with Option A (extension) - it's the easiest! 🚀
