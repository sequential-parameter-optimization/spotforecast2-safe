<!--
SPDX-FileCopyrightText: 2026 bartzbeielstein
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# OpenSSF Scorecard Improvement Guide
## Steps 2-3: Branch Protection & GPG Signing

**Date:** February 2026  
**Project:** spotforecast2-safe

---

## TABLE OF CONTENTS
1. [Step 2: Branch Protection Rules](#step-2-branch-protection-rules)
2. [Step 3: GPG Signing Setup](#step-3-gpg-signing-setup)
3. [Verification Checklist](#verification-checklist)
4. [Troubleshooting](#troubleshooting)

---

## STEP 2: Branch Protection Rules

### Why This Matters
- **Scorecard Impact:** HIGH (Branch-Protection & Code-Review checks)
- **Security:** Prevents direct pushes to `main`, requires code review
- **Quality:** Ensures all status checks pass before merging
- **Visibility:** Tracks who reviewed what code

### Section 2.1: Navigate to Repository Settings

**Steps:**
1. Go to: https://github.com/sequential-parameter-optimization/spotforecast2-safe
2. Click **Settings** tab (top menu bar)
3. Left sidebar → **Branches**
4. You'll see "Branch protection rules" section

### Section 2.2: Create Rule for `main` Branch

**Click:** "Add rule" button

**Fill in the dialog:**

#### Part A: Basic Information
```
Pattern: main
```
This applies rules only to the main branch

#### Part B: Require Pull Request Review

**Checkbox:** ✅ "Require a pull request before merging"

**Sub-options to enable:**
- ✅ "Require approvals" → Set to: **1**
  - Means at least 1 person must approve
  
- ✅ "Dismiss stale pull request approvals when new commits are pushed"
  - Auto-dismisses approvals when code changes
  
- ✅ "Require review from code owners"
  - Will use the `.github/CODEOWNERS` file we created
  - Routes reviews to you (@bartzbeielstein)

#### Part C: Require Status Checks to Pass

**Checkbox:** ✅ "Require status checks to pass before merging"

**Sub-options:**
- ✅ "Require branches to be up to date before merging"
  - Prevents merging stale branches
  
**Search for and select these status checks:**
```
Checks that must pass:
  ✅ CodeQL Analysis
  ✅ REUSE Compliance
  ✅ Code Quality
  ✅ Security Scan
  ✅ Test on Python 3.13
  ✅ Test on Python 3.14
  ✅ codeql (if listed separately)
```

**To find status checks:**
1. Start typing "codeql" → GitHub autocompletes
2. Click each one as it appears
3. Repeat for all others listed above

#### Part D: Security Settings

**Checkbox:** ✅ "Dismiss actors"
- Allows enforcement even for admins
- Optional but recommended for maximum security

**Checkbox:** ✅ "Require code reviews"
- Part of "Require pull request" (should be auto-checked)

**Checkbox:** ✅ "Include administrators"
- Branch protection applies to everyone including you
- HIGH SECURITY: Prevents accidental direct pushes

#### Part E: Restrict Direct Pushes (Optional but Recommended)

**Checkbox:** ✅ "Restrict who can push to matching branches"
- **Allow pushes from:** Leave empty OR specify GitHub Apps
- This makes it truly impossible to push directly
- Recommended for `main` branch

#### Part F: Save the Rule

**Button:** "Create" (at bottom of dialog)

### Section 2.3: Verify Branch Protection Applied

After clicking Create, you should see:
```
Branch protection rule
  Pattern: main
  ✅ Pull request review required (1)
  ✅ Code owner review required
  ✅ Status checks required (6 items)
  ✅ Stale approvals dismissed
  ✅ Include administrators
```

### Common Issues & Solutions

**Issue 1: "Status checks not found"**
- The checks don't exist until first CI run
- Push changes first, wait for CI to pass
- Then return and add status checks

**Issue 2: "Can't select status checks"**
- Run tests on `develop` branch first
- Status checks are created by successful CI runs

**Issue 3: "Apply to administrator too" checkbox missing**
- This is under the "Include administrators" section
- Scroll down in the dialog

---

## STEP 3: GPG Signing Setup

### Why This Matters
- **Scorecard Impact:** MEDIUM (Signed-Releases check)
- **Security:** Cryptographically proves commits are from you
- **Trust:** GitHub shows "Verified" badge on signed commits
- **Compliance:** Required for security-critical projects

### Section 3.1: Generate GPG Key (If You Don't Have One)

**On your local machine, open terminal:**

```bash
gpg --full-generate-key
```

**Answer the prompts:**

```
Please select what kind of key you want:
  (1) RSA and RSA (default)     ← Press 1, then Enter

What keysize do you want? (3072)
  4096                           ← Type 4096, then Enter

Please specify how long the key should be valid.
  0 = key does not expire
  <n>  = key expires in n days
  <n>w = key expires in n weeks
  <n>m = key expires in n months
  <n>y = key expires in n years
  Key is valid for? (0)
  0                             ← Press Enter (no expiry)
  
Is this correct? (y/N)
  y                             ← Press y, then Enter

GnuPG needs to construct a user ID to identify your key.

Real name:
  [Your Name]                   ← Use: bartzbeielstein

Email address:
  [Your Email]                  ← Use your GitHub email
  32470350+bartzbeielstein@users.noreply.github.com

Comment:
  [Optional comment]            ← Leave empty or add context

You selected this user ID:
    "bartzbeielstein <32470350+bartzbeielstein@users.noreply.github.com>"

Change (N)ame, (C)omment, (E)mail or (O)kay/(Q)uit?
  O                             ← Press o, then Enter

Enter passphrase:
  [SECURE PASSPHRASE]           ← Create strong passphrase (save it!)
  
Repeat passphrase:
  [SECURE PASSPHRASE]           ← Type again
```

**After generation:**
```
gpg: key XXXXXXXXXXXXXXXX marked as ultimately trusted
public key created and signed.
pub   rsa4096/XXXXXXXXXXXXXXXX 2026-02-11 [SC] [expires: ...]
      XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
uid                     bartzbeielstein <...@users.noreply.github.com>
sub   rsa4096/XXXXXXXXXXXXXXXX 2026-02-11 [E] [expires: ...]
```

Save the key ID: `XXXXXXXXXXXXXXXX`

### Section 3.2: Get Your GPG Key ID

**In terminal:**

```bash
gpg --list-secret-keys --keyid-format=long
```

**Output will look like:**
```
/Users/bartz/.gnupg/privkey.gpg
-------------------------------
sec   rsa4096/0D1234567890ABCD 2026-02-11 [SC]
      XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
uid                             bartzbeielstein <32470350+bartzbeielstein@users.noreply.github.com>
ssb   rsa4096/YYYYYYYYYYYYYYYY 2026-02-11 [E]
```

**Copy the key ID:** `0D1234567890ABCD` (after the `/`)

### Section 3.3: Export Your Public Key

**In terminal:**

```bash
gpg --armor --export 0D1234567890ABCD > gpg-public-key.txt
```

Replace `0D1234567890ABCD` with your actual key ID.

**View the exported key:**
```bash
cat gpg-public-key.txt
```

**Output looks like:**
```
-----BEGIN PGP PUBLIC KEY BLOCK-----

mQINBGXXXXXXBEAA...
[Long encoded string]
...XXXXXXX=
-----END PGP PUBLIC KEY BLOCK-----
```

### Section 3.4: Add Public Key to GitHub

**On GitHub.com:**

1. Click your profile photo (top right)
2. Click **Settings**
3. Left sidebar → **SSH and GPG keys**
4. Click **New GPG key**
5. Title: `GPG Key - spotforecast2-safe`
6. Paste the entire content from `gpg-public-key.txt`
7. Click **Add GPG key**

**Verify:** You should see your key listed with fingerprint

### Section 3.5: Export Your Private Key (For GitHub Actions)

**In terminal:**

```bash
gpg --armor --export-secret-keys 0D1234567890ABCD > gpg-private-key.txt
```

**⚠️ IMPORTANT: Keep this private!**

**View the key:**
```bash
cat gpg-private-key.txt
```

Output looks like:
```
-----BEGIN PGP PRIVATE KEY BLOCK-----

lQWGBGXXXXXBEAA...
[Very long encoded string - contains your PRIVATE key]
...XXXXXXX=
-----END PGP PRIVATE KEY BLOCK-----
```

### Section 3.6: Add Private Key to GitHub Secrets

**On GitHub.com:**

1. Go to: https://github.com/sequential-parameter-optimization/spotforecast2-safe
2. Click **Settings** tab
3. Left sidebar → **Secrets and variables** → **Actions**
4. Click **New repository secret**

**Fill in:**
- **Name:** `GPG_PRIVATE_KEY`
- **Secret:** Paste entire content from `gpg-private-key.txt`

**Click:** Add secret

### Section 3.7: Configure Git Locally

**Set your signing key globally:**

```bash
git config --global user.signingkey 0D1234567890ABCD
git config --global commit.gpgsign true
git config --global tag.gpgsign true
```

Replace `0D1234567890ABCD` with your key ID.

**For spotforecast2-safe only (if you prefer):**

```bash
cd /Users/bartz/workspace/spotforecast2-safe

git config user.signingkey 0D1234567890ABCD
git config commit.gpgsign true
git config tag.gpgsign true
```

### Section 3.8: Configure Release Workflow

The `release.yml` file already has GPG setup instructions. To enable it:

1. File: `.github/workflows/release.yml`
2. Add this step after "Set up Python":

```yaml
      - name: Import GPG key
        uses: crazy-max/ghaction-import-gpg@v6
        with:
          gpg_private_key: ${{ secrets.GPG_PRIVATE_KEY }}
          git_config_global: true
          git_user_signingkey: true
          git_commit_gpgsign: true
          git_tag_gpgsign: true
```

This is already added in our updated workflows!

### Section 3.9: Test Your Setup Locally

**Make a test commit:**

```bash
cd /Users/bartz/workspace/spotforecast2-safe

# Create a test file
echo "# Test GPG signing" > test_gpg.md

# Stage it
git add test_gpg.md

# Commit with signing (will prompt for passphrase)
git commit -m "test: verify GPG signing setup"
```

**You'll see:**
```
gpg: using RSA key 0D1234567890ABCD
gpg: signing with key 0D1234567890ABCD
```

**Verify the signature:**

```bash
git log --show-signature -1
```

**Output should show:**
```
commit [hash]
gpg: Signature made ... using RSA key ...
gpg: Good signature from "bartzbeielstein <...@users.noreply.github.com>"
Author: bartzbeielstein <...>
Date:   [date]

    test: verify GPG signing setup
```

### Section 3.10: Clean Up Test Commit

**If everything works, remove the test commit:**

```bash
git reset HEAD~1
rm test_gpg.md
```

**Clean up exported key files:**

```bash
rm ~/gpg-private-key.txt ~/gpg-public-key.txt
```

⚠️ **NEVER commit the private key file!**

---

## VERIFICATION CHECKLIST

### After Step 2 (Branch Protection)
- [ ] Visit: Settings → Branches
- [ ] See rule for `main` branch
- [ ] Status shows all 6 checks required
- [ ] Code owner review enabled
- [ ] Administrator included

### After Step 3 (GPG Signing)
- [ ] Public key visible in Settings → SSH and GPG keys
- [ ] `GPG_PRIVATE_KEY` secret added to Actions secrets
- [ ] Local git configured with signing key
- [ ] Test commit shows "Good signature" when verified
- [ ] GitHub shows "Verified" badge on new commits

### After Pushing Changes
- [ ] All CI checks pass (CodeQL, tests, REUSE, etc.)
- [ ] New commits show "Verified" badge on GitHub
- [ ] Branch protection prevents direct `main` pushes
- [ ] Scorecard updated within 24 hours

---

## TROUBLESHOOTING

### GPG Issues

**Problem:** "error: gpg failed to sign the data"
```bash
# Solution: Export GPG_TTY
export GPG_TTY=$(tty)

# Add to ~/.zshrc to persist:
echo 'export GPG_TTY=$(tty)' >> ~/.zshrc
```

**Problem:** "key [ID] is not usable - skipped"
```bash
# Solution: Trust the key
gpg --edit-key 0D1234567890ABCD
# Type: trust
# Select: 5 (I trust ultimately)
# Type: quit
```

**Problem:** Passphrase prompt doesn't appear
```bash
# Solution: Add pinentry
brew install pinentry-mac

# Add to ~/.gnupg/gpg-agent.conf:
pinentry-program /usr/local/bin/pinentry-mac
```

### GitHub Branch Protection Issues

**Problem:** "Unable to add status checks"
- Reason: Status checks don't exist until first successful run
- Solution: Create a PR, let CI run, then edit the rule

**Problem:** "Administrator included" checkbox missing
- Reason: Check "Dismiss actors" option first
- Solution: Scroll down in the branch protection dialog

**Problem:** Can't merge because status check failed
- Check the specific failed job
- Fix the code or workflow
- Push to PR, CI re-runs automatically

### Git/Commit Issues

**Problem:** Commits not showing as "Verified"
- Verify email in GitHub matches git config: `git config user.email`
- Check public key shows correct email
- Wait a few minutes for GitHub to refresh

**Problem:** "fatal: no changes added to commit"
- Make sure you actually changed files: `git status`
- Or: `git commit --allow-empty -m "message"`

---

## NEXT STEPS

### After You Complete Steps 2-3:

1. **Verify Everything Works**
   - Make a small change to `SECURITY.md`
   - Commit and push to new branch
   - Create PR to `main`
   - Verify all checks pass
   - Verify commit is "Verified"

2. **Monitor OpenSSF Scorecard**
   - Visit: https://scorecard.dev/viewer/?uri=github.com/sequential-parameter-optimization/spotforecast2-safe
   - Should update within 24 hours
   - Expected score: 8-9/10 (up from ~5/10)

3. **Document Your Process** (Optional)
   - Add notes to `MODEL_CARD.md` about security setup
   - Update `CHANGELOG.md` with security improvements

---

## REFERENCE LINKS

- **OpenSSF Scorecard:** https://scorecard.dev/
- **GitHub Branch Protection:** https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches
- **GPG in Git:** https://docs.github.com/en/authentication/managing-commit-signature-verification
- **CodeQL:** https://codeql.github.com/

---

**Questions?** Check the [Troubleshooting](#troubleshooting) section above.
