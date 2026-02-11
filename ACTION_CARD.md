<!--
SPDX-FileCopyrightText: 2026 bartzbeielstein
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# 🎬 ACTION CARD - WHAT TO DO RIGHT NOW

## You're Almost Done! ✅

Your code changes are already pushed to GitHub. Just a few final steps before your scorecard improves.

---

## 📌 IMMEDIATE ACTION (Next 5-10 minutes)

### 1. View the GitHub PR Creation Page
You should see it in the **Simple Browser window** (right side panel)
- URL: `https://github.com/.../pull/new/feat/openssf-scorecard-improvements`

### 2. Fill in the PR Form

**Copy this PR Title:**
```
feat(security): Improve OpenSSF Scorecard compliance to 8-9/10
```

**Copy the PR Description from:**
- Open: `PR_CREATION_GUIDE.md` (in your editor)
- Go to section: "2. Fill in PR Title and Description"
- Copy the entire Description section
- Paste into GitHub PR form

**Why?** The description explains all your changes to reviewers and tracking systems.

### 3. Click "Create pull request" on GitHub

GitHub will automatically:
- Run all CI checks ✓
- Report if anything fails (unlikely)
- Show you the PR page

---

## ⏱️ WHILE GITHUB RUNS CI (5-10 minutes)

**What to expect:**
```
✓ REUSE Compliance      (should pass)
✓ Code Quality          (should pass)  
✓ Security Scan         (should pass)
✓ Test - Python 3.13    (should pass)
✓ Test - Python 3.14    (should pass)
✓ CodeQL Analysis       (should pass - NEW!)
```

**If all pass:** Green checkmark ✅

**If any fail:** 
- Click the check
- Read the error
- Let me know (unlikely to happen)

---

## ✅ AFTER CI PASSES (When green ✅)

### A. Merge the PR
Find the green "Merge pull request" button:

1. Click dropdown → "Squash and merge" ⭐ (recommended)
2. Click "Confirm squash and merge"
3. Delete the feature branch (GitHub offers this)

### B. Verify on Your Local Machine
```bash
git checkout main
git pull
ls -la .github/SECURITY.md
```

Should output:
```
-rw-r--r--  .github/SECURITY.md
```

---

## 📚 NEXT SESSION (This week)

After the PR merges, follow steps 2-3 using this guide:
- **File:** `SETUP_OPENSSF_STEPS_2_3.md`
- **Step 2:** Branch Protection (10 min, GitHub UI)
- **Step 3:** GPG Signing (30 min, local + GitHub)

---

## 📊 EXPECTED OUTCOME

| When | Scorecard |
|------|-----------|
| Now | ~5/10 ⭐⭐⭐⭐⭐ |
| After PR merges | ~6/10 ⭐⭐⭐⭐⭐⭐ |
| After Step 2 | ~7/10 ⭐⭐⭐⭐⭐⭐⭐ |
| After Step 3 | ~8/10 ⭐⭐⭐⭐⭐⭐⭐⭐ |
| Updates in 24h | Final score visible |

---

## ✨ YOU'VE DONE THE HARD PART!

All the security code is done. Now just:
1. Create PR ← **You are here**
2. Merge PR
3. Follow guides for steps 2-3

Everything else is automated. 🚀

---

## 🆘 IF YOU GET STUCK

- **CI check failed:** Check `PR_CREATION_GUIDE.md` → Troubleshooting
- **Can't find PR form:** Simple Browser might have closed, open:
  ```
  https://github.com/sequential-parameter-optimization/spotforecast2-safe/pull/new/feat/openssf-scorecard-improvements
  ```
- **Git conflict:** Very unlikely, but let me know
- **Something else:** Check the detailed guide files

---

## 📋 CHECKLIST FOR THIS SESSION

- [ ] Simple Browser shows GitHub PR creation page
- [ ] Copied PR title: "feat(security): Improve OpenSSF Scorecard compliance to 8-9/10"
- [ ] Copied PR description from PR_CREATION_GUIDE.md
- [ ] Pasted both into GitHub form
- [ ] Clicked "Create pull request"
- [ ] Waited for CI checks (5-10 min)
- [ ] All checks turned GREEN ✅
- [ ] Clicked "Squash and merge"
- [ ] Confirmed merge
- [ ] Deleted feature branch

**After all checked:** Go to `SETUP_OPENSSF_STEPS_2_3.md` for next steps!

---

**That's it! Your scorecard improvements are rolling.** 🎉
