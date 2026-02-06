# GitHub Actions Workflow Templates

## Quick Reference für spotforecast2_safe Team

### Erstelle einen Feature-Branch und Pull Request

```bash
# 1. Neuesten Stand holen
git checkout main
git pull origin main

# 2. Feature-Branch erstellen
git checkout -b feature/beschreibung

# 3. Änderungen machen und committen
git add .
git commit -m "feat: kurze Beschreibung der Änderung"

# 4. Pushen
git push origin feature/beschreibung

# 5. Auf GitHub Pull Request erstellen
# → Tests laufen automatisch
# → Nach Merge: Automatisches Release!
```

### Commit-Message-Templates

Kopiere diese in deine Commits:

```bash
# Neue Funktion (Minor-Version erhöhen)
git commit -m "feat: [Beschreibung]"
git commit -m "feat(modul): [Beschreibung]"

# Bug Fix (Patch-Version erhöhen)
git commit -m "fix: [Beschreibung]"
git commit -m "fix(modul): [Beschreibung]"

# Breaking Change (Major-Version erhöhen)
git commit -m "feat!: [Beschreibung]"
git commit -m "feat(modul)!: [Beschreibung]"

# Dokumentation (kein Release)
git commit -m "docs: [Beschreibung]"

# Tests (kein Release)
git commit -m "test: [Beschreibung]"

# Refactoring (Patch-Version)
git commit -m "refactor: [Beschreibung]"

# Performance (Patch-Version)
git commit -m "perf: [Beschreibung]"
```

### Module/Bereiche für Scope (optional)

- `feat(forecaster): ...`
- `feat(data): ...`
- `feat(preprocessing): ...`
- `feat(model_selection): ...`
- `feat(utils): ...`
- `feat(weather): ...`

### Was passiert nach einem Merge zu main?

1. ✅ CI Tests laufen
2. ✅ Semantic-Release analysiert Commits
3. ✅ Version wird automatisch bestimmt
4. ✅ CHANGELOG.md wird aktualisiert
5. ✅ pyproject.toml Version wird aktualisiert
6. ✅ Python Package wird gebaut
7. ✅ Upload zu PyPI
8. ✅ GitHub Release wird erstellt
9. ✅ Dokumentation wird deployed
10. ✅ Git Tag wird erstellt

### Typische Workflows

#### Feature hinzufügen

```bash
git checkout -b feature/neue-prognose-methode
# ... Code schreiben ...
git add src/spotforecast2_safe/forecaster/new_method.py
git commit -m "feat(forecaster): neue Prognose-Methode für XGBoost"
git push origin feature/neue-prognose-methode
# → Pull Request erstellen
# → Nach Merge: Version 1.2.0 → 1.3.0
```

#### Bug fixen

```bash
git checkout -b fix/nan-handling
# ... Bug fixen ...
git add src/spotforecast2_safe/preprocessing/imputation.py
git commit -m "fix(preprocessing): korrekte Behandlung von NaN-Werten in imputation"
git push origin fix/nan-handling
# → Pull Request erstellen
# → Nach Merge: Version 1.2.0 → 1.2.1
```

#### Dokumentation aktualisieren

```bash
git checkout -b docs/update-readme
# ... Doku schreiben ...
git add README.md docs/
git commit -m "docs: API-Beispiele und Tutorials hinzugefügt"
git push origin docs/update-readme
# → Pull Request erstellen
# → Nach Merge: KEIN neues Release, nur Doku-Update
```

#### Breaking Change

```bash
git checkout -b refactor/api-redesign
# ... API ändern ...
git add src/spotforecast2_safe/
git commit -m "feat!: API komplett überarbeitet für bessere Usability

BREAKING CHANGE: Die alte predict() Methode wurde durch forecast() ersetzt.
Siehe Migration Guide in der Dokumentation."
git push origin refactor/api-redesign
# → Pull Request erstellen
# → Nach Merge: Version 1.9.5 → 2.0.0
```

### Troubleshooting

#### Tests schlagen fehl

```bash
# Lokal testen vor dem Push
pytest tests/ -v

# Bestimmte Tests
pytest tests/test_forecaster.py -v

# Mit Coverage
pytest tests/ --cov=src/spotforecast2_safe
```

#### Commit-Message vergessen oder falsch

```bash
# Letzten Commit ändern (vor dem Push!)
git commit --amend -m "feat: korrekte Message"

# Mehrere Commits zusammenfassen
git rebase -i HEAD~3
```

#### Release überspringen

```bash
# Verwende Typen ohne Release:
git commit -m "chore: Dependencies aktualisiert"
git commit -m "docs: Typos korrigiert"
git commit -m "ci: Workflow optimiert"
```

### GitHub Actions Status

Alle Workflows: https://github.com/sequential-parameter-optimization/spotforecast2-safe/actions

- 🟢 Grün = Alles OK
- 🔴 Rot = Fehler (klicken für Details)
- 🟡 Gelb = Läuft gerade

### Wichtige Links

- **Repository:** https://github.com/sequential-parameter-optimization/spotforecast2-safe
- **PyPI:** https://pypi.org/project/spotforecast2-safe/
- **Dokumentation:** https://sequential-parameter-optimization.github.io/spotforecast2-safe/
- **Releases:** https://github.com/sequential-parameter-optimization/spotforecast2-safe/releases
- **Actions:** https://github.com/sequential-parameter-optimization/spotforecast2-safe/actions

### Git Config Empfehlung

```bash
# Für bessere Commit-Messages
git config commit.template .gitmessage

# Automatisches Signieren (optional)
git config commit.gpgsign true
```

### Erstelle .gitmessage Template (optional)

```bash
cat > .gitmessage << 'EOF'
# <type>(<scope>): <subject>
#
# <body>
#
# <footer>
#
# Type: feat, fix, docs, style, refactor, perf, test, chore, ci
# Scope: forecaster, data, preprocessing, model_selection, utils, weather
# Subject: Kurze Beschreibung (max 50 Zeichen)
# Body: Detaillierte Erklärung (optional)
# Footer: BREAKING CHANGE, Closes #123 (optional)
EOF

git config commit.template .gitmessage
```

### Nützliche Git Aliases

```bash
# Shortcuts für häufige Befehle
git config --global alias.feat '!f() { git commit -m "feat: $1"; }; f'
git config --global alias.fix '!f() { git commit -m "fix: $1"; }; f'
git config --global alias.docs '!f() { git commit -m "docs: $1"; }; f'

# Verwendung:
# git feat "neue Funktion"
# git fix "Bug behoben"
# git docs "README aktualisiert"
```
