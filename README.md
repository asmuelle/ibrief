# ibrief

[![CI](https://github.com/asmuelle/ibrief/actions/workflows/ci.yml/badge.svg)](https://github.com/asmuelle/ibrief/actions/workflows/ci.yml)
[![Rust 1.95](https://img.shields.io/badge/rust-1.95%20edition%202024-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Selbstverbesserndes, personalisiertes Morning Briefing — lokal-first in Rust, mit lokalen LLMs.
Siehe [SPEC.md](SPEC.md) · **[Projektseite ↗](https://asmuelle.github.io/ibrief/)**

**Stand: M1–M6 ✓** — vollständige Pipeline plus Selbstverbesserung auf drei Achsen (Gewichte,
Prompts, Quellen), jede abgesichert. Anti-Blase-Sektionen (Gegenperspektive, Wildcard) sind
nicht abschaltbar. Build + Tests grün unter Rust 1.95 / Edition 2024.

## Voraussetzungen

- Rust (1.80+)
- [Ollama](https://ollama.com) lokal laufend, mit den Modellen aus `config/profile.toml`:

```bash
ollama pull qwen2.5:14b      # Massen-Tier (Enrich)
ollama pull llama3.3:70b     # Synthese-Tier (TL;DR)
```

> Auf kleinerer Hardware kleinere Modelle eintragen (z.B. `llama3.1:8b` für beide).

## Lauf

```bash
cargo run -p ibrief-app                 # = `brief`: Briefing erzeugen
cargo run -p ibrief-app -- brief ./config
```

Ergebnis: `out/briefing-YYYY-MM-DD.md` (+ stdout) und Persistenz in `ibrief.db`
(Items werden dedupliziert — bereits gesehene erscheinen nicht erneut).

### Feedback via Telegram (optional)

1. Bot bei [@BotFather](https://t.me/BotFather) anlegen, Token holen.
2. `chat_id` in `config/profile.toml` setzen, `enabled = true`.
3. Token als Env-Var bereitstellen und beide Prozesse starten:

```bash
export IBRIEF_TELEGRAM_TOKEN=123456:ABC...
cargo run -p ibrief-app -- brief        # pusht das Briefing mit 👍/👎-Buttons
cargo run -p ibrief-app -- feedback     # Long-Polling: Klicks → ibrief.db (Tabelle feedback)
```

Die erfassten 👍/👎 sind die Datengrundlage für den Lern-Loop ab M3/M4.

## Workspace

| Crate | Zweck |
|-------|-------|
| `ibrief-core` | Domänentypen (`ContentItem`, `Briefing`, …) |
| `ibrief-llm` | `LanguageModel`-Trait + Backends: `OllamaClient` (lokal), `ClaudeCodeModel` (Abo via `claude -p`) |
| `ibrief-ingest` | RSS/Atom-Fetch (`feed-rs`) |
| `ibrief-pipeline` | Stages: Enrich, Curate, Render |
| `ibrief-store` | SQLite (sqlx): Content, Briefing-Records, Feedback, Evals, Dedup |
| `ibrief-telegram` | Telegram-Push + Feedback-Buttons (Bot-API via reqwest) |
| `ibrief-eval` | Eval Engine: Verhaltens-Score + LLM-Judge + Strukturchecks |
| `ibrief-learn` | Gewichts-Lernen: Thompson-Sampling + Safety Gate + A/B-Entscheidung + Config-Versionierung |
| `ibrief-optimize` | Prompt-Optimierung: Optimizer-LLM + Schatten-Test + Prompt-Versionierung |
| `ibrief-sources` | Quellen-Evolution: Qualitäts-Scoring + Pruning + Drift-Wächter |
| `ibrief-research` | AutoResearch (§14): gated Loop + Beleg-Verifikation über lokalen Korpus |
| `ibrief-app` | Binary `ibrief` (`brief`/`feedback`/`eval`/`learn`/`config`/`optimize`/`experiment`/`sources`/`research`) |

## Eval (M3)

```bash
cargo run -p ibrief-app -- eval               # heutiges Briefing bewerten (Judge lokal)
cargo run -p ibrief-app -- eval 2026-06-28    # bestimmtes Datum
cargo run -p ibrief-app -- eval calibrate     # Judge via Abo (claude -p) statt lokal
```

Ergebnis: `total`-Note (0–1) aus `behavior`/`judge`/`structure` (Gewichte 0.5/0.3/0.2),
gespeichert in der `evals`-Tabelle pro `date × config_version`. Diagnose-Notizen inklusive.

## Lernen (M4)

```bash
cargo run -p ibrief-app -- learn              # Gewichte aus Feedback lernen (Thompson + Safety Gate)
cargo run -p ibrief-app -- config list        # Config-Historie (aktive mit * markiert)
cargo run -p ibrief-app -- config rollback cfg-xxxx   # auf frühere Version zurück
```

`learn` aggregiert das Feedback je Quelle/Thema, sampelt neue Gewichte (Beta/Thompson,
geklemmt auf [0.2, 2.0] → Exploration-Floor gegen Aussterben), prüft das **Safety Gate**
(Grenzen + Quellen-Diversität) und übernimmt nur bei PASS als neue, versionierte Config.
`brief` nutzt die aktive Config beim Ranking (`recency × source_weight × topic_weight`).

## Prompt-Optimierung (M5)

```bash
cargo run -p ibrief-app -- optimize 2026-06-28          # TL;DR-Prompt im Schatten-Test verbessern
cargo run -p ibrief-app -- optimize 2026-06-28 calibrate # Optimizer+Judge via Abo (claude -p)
cargo run -p ibrief-app -- experiment list               # A/B-Experiment-Historie
```

Ein Optimizer-LLM erzeugt eine Prompt-Variante; im **Schatten-Test** rendern aktiver und
Kandidaten-Prompt dasselbe Beispiel-Briefing, der Judge bewertet beide. Nur bei klarem
Vorsprung (Marge 0.05) wird der Kandidat neue, versionierte Default-Prompt-Version.
`brief` nutzt stets den aktiven Prompt. Die temporale A/B-Entscheidung
(`ibrief-learn::ab_decision`) ist die Grundlage für mehrtägige Behavioral-Tests.

## Quellen-Evolution & AutoResearch (M6)

```bash
cargo run -p ibrief-app -- sources list      # Registry: Qualität, aktiv/inaktiv
cargo run -p ibrief-app -- sources evolve     # bewerten + aussortieren (mit Drift-Wächter)
cargo run -p ibrief-app -- research "Was ist neu bei lokalen LLMs?"
```

`sources evolve` bewertet Quellen aus Feedback + Selektionshäufigkeit, deaktiviert schwache
(nie unter 3 aktive), und der **Drift-Wächter** setzt Pruning aus, wenn die Quellen-Diversität
zu stark sinkt (HHI > 0.5) — Anti-Blase vor Engagement.
`research` ist der gated AutoResearch-Loop (§14) über den lokalen Korpus: belegpflichtig,
unbelegte Aussagen werden verworfen. Die `ResearchSource`-Trait erlaubt später ein Web-Backend.

## Roadmap

M1 statisch ✓ · M2 Persistenz + Feedback ✓ · M3 Eval ✓ · M4 Lernen (Gewichte) ✓ · M5 Prompt-Opt ✓ · M6 Quellen-Evolution + AutoResearch ✓.
Details in [SPEC.md](SPEC.md). Alle Meilensteine implementiert; Build + 19 Tests grün.
