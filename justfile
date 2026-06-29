# ibrief — Task-Runner (https://github.com/casey/just)
# `just` ohne Argument listet alle Rezepte.

# Schneller Alias für den App-Binary (vermeidet Tippen von `cargo run -p ibrief-app --`).
cargo := "cargo"
run := cargo + " run -q -p ibrief-app --"

# Standard: Übersicht aller Rezepte.
default:
    @just --list

# --- Qualität (spiegelt die CI: fmt · clippy · test) -----------------------

# Alle CI-Checks lokal, in CI-Reihenfolge. Vor jedem Commit ausführen.
ci: fmt-check clippy test

# Code formatieren.
fmt:
    {{cargo}} fmt --all

# Formatierung prüfen (wie CI, ohne zu ändern).
fmt-check:
    {{cargo}} fmt --all --check

# Lints als Fehler (wie CI).
clippy:
    {{cargo}} clippy --all-targets --all-features -- -D warnings

# Tests über den gesamten Workspace.
test:
    {{cargo}} test --workspace --all-features

# Einzelnes Crate testen, z.B. `just test-one ibrief-eval`.
test-one crate:
    {{cargo}} test -p {{crate}}

# Schneller Kompilier-Check ohne Artefakte.
check:
    {{cargo}} check --workspace --all-targets

# Release-Build des Binaries.
build:
    {{cargo}} build --release -p ibrief-app

# Generierte Artefakte entfernen.
clean:
    {{cargo}} clean

# --- ibrief-Pipeline -------------------------------------------------------

# Tagesbriefing erzeugen (Ingest → … → Render → Persist → Push).
brief:
    {{run}} brief

# Telegram-Feedback-Loop starten (braucht IBRIEF_TELEGRAM_TOKEN).
feedback:
    {{run}} feedback

# Briefing bewerten. Optional Datum; `just eval "" calibrate` für Abo-Judge.
eval date="" mode="":
    {{run}} eval {{date}} {{mode}}

# Gewichte lernen (Thompson + Safety Gate) → ggf. neue Config-Version.
learn:
    {{run}} learn

# TL;DR-Prompt optimieren (Schatten-Test). `just optimize "" calibrate` für Abo.
optimize date="" mode="":
    {{run}} optimize {{date}} {{mode}}

# --- Modell-Bakeoff (A/B) --------------------------------------------------

# Synth-Tier vergleichen (TL;DR/Gegenperspektive). `just bench "" calibrate` für Abo-Judge.
bench date="" mode="":
    {{run}} bench {{date}} {{mode}}

# Enrich-Tier vergleichen (Ein-Satz-Zusammenfassungen je Item).
bench-enrich date="" mode="":
    {{run}} bench enrich {{date}} {{mode}}

# Bakeoff-Historie anzeigen (Modell-Vergleiche über Tage).
bench-list:
    {{run}} bench list

# --- Registry & Recherche --------------------------------------------------

# Config-Historie (aktive markiert).
config-list:
    {{run}} config list

# Auf eine frühere Config-Version zurücksetzen.
config-rollback version:
    {{run}} config rollback {{version}}

# A/B-Experiment-Historie.
experiment-list:
    {{run}} experiment list

# Quellen-Registry anzeigen.
sources-list:
    {{run}} sources list

# Quellen bewerten/aussortieren + Drift-Wächter.
sources-evolve:
    {{run}} sources evolve

# AutoResearch über den lokalen Korpus, z.B. `just research "Stand lokaler LLMs"`.
research +question:
    {{run}} research {{question}}

# --- Ollama-Helfer ---------------------------------------------------------

# Alle in profile.toml deklarierten Modelle (aktiv + Kandidaten) herunterladen.
# Greift Tags der Form name:<größe>b (z.B. gemma4:26b, qwen3:30b-a3b) — ignoriert Ports/Pfade.
pull-models:
    #!/usr/bin/env bash
    set -euo pipefail
    grep -hoE '[a-zA-Z0-9._-]+:[0-9]+b[a-zA-Z0-9._-]*' config/profile.toml \
        | sort -u \
        | while read -r m; do echo "→ ollama pull $m"; ollama pull "$m"; done

# Lokal installierte Ollama-Modelle auflisten.
models:
    ollama list
