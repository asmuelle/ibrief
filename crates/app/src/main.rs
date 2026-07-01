//! ibrief M3 — Briefing (M1) + Persistenz/Feedback (M2) + Eval Engine (M3).
//!
//! Befehle:
//!   ibrief [brief] [--force]  Ingest → Dedup → Score → Enrich(Top-Kandidaten) → Curate → Render → Persist → (Push)
//!                             `--force`: Cross-Day-Dedup überspringen, heutiges Briefing neu aufbauen
//!   ibrief feedback           Telegram-Feedback-Loop: Button-Klicks → Store
//!   ibrief feedback list [datum]              Items des Briefings mit Position anzeigen
//!   ibrief feedback add <pos> <art> [datum]   Feedback ohne Telegram (art: up|down|more|less|open)
//!   ibrief eval [datum] [calibrate]
//!                             Briefing bewerten (Verhalten + Judge + Struktur) → evals-Tabelle
//!   ibrief bench [enrich] [datum] [calibrate]
//!                             Modelle gegeneinander testen (A/B-Bakeoff) → Ranking.
//!                             Default: Synth-Tier (TL;DR/Gegenperspektive).
//!                             `enrich`: Massen-Tier (Ein-Satz-Zusammenfassungen je Item).
//!   ibrief bench list         Bakeoff-Historie anzeigen (Modell-Vergleiche über Tage)
//!   ibrief learn              Gewichte lernen (Thompson + Safety Gate) → neue Config-Version
//!   ibrief config list        Config-Historie anzeigen (aktive markiert)
//!   ibrief config rollback <version>   Auf frühere Config-Version zurücksetzen
//!   ibrief optimize [datum] [calibrate]  TL;DR-Prompt optimieren (Schatten-Test) → ggf. neue Version
//!   ibrief experiment list    A/B-Experiment-Historie anzeigen
//!   ibrief sources list       Quellen-Registry anzeigen (Qualität, aktiv/inaktiv)
//!   ibrief sources evolve     Quellen bewerten/aussortieren + Drift-Wächter
//!   ibrief research <frage>   AutoResearch über den lokalen Korpus (§14, belegpflichtig)
//!
//! Config-Verzeichnis: Env IBRIEF_CONFIG_DIR (Default ./config).
//! Voraussetzungen: Ollama (brief/eval), IBRIEF_TELEGRAM_TOKEN (feedback/push).

use anyhow::{Context, Result};
use ibrief_core::FeedbackKind;
use ibrief_eval::{EvalWeights, RUBRIC_VERSION, bakeoff::Candidate};
use ibrief_ingest::Source;
use ibrief_llm::{ClaudeCodeModel, Embedder, LanguageModel, OllamaClient, OllamaEmbedder};
use ibrief_store::{BenchRunRow, EvalRow, Store};
use ibrief_telegram::Telegram;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const TOKEN_ENV: &str = "IBRIEF_TELEGRAM_TOKEN";
const CONFIG_DIR_ENV: &str = "IBRIEF_CONFIG_DIR";
/// Max. Anteil einer einzelnen Quelle im Briefing (muss zum Diversitäts-Check in ibrief_eval passen).
const MAX_SOURCE_SHARE: f64 = 0.6;

#[derive(Deserialize)]
struct ProfileFile {
    profile: Profile,
    llm: LlmConfig,
    store: StoreConfig,
    telegram: TelegramConfig,
}

#[derive(Deserialize)]
struct Profile {
    #[allow(dead_code)]
    language: String,
    reading_time_min: u32,
}

#[derive(Deserialize)]
struct LlmConfig {
    ollama_url: String,
    enrich_model: String,
    synth_model: String,
    /// Embedding-Modell für semantische Dedup/Diversität (§T2.2). Leer = Feature aus.
    #[serde(default)]
    embed_model: String,
    /// Lokales Judge-Modell für Schatten-Tests (§T2.4). Leer = synth_model. Idealerweise
    /// eine ANDERE Modellfamilie als das synth_model, um Selbst-Präferenz zu dämpfen.
    #[serde(default)]
    judge_model: String,
    max_items_enrich: usize,
    top_n: usize,
    /// Quellen mit `quality` unter diesem Wert werden nicht kuratiert (0.0 = Filter aus).
    /// Greift mit `sources evolve` zusammen, das die Qualität pflegt.
    #[serde(default)]
    min_source_quality: f64,
    /// A/B-Kandidaten für den Synthese-Tier (`ibrief bench`).
    #[serde(default)]
    synth_candidates: Vec<String>,
    /// A/B-Kandidaten für den Enrich-Tier (`ibrief bench enrich`).
    #[serde(default)]
    enrich_candidates: Vec<String>,
}

#[derive(Deserialize)]
struct StoreConfig {
    path: String,
}

#[derive(Deserialize)]
struct TelegramConfig {
    enabled: bool,
    chat_id: String,
}

#[derive(Deserialize)]
struct SourcesFile {
    source: Vec<Source>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Default INFO; per RUST_LOG feiner steuerbar (z.B. `RUST_LOG=ibrief_pipeline=debug` zeigt
    // die Enrich-Zeit pro Item — Grundlage der Latenz-Diagnose).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "brief".to_string());
    let rest: Vec<String> = args.collect();

    let cfg_dir = std::env::var(CONFIG_DIR_ENV).unwrap_or_else(|_| "config".to_string());
    let cfg_dir = Path::new(&cfg_dir);
    let profile: ProfileFile = load_toml(cfg_dir.join("profile.toml"))?;

    match command.as_str() {
        "brief" => run_brief(cfg_dir, &profile, &rest).await,
        "feedback" => run_feedback(&profile, &rest).await,
        "eval" => run_eval(&profile, &rest).await,
        "bench" => run_bench(&profile, &rest).await,
        "learn" => run_learn(&profile).await,
        "config" => run_config(&profile, &rest).await,
        "optimize" => run_optimize(&profile, &rest).await,
        "experiment" => run_experiment(&profile, &rest).await,
        "sources" => run_sources(cfg_dir, &profile, &rest).await,
        "research" => run_research(&profile, &rest).await,
        other => anyhow::bail!(
            "unbekannter Befehl '{other}' (erwartet: brief | feedback | eval | bench | learn | config | optimize | experiment | sources | research)"
        ),
    }
}

async fn run_brief(cfg_dir: &Path, profile: &ProfileFile, rest: &[String]) -> Result<()> {
    let force = rest.iter().any(|a| a == "--force" || a == "force");
    let store = Store::open(&profile.store.path).await?;

    // Quellen-Registry: aus sources.toml seeden (idempotent), dann aktive Quellen laden.
    seed_sources(cfg_dir, &store).await?;
    let active = store.active_sources().await?;
    // Qualität je Quelle merken (für den Kurations-Floor weiter unten).
    let quality: HashMap<String, f64> = active.iter().map(|s| (s.id.clone(), s.quality)).collect();
    let sources: Vec<Source> = active
        .into_iter()
        .map(|s| Source {
            id: s.id,
            url: s.url,
        })
        .collect();

    // INGEST + DEDUP (Intra-Batch gegen Feed-Doppel, dann Cross-Day gegen den Store)
    tracing::info!(sources = sources.len(), "INGEST");
    let t_ingest = Instant::now();
    let fetched = ibrief_ingest::fetch_all(&sources).await;
    let deduped = ibrief_pipeline::dedup_batch(fetched);
    let fresh = if force {
        tracing::warn!("--force: Cross-Day-Dedup übersprungen, Briefing wird neu aufgebaut");
        deduped
    } else {
        store.filter_unseen(deduped).await?
    };

    // Qualitäts-Floor: Items aus Quellen unter dem Schwellwert nicht kuratieren.
    // Unbekannte Quellen (sollte nicht vorkommen) passieren mit 1.0.
    let min_q = profile.llm.min_source_quality;
    let fresh: Vec<_> = if min_q > 0.0 {
        let before = fresh.len();
        let kept: Vec<_> = fresh
            .into_iter()
            .filter(|it| quality.get(&it.source_id).copied().unwrap_or(1.0) >= min_q)
            .collect();
        if kept.len() < before {
            tracing::info!(
                dropped = before - kept.len(),
                min_quality = min_q,
                "Quellen-Qualitätsfilter angewandt"
            );
        }
        kept
    } else {
        fresh
    };

    tracing::info!(
        fresh = fresh.len(),
        elapsed_ms = t_ingest.elapsed().as_millis() as u64,
        "INGEST fertig (nach Dedup/Filter)"
    );
    if fresh.is_empty() {
        // Unterscheiden: heute schon gebrieft vs. tatsächlich nichts Neues (Recovery-Hinweis).
        if store.load_briefing(&today()).await?.is_some() {
            tracing::info!("keine neuen Items — Briefing für heute existiert bereits");
        } else {
            tracing::warn!(
                "keine neuen Items — nichts zu briefen (Tipp: `brief --force` baut heute neu)"
            );
        }
        return Ok(());
    }

    let cfg = ibrief_learn::load_active(&store).await?;
    // Alle geholten Items merken, um NICHT gezeigte erst NACH dem durablen save_briefing als
    // „gesehen" zu markieren (Crash-Resumability §T1.2 — vorher wird nichts markiert).
    let all_fetched = fresh.clone();

    let enrich_model = OllamaClient::new(
        profile.llm.ollama_url.clone(),
        profile.llm.enrich_model.clone(),
    );
    let synth_model = OllamaClient::new(
        profile.llm.ollama_url.clone(),
        profile.llm.synth_model.clone(),
    );
    // Embedder ist optional (leerer Modellname = aus) und sein Ausfall nie fatal —
    // assemble_briefing degradiert dann auf URL-Dedup/Quellen-Diversität.
    let embedder = (!profile.llm.embed_model.is_empty()).then(|| {
        OllamaEmbedder::new(
            profile.llm.ollama_url.clone(),
            profile.llm.embed_model.clone(),
        )
    });
    let tldr_prompt = ibrief_optimize::active_tldr(&store).await?;
    tracing::info!(prompt = %tldr_prompt.version, "aktiver TL;DR-Prompt");

    let opts = ibrief_pipeline::AssembleOpts {
        max_items_enrich: profile.llm.max_items_enrich,
        top_n: profile.llm.top_n,
        max_per_source: ((profile.llm.top_n as f64) * MAX_SOURCE_SHARE)
            .floor()
            .max(1.0) as usize,
    };
    // Kern-Pipeline (SCORE→ENRICH→RE-RANK→CURATE→SYNTHESE→WILDCARD) — testbarer Seam,
    // ohne Persistenz/Ingest. Modelle werden injiziert.
    let briefing = ibrief_pipeline::assemble_briefing(
        fresh,
        &cfg,
        &ibrief_pipeline::Models {
            enrich: &enrich_model,
            synth: &synth_model,
            embedder: embedder.as_ref().map(|e| e as &dyn Embedder),
        },
        &tldr_prompt.template,
        today(),
        &opts,
    )
    .await;

    // PERSIST Briefing-Record
    let config_version = store
        .active_config_version()
        .await?
        .unwrap_or_else(|| "default".to_string());
    store.save_briefing(&briefing, &config_version).await?;

    // Erst JETZT (Briefing durabel gespeichert) die geholten, aber NICHT gezeigten Items als
    // „gesehen" markieren, damit sie nicht erneut auftauchen. Gezeigte Items hat save_briefing
    // bereits atomar persistiert; --force baut heute bewusst neu und markiert nicht nach.
    if !force {
        let shown: std::collections::HashSet<&str> = briefing
            .sections
            .iter()
            .flat_map(|s| s.items.iter())
            .map(|it| it.id.as_str())
            .collect();
        for it in all_fetched
            .iter()
            .filter(|it| !shown.contains(it.id.as_str()))
        {
            store.upsert_item(it).await?;
        }
    }

    // RENDER (Datei)
    let md = ibrief_pipeline::render(&briefing);
    std::fs::create_dir_all("out").ok();
    let path = format!("out/briefing-{}.md", briefing.date);
    std::fs::write(&path, &md).with_context(|| format!("{path} schreiben"))?;
    tracing::info!(path = %path, "Briefing geschrieben");

    // PUSH (optional)
    if profile.telegram.enabled {
        match telegram_from(profile) {
            Ok(tg) => {
                tg.send_briefing(&briefing).await.context("Telegram-Push")?;
                tracing::info!("Briefing an Telegram gepusht");
            }
            Err(e) => {
                tracing::warn!(error = %e, "Telegram aktiviert, aber nicht konfiguriert — Push übersprungen")
            }
        }
    }

    println!("{md}");
    Ok(())
}

async fn run_feedback(profile: &ProfileFile, rest: &[String]) -> Result<()> {
    match rest.first().map(String::as_str) {
        Some("list") => feedback_list(profile, &rest[1..]).await,
        Some("add") => feedback_add(profile, &rest[1..]).await,
        // Ohne Subbefehl: Telegram-Loop (braucht Token + chat_id).
        _ => {
            let store = Store::open(&profile.store.path).await?;
            let tg = telegram_from(profile)?;
            tg.run_feedback_loop(&store).await
        }
    }
}

/// Zeigt die Items des Briefings mit ihrer Position (Eingabehilfe für `feedback add`).
async fn feedback_list(profile: &ProfileFile, args: &[String]) -> Result<()> {
    let date = args.first().cloned().unwrap_or_else(today);
    let store = Store::open(&profile.store.path).await?;
    let items = store.briefing_item_list(&date).await?;
    if items.is_empty() {
        anyhow::bail!("kein Briefing für {date} im Store");
    }
    println!("Briefing {date} — Items (Position für `feedback add`):");
    for it in items {
        println!(
            "  [{:>2}] {:<12} {:<14} {}",
            it.position, it.section_id, it.source_id, it.title
        );
    }
    println!("\nFeedback geben: ibrief feedback add <position> <up|down|more|less|open> [datum]");
    Ok(())
}

/// Schreibt ein Feedback-Ereignis für die Item-Position eines Briefings (ohne Telegram).
async fn feedback_add(profile: &ProfileFile, args: &[String]) -> Result<()> {
    let pos: i64 = args
        .first()
        .context("Position fehlt: ibrief feedback add <position> <art> [datum]")?
        .parse()
        .context("Position muss eine Zahl sein")?;
    let kind_str = args
        .get(1)
        .context("Art fehlt (up|down|more|less|open)")?
        .as_str();
    let kind = FeedbackKind::parse(kind_str)
        .with_context(|| format!("ungültige Art '{kind_str}' (up|down|more|less|open)"))?;
    let date = args.get(2).cloned().unwrap_or_else(today);

    let store = Store::open(&profile.store.path).await?;
    let Some(item_id) = store.item_at(&date, pos).await? else {
        anyhow::bail!("keine Position {pos} im Briefing {date} (siehe `feedback list {date}`)");
    };
    store.record_feedback(&date, &item_id, kind).await?;
    println!(
        "✓ {} für Position {pos} gespeichert ({item_id})",
        kind.as_str()
    );
    Ok(())
}

async fn run_eval(profile: &ProfileFile, rest: &[String]) -> Result<()> {
    let calibrate = rest.iter().any(|a| a == "calibrate");
    let date = rest
        .iter()
        .find(|a| a.as_str() != "calibrate")
        .cloned()
        .unwrap_or_else(today);

    let store = Store::open(&profile.store.path).await?;
    let Some(briefing) = store.load_briefing(&date).await? else {
        anyhow::bail!("kein Briefing für {date} im Store");
    };
    let feedback = store.feedback_counts(&date).await?;

    // Judge-Backend: lokal (Ollama) oder Abo-Kalibrierung (claude -p).
    let judge: Box<dyn LanguageModel> = if calibrate {
        tracing::info!("Judge via Abo-Kalibrierung (claude -p)");
        Box::new(ClaudeCodeModel::new(Some("opus".to_string())))
    } else {
        Box::new(local_judge(profile))
    };

    let weights = EvalWeights::default();
    let res = ibrief_eval::evaluate(
        &briefing,
        &feedback,
        profile.profile.reading_time_min,
        &weights,
        judge.as_ref(),
    )
    .await;

    let config_version = store
        .active_config_version()
        .await?
        .unwrap_or_else(|| "default".to_string());
    store
        .save_eval(&EvalRow {
            date: date.clone(),
            config_version: config_version.clone(),
            rubric_version: RUBRIC_VERSION.to_string(),
            behavior: res.behavior,
            judge: res.judge,
            structure: res.structure,
            total: res.total,
            notes: res.notes.clone(),
        })
        .await?;

    println!("Eval {date} (config {config_version}, rubric {RUBRIC_VERSION}):");
    println!(
        "  total={:.2}  [behavior={:.2} judge={:.2} structure={:.2}]",
        res.total, res.behavior, res.judge, res.structure
    );
    for n in &res.notes {
        println!("  · {n}");
    }
    Ok(())
}

async fn run_bench(profile: &ProfileFile, rest: &[String]) -> Result<()> {
    if rest.iter().any(|a| a == "list") {
        return run_bench_list(profile).await;
    }

    let calibrate = rest.iter().any(|a| a == "calibrate");
    let enrich_mode = rest.iter().any(|a| a == "enrich");
    // Datum = erstes Argument, das kein Schlüsselwort ist.
    let keywords = ["calibrate", "enrich", "synth", "list"];
    let date = rest
        .iter()
        .find(|a| !keywords.contains(&a.as_str()))
        .cloned()
        .unwrap_or_else(today);

    let store = Store::open(&profile.store.path).await?;
    let Some(briefing) = store.load_briefing(&date).await? else {
        anyhow::bail!("kein Briefing für {date} im Store (erst `ibrief brief` für diesen Tag)");
    };

    // Judge: lokal (judge_model, Fallback synth_model) oder Abo-Kalibrierung (claude -p).
    // Für alle Kandidaten gleich.
    let judge: Box<dyn LanguageModel> = if calibrate {
        tracing::info!("Bakeoff-Judge via Abo-Kalibrierung (claude -p)");
        Box::new(ClaudeCodeModel::new(Some("opus".to_string())))
    } else {
        Box::new(local_judge(profile))
    };

    if enrich_mode {
        run_bench_enrich(profile, &store, &briefing, judge.as_ref(), calibrate, &date).await
    } else {
        run_bench_synth(profile, &store, &briefing, judge.as_ref(), calibrate, &date).await
    }
}

/// Baut Kandidaten-Clients: das aktive `baseline`-Modell ist immer dabei.
fn bench_candidates<'a>(
    ollama_url: &str,
    baseline: &str,
    configured: &[String],
    models: &'a mut Vec<OllamaClient>,
) -> Vec<Candidate<'a>> {
    let mut names: Vec<String> = configured.to_vec();
    if !names.iter().any(|n| n == baseline) {
        names.insert(0, baseline.to_string());
    }
    *models = names
        .iter()
        .map(|m| OllamaClient::new(ollama_url.to_string(), m.clone()))
        .collect();
    names
        .into_iter()
        .zip(models.iter())
        .map(|(name, model)| Candidate { name, model })
        .collect()
}

async fn run_bench_synth(
    profile: &ProfileFile,
    store: &Store,
    briefing: &ibrief_core::Briefing,
    judge: &dyn LanguageModel,
    calibrate: bool,
    date: &str,
) -> Result<()> {
    let feedback = store.feedback_counts(date).await?;
    let mut models = Vec::new();
    let candidates = bench_candidates(
        &profile.llm.ollama_url,
        &profile.llm.synth_model,
        &profile.llm.synth_candidates,
        &mut models,
    );

    // Feste TL;DR-Vorlage (aktiver, gelernter Prompt) — nur das Modell variiert.
    let tldr_prompt = ibrief_optimize::active_tldr(store).await?;
    let weights = EvalWeights::default();

    let outcome = ibrief_eval::bakeoff::run(
        briefing,
        &feedback,
        profile.profile.reading_time_min,
        &weights,
        judge,
        &tldr_prompt.template,
        &candidates,
    )
    .await;

    let judge_mode = if calibrate { "abo" } else { "lokal" };
    println!(
        "Synth-Bakeoff {date} (judge={judge_mode}, prompt={}, baseline={}):",
        tldr_prompt.version, profile.llm.synth_model
    );
    for (i, e) in outcome.entries.iter().enumerate() {
        let star = baseline_marker(&e.name, &profile.llm.synth_model);
        println!(
            "  {}. {:<18} total={:.2}  [judge={:.2} struct={:.2}]  {} ms{star}",
            i + 1,
            e.name,
            e.eval.total,
            e.eval.judge,
            e.eval.structure,
            e.elapsed_ms,
        );
        store
            .save_bench_run(&BenchRunRow {
                date: date.to_string(),
                tier: "synth".into(),
                model: e.name.clone(),
                judge_mode: judge_mode.into(),
                total: e.eval.total,
                scores: vec![
                    ("judge".into(), e.eval.judge),
                    ("struct".into(), e.eval.structure),
                    ("behavior".into(), e.eval.behavior),
                ],
                items_scored: 0,
                elapsed_ms: e.elapsed_ms as i64,
                is_winner: i == 0,
            })
            .await?;
    }
    if let Some(w) = outcome.winner() {
        println!(
            "→ Gewinner: {} (total {:.2}) · in Historie gespeichert",
            w.name, w.eval.total
        );
    }
    Ok(())
}

async fn run_bench_enrich(
    profile: &ProfileFile,
    store: &Store,
    briefing: &ibrief_core::Briefing,
    judge: &dyn LanguageModel,
    calibrate: bool,
    date: &str,
) -> Result<()> {
    let mut models = Vec::new();
    let candidates = bench_candidates(
        &profile.llm.ollama_url,
        &profile.llm.enrich_model,
        &profile.llm.enrich_candidates,
        &mut models,
    );

    let outcome = ibrief_eval::bakeoff::run_enrich(
        briefing,
        judge,
        profile.llm.max_items_enrich,
        &candidates,
    )
    .await;

    let judge_mode = if calibrate { "abo" } else { "lokal" };
    println!(
        "Enrich-Bakeoff {date} (judge={judge_mode}, baseline={}):",
        profile.llm.enrich_model
    );
    for (i, e) in outcome.entries.iter().enumerate() {
        let star = baseline_marker(&e.name, &profile.llm.enrich_model);
        println!(
            "  {}. {:<18} total={:.2}  [treue={:.2} prägnanz={:.2} tags={:.2}]  {} Items  {} ms{star}",
            i + 1,
            e.name,
            e.total,
            e.faithfulness,
            e.concision,
            e.tags,
            e.items_scored,
            e.elapsed_ms,
        );
        store
            .save_bench_run(&BenchRunRow {
                date: date.to_string(),
                tier: "enrich".into(),
                model: e.name.clone(),
                judge_mode: judge_mode.into(),
                total: e.total,
                scores: vec![
                    ("treue".into(), e.faithfulness),
                    ("prägnanz".into(), e.concision),
                    ("tags".into(), e.tags),
                ],
                items_scored: e.items_scored as i64,
                elapsed_ms: e.elapsed_ms as i64,
                is_winner: i == 0,
            })
            .await?;
    }
    if let Some(w) = outcome.winner() {
        println!(
            "→ Gewinner: {} (total {:.2}) · in Historie gespeichert",
            w.name, w.total
        );
    }
    Ok(())
}

/// Zeigt die Bakeoff-Historie (`ibrief bench list`) — neueste Läufe zuerst.
async fn run_bench_list(profile: &ProfileFile) -> Result<()> {
    let store = Store::open(&profile.store.path).await?;
    let runs = store.recent_bench_runs(40).await?;
    if runs.is_empty() {
        println!("noch keine Bakeoff-Läufe (erst `ibrief bench` oder `ibrief bench enrich`)");
        return Ok(());
    }
    for r in runs {
        let star = if r.is_winner { "★" } else { " " };
        let subs: Vec<String> = r
            .scores
            .iter()
            .map(|(k, v)| format!("{k}={v:.2}"))
            .collect();
        println!(
            "{star} {} {:<7} {:<18} total={:.2}  [{}]  {} ms  (judge {})",
            r.date,
            r.tier,
            r.model,
            r.total,
            subs.join(" "),
            r.elapsed_ms,
            r.judge_mode,
        );
    }
    Ok(())
}

fn baseline_marker(name: &str, baseline: &str) -> &'static str {
    if name == baseline { " (baseline)" } else { "" }
}

async fn run_learn(profile: &ProfileFile) -> Result<()> {
    let store = Store::open(&profile.store.path).await?;
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    let outcome = ibrief_learn::learn_once(&store, seed).await?;

    if outcome.adopted {
        println!("✓ neue Config übernommen: {}", outcome.version);
        if let Some(parent) = &outcome.parent {
            println!("  Eltern-Version: {parent}");
        }
        println!("  Grund: {}", outcome.reason);
    } else {
        println!("✗ kein Update — aktive Config bleibt");
        println!("  {}", outcome.reason);
    }
    println!("  Feedback-Ereignisse: {}", outcome.n_feedback);
    match (outcome.eval_active, outcome.eval_candidate) {
        (Some(a), Some(c)) => {
            println!("  Präferenz-AUC (reales Feedback): aktiv {a:.3} → Kandidat {c:.3}")
        }
        _ => println!("  Präferenz-AUC: nicht beurteilbar (einseitiges/dünnes Feedback)"),
    }
    for r in &outcome.gate_reasons {
        println!("  · Gate: {r}");
    }
    Ok(())
}

async fn run_config(profile: &ProfileFile, rest: &[String]) -> Result<()> {
    let store = Store::open(&profile.store.path).await?;
    let sub = rest.first().map(String::as_str).unwrap_or("list");

    match sub {
        "list" => {
            let active = store.active_config_version().await?;
            let configs = store.recent_configs(20).await?;
            if configs.is_empty() {
                println!("noch keine gelernten Configs (Default-Gewichte aktiv)");
                return Ok(());
            }
            for c in configs {
                let marker = if active.as_deref() == Some(c.version.as_str()) {
                    "* "
                } else {
                    "  "
                };
                println!("{marker}{}  {}  ({})", c.version, c.reason, c.created_at);
            }
            Ok(())
        }
        "rollback" => {
            let version = rest
                .get(1)
                .context("Verwendung: ibrief config rollback <version>")?;
            ibrief_learn::rollback(&store, version).await?;
            println!("✓ aktive Config auf {version} zurückgesetzt");
            Ok(())
        }
        other => anyhow::bail!("unbekannter config-Befehl '{other}' (erwartet: list | rollback)"),
    }
}

/// Lokales Judge-Modell: `judge_model` aus dem Profil, Fallback synth_model (§T2.4).
fn local_judge(profile: &ProfileFile) -> OllamaClient {
    let model = if profile.llm.judge_model.is_empty() {
        &profile.llm.synth_model
    } else {
        &profile.llm.judge_model
    };
    OllamaClient::new(profile.llm.ollama_url.clone(), model.clone())
}

async fn run_optimize(profile: &ProfileFile, rest: &[String]) -> Result<()> {
    let calibrate = rest.iter().any(|a| a == "calibrate");
    // Optionales Datum: beschränkt den Schatten-Test auf diesen Tag; ohne Datum laufen
    // die jüngsten Briefing-Tage (Multi-Tag-Urteil, §T2.4).
    let date = rest.iter().find(|a| a.as_str() != "calibrate").cloned();

    let store = Store::open(&profile.store.path).await?;
    let synth = OllamaClient::new(
        profile.llm.ollama_url.clone(),
        profile.llm.synth_model.clone(),
    );

    // Optimizer & Judge: lokal, oder via Abo (claude -p) bei `calibrate`.
    let optimizer: Box<dyn LanguageModel> = if calibrate {
        Box::new(ClaudeCodeModel::new(Some("opus".to_string())))
    } else {
        Box::new(OllamaClient::new(
            profile.llm.ollama_url.clone(),
            profile.llm.synth_model.clone(),
        ))
    };
    let judge: Box<dyn LanguageModel> = if calibrate {
        Box::new(ClaudeCodeModel::new(Some("opus".to_string())))
    } else {
        Box::new(local_judge(profile))
    };

    let outcome = ibrief_optimize::optimize_tldr(
        &store,
        optimizer.as_ref(),
        &synth,
        judge.as_ref(),
        date.as_deref(),
    )
    .await?;

    if outcome.adopted {
        println!(
            "✓ neuer TL;DR-Prompt übernommen: {}",
            outcome.candidate_version
        );
        println!("  vorher: {}", outcome.active_version);
    } else {
        println!(
            "✗ kein Update — aktiver Prompt bleibt: {}",
            outcome.active_version
        );
    }
    println!("  {}", outcome.reason);
    Ok(())
}

async fn run_experiment(profile: &ProfileFile, rest: &[String]) -> Result<()> {
    let store = Store::open(&profile.store.path).await?;
    let sub = rest.first().map(String::as_str).unwrap_or("list");
    match sub {
        "list" => {
            let experiments = store.recent_experiments(20).await?;
            if experiments.is_empty() {
                println!("noch keine Experimente");
                return Ok(());
            }
            for e in experiments {
                println!(
                    "[{}] {}/{}  {} → {}  ({})",
                    e.status, e.kind, e.slot, e.control, e.candidate, e.created_at
                );
            }
            Ok(())
        }
        other => anyhow::bail!("unbekannter experiment-Befehl '{other}' (erwartet: list)"),
    }
}

async fn seed_sources(cfg_dir: &Path, store: &Store) -> Result<()> {
    let sources: SourcesFile = load_toml(cfg_dir.join("sources.toml"))?;
    for s in &sources.source {
        store.seed_source(&s.id, &s.url, "seed").await?;
    }
    Ok(())
}

async fn run_sources(cfg_dir: &Path, profile: &ProfileFile, rest: &[String]) -> Result<()> {
    let store = Store::open(&profile.store.path).await?;
    seed_sources(cfg_dir, &store).await?;
    let sub = rest.first().map(String::as_str).unwrap_or("list");

    match sub {
        "list" => {
            for s in store.all_sources().await? {
                let marker = if s.active { "*" } else { " " };
                println!("{marker} {:<12} q={:.2}  {}", s.id, s.quality, s.url);
            }
            Ok(())
        }
        "evolve" => {
            let o = ibrief_sources::evolve_once(&store).await?;
            println!("Quellen-Evolution:");
            println!("  Qualität aktualisiert: {}", o.quality_updates);
            println!("  Drift: {:?}", o.drift);
            println!("  {}", o.note);
            if !o.deactivated.is_empty() {
                println!("  deaktiviert: {}", o.deactivated.join(", "));
            }
            Ok(())
        }
        other => anyhow::bail!("unbekannter sources-Befehl '{other}' (erwartet: list | evolve)"),
    }
}

async fn run_research(profile: &ProfileFile, rest: &[String]) -> Result<()> {
    if rest.is_empty() {
        anyhow::bail!("Verwendung: ibrief research <frage>");
    }
    let question = rest.join(" ");
    let store = Store::open(&profile.store.path).await?;
    let source = ibrief_research::StoreResearchSource::new(&store);
    let model = OllamaClient::new(
        profile.llm.ollama_url.clone(),
        profile.llm.synth_model.clone(),
    );

    let res = ibrief_research::research(
        &question,
        &source,
        &model,
        &ibrief_research::Budget::default(),
    )
    .await?;

    println!(
        "AutoResearch: {:?} ({} Iterationen, {} Quellen)",
        res.status,
        res.iterations,
        res.sources_used.len()
    );
    if !res.answer_md.is_empty() {
        println!("\n{}\n", res.answer_md);
    }
    if !res.claims.is_empty() {
        println!("Belegte Aussagen:");
        for c in &res.claims {
            println!("  • {}  [{}]", c.text, c.source);
        }
    }
    if !res.unverified_claims.is_empty() {
        println!("\n⚠ unbelegt (nicht übernommen):");
        for u in &res.unverified_claims {
            println!("  · {u}");
        }
    }
    Ok(())
}

fn telegram_from(profile: &ProfileFile) -> Result<Telegram> {
    let token =
        std::env::var(TOKEN_ENV).with_context(|| format!("Env-Var {TOKEN_ENV} nicht gesetzt"))?;
    if profile.telegram.chat_id.is_empty() {
        anyhow::bail!("telegram.chat_id ist leer (config/profile.toml)");
    }
    Ok(Telegram::new(token, profile.telegram.chat_id.clone()))
}

fn load_toml<T: DeserializeOwned>(path: PathBuf) -> Result<T> {
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("{} lesen", path.display()))?;
    let value = toml::from_str(&text).with_context(|| format!("{} parsen", path.display()))?;
    Ok(value)
}

fn today() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}
