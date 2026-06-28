//! ibrief M3 — Briefing (M1) + Persistenz/Feedback (M2) + Eval Engine (M3).
//!
//! Befehle:
//!   ibrief [brief]            Ingest → Dedup → Enrich → Score(Gewichte) → Curate → Render → Persist → (Push)
//!   ibrief feedback           Telegram-Feedback-Loop: Button-Klicks → Store
//!   ibrief eval [datum] [calibrate]
//!                             Briefing bewerten (Verhalten + Judge + Struktur) → evals-Tabelle
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
use ibrief_core::BriefingSection;
use ibrief_eval::{EvalWeights, RUBRIC_VERSION};
use ibrief_ingest::Source;
use ibrief_llm::{ClaudeCodeModel, LanguageModel, OllamaClient};
use ibrief_store::{EvalRow, Store};
use ibrief_telegram::Telegram;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const CONFIG_VERSION: &str = "m3-dev";
const TOKEN_ENV: &str = "IBRIEF_TELEGRAM_TOKEN";
const CONFIG_DIR_ENV: &str = "IBRIEF_CONFIG_DIR";

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
    max_items_enrich: usize,
    top_n: usize,
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
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "brief".to_string());
    let rest: Vec<String> = args.collect();

    let cfg_dir = std::env::var(CONFIG_DIR_ENV).unwrap_or_else(|_| "config".to_string());
    let cfg_dir = Path::new(&cfg_dir);
    let profile: ProfileFile = load_toml(cfg_dir.join("profile.toml"))?;

    match command.as_str() {
        "brief" => run_brief(cfg_dir, &profile).await,
        "feedback" => run_feedback(&profile).await,
        "eval" => run_eval(&profile, &rest).await,
        "learn" => run_learn(&profile).await,
        "config" => run_config(&profile, &rest).await,
        "optimize" => run_optimize(&profile, &rest).await,
        "experiment" => run_experiment(&profile, &rest).await,
        "sources" => run_sources(cfg_dir, &profile, &rest).await,
        "research" => run_research(&profile, &rest).await,
        other => anyhow::bail!(
            "unbekannter Befehl '{other}' (erwartet: brief | feedback | eval | learn | config | optimize | experiment | sources | research)"
        ),
    }
}

async fn run_brief(cfg_dir: &Path, profile: &ProfileFile) -> Result<()> {
    let store = Store::open(&profile.store.path).await?;

    // Quellen-Registry: aus sources.toml seeden (idempotent), dann aktive Quellen laden.
    seed_sources(cfg_dir, &store).await?;
    let active = store.active_sources().await?;
    let sources: Vec<Source> = active
        .into_iter()
        .map(|s| Source {
            id: s.id,
            url: s.url,
        })
        .collect();

    // INGEST + DEDUP
    tracing::info!(sources = sources.len(), "INGEST");
    let fetched = ibrief_ingest::fetch_all(&sources).await;
    let fresh = store.filter_unseen(fetched).await?;
    tracing::info!(fresh = fresh.len(), "neue Items nach Dedup");
    if fresh.is_empty() {
        tracing::warn!("keine neuen Items — nichts zu briefen");
        return Ok(());
    }

    // ENRICH (Massen-Tier) + Persist
    let enrich_model = OllamaClient::new(
        profile.llm.ollama_url.clone(),
        profile.llm.enrich_model.clone(),
    );
    tracing::info!(model = enrich_model.label(), "ENRICH");
    let enriched =
        ibrief_pipeline::enrich(fresh, &enrich_model, profile.llm.max_items_enrich).await;
    for it in &enriched {
        store.upsert_item(it).await?;
    }

    // SCORE (gelernte Gewichte) + CURATE
    let cfg = ibrief_learn::load_active(&store).await?;
    let ranked = ibrief_pipeline::rank(enriched, &cfg);
    let wildcard = ibrief_pipeline::pick_wildcard(&ranked, profile.llm.top_n);
    let mut briefing = ibrief_pipeline::curate(ranked, profile.llm.top_n);
    briefing.date = today();

    // TL;DR (Synthese-Tier, mit aktivem/gelerntem Prompt)
    let synth_model = OllamaClient::new(
        profile.llm.ollama_url.clone(),
        profile.llm.synth_model.clone(),
    );
    let tldr_prompt = ibrief_optimize::active_tldr(&store).await?;
    tracing::info!(model = synth_model.label(), prompt = %tldr_prompt.version, "TL;DR");
    match ibrief_pipeline::make_tldr(&briefing, &synth_model, &tldr_prompt.template).await {
        Ok(tldr) => briefing.tldr = tldr,
        Err(e) => tracing::warn!(error = %e, "TL;DR-Erzeugung fehlgeschlagen, fahre ohne fort"),
    }

    // GEGENPERSPEKTIVE (§3, nicht abschaltbar — Anti-Blase)
    match ibrief_pipeline::make_counterpoint(&briefing, &synth_model, &briefing.date).await {
        Ok(Some(cp)) => {
            store.upsert_item(&cp).await?; // persistieren, damit load_briefing es findet
            briefing.sections.push(BriefingSection {
                id: "counterpoint".into(),
                title: "Gegenperspektive".into(),
                items: vec![cp],
            });
        }
        Ok(None) => tracing::warn!("Gegenperspektive leer — übersprungen"),
        Err(e) => tracing::warn!(error = %e, "Gegenperspektive fehlgeschlagen"),
    }

    // WILDCARD (§3, nicht abschaltbar — bewusste Überraschung)
    if let Some(w) = wildcard {
        briefing.sections.push(BriefingSection {
            id: "wildcard".into(),
            title: "Wildcard — über den Tellerrand".into(),
            items: vec![w],
        });
    }

    // PERSIST Briefing-Record
    store.save_briefing(&briefing, CONFIG_VERSION).await?;

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

async fn run_feedback(profile: &ProfileFile) -> Result<()> {
    let store = Store::open(&profile.store.path).await?;
    let tg = telegram_from(profile)?;
    tg.run_feedback_loop(&store).await
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
        Box::new(OllamaClient::new(
            profile.llm.ollama_url.clone(),
            profile.llm.synth_model.clone(),
        ))
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

    store
        .save_eval(&EvalRow {
            date: date.clone(),
            config_version: CONFIG_VERSION.to_string(),
            rubric_version: RUBRIC_VERSION.to_string(),
            behavior: res.behavior,
            judge: res.judge,
            structure: res.structure,
            total: res.total,
            notes: res.notes.clone(),
        })
        .await?;

    println!("Eval {date} (config {CONFIG_VERSION}, rubric {RUBRIC_VERSION}):");
    println!(
        "  total={:.2}  [behavior={:.2} judge={:.2} structure={:.2}]",
        res.total, res.behavior, res.judge, res.structure
    );
    for n in &res.notes {
        println!("  · {n}");
    }
    Ok(())
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

async fn run_optimize(profile: &ProfileFile, rest: &[String]) -> Result<()> {
    let calibrate = rest.iter().any(|a| a == "calibrate");
    let date = rest
        .iter()
        .find(|a| a.as_str() != "calibrate")
        .cloned()
        .unwrap_or_else(today);

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
        Box::new(OllamaClient::new(
            profile.llm.ollama_url.clone(),
            profile.llm.synth_model.clone(),
        ))
    };

    let outcome =
        ibrief_optimize::optimize_tldr(&store, optimizer.as_ref(), &synth, judge.as_ref(), &date)
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
