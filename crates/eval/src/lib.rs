//! Eval Engine (§6.2): "Was ist ein gutes Briefing?" als Zahl + Diagnose.
//!
//! Drei Quellen, gewichtet kombiniert:
//!   1. Verhaltens-Score — aus dem realen Feedback (Ground Truth).
//!   2. LLM-Judge        — Bewertung gegen eine versionierte Rubrik (lokal oder via Abo).
//!   3. Strukturchecks   — deterministisch (Lesezeit, Diversität, Vollständigkeit).
//!
//! Wichtig: Der Verhaltens-Score dominiert; der Judge dient v.a. im Cold-Start
//! (wenig Feedback) und wird langfristig gegen reales Feedback kalibriert.

use anyhow::Result;
use ibrief_core::Briefing;
use ibrief_llm::{Completion, LanguageModel};
use ibrief_store::FeedbackCounts;
use serde::Deserialize;
use std::collections::HashMap;

pub const RUBRIC_VERSION: &str = "r1";

const WORDS_PER_MIN: f64 = 200.0;
const MAX_SOURCE_SHARE: f64 = 0.6;
const OPEN_WEIGHT: f64 = 0.3;

const JUDGE_SYSTEM: &str =
    "Du bist ein strenger Redaktions-Gutachter. Antworte ausschließlich mit JSON.";

const RUBRIC: &str = "Bewerte das folgende Morning Briefing auf einer Skala von 0.0 bis 1.0 \
je Kriterium: relevance (Relevanz für einen KI-Unternehmer & Entwickler), novelty (Neuigkeitswert), \
diversity (thematische Vielfalt), anti_bubble (fordert es heraus statt nur zu bestätigen), \
concision (Prägnanz), actionability (Anschlussfähigkeit für Entscheidungen/Gespräche). \
Berechne overall als Gesamteindruck (0.0-1.0).";

/// Gewichte der drei Eval-Quellen (Default 0.5 / 0.3 / 0.2 laut SPEC §6.2).
#[derive(Debug, Clone, Copy)]
pub struct EvalWeights {
    pub behavior: f64,
    pub judge: f64,
    pub structure: f64,
}

impl Default for EvalWeights {
    fn default() -> Self {
        Self {
            behavior: 0.5,
            judge: 0.3,
            structure: 0.2,
        }
    }
}

/// Ergebnis einer Bewertung.
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub behavior: f64,
    pub judge: f64,
    pub structure: f64,
    pub total: f64,
    pub notes: Vec<String>,
}

/// Bewertet ein Briefing. Schlägt der Judge fehl, fällt er auf den Verhaltens-Score zurück.
pub async fn evaluate(
    briefing: &Briefing,
    feedback: &FeedbackCounts,
    reading_time_budget_min: u32,
    weights: &EvalWeights,
    judge_model: &dyn LanguageModel,
) -> EvalResult {
    let mut notes = Vec::new();

    let behavior = behavior_score(feedback);
    let (structure, mut struct_notes) = structure_score(briefing, reading_time_budget_min);
    notes.append(&mut struct_notes);

    let judge = match judge_score(briefing, judge_model).await {
        Ok((score, comment)) => {
            notes.push(comment);
            score
        }
        Err(e) => {
            notes.push(format!(
                "judge:fehlgeschlagen ({e}) → Fallback auf Verhaltens-Score"
            ));
            behavior
        }
    };

    let sum = weights.behavior + weights.judge + weights.structure;
    let total = if sum > 0.0 {
        (weights.behavior * behavior + weights.judge * judge + weights.structure * structure) / sum
    } else {
        0.0
    };

    EvalResult {
        behavior,
        judge,
        structure,
        total,
        notes,
    }
}

/// Verhaltens-Score aus Feedback: pos/(pos+neg). Ohne Signale neutral (0.5, Cold-Start).
fn behavior_score(f: &FeedbackCounts) -> f64 {
    let pos = f.up as f64 + f.more as f64 + OPEN_WEIGHT * f.open as f64;
    let neg = f.down as f64 + f.less as f64;
    if pos + neg == 0.0 {
        0.5
    } else {
        pos / (pos + neg)
    }
}

/// Deterministische Strukturchecks → Anteil bestandener Checks + Diagnose-Notizen.
fn structure_score(b: &Briefing, budget_min: u32) -> (f64, Vec<String>) {
    let n_items: usize = b.sections.iter().map(|s| s.items.len()).sum();
    let est_min = word_count(b) as f64 / WORDS_PER_MIN;

    let has_section = |id: &str| b.sections.iter().any(|s| s.id == id);

    let checks: [(&str, bool); 6] = [
        ("reading_time", est_min <= budget_min as f64 + 0.5),
        ("has_items", n_items > 0),
        ("tldr_present", !b.tldr.is_empty()),
        (
            "source_diversity",
            n_items < 3 || source_max_share(b) <= MAX_SOURCE_SHARE,
        ),
        // Anti-Blase-Invarianten (§3): jetzt erzeugt und damit gescort.
        ("counterpoint_present", has_section("counterpoint")),
        ("wildcard_present", has_section("wildcard")),
    ];

    let passed = checks.iter().filter(|(_, ok)| *ok).count();
    let score = passed as f64 / checks.len() as f64;

    let notes: Vec<String> = checks
        .iter()
        .filter(|(_, ok)| !*ok)
        .map(|(label, _)| format!("struct:fail:{label}"))
        .collect();

    (score, notes)
}

fn word_count(b: &Briefing) -> usize {
    let mut n: usize = b.tldr.iter().map(|t| t.split_whitespace().count()).sum();
    for s in &b.sections {
        for it in &s.items {
            n += it.title.split_whitespace().count();
            if let Some(sum) = &it.summary {
                n += sum.split_whitespace().count();
            }
        }
    }
    n
}

fn source_max_share(b: &Briefing) -> f64 {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut total = 0usize;
    for s in &b.sections {
        for it in &s.items {
            *counts.entry(it.source_id.as_str()).or_default() += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    counts.values().copied().max().unwrap_or(0) as f64 / total as f64
}

#[derive(Deserialize)]
struct JudgeOut {
    overall: f64,
    #[serde(default)]
    comment: String,
}

async fn judge_score(b: &Briefing, model: &dyn LanguageModel) -> Result<(f64, String)> {
    let prompt = format!(
        "{RUBRIC}\n\nBRIEFING:\n{}\n\nAntworte NUR mit JSON der Form \
{{\"relevance\":0.0,\"novelty\":0.0,\"diversity\":0.0,\"anti_bubble\":0.0,\
\"concision\":0.0,\"actionability\":0.0,\"overall\":0.0,\"comment\":\"kurz\"}}.",
        brief_to_text(b)
    );
    let req = Completion::new(prompt)
        .with_system(JUDGE_SYSTEM)
        .temperature(0.2);
    let raw = model.complete(&req).await?;
    let out: JudgeOut = serde_json::from_str(&extract_json(&raw))?;
    Ok((
        out.overall.clamp(0.0, 1.0),
        format!("judge: {}", out.comment),
    ))
}

fn brief_to_text(b: &Briefing) -> String {
    let mut t = String::new();
    if !b.tldr.is_empty() {
        t.push_str("TL;DR:\n");
        for x in &b.tldr {
            t.push_str(&format!("- {x}\n"));
        }
    }
    for s in &b.sections {
        t.push_str(&format!("\n## {}\n", s.title));
        for it in &s.items {
            t.push_str(&format!("- {}", it.title));
            if let Some(sum) = &it.summary {
                t.push_str(&format!(": {sum}"));
            }
            t.push('\n');
        }
    }
    t
}

/// Toleriert ```json-Fences / umgebende Prosa um das JSON-Objekt.
fn extract_json(s: &str) -> String {
    match (s.find('{'), s.rfind('}')) {
        (Some(a), Some(b)) if b >= a => s[a..=b].to_string(),
        _ => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ibrief_core::{Briefing, BriefingSection, ContentItem};

    fn item(source: &str) -> ContentItem {
        ContentItem {
            id: source.into(),
            source_id: source.into(),
            title: "Kurzer Titel".into(),
            url: "https://example.com".into(),
            published_at: None,
            raw_summary: None,
            summary: Some("Ein Satz Zusammenfassung.".into()),
            topics: vec![],
        }
    }

    #[test]
    fn behavior_neutral_without_feedback() {
        let f = FeedbackCounts::default();
        assert!((behavior_score(&f) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn behavior_reflects_thumbs() {
        let f = FeedbackCounts {
            up: 3,
            down: 1,
            ..Default::default()
        };
        assert!((behavior_score(&f) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn structure_flags_source_monoculture() {
        // 4 Items, alle aus derselben Quelle → source_diversity-Check fällt.
        let b = Briefing {
            date: "2026-06-28".into(),
            tldr: vec!["a".into()],
            sections: vec![BriefingSection {
                id: "ai_tech".into(),
                title: "T".into(),
                items: vec![item("verge"), item("verge"), item("verge"), item("verge")],
            }],
        };
        let (score, notes) = structure_score(&b, 5);
        assert!(score < 1.0);
        assert!(notes.iter().any(|n| n.contains("source_diversity")));
    }
}
