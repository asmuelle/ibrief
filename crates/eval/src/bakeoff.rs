//! Modell-Bakeoff (§6.2 / A/B): gleicher Tag, verschiedene Synthese-Modelle, fair gescort.
//!
//! Nur die **Synthese-Stufe** (TL;DR + Gegenperspektive) variiert pro Kandidat — Items,
//! Ranking und Wildcard sind identisch. So vergleicht der Judge wirklich das Modell und
//! nicht die Inhaltsauswahl. Der Judge und die TL;DR-Vorlage sind für alle Kandidaten gleich.
//!
//! Der Verhaltens-Score (aus realem Feedback des ausgelieferten Briefings) ist über alle
//! Kandidaten konstant und verschiebt damit nur das Niveau, nicht die Reihenfolge — die
//! Differenzierung kommt aus Judge + Strukturchecks der jeweiligen Variante.

use std::time::Instant;

use anyhow::Result;
use ibrief_core::{Briefing, BriefingSection, ContentItem};
use ibrief_llm::{Completion, LanguageModel};
use ibrief_pipeline::{make_counterpoint, make_tldr};
use ibrief_store::FeedbackCounts;
use serde::Deserialize;

use crate::{EvalResult, EvalWeights, evaluate, extract_json};

const COUNTERPOINT_ID: &str = "counterpoint";

/// Ein zu testendes Synthese-Modell mit sprechendem Namen (z.B. "gemma4:31b").
pub struct Candidate<'a> {
    pub name: String,
    pub model: &'a dyn LanguageModel,
}

/// Ergebnis eines Kandidaten im Bakeoff.
#[derive(Debug, Clone)]
pub struct BakeoffEntry {
    pub name: String,
    pub model_label: String,
    pub eval: EvalResult,
    pub elapsed_ms: u128,
}

/// Gesamtergebnis: Kandidaten, bestes zuerst (nach `eval.total`).
#[derive(Debug, Clone)]
pub struct BakeoffOutcome {
    pub date: String,
    pub entries: Vec<BakeoffEntry>,
}

impl BakeoffOutcome {
    /// Bester Kandidat (höchster `total`), falls überhaupt einer gelaufen ist.
    pub fn winner(&self) -> Option<&BakeoffEntry> {
        self.entries.first()
    }
}

/// Führt den Bakeoff aus: regeneriert die Synthese je Kandidat und bewertet sie mit demselben Judge.
pub async fn run(
    base: &Briefing,
    feedback: &FeedbackCounts,
    reading_time_budget_min: u32,
    weights: &EvalWeights,
    judge: &dyn LanguageModel,
    tldr_template: &str,
    candidates: &[Candidate<'_>],
) -> BakeoffOutcome {
    let mut entries = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let started = Instant::now();
        let variant = synthesize_variant(base, cand.model, tldr_template).await;
        let eval = evaluate(&variant, feedback, reading_time_budget_min, weights, judge).await;
        entries.push(BakeoffEntry {
            name: cand.name.clone(),
            model_label: cand.model.label().to_string(),
            eval,
            elapsed_ms: started.elapsed().as_millis(),
        });
    }

    entries.sort_by(|a, b| {
        b.eval
            .total
            .partial_cmp(&a.eval.total)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    BakeoffOutcome {
        date: base.date.clone(),
        entries,
    }
}

/// Baut eine Briefing-Variante: identische Items/Wildcard, aber TL;DR + Gegenperspektive
/// frisch vom Kandidatenmodell. Schlägt eine Synthese-Stufe fehl, läuft es ohne weiter —
/// die Strukturchecks im Eval bestrafen die fehlenden Teile dann fair.
async fn synthesize_variant(
    base: &Briefing,
    model: &dyn LanguageModel,
    tldr_template: &str,
) -> Briefing {
    let mut variant = Briefing {
        date: base.date.clone(),
        tldr: Vec::new(),
        sections: base
            .sections
            .iter()
            .filter(|s| s.id != COUNTERPOINT_ID)
            .cloned()
            .collect(),
    };

    match make_tldr(&variant, model, tldr_template).await {
        Ok(tldr) => variant.tldr = tldr,
        Err(e) => {
            tracing::warn!(model = model.label(), error = %e, "bakeoff: TL;DR fehlgeschlagen")
        }
    }

    match make_counterpoint(&variant, model, &variant.date).await {
        Ok(Some(cp)) => variant.sections.push(BriefingSection {
            id: COUNTERPOINT_ID.into(),
            title: "Gegenperspektive".into(),
            items: vec![cp],
        }),
        Ok(None) => tracing::warn!(model = model.label(), "bakeoff: Gegenperspektive leer"),
        Err(e) => {
            tracing::warn!(model = model.label(), error = %e, "bakeoff: Gegenperspektive fehlgeschlagen")
        }
    }

    variant
}

// ---------------------------------------------------------------------------
// Enrich-Tier-Bakeoff: das schnelle Massen-Modell testen.
//
// Anders als beim Synth-Bakeoff gibt es kein einzelnes Urteilsobjekt — bewertet
// wird jede erzeugte Ein-Satz-Zusammenfassung einzeln gegen ihren Quelltext
// (Variante 1: ehrlicher Test der Zusammenfassungs-Qualität, nicht des Gesamteffekts).
// ---------------------------------------------------------------------------

const ENRICH_JUDGE_SYSTEM: &str = "Du bist ein strenger Redaktions-Gutachter für Kurzzusammenfassungen. \
Antworte ausschließlich mit JSON.";

/// Mittel-Noten eines Enrich-Kandidaten über alle bewerteten Items.
#[derive(Debug, Clone)]
pub struct EnrichEntry {
    pub name: String,
    pub model_label: String,
    /// Anzahl Items, die tatsächlich bewertet werden konnten.
    pub items_scored: usize,
    pub faithfulness: f64,
    pub concision: f64,
    pub tags: f64,
    /// Gesamt-Mittelwert (`overall`) — Sortierschlüssel.
    pub total: f64,
    pub elapsed_ms: u128,
}

/// Gesamtergebnis des Enrich-Bakeoffs, bestes zuerst.
#[derive(Debug, Clone)]
pub struct EnrichOutcome {
    pub entries: Vec<EnrichEntry>,
}

impl EnrichOutcome {
    pub fn winner(&self) -> Option<&EnrichEntry> {
        self.entries.first()
    }
}

#[derive(Deserialize)]
struct EnrichJudgeOut {
    faithfulness: f64,
    concision: f64,
    tags: f64,
    overall: f64,
}

/// Führt den Enrich-Bakeoff aus: jedes Kandidatenmodell reichert dieselben Items an,
/// dann benotet derselbe Judge jede Zusammenfassung gegen ihren Quelltext.
pub async fn run_enrich(
    base: &Briefing,
    judge: &dyn LanguageModel,
    max_items: usize,
    candidates: &[Candidate<'_>],
) -> EnrichOutcome {
    let inputs = enrich_inputs(base);
    let mut entries = Vec::with_capacity(candidates.len());

    for cand in candidates {
        let started = Instant::now();
        let enriched = ibrief_pipeline::enrich(inputs.clone(), cand.model, max_items).await;
        let elapsed_ms = started.elapsed().as_millis();

        let mut n = 0usize;
        let (mut faith, mut conc, mut tags, mut overall) = (0.0, 0.0, 0.0, 0.0);
        for it in enriched.iter().take(max_items) {
            let Some(summary) = it.summary.as_deref() else {
                continue;
            };
            match enrich_judge(judge, it, summary).await {
                Ok(s) => {
                    faith += s.faithfulness.clamp(0.0, 1.0);
                    conc += s.concision.clamp(0.0, 1.0);
                    tags += s.tags.clamp(0.0, 1.0);
                    overall += s.overall.clamp(0.0, 1.0);
                    n += 1;
                }
                Err(e) => {
                    tracing::warn!(model = cand.model.label(), error = %e, "enrich-bakeoff: Judge fehlgeschlagen")
                }
            }
        }

        // §T1.5: Ein Kandidat ohne EINE bewertbare Zusammenfassung wird NICHT als Qualität 0.0
        // gewertet (das würde einen transienten Ausfall wie ein wirklich schlechtes Modell ganz
        // nach unten sortieren) — er fällt aus dem Ranking, mit klarem Log.
        if n == 0 {
            tracing::warn!(
                model = cand.model.label(),
                "enrich-bakeoff: keine Zusammenfassung bewertbar (Modell/Judge nicht verfügbar?) — Kandidat ausgeschlossen, NICHT als 0.0 gewertet"
            );
            continue;
        }
        let avg = |sum: f64| sum / n as f64;
        entries.push(EnrichEntry {
            name: cand.name.clone(),
            model_label: cand.model.label().to_string(),
            items_scored: n,
            faithfulness: avg(faith),
            concision: avg(conc),
            tags: avg(tags),
            total: avg(overall),
            elapsed_ms,
        });
    }

    entries.sort_by(|a, b| {
        b.total
            .partial_cmp(&a.total)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    EnrichOutcome { entries }
}

/// Echte Inhalts-Items aus dem Briefing (ohne synthetische wie die Gegenperspektive),
/// mit zurückgesetzter Anreicherung — so wird ein fehlgeschlagenes Enrich nicht aus
/// alten Daten „gerettet" und der Vergleich bleibt fair.
fn enrich_inputs(base: &Briefing) -> Vec<ContentItem> {
    base.sections
        .iter()
        .flat_map(|s| s.items.iter())
        .filter(|it| !it.url.is_empty())
        .map(|it| ContentItem {
            summary: None,
            topics: Vec::new(),
            ..it.clone()
        })
        .collect()
}

async fn enrich_judge(
    judge: &dyn LanguageModel,
    item: &ContentItem,
    summary: &str,
) -> Result<EnrichJudgeOut> {
    let source = item.raw_summary.clone().unwrap_or_default();
    let tags = item.topics.join(", ");
    let prompt = format!(
        "Quelle:\nTitel: {}\nText: {}\n\nErzeugte Zusammenfassung: {}\nErzeugte Tags: {}\n\n\
Bewerte 0.0-1.0: faithfulness (sachlich treu, nichts erfunden), concision (genau ein knapper Satz), \
tags (treffende, konsistente Schlagworte). overall = Gesamteindruck. Antworte NUR mit JSON der Form \
{{\"faithfulness\":0.0,\"concision\":0.0,\"tags\":0.0,\"overall\":0.0}}.",
        item.title, source, summary, tags
    );
    let req = Completion::new(prompt)
        .with_system(ENRICH_JUDGE_SYSTEM)
        .temperature(0.2);
    let raw = judge.complete(&req).await?;
    Ok(serde_json::from_str(&extract_json(&raw))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use ibrief_core::{BriefingSection, ContentItem};
    use ibrief_llm::Completion;

    /// Liefert immer denselben String — steht für ein Synthese-Modell.
    struct Scripted {
        label: String,
        out: String,
    }

    #[async_trait]
    impl LanguageModel for Scripted {
        async fn complete(&self, _req: &Completion) -> Result<String, ibrief_llm::ModelError> {
            Ok(self.out.clone())
        }
        fn label(&self) -> &str {
            &self.label
        }
    }

    /// Judge, der eine Variante höher bewertet, wenn ihr Text den Marker "WIN" enthält.
    struct MarkerJudge;

    #[async_trait]
    impl LanguageModel for MarkerJudge {
        async fn complete(&self, req: &Completion) -> Result<String, ibrief_llm::ModelError> {
            let score = if req.prompt.contains("WIN") { 0.9 } else { 0.2 };
            Ok(format!("{{\"overall\":{score},\"comment\":\"t\"}}"))
        }
        fn label(&self) -> &str {
            "judge"
        }
    }

    fn item(source: &str) -> ContentItem {
        ContentItem {
            id: source.into(),
            source_id: source.into(),
            title: "Titel".into(),
            url: format!("https://example.com/{source}"),
            published_at: None,
            raw_summary: Some("Quelltext des Artikels.".into()),
            summary: Some("Ein Satz.".into()),
            topics: vec![],
            entities: vec![],
            embedding: None,
        }
    }

    /// Synthetisches Item wie die Gegenperspektive: ohne URL → kein echter Artikel.
    fn synthetic(id: &str) -> ContentItem {
        ContentItem {
            url: String::new(),
            source_id: "ibrief".into(),
            ..item(id)
        }
    }

    fn base_with_counterpoint() -> Briefing {
        Briefing {
            date: "2026-06-29".into(),
            tldr: vec!["alte Zusammenfassung".into()],
            sections: vec![
                BriefingSection {
                    id: "ai_tech".into(),
                    title: "KI & Tech".into(),
                    items: vec![item("verge"), item("hn")],
                },
                BriefingSection {
                    id: "counterpoint".into(),
                    title: "Gegenperspektive".into(),
                    items: vec![synthetic("cp")],
                },
            ],
        }
    }

    #[tokio::test]
    async fn variant_resets_tldr_and_replaces_counterpoint() {
        let base = base_with_counterpoint();
        let model = Scripted {
            label: "cand".into(),
            out: "- x\n- y\n- z".into(),
        };

        let v = synthesize_variant(&base, &model, "Fasse zusammen:\n{items}").await;

        // Frisches TL;DR (3 Bullets), nicht das alte.
        assert_eq!(v.tldr.len(), 3);
        assert!(!v.tldr.iter().any(|t| t.contains("alte")));
        // Genau eine Gegenperspektive — die alte wurde ersetzt, nicht dupliziert.
        let cps = v.sections.iter().filter(|s| s.id == "counterpoint").count();
        assert_eq!(cps, 1);
    }

    #[tokio::test]
    async fn ranks_best_synth_model_first() {
        let base = base_with_counterpoint();
        let win = Scripted {
            label: "ollama:A".into(),
            out: "- WIN\n- b\n- c".into(),
        };
        let lose = Scripted {
            label: "ollama:B".into(),
            out: "- a\n- b\n- c".into(),
        };
        let judge = MarkerJudge;
        let candidates = vec![
            Candidate {
                name: "B".into(),
                model: &lose,
            },
            Candidate {
                name: "A".into(),
                model: &win,
            },
        ];

        let out = run(
            &base,
            &FeedbackCounts::default(),
            5,
            &EvalWeights::default(),
            &judge,
            "Fasse zusammen:\n{items}",
            &candidates,
        )
        .await;

        assert_eq!(out.entries.len(), 2);
        assert_eq!(out.winner().unwrap().name, "A");
        // Reihenfolge absteigend nach total.
        assert!(out.entries[0].eval.total >= out.entries[1].eval.total);
    }

    /// Judge mit festen Enrich-Noten (für den Mechanik-Test).
    struct FixedEnrichJudge;

    #[async_trait]
    impl LanguageModel for FixedEnrichJudge {
        async fn complete(&self, _req: &Completion) -> Result<String, ibrief_llm::ModelError> {
            Ok(r#"{"faithfulness":0.9,"concision":0.8,"tags":0.7,"overall":0.85}"#.into())
        }
        fn label(&self) -> &str {
            "enrich-judge"
        }
    }

    /// Enrich-Judge, der hoch bewertet, wenn die Zusammenfassung "GUT" enthält.
    struct MarkerEnrichJudge;

    #[async_trait]
    impl LanguageModel for MarkerEnrichJudge {
        async fn complete(&self, req: &Completion) -> Result<String, ibrief_llm::ModelError> {
            let s = if req.prompt.contains("GUT") { 0.9 } else { 0.2 };
            Ok(format!(
                "{{\"faithfulness\":{s},\"concision\":{s},\"tags\":{s},\"overall\":{s}}}"
            ))
        }
        fn label(&self) -> &str {
            "enrich-judge"
        }
    }

    #[tokio::test]
    async fn enrich_scores_each_real_item() {
        let base = base_with_counterpoint(); // 2 echte Items + 1 synthetisches (ohne URL)
        let model = Scripted {
            label: "ollama:cand".into(),
            out: r#"{"summary":"Treuer Satz.","topics":["ki"]}"#.into(),
        };
        let judge = FixedEnrichJudge;
        let candidates = vec![Candidate {
            name: "cand".into(),
            model: &model,
        }];

        let out = run_enrich(&base, &judge, 15, &candidates).await;

        let e = out.winner().unwrap();
        // Nur die zwei echten Artikel werden bewertet, nicht die synthetische Gegenperspektive.
        assert_eq!(e.items_scored, 2);
        assert!((e.total - 0.85).abs() < 1e-9);
        assert!((e.faithfulness - 0.9).abs() < 1e-9);
    }

    #[tokio::test]
    async fn enrich_ranks_better_summaries_first() {
        let base = base_with_counterpoint();
        let good = Scripted {
            label: "ollama:A".into(),
            out: r#"{"summary":"GUT zusammengefasst.","topics":["x"]}"#.into(),
        };
        let bad = Scripted {
            label: "ollama:B".into(),
            out: r#"{"summary":"mies.","topics":["x"]}"#.into(),
        };
        let judge = MarkerEnrichJudge;
        let candidates = vec![
            Candidate {
                name: "B".into(),
                model: &bad,
            },
            Candidate {
                name: "A".into(),
                model: &good,
            },
        ];

        let out = run_enrich(&base, &judge, 15, &candidates).await;

        assert_eq!(out.entries.len(), 2);
        assert_eq!(out.winner().unwrap().name, "A");
        assert!(out.entries[0].total >= out.entries[1].total);
    }
}
