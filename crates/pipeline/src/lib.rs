//! Pipeline-Stages: ENRICH -> CURATE -> RENDER.
//!
//! M1 bewusst schlank: noch kein Scoring/Gewichte (kommt in M3/M4), eine Sektion,
//! TL;DR per Synthese-Modell. Alle LLM-Aufrufe laufen über das [`LanguageModel`]-Trait.

use anyhow::Result;
use ibrief_core::{Briefing, BriefingSection, Config, ContentItem};
use ibrief_llm::{Completion, LanguageModel};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

const ENRICH_SYSTEM: &str =
    "Du bist ein präziser Redaktions-Assistent. Antworte ausschließlich mit JSON.";

/// Enrich liefert nur einen Satz + max. 3 Tags — Obergrenze gegen ausuferndes Generieren
/// (großzügig, damit das JSON nie mitten im Objekt abgeschnitten wird und unparsebar wird).
const ENRICH_MAX_TOKENS: u32 = 200;
/// Synthese (TL;DR: 3 Bullets · Gegenperspektive: 2-3 Sätze) — deckelt den Generierungs-Schwanz.
const SYNTH_MAX_TOKENS: u32 = 400;

const COUNTERPOINT_SYSTEM: &str =
    "Du bist ein fairer, intellektuell ehrlicher Sparringspartner — kein Provokateur.";

const COUNTERPOINT_PROMPT: &str = "Hier sind die heutigen Meldungen:\n{items}\n\n\
Formuliere EINE ernsthafte, fair dargestellte Gegenperspektive zu einer eher links-liberalen \
Sicht auf eines dieser Themen — das stärkste Gegenargument (Steelman), keine Karikatur. \
2–3 Sätze auf Deutsch.";

#[derive(Deserialize)]
struct EnrichOut {
    summary: String,
    #[serde(default)]
    topics: Vec<String>,
}

/// ENRICH: für die ersten `max` Items eine Ein-Satz-Zusammenfassung + Tags erzeugen.
/// Läuft parallel in Batches à 4, um Ollama nicht zu überlasten.
pub async fn enrich(
    mut items: Vec<ContentItem>,
    model: &dyn LanguageModel,
    max: usize,
) -> Vec<ContentItem> {
    let n = max.min(items.len());
    let started = std::time::Instant::now();
    const CONCURRENCY: usize = 4;
    for chunk_start in (0..n).step_by(CONCURRENCY) {
        let chunk_end = (chunk_start + CONCURRENCY).min(n);
        let futs: Vec<_> = (chunk_start..chunk_end)
            .map(|i| enrich_one(&items[i], model))
            .collect();
        let results = futures::future::join_all(futs).await;
        for (i, result) in (chunk_start..).zip(results) {
            match result {
                Ok(out) => {
                    items[i].summary = Some(out.summary);
                    items[i].topics = out.topics;
                }
                Err(e) => {
                    tracing::warn!(title = %items[i].title, error = %e, "enrich fehlgeschlagen")
                }
            }
        }
    }
    let ok = items
        .iter()
        .take(n)
        .filter(|it| it.summary.is_some())
        .count();
    tracing::info!(
        items = n,
        ok,
        failed = n - ok,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "ENRICH abgeschlossen"
    );
    items
}

async fn enrich_one(item: &ContentItem, model: &dyn LanguageModel) -> Result<EnrichOut> {
    let context = item.raw_summary.clone().unwrap_or_default();
    let prompt = format!(
        "Titel: {}\nText: {}\n\nGib JSON zurück mit den Feldern \"summary\" \
(ein prägnanter Satz auf Deutsch) und \"topics\" (max. 3 Schlagworte). Nur JSON.",
        item.title, context
    );
    let req = Completion::new(prompt)
        .with_system(ENRICH_SYSTEM)
        .temperature(0.3)
        .max_tokens(ENRICH_MAX_TOKENS);
    let t0 = std::time::Instant::now();
    let raw = model.complete(&req).await?;
    tracing::debug!(
        title = %item.title,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "enrich_one"
    );
    let json = extract_json(&raw);
    Ok(serde_json::from_str(&json)?)
}

/// Toleriert ```json-Fences und umgebenden Prosatext um das JSON-Objekt herum.
fn extract_json(s: &str) -> String {
    match (s.find('{'), s.rfind('}')) {
        (Some(a), Some(b)) if b >= a => s[a..=b].to_string(),
        _ => s.to_string(),
    }
}

/// DEDUP (innerhalb eines Laufs): behält je `id` (= URL) nur das erste Vorkommen.
/// `filter_unseen` dedupliziert nur gegen frühere Tage — Feeds liefern aber auch
/// im selben Batch Doppel (z.B. dieselbe Meldung mehrfach), die hier herausfallen.
pub fn dedup_batch(items: Vec<ContentItem>) -> Vec<ContentItem> {
    let mut seen = HashSet::new();
    items
        .into_iter()
        .filter(|it| seen.insert(it.id.clone()))
        .collect()
}

/// SCORE (§ Pipeline / M4): Items nach `recency × source_weight × topic_weight` ordnen.
/// Neutrale Config (alle Gewichte 1.0) ⇒ reine Aktualitäts-Reihenfolge.
pub fn rank(mut items: Vec<ContentItem>, cfg: &Config) -> Vec<ContentItem> {
    items.sort_by_key(|i| std::cmp::Reverse(i.published_at));
    let n = items.len().max(1) as f64;

    let mut scored: Vec<(f64, ContentItem)> = items
        .into_iter()
        .enumerate()
        .map(|(i, it)| {
            let recency = 1.0 - (i as f64) / n;
            let source_w = cfg.source_weight(&it.source_id);
            let topic_w = if it.topics.is_empty() {
                1.0
            } else {
                it.topics.iter().map(|t| cfg.topic_weight(t)).sum::<f64>() / it.topics.len() as f64
            };
            (recency * source_w * topic_w, it)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, it)| it).collect()
}

/// CURATE: Top-N der bereits gerankten Items in eine Sektion legen — mit Quellen-Limit
/// gegen Monokultur. `max_per_source` begrenzt, wie viele Items eine einzelne Quelle
/// stellen darf. Bleiben danach Plätze frei (zu wenig Vielfalt vorhanden), werden sie
/// in Rangfolge mit den besten verbleibenden Items aufgefüllt, statt das Briefing zu kürzen.
pub fn curate(items: Vec<ContentItem>, top_n: usize, max_per_source: usize) -> Briefing {
    let cap = max_per_source.max(1);
    let mut per_source: HashMap<String, usize> = HashMap::new();
    let mut picked: Vec<ContentItem> = Vec::with_capacity(top_n);
    let mut overflow: Vec<ContentItem> = Vec::new();

    for it in items {
        if picked.len() >= top_n {
            break;
        }
        let count = per_source.entry(it.source_id.clone()).or_default();
        if *count < cap {
            *count += 1;
            picked.push(it);
        } else {
            overflow.push(it);
        }
    }

    // Restplätze auffüllen, falls die Vielfalt nicht reichte (Rangfolge bleibt erhalten).
    for it in overflow {
        if picked.len() >= top_n {
            break;
        }
        picked.push(it);
    }

    Briefing {
        date: String::new(), // wird vom Aufrufer gesetzt
        tldr: Vec::new(),
        sections: vec![BriefingSection {
            id: "ai_tech".into(),
            title: "KI & Tech — Highlights".into(),
            items: picked,
        }],
    }
}

/// "Die 3 Dinge heute" vom Synthese-Modell ableiten.
/// `template` ist die (lernbare, versionierte) Prompt-Vorlage; `{items}` wird ersetzt.
pub async fn make_tldr(
    briefing: &Briefing,
    model: &dyn LanguageModel,
    template: &str,
) -> Result<Vec<String>> {
    let mut lines = String::new();
    for sec in &briefing.sections {
        for it in &sec.items {
            let s = it.summary.clone().unwrap_or_else(|| it.title.clone());
            lines.push_str("- ");
            lines.push_str(&s);
            lines.push('\n');
        }
    }

    let prompt = if template.contains("{items}") {
        template.replace("{items}", &lines)
    } else {
        format!("{template}\n\n{lines}")
    };
    let raw = model
        .complete(
            &Completion::new(prompt)
                .temperature(0.4)
                .max_tokens(SYNTH_MAX_TOKENS),
        )
        .await?;

    let bullets = raw
        .lines()
        .map(|l| l.trim_start_matches(['-', '*', '•', ' ']).trim())
        .filter(|l| !l.is_empty())
        .take(3)
        .map(|l| l.to_string())
        .collect();
    Ok(bullets)
}

/// Zusammenfassungen (bzw. Titel) aller bisherigen Sektions-Items als Zeilenliste.
fn section_items_text(briefing: &Briefing) -> String {
    let mut lines = String::new();
    for sec in &briefing.sections {
        for it in &sec.items {
            let s = it.summary.clone().unwrap_or_else(|| it.title.clone());
            lines.push_str(&format!("- {s}\n"));
        }
    }
    lines
}

/// Erzeugt die **Gegenperspektive** (§3, nicht abschaltbar): ein faires Steelman-Gegenargument.
/// Synthetisches Item (Quelle `ibrief`, keine URL). `None`, wenn das Modell nichts liefert.
pub async fn make_counterpoint(
    briefing: &Briefing,
    model: &dyn LanguageModel,
    date: &str,
) -> Result<Option<ContentItem>> {
    let prompt = COUNTERPOINT_PROMPT.replace("{items}", &section_items_text(briefing));
    let text = model
        .complete(
            &Completion::new(prompt)
                .with_system(COUNTERPOINT_SYSTEM)
                .temperature(0.6)
                .max_tokens(SYNTH_MAX_TOKENS),
        )
        .await?
        .trim()
        .to_string();
    if text.is_empty() {
        return Ok(None);
    }
    Ok(Some(ContentItem {
        id: format!("counterpoint-{date}"),
        source_id: "ibrief".into(),
        title: "Gegenperspektive".into(),
        url: String::new(),
        published_at: None,
        raw_summary: None,
        summary: Some(text),
        topics: vec![],
        entities: vec![],
        embedding: None,
    }))
}

/// Wählt die **Wildcard** (§3): ein echter Artikel jenseits der Top-N, bevorzugt aus einer
/// Quelle, die nicht schon in der Hauptauswahl steckt — bewusste Überraschung gegen die Blase.
pub fn pick_wildcard(ranked: &[ContentItem], top_n: usize) -> Option<ContentItem> {
    if ranked.len() <= top_n {
        return None;
    }
    let top_sources: HashSet<&str> = ranked[..top_n]
        .iter()
        .map(|i| i.source_id.as_str())
        .collect();
    let leftover = &ranked[top_n..];
    leftover
        .iter()
        .find(|i| !top_sources.contains(i.source_id.as_str()))
        .or_else(|| leftover.first())
        .cloned()
}

/// RENDER: Briefing als Markdown.
pub fn render(briefing: &Briefing) -> String {
    let mut md = format!("# Morning Briefing — {}\n\n", briefing.date);

    if !briefing.tldr.is_empty() {
        md.push_str("## Die 3 Dinge heute\n\n");
        for t in &briefing.tldr {
            md.push_str(&format!("- {t}\n"));
        }
        md.push('\n');
    }

    for sec in &briefing.sections {
        md.push_str(&format!("## {}\n\n", sec.title));
        for it in &sec.items {
            md.push_str(&format!("### {}\n", it.title));
            if let Some(s) = &it.summary {
                md.push_str(&format!("{s}\n\n"));
            }
            if !it.url.is_empty() {
                md.push_str(&format!("[Quelle]({}) · _{}_\n\n", it.url, it.source_id));
            }
        }
    }

    md
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, source: &str) -> ContentItem {
        ContentItem {
            id: id.into(),
            source_id: source.into(),
            title: "t".into(),
            url: format!("https://x/{id}"),
            published_at: None,
            raw_summary: None,
            summary: Some("s".into()),
            topics: vec![],
            entities: vec![],
            embedding: None,
        }
    }

    #[test]
    fn wildcard_prefers_unseen_source() {
        // Top-2 stammen aus "a"; Wildcard soll "c" (neue Quelle) vor "a" bevorzugen.
        let ranked = vec![
            item("1", "a"),
            item("2", "a"),
            item("3", "a"),
            item("4", "c"),
        ];
        let w = pick_wildcard(&ranked, 2).unwrap();
        assert_eq!(w.source_id, "c");
    }

    #[test]
    fn wildcard_none_without_leftover() {
        let ranked = vec![item("1", "a"), item("2", "b")];
        assert!(pick_wildcard(&ranked, 2).is_none());
    }

    #[test]
    fn dedup_batch_removes_same_id_keeps_order() {
        let items = vec![item("1", "a"), item("1", "a"), item("2", "b")];
        let out = dedup_batch(items);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, "1");
        assert_eq!(out[1].id, "2");
    }

    #[test]
    fn curate_caps_per_source_for_diversity() {
        // 5 Items aus "a", 2 aus "b"; top_n=4, cap=2 → genau 2a + 2b statt 4a.
        let items = vec![
            item("1", "a"),
            item("2", "a"),
            item("3", "a"),
            item("4", "a"),
            item("5", "a"),
            item("6", "b"),
            item("7", "b"),
        ];
        let b = curate(items, 4, 2);
        let picked = &b.sections[0].items;
        assert_eq!(picked.len(), 4);
        assert_eq!(picked.iter().filter(|i| i.source_id == "a").count(), 2);
        assert_eq!(picked.iter().filter(|i| i.source_id == "b").count(), 2);
    }

    #[test]
    fn curate_fills_overflow_when_not_diverse() {
        // Nur eine Quelle, cap=2, top_n=4 → Overflow füllt auf 4 auf, statt zu kürzen.
        let items = vec![
            item("1", "a"),
            item("2", "a"),
            item("3", "a"),
            item("4", "a"),
        ];
        let b = curate(items, 4, 2);
        assert_eq!(b.sections[0].items.len(), 4);
    }

    mod integration {
        use super::*;
        use async_trait::async_trait;
        use ibrief_llm::Completion;
        use std::sync::Mutex;

        /// Mock-Modell, das pro Aufruf die nächste Zeile eines Skripts liefert.
        struct ScriptedModel {
            label: String,
            lines: Mutex<Vec<String>>,
        }

        impl ScriptedModel {
            fn new(label: &str, mut responses: Vec<String>) -> Self {
                responses.reverse();
                Self {
                    label: label.into(),
                    lines: Mutex::new(responses),
                }
            }
        }

        #[async_trait]
        impl LanguageModel for ScriptedModel {
            async fn complete(&self, _req: &Completion) -> anyhow::Result<String> {
                self.lines
                    .lock()
                    .unwrap()
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("mock exhausted"))
            }
            fn label(&self) -> &str {
                &self.label
            }
        }

        #[tokio::test]
        async fn full_pipeline_enrich_curate_render() {
            let items: Vec<ContentItem> = (0..12)
                .map(|i| {
                    let src = if i < 6 { "verge" } else { "hn" };
                    ContentItem {
                        id: format!("item-{i}"),
                        source_id: src.into(),
                        title: format!("Artikel {i}"),
                        url: format!("https://{src}.com/{i}"),
                        published_at: None,
                        raw_summary: Some(format!("Roh-Text von Artikel {i}.")),
                        summary: None,
                        topics: vec![],
                        entities: vec![],
                        embedding: None,
                    }
                })
                .collect();

            // Mock: gibt für jedes Item eine Zusammenfassung + Topics zurück.
            let mut enrich_responses = Vec::new();
            for i in 0..12 {
                enrich_responses.push(format!(
                    r#"{{"summary":"Zusammenfassung von Artikel {i}.","topics":["thema{i}"]}}"#
                ));
            }
            let enrich_model = ScriptedModel::new("enrich-mock", enrich_responses);

            // Mock: TL;DR gibt 3 Bullet Points.
            let synth_model = ScriptedModel::new(
                "synth-mock",
                vec!["- Erste Sache\n- Zweite Sache\n- Dritte Sache".into()],
            );

            let counterpoint_model = ScriptedModel::new(
                "counterpoint-mock",
                vec!["Eine faire Gegenperspektive mit Substanz.".into()],
            );

            // ENRICH (parallel, batchweise)
            let enriched = enrich(items, &enrich_model, 12).await;
            assert_eq!(enriched.len(), 12);
            for (i, it) in enriched.iter().enumerate() {
                assert!(it.summary.is_some(), "Item {i} sollte eine Summary haben");
                assert!(!it.topics.is_empty(), "Item {i} sollte Topics haben");
            }

            // SCORE → CURATE
            let cfg = Config::default();
            let ranked = rank(enriched, &cfg);
            let mut briefing = curate(ranked, 8, 3);
            briefing.date = "2026-06-29".into();

            // TL;DR
            let tldr = make_tldr(
                &briefing,
                &synth_model,
                "Fasse diese Meldungen in 3 Kernpunkten zusammen:\n{items}",
            )
            .await
            .unwrap();
            assert_eq!(tldr.len(), 3);
            briefing.tldr = tldr;

            // COUNTERPOINT
            let cp = make_counterpoint(&briefing, &counterpoint_model, &briefing.date)
                .await
                .unwrap();
            assert!(cp.is_some(), "Gegenperspektive sollte erzeugt werden");
            briefing.sections.push(BriefingSection {
                id: "counterpoint".into(),
                title: "Gegenperspektive".into(),
                items: vec![cp.unwrap()],
            });

            // WILDCARD aus den restlichen Items
            let wildcard = pick_wildcard(
                &briefing.sections[0].items,
                briefing.sections[0].items.len().max(8),
            );
            if let Some(w) = wildcard {
                briefing.sections.push(BriefingSection {
                    id: "wildcard".into(),
                    title: "Wildcard".into(),
                    items: vec![w],
                });
            }

            // RENDER
            let md = render(&briefing);
            assert!(md.contains("# Morning Briefing — 2026-06-29"));
            assert!(md.contains("## Die 3 Dinge heute"));
            assert!(md.contains("Erste Sache"));
            assert!(md.contains("Zweite Sache"));
            assert!(md.contains("Dritte Sache"));
            assert!(md.contains("## KI & Tech — Highlights"));
            assert!(md.contains("Gegenperspektive"));
            let n_sections = briefing.sections.len();
            assert!(n_sections >= 2, "mindestens Haupt- und Gegenperspektive");

            // Keine Items ohne Summary in der Ausgabe
            for sec in &briefing.sections {
                for it in &sec.items {
                    assert!(
                        it.summary.is_some() || it.title == "Gegenperspektive",
                        "Item {} ({}) in Sektion {} hat keine Summary",
                        it.id,
                        it.title,
                        sec.id
                    );
                }
            }
        }

        #[tokio::test]
        async fn enrich_handles_partial_failures() {
            let items: Vec<ContentItem> = (0..5)
                .map(|i| ContentItem {
                    id: format!("fail-{i}"),
                    source_id: "s".into(),
                    title: format!("Titel {i}"),
                    url: format!("https://s.com/{i}"),
                    published_at: None,
                    raw_summary: Some("Text.".into()),
                    summary: None,
                    topics: vec![],
                    entities: vec![],
                    embedding: None,
                })
                .collect();

            // Nur 3 von 5 Responses — die letzten beiden schlagen fehl.
            let mut responses = Vec::new();
            for i in 0..3 {
                responses.push(format!(r#"{{"summary":"Summary {i}","topics":["t{i}"]}}"#));
            }
            let model = ScriptedModel::new("partial", responses);

            let enriched = enrich(items, &model, 5).await;
            assert_eq!(enriched.len(), 5);
            assert!(enriched[0].summary.is_some());
            assert!(enriched[1].summary.is_some());
            assert!(enriched[2].summary.is_some());
            assert!(enriched[3].summary.is_none());
            assert!(enriched[4].summary.is_none());
        }
    }
}
