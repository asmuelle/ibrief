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
/// Sequentiell gehalten (M1), um Ollama nicht zu überlasten.
pub async fn enrich(
    mut items: Vec<ContentItem>,
    model: &dyn LanguageModel,
    max: usize,
) -> Vec<ContentItem> {
    for item in items.iter_mut().take(max) {
        match enrich_one(item, model).await {
            Ok(out) => {
                item.summary = Some(out.summary);
                item.topics = out.topics;
            }
            Err(e) => tracing::warn!(title = %item.title, error = %e, "enrich fehlgeschlagen"),
        }
    }
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
        .temperature(0.3);
    let raw = model.complete(&req).await?;
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
        .complete(&Completion::new(prompt).temperature(0.4))
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
                .temperature(0.6),
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
}
