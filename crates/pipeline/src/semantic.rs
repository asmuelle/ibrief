//! Semantische Nähe (§T2.2): Cosine-Ähnlichkeit über Embeddings.
//!
//! Zwei Anwendungen in der Pipeline:
//! 1. **Cross-Source-Dedup** — dieselbe Story aus drei Feeds belegt sonst drei der
//!    wenigen Briefing-Plätze; URL-Dedup sieht das nicht.
//! 2. **Diversitäts-Kappe in der Kuration** — thematische Quasi-Doppel werden wie die
//!    Quellen-Kappe behandelt (Overflow statt harter Verlust).
//!
//! Items ohne Embedding (Embedder aus/ausgefallen) passieren alle Checks — das Feature
//! degradiert sauber zum bisherigen Verhalten.

use ibrief_core::ContentItem;

/// Ab dieser Cosine-Ähnlichkeit gelten zwei Items als DIESELBE Story (Dubletten-Kollaps).
/// nomic-embed: Quasi-Identisches liegt typisch >0.95, verwandte-aber-eigene Stories darunter.
pub const DEDUP_SIMILARITY: f64 = 0.92;
/// Ab dieser Ähnlichkeit gelten zwei Items in der Kuration als thematisches Doppel
/// (weicher als Dedup: erst Overflow, aufgefüllt wird nur bei Platzmangel).
pub const DIVERSITY_SIMILARITY: f64 = 0.85;
/// Nomic-Task-Präfix für symmetrische Ähnlichkeit (Empfehlung des Modellherstellers).
const EMBED_PREFIX: &str = "clustering: ";

/// Der Text, der für ein Item eingebettet wird: Titel + Roh-Zusammenfassung.
/// Bewusst VOR dem Enrich verfügbar — Dedup muss laufen, bevor Enrich-Slots verbraucht werden.
pub fn embed_text(it: &ContentItem) -> String {
    match &it.raw_summary {
        Some(raw) => format!("{EMBED_PREFIX}{}\n{raw}", it.title),
        None => format!("{EMBED_PREFIX}{}", it.title),
    }
}

/// Cosine-Ähnlichkeit zweier Vektoren. 0.0 bei Längen-Mismatch oder Null-Vektor.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Ähnlichkeit zweier Items. `None`, wenn mindestens ein Embedding fehlt.
pub fn item_similarity(a: &ContentItem, b: &ContentItem) -> Option<f64> {
    match (&a.embedding, &b.embedding) {
        (Some(ea), Some(eb)) => Some(cosine(ea, eb)),
        _ => None,
    }
}

/// True, wenn `it` semantisch zu nah an einem der `kept`-Items liegt.
pub fn too_similar(it: &ContentItem, kept: &[ContentItem], threshold: f64) -> bool {
    kept.iter()
        .any(|k| item_similarity(k, it).is_some_and(|s| s >= threshold))
}

/// Kollabiert Quasi-Dubletten in einer RANG-geordneten Liste: das frühere (besser
/// gerankte) Item gewinnt, spätere Wiederholungen derselben Story fallen weg.
pub fn collapse_near_duplicates(items: Vec<ContentItem>, threshold: f64) -> Vec<ContentItem> {
    let mut kept: Vec<ContentItem> = Vec::with_capacity(items.len());
    let mut dropped = 0usize;
    for it in items {
        if too_similar(&it, &kept, threshold) {
            tracing::debug!(title = %it.title, source = %it.source_id, "semantische Dublette — kollabiert");
            dropped += 1;
        } else {
            kept.push(it);
        }
    }
    if dropped > 0 {
        tracing::info!(dropped, threshold, "Cross-Source-Dubletten kollabiert");
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, emb: Option<Vec<f32>>) -> ContentItem {
        ContentItem {
            id: id.into(),
            source_id: "s".into(),
            title: id.into(),
            url: format!("https://x/{id}"),
            published_at: None,
            raw_summary: None,
            summary: None,
            topics: vec![],
            entities: vec![],
            embedding: emb,
        }
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
        assert_eq!(cosine(&[1.0], &[1.0, 0.0]), 0.0); // Längen-Mismatch
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0); // Null-Vektor
    }

    #[test]
    fn collapse_keeps_higher_ranked_of_near_duplicates() {
        // a und b sind quasi identisch; c ist orthogonal. a steht vorn (besser gerankt).
        let items = vec![
            item("a", Some(vec![1.0, 0.0, 0.0])),
            item("b", Some(vec![0.999, 0.04, 0.0])),
            item("c", Some(vec![0.0, 1.0, 0.0])),
        ];
        let out = collapse_near_duplicates(items, DEDUP_SIMILARITY);
        let ids: Vec<&str> = out.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "c"]);
    }

    #[test]
    fn items_without_embedding_always_pass() {
        let items = vec![
            item("a", Some(vec![1.0, 0.0])),
            item("b", None), // kein Embedding → kein Urteil → behalten
            item("c", None),
        ];
        let out = collapse_near_duplicates(items, DEDUP_SIMILARITY);
        assert_eq!(out.len(), 3);
    }
}
