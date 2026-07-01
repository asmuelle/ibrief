//! Quellen-Evolution (§6.3 C): Quellen bewerten, schwache aussortieren, Drift überwachen.
//!
//! Der **Drift-Wächter** (§9) ist hier zentral: sinkt die Quellen-Diversität zu stark,
//! wird Pruning ausgesetzt (Exploration erzwungen) — auch auf Kosten des kurzfristigen
//! Eval-Scores. Anti-Blase schlägt Engagement.

use anyhow::Result;
use ibrief_store::{FeedbackMeta, Store};
use std::collections::HashMap;

/// Mindestzahl aktiver Quellen (Diversitäts-Floor — nie darunter prunen).
pub const MIN_ACTIVE_SOURCES: usize = 3;
/// Quellen unter dieser Qualität sind Prune-Kandidaten.
pub const PRUNE_THRESHOLD: f64 = 0.25;
/// HHI-Schwelle: darüber gilt die Quellenverteilung als zu konzentriert.
pub const MAX_HHI: f64 = 0.5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Drift {
    Ok,
    ForceExploration,
}

#[derive(Debug, Clone)]
pub struct EvolveOutcome {
    pub quality_updates: usize,
    pub deactivated: Vec<String>,
    pub drift: Drift,
    pub note: String,
}

/// Qualitäts-Score je Quelle in [0,1]: Feedback-Verhältnis (0.7) + Selektions-Evidenz (0.3).
pub fn score_sources(
    feedback: &[FeedbackMeta],
    selections: &[(String, i64)],
) -> HashMap<String, f64> {
    let mut pos: HashMap<&str, f64> = HashMap::new();
    let mut neg: HashMap<&str, f64> = HashMap::new();
    for f in feedback {
        let (dp, dn) = match f.kind.as_str() {
            "up" | "more" => (1.0, 0.0),
            "open" => (0.3, 0.0),
            "down" | "less" => (0.0, 1.0),
            _ => (0.0, 0.0),
        };
        *pos.entry(f.source_id.as_str()).or_default() += dp;
        *neg.entry(f.source_id.as_str()).or_default() += dn;
    }
    let sel: HashMap<&str, i64> = selections.iter().map(|(s, n)| (s.as_str(), *n)).collect();

    let mut ids: Vec<&str> = pos
        .keys()
        .chain(neg.keys())
        .chain(sel.keys())
        .copied()
        .collect();
    ids.sort();
    ids.dedup();

    let mut out = HashMap::new();
    for id in ids {
        let p = pos.get(id).copied().unwrap_or(0.0);
        let n = neg.get(id).copied().unwrap_or(0.0);
        let ratio = if p + n == 0.0 { 0.5 } else { p / (p + n) };
        let s = sel.get(id).copied().unwrap_or(0) as f64;
        let selection_norm = s / (s + 3.0); // diminishing returns
        out.insert(id.to_string(), 0.7 * ratio + 0.3 * selection_norm);
    }
    out
}

/// Empfiehlt Deaktivierungen: schwächste zuerst, aber nie unter [`MIN_ACTIVE_SOURCES`].
pub fn prune_recommendations(
    scores: &HashMap<String, f64>,
    active: &[String],
    min_active: usize,
    threshold: f64,
) -> Vec<String> {
    let mut ranked: Vec<(&String, f64)> = active
        .iter()
        .map(|id| (id, scores.get(id).copied().unwrap_or(0.5)))
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut remaining = active.len();
    let mut deactivate = Vec::new();
    for (id, score) in ranked {
        if remaining <= min_active {
            break;
        }
        if score < threshold {
            deactivate.push(id.clone());
            remaining -= 1;
        }
    }
    deactivate
}

/// Drift-Wächter: HHI (Herfindahl) der Quellen-Anteile. Zu konzentriert ⇒ Exploration.
pub fn drift_status(shares: &[f64], max_hhi: f64) -> Drift {
    let hhi: f64 = shares.iter().map(|s| s * s).sum();
    if hhi > max_hhi {
        Drift::ForceExploration
    } else {
        Drift::Ok
    }
}

/// Ein Evolutions-Schritt: Qualität aktualisieren, prunen (außer bei Drift), Status melden.
pub async fn evolve_once(store: &Store) -> Result<EvolveOutcome> {
    let feedback = store.feedback_join_meta().await?;
    let selections = store.selection_counts().await?;
    let active: Vec<String> = store
        .active_sources()
        .await?
        .into_iter()
        .map(|s| s.id)
        .collect();

    let scores = score_sources(&feedback, &selections);
    for (id, q) in &scores {
        store.set_source_quality(id, *q).await?;
    }

    // Drift aus den tatsächlichen Selektions-Anteilen.
    let total: i64 = selections.iter().map(|(_, n)| *n).sum();
    let shares: Vec<f64> = if total > 0 {
        selections
            .iter()
            .map(|(_, n)| *n as f64 / total as f64)
            .collect()
    } else {
        vec![]
    };
    let drift = drift_status(&shares, MAX_HHI);

    let mut deactivated = Vec::new();
    let note = if drift == Drift::ForceExploration {
        "Drift erkannt → Pruning ausgesetzt, Breite erhalten (Anti-Blase)".to_string()
    } else {
        for id in prune_recommendations(&scores, &active, MIN_ACTIVE_SOURCES, PRUNE_THRESHOLD) {
            store.set_source_active(&id, false).await?;
            deactivated.push(id);
        }
        format!("{} Quelle(n) deaktiviert", deactivated.len())
    };

    Ok(EvolveOutcome {
        quality_updates: scores.len(),
        deactivated,
        drift,
        note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fb(source: &str, kind: &str) -> FeedbackMeta {
        FeedbackMeta {
            source_id: source.into(),
            topics: vec![],
            kind: kind.into(),
            created_at: "2026-07-01T12:00:00Z".into(),
        }
    }

    #[test]
    fn scores_reward_positive_feedback() {
        let feedback = vec![fb("verge", "up"), fb("verge", "up"), fb("hn", "down")];
        let selections = vec![("verge".to_string(), 10), ("hn".to_string(), 1)];
        let scores = score_sources(&feedback, &selections);
        assert!(scores["verge"] > scores["hn"]);
    }

    #[test]
    fn prune_respects_min_active() {
        let mut scores = HashMap::new();
        scores.insert("a".to_string(), 0.1);
        scores.insert("b".to_string(), 0.1);
        scores.insert("c".to_string(), 0.1);
        scores.insert("d".to_string(), 0.1);
        let active = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let deact = prune_recommendations(&scores, &active, 3, 0.25);
        assert_eq!(deact.len(), 1); // nur eine darf weg, min_active=3
    }

    #[test]
    fn drift_detects_concentration() {
        assert_eq!(
            drift_status(&[0.9, 0.05, 0.05], MAX_HHI),
            Drift::ForceExploration
        );
        assert_eq!(drift_status(&[0.34, 0.33, 0.33], MAX_HHI), Drift::Ok);
    }
}
