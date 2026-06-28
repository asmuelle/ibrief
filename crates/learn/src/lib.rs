//! Lern-Hebel (§6.3 A): Gewichte via Multi-Armed-Bandit (Thompson-Sampling),
//! abgesichert durch das Safety Gate (§6.4) und versioniert im Config-Store (§8).
//!
//! Ablauf von [`learn_once`]:
//!   aktive Config laden → Feedback je Quelle/Thema aggregieren → Kandidat samplen
//!   → Safety Gate → bei PASS: neue Version speichern & aktivieren, sonst verwerfen.
//!
//! Exploration ist eingebaut (Beta-Sampling) und ein Gewichts-Floor verhindert,
//! dass Arme aussterben — das ist Anti-Blase auf Mechanismus-Ebene.

use anyhow::Result;
use ibrief_core::Config;
use ibrief_store::{FeedbackMeta, Store};
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand_distr::Beta;
use std::collections::HashMap;

/// Gewichts-Untergrenze (Exploration-Floor: kein Arm stirbt → Anti-Blase).
pub const MIN_WEIGHT: f64 = 0.2;
/// Gewichts-Obergrenze (verhindert Monokultur einer Quelle/eines Themas).
pub const MAX_WEIGHT: f64 = 2.0;
/// Maximaler Anteil einer einzelnen Quelle am Gewichtsbudget (Diversitäts-Gate).
const MAX_SOURCE_SHARE: f64 = 0.5;

/// Positive/negative Evidenz eines Bandit-Arms (Quelle oder Thema).
#[derive(Debug, Default, Clone, Copy)]
pub struct ArmStats {
    pub pos: f64,
    pub neg: f64,
}

/// Ergebnis des Safety Gate.
#[derive(Debug, Clone)]
pub struct GateResult {
    pub passed: bool,
    pub reasons: Vec<String>,
}

/// Ergebnis eines Lern-Laufs.
#[derive(Debug, Clone)]
pub struct LearnOutcome {
    pub adopted: bool,
    pub version: String,
    pub parent: Option<String>,
    pub reason: String,
    pub gate_reasons: Vec<String>,
    pub n_feedback: usize,
}

/// Lädt die aktive Config (neutraler Default, falls noch nie gelernt).
pub async fn load_active(store: &Store) -> Result<Config> {
    let Some(version) = store.active_config_version().await? else {
        return Ok(Config::default());
    };
    match store.load_config_payload(&version).await? {
        Some(payload) => Ok(serde_json::from_str(&payload)?),
        None => Ok(Config::default()),
    }
}

/// Ein kompletter Lern-Schritt. `seed` macht das Sampling reproduzierbar.
pub async fn learn_once(store: &Store, seed: u64) -> Result<LearnOutcome> {
    let active = load_active(store).await?;
    let rows = store.feedback_join_meta().await?;
    let (sources, topics) = aggregate(&rows);

    let mut rng = StdRng::seed_from_u64(seed);
    let candidate = propose(&active, &sources, &topics, &mut rng);
    let gate = gate(&candidate);
    let version = config_version(&candidate);
    let parent = store.active_config_version().await?;

    if !gate.passed {
        return Ok(LearnOutcome {
            adopted: false,
            version,
            parent,
            reason: "Safety Gate fehlgeschlagen — Kandidat verworfen".into(),
            gate_reasons: gate.reasons,
            n_feedback: rows.len(),
        });
    }

    let payload = serde_json::to_string(&candidate)?;
    let reason = format!("Thompson-Update aus {} Feedback-Ereignissen", rows.len());
    store
        .save_config(&version, parent.as_deref(), &reason, &payload)
        .await?;
    store.set_active_config(&version).await?;

    Ok(LearnOutcome {
        adopted: true,
        version,
        parent,
        reason,
        gate_reasons: gate.reasons,
        n_feedback: rows.len(),
    })
}

/// Setzt die aktive Config auf eine frühere Version zurück (§6.4 Rollback).
pub async fn rollback(store: &Store, version: &str) -> Result<()> {
    if !store.config_exists(version).await? {
        anyhow::bail!("Config-Version '{version}' existiert nicht");
    }
    store.set_active_config(version).await?;
    Ok(())
}

/// Aggregiert Feedback je Quelle und je Thema zu Beta-Evidenz.
pub fn aggregate(rows: &[FeedbackMeta]) -> (HashMap<String, ArmStats>, HashMap<String, ArmStats>) {
    let mut sources: HashMap<String, ArmStats> = HashMap::new();
    let mut topics: HashMap<String, ArmStats> = HashMap::new();

    for r in rows {
        let (dp, dn) = match r.kind.as_str() {
            "up" | "more" => (1.0, 0.0),
            "open" => (0.3, 0.0),
            "down" | "less" => (0.0, 1.0),
            _ => (0.0, 0.0),
        };
        let s = sources.entry(r.source_id.clone()).or_default();
        s.pos += dp;
        s.neg += dn;
        for t in &r.topics {
            let e = topics.entry(t.clone()).or_default();
            e.pos += dp;
            e.neg += dn;
        }
    }
    (sources, topics)
}

/// Schlägt eine neue Config vor: je Arm ein Beta-Sample (Exploration + Exploitation).
pub fn propose(
    active: &Config,
    sources: &HashMap<String, ArmStats>,
    topics: &HashMap<String, ArmStats>,
    rng: &mut impl rand::Rng,
) -> Config {
    let mut source_weights = active.source_weights.clone();
    for (id, st) in sources {
        source_weights.insert(id.clone(), sample_weight(st, rng));
    }
    let mut topic_weights = active.topic_weights.clone();
    for (t, st) in topics {
        topic_weights.insert(t.clone(), sample_weight(st, rng));
    }
    Config {
        source_weights,
        topic_weights,
    }
}

/// Beta(1+pos, 1+neg)-Sample, skaliert auf neutral≈1.0 und auf [MIN,MAX] geklemmt.
fn sample_weight(st: &ArmStats, rng: &mut impl rand::Rng) -> f64 {
    let beta = Beta::new(1.0 + st.pos, 1.0 + st.neg).expect("alpha,beta > 0");
    let theta = beta.sample(rng); // 0..1, Mittel 0.5 bei keiner Evidenz
    (theta * 2.0).clamp(MIN_WEIGHT, MAX_WEIGHT)
}

/// Safety Gate (§6.4): deterministische Invarianten-Prüfung des Kandidaten.
pub fn gate(c: &Config) -> GateResult {
    let mut reasons = Vec::new();
    let mut passed = true;

    let mut check_bounds = |label: &str, w: f64| {
        if !(MIN_WEIGHT - 1e-9..=MAX_WEIGHT + 1e-9).contains(&w) {
            passed = false;
            reasons.push(format!(
                "{label}: Gewicht {w:.2} außerhalb [{MIN_WEIGHT}, {MAX_WEIGHT}]"
            ));
        }
    };
    for (id, w) in &c.source_weights {
        check_bounds(&format!("source:{id}"), *w);
    }
    for (t, w) in &c.topic_weights {
        check_bounds(&format!("topic:{t}"), *w);
    }

    // Diversität: keine Quelle darf >50 % des Quellen-Gewichtsbudgets binden (ab ≥3 Quellen).
    if c.source_weights.len() >= 3 {
        let sum: f64 = c.source_weights.values().sum();
        if sum > 0.0 {
            let max = c.source_weights.values().cloned().fold(0.0_f64, f64::max);
            if max / sum > MAX_SOURCE_SHARE {
                passed = false;
                reasons.push(format!(
                    "Quellen-Monokultur: Top-Quelle hält {:.0} % des Gewichts (>{:.0} %)",
                    max / sum * 100.0,
                    MAX_SOURCE_SHARE * 100.0
                ));
            }
        }
    }

    if passed {
        reasons.push("alle Invarianten erfüllt".into());
    }
    GateResult { passed, reasons }
}

/// Entscheidung eines temporalen A/B-Tests (§6.5) — die Verschärfung des Safety Gate:
/// eine Variante wird nur übernommen, wenn sie die Kontrolle über genügend Tage
/// im Eval-Score schlägt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbDecision {
    Promote,
    Reject,
    KeepRunning,
}

/// Vergleicht Eval-Scores zweier Varianten. `min_samples` Tage pro Arm nötig,
/// `margin` ist der geforderte Mindestabstand der Mittelwerte.
pub fn ab_decision(
    control: &[f64],
    candidate: &[f64],
    min_samples: usize,
    margin: f64,
) -> AbDecision {
    if control.len() < min_samples || candidate.len() < min_samples {
        return AbDecision::KeepRunning;
    }
    let mc = mean(control);
    let mk = mean(candidate);
    if mk >= mc + margin {
        AbDecision::Promote
    } else if mk <= mc - margin {
        AbDecision::Reject
    } else {
        AbDecision::KeepRunning
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Stabile, inhaltsbasierte Versions-ID (FNV-1a über kanonisches JSON).
pub fn config_version(c: &Config) -> String {
    let json = serde_json::to_string(c).unwrap_or_default();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in json.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("cfg-{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ibrief_core::Config;
    use std::collections::BTreeMap;

    fn fb(source: &str, topic: &str, kind: &str) -> FeedbackMeta {
        FeedbackMeta {
            source_id: source.into(),
            topics: vec![topic.into()],
            kind: kind.into(),
        }
    }

    #[test]
    fn aggregate_counts_pos_neg() {
        let rows = vec![
            fb("verge", "ai", "up"),
            fb("verge", "ai", "up"),
            fb("hn", "rust", "down"),
        ];
        let (sources, topics) = aggregate(&rows);
        assert_eq!(sources["verge"].pos, 2.0);
        assert_eq!(sources["hn"].neg, 1.0);
        assert_eq!(topics["ai"].pos, 2.0);
    }

    #[test]
    fn proposal_stays_within_bounds_and_is_deterministic() {
        let mut sources = HashMap::new();
        sources.insert("verge".to_string(), ArmStats { pos: 5.0, neg: 0.0 });
        sources.insert("hn".to_string(), ArmStats { pos: 0.0, neg: 5.0 });
        let topics = HashMap::new();

        let mut rng1 = StdRng::seed_from_u64(7);
        let mut rng2 = StdRng::seed_from_u64(7);
        let a = propose(&Config::default(), &sources, &topics, &mut rng1);
        let b = propose(&Config::default(), &sources, &topics, &mut rng2);

        assert_eq!(config_version(&a), config_version(&b)); // reproduzierbar
        for w in a.source_weights.values() {
            assert!(*w >= MIN_WEIGHT && *w <= MAX_WEIGHT);
        }
    }

    #[test]
    fn gate_rejects_source_monoculture() {
        let mut source_weights = BTreeMap::new();
        source_weights.insert("verge".into(), 2.0);
        source_weights.insert("hn".into(), 0.2);
        source_weights.insert("ars".into(), 0.2);
        let c = Config {
            source_weights,
            topic_weights: BTreeMap::new(),
        };
        let g = gate(&c);
        assert!(!g.passed);
        assert!(g.reasons.iter().any(|r| r.contains("Monokultur")));
    }

    #[test]
    fn gate_accepts_balanced() {
        let mut source_weights = BTreeMap::new();
        source_weights.insert("verge".into(), 1.0);
        source_weights.insert("hn".into(), 1.1);
        source_weights.insert("ars".into(), 0.9);
        let c = Config {
            source_weights,
            topic_weights: BTreeMap::new(),
        };
        assert!(gate(&c).passed);
    }

    #[test]
    fn ab_keeps_running_until_enough_samples() {
        assert_eq!(
            ab_decision(&[0.6], &[0.8], 3, 0.05),
            AbDecision::KeepRunning
        );
    }

    #[test]
    fn ab_promotes_clear_winner_and_rejects_loser() {
        let control = [0.50, 0.52, 0.48];
        let winner = [0.70, 0.72, 0.71];
        let loser = [0.30, 0.31, 0.29];
        assert_eq!(ab_decision(&control, &winner, 3, 0.05), AbDecision::Promote);
        assert_eq!(ab_decision(&control, &loser, 3, 0.05), AbDecision::Reject);
        assert_eq!(
            ab_decision(&control, &[0.50, 0.51, 0.49], 3, 0.05),
            AbDecision::KeepRunning
        );
    }
}
