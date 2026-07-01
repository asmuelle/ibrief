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
use chrono::{DateTime, Utc};
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
/// Halbwertszeit des Feedback-Zeitverfalls (§T2.5): ein 👍 von vor einem Monat zählt halb,
/// von vor zwei Monaten ein Viertel. Präferenzen dürfen sich ändern — alte Evidenz soll
/// neue nicht dauerhaft überstimmen.
pub const FEEDBACK_HALF_LIFE_DAYS: f64 = 30.0;

/// Zeitverfall eines Feedback-Ereignisses relativ zu `now`: 0.5^(Alter/Halbwertszeit).
/// Unparsebarer/fehlender Zeitstempel zählt voll (fail-open — Alt-Zeilen ohne saubere
/// Timestamps verlieren keine Stimme).
fn decay(created_at: &str, now: DateTime<Utc>) -> f64 {
    let Ok(t) = DateTime::parse_from_rfc3339(created_at) else {
        return 1.0;
    };
    let age_days = (now - t.with_timezone(&Utc)).num_seconds().max(0) as f64 / 86_400.0;
    0.5_f64.powf(age_days / FEEDBACK_HALF_LIFE_DAYS)
}

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
    /// Präferenz-AUC der AKTIVEN Config gegen reales Feedback (§T1.1). `None` = zu
    /// dünnes/einseitiges Feedback für ein Urteil (dann nur Invarianten geprüft).
    pub eval_active: Option<f64>,
    /// Präferenz-AUC der KANDIDATEN-Config gegen dasselbe Feedback.
    pub eval_candidate: Option<f64>,
}

/// Toleranz des Eval-Gates: der Kandidat wird nur verworfen, wenn er die Präferenz-AUC
/// echt (jenseits FP-Rauschen) verschlechtert. Gleichstand/Verbesserung → erlaubt.
const EVAL_TOLERANCE: f64 = 1e-9;

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
///
/// Ablauf: Kandidat samplen → (1) strukturelle Invarianten (Safety Gate) →
/// (2) Eval-Gate: der Kandidat darf die aus REALEM Feedback abgeleitete Präferenz-AUC
/// nicht verschlechtern (§6.4.1/§T1.1). Erst wenn beide Gates passen, wird übernommen.
pub async fn learn_once(store: &Store, seed: u64) -> Result<LearnOutcome> {
    let now = Utc::now();
    let active = load_active(store).await?;
    let rows = store.feedback_join_meta().await?;
    let (sources, topics) = aggregate(&rows, now);

    let mut rng = StdRng::seed_from_u64(seed);
    let candidate = propose(&active, &sources, &topics, &mut rng);
    let gate = gate(&candidate);
    let version = config_version(&candidate);
    let parent = store.active_config_version().await?;

    // Präferenz-Übereinstimmung beider Configs gegen dasselbe Feedback (in-sample; bei
    // Single-User gibt es keinen Holdout — fängt aber Explorationsproben, die der
    // gezeigten Präferenz widersprechen, statt sie blind zu übernehmen).
    let eval_active = preference_auc(&active, &rows, now);
    let eval_candidate = preference_auc(&candidate, &rows, now);

    let outcome = |adopted: bool, reason: String, gate_reasons: Vec<String>| LearnOutcome {
        adopted,
        version: version.clone(),
        parent: parent.clone(),
        reason,
        gate_reasons,
        n_feedback: rows.len(),
        eval_active,
        eval_candidate,
    };

    // (1) Strukturelle Invarianten.
    if !gate.passed {
        return Ok(outcome(
            false,
            "Safety Gate fehlgeschlagen — Kandidat verworfen".into(),
            gate.reasons,
        ));
    }

    // (2) Eval-Gate: keine Verschlechterung der Präferenz-AUC (falls beurteilbar).
    if let (Some(base), Some(cand)) = (eval_active, eval_candidate)
        && cand + EVAL_TOLERANCE < base
    {
        return Ok(outcome(
            false,
            format!("Eval-Gate: Präferenz-AUC verschlechtert ({base:.3} → {cand:.3}) — verworfen"),
            gate.reasons,
        ));
    }

    // Gleichstand explizit als Explorations-Schritt ausweisen (§T2.5): die Übernahme
    // stützt sich dann NICHT auf Evidenz — das soll in `config list` sichtbar sein.
    let eval_note = match (eval_active, eval_candidate) {
        (Some(b), Some(c)) if (c - b).abs() <= EVAL_TOLERANCE => {
            format!("Präferenz-AUC unverändert ({b:.3}) — Übernahme ist Explorations-Schritt")
        }
        (Some(b), Some(c)) => format!("Präferenz-AUC {b:.3} → {c:.3}"),
        _ => "Eval inconclusive (einseitiges/dünnes Feedback) — nur Invarianten".into(),
    };
    let reason = format!(
        "Thompson-Update aus {} Feedback-Ereignissen · {eval_note}",
        rows.len()
    );
    let payload = serde_json::to_string(&candidate)?;
    store
        .save_config(&version, parent.as_deref(), &reason, &payload)
        .await?;
    store.set_active_config(&version).await?;

    Ok(outcome(true, reason, gate.reasons))
}

/// Reward eines Feedback-Ereignisses (konsistent mit [`aggregate`]): explizit positiv/negativ
/// dominiert, Öffnen zählt schwach positiv.
fn event_reward(kind: &str) -> f64 {
    match kind {
        "up" | "more" => 1.0,
        "open" => 0.3,
        "down" | "less" => -1.0,
        _ => 0.0,
    }
}

/// Ranking-Score, den eine Config einem Item (Quelle + Themen) gibt — der lernbare Teil von
/// `pipeline::rank` (ohne die config-unabhängige Aktualität).
fn config_score(cfg: &Config, source_id: &str, topics: &[String]) -> f64 {
    let source_w = cfg.source_weight(source_id);
    let topic_w = if topics.is_empty() {
        1.0
    } else {
        topics.iter().map(|t| cfg.topic_weight(t)).sum::<f64>() / topics.len() as f64
    };
    source_w * topic_w
}

/// Präferenz-AUC (§T1.1, gewichtet §T2.5): Wahrscheinlichkeit, dass die Config ein positiv
/// bewertetes Ereignis höher rankt als ein negativ bewertetes — die Ground-Truth-Metrik für
/// den Eval-Gate. Jedes Paar zählt mit `|reward| × Zeitverfall` beider Ereignisse: ein
/// frisches 👎 wiegt schwerer als ein altes „geöffnet". `None`, wenn Positiva ODER Negativa
/// fehlen (kein diskriminierendes Signal).
pub fn preference_auc(cfg: &Config, rows: &[FeedbackMeta], now: DateTime<Utc>) -> Option<f64> {
    // (Score der Config, Gewicht des Ereignisses)
    let mut pos: Vec<(f64, f64)> = Vec::new();
    let mut neg: Vec<(f64, f64)> = Vec::new();
    for r in rows {
        let reward = event_reward(&r.kind);
        if reward == 0.0 {
            continue;
        }
        let weight = reward.abs() * decay(&r.created_at, now);
        let score = config_score(cfg, &r.source_id, &r.topics);
        if reward > 0.0 {
            pos.push((score, weight));
        } else {
            neg.push((score, weight));
        }
    }
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut agree = 0.0;
    let mut total = 0.0;
    for &(p, wp) in &pos {
        for &(n, wn) in &neg {
            let w = wp * wn;
            total += w;
            if p > n + EVAL_TOLERANCE {
                agree += w;
            } else if (p - n).abs() <= EVAL_TOLERANCE {
                agree += 0.5 * w;
            }
        }
    }
    if total <= 0.0 {
        return None;
    }
    Some(agree / total)
}

/// Setzt die aktive Config auf eine frühere Version zurück (§6.4 Rollback).
pub async fn rollback(store: &Store, version: &str) -> Result<()> {
    if !store.config_exists(version).await? {
        anyhow::bail!("Config-Version '{version}' existiert nicht");
    }
    store.set_active_config(version).await?;
    Ok(())
}

/// Mindest-Eval-Tage je Arm, bevor der äußere Loop urteilt (§T2.6).
pub const OUTER_MIN_SAMPLES: usize = 3;
/// Geforderter Mindestabstand der Eval-Mittelwerte für Demotion/Bestätigung.
pub const OUTER_MARGIN: f64 = 0.05;
/// Wie viele jüngste Eval-Noten je Config in den Vergleich eingehen.
const OUTER_EVAL_WINDOW: i64 = 30;

/// Ergebnis des äußeren Eval-Loops.
#[derive(Debug, Clone)]
pub struct OuterLoopOutcome {
    pub decision: AbDecision,
    pub active: String,
    pub parent: Option<String>,
    /// True, wenn die aktive Config demotet und der Parent reaktiviert wurde.
    pub rolled_back: bool,
    pub note: String,
}

/// Äußerer A/B-Loop (§6.5/§T2.6): vergleicht die **täglichen Eval-Noten** der aktiven
/// Config mit denen ihres Parents ([`ab_decision`]). Das ist die langsame Verschärfung des
/// schnellen Präferenz-Gates in [`learn_once`]: eine Config, die das Gate passierte, aber
/// über Tage messbar schlechter briefed als ihr Vorgänger, wird demotet (Rollback auf den
/// Parent, als Experiment protokolliert). `Promote` bestätigt nur; `KeepRunning` sammelt.
pub async fn outer_loop_check(store: &Store) -> Result<OuterLoopOutcome> {
    let keep = |active: String, parent: Option<String>, note: String| OuterLoopOutcome {
        decision: AbDecision::KeepRunning,
        active,
        parent,
        rolled_back: false,
        note,
    };

    let Some(active) = store.active_config_version().await? else {
        return Ok(keep(
            "default".into(),
            None,
            "keine gelernte Config aktiv — nichts zu prüfen".into(),
        ));
    };
    let Some(parent) = store.config_parent(&active).await? else {
        return Ok(keep(
            active,
            None,
            "aktive Config ist Wurzel (kein Parent) — kein Vergleichsarm".into(),
        ));
    };

    let active_evals = store.evals_for_config(&active, OUTER_EVAL_WINDOW).await?;
    let parent_evals = store.evals_for_config(&parent, OUTER_EVAL_WINDOW).await?;
    let decision = ab_decision(
        &parent_evals,
        &active_evals,
        OUTER_MIN_SAMPLES,
        OUTER_MARGIN,
    );

    let (rolled_back, note) = match decision {
        AbDecision::Reject => {
            // Demotion: Parent reaktivieren + als Experiment protokollieren.
            rollback(store, &parent).await?;
            store
                .save_experiment(&ibrief_store::ExperimentRow {
                    id: format!("outer-{active}-n{}", active_evals.len()),
                    kind: "config".into(),
                    slot: "weights".into(),
                    control: parent.clone(),
                    candidate: active.clone(),
                    status: "demoted".into(),
                    created_at: String::new(), // vom Store gesetzt
                })
                .await?;
            (
                true,
                format!(
                    "Demotion: {active} evaluiert über {} Tage schlechter als Parent {parent} \
(Marge {OUTER_MARGIN}) — Rollback",
                    active_evals.len()
                ),
            )
        }
        AbDecision::Promote => (
            false,
            format!(
                "aktive Config {active} über {} Tage besser als Parent — bestätigt",
                active_evals.len()
            ),
        ),
        AbDecision::KeepRunning => (
            false,
            format!(
                "zu wenig Evidenz (aktiv {} / Parent {} Eval-Tage, min {OUTER_MIN_SAMPLES}) — \
weiter beobachten",
                active_evals.len(),
                parent_evals.len()
            ),
        ),
    };

    Ok(OuterLoopOutcome {
        decision,
        active,
        parent: Some(parent),
        rolled_back,
        note,
    })
}

/// Aggregiert Feedback je Quelle und je Thema zu Beta-Evidenz — zeitverfallen (§T2.5):
/// die Evidenz eines Ereignisses schrumpft mit 0.5^(Alter/[`FEEDBACK_HALF_LIFE_DAYS`]).
pub fn aggregate(
    rows: &[FeedbackMeta],
    now: DateTime<Utc>,
) -> (HashMap<String, ArmStats>, HashMap<String, ArmStats>) {
    let mut sources: HashMap<String, ArmStats> = HashMap::new();
    let mut topics: HashMap<String, ArmStats> = HashMap::new();

    for r in rows {
        let (dp, dn) = match r.kind.as_str() {
            "up" | "more" => (1.0, 0.0),
            "open" => (0.3, 0.0),
            "down" | "less" => (0.0, 1.0),
            _ => (0.0, 0.0),
        };
        let d = decay(&r.created_at, now);
        let (dp, dn) = (dp * d, dn * d);
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

    /// Fixe „Jetzt"-Zeit für deterministische Decay-Tests.
    fn test_now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-07-01T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn fb(source: &str, topic: &str, kind: &str) -> FeedbackMeta {
        fb_at(source, topic, kind, "2026-07-01T12:00:00Z")
    }

    fn fb_at(source: &str, topic: &str, kind: &str, created_at: &str) -> FeedbackMeta {
        FeedbackMeta {
            source_id: source.into(),
            topics: vec![topic.into()],
            kind: kind.into(),
            created_at: created_at.into(),
        }
    }

    fn cfg_sources(pairs: &[(&str, f64)]) -> Config {
        Config {
            source_weights: pairs.iter().map(|(s, w)| (s.to_string(), *w)).collect(),
            topic_weights: BTreeMap::new(),
        }
    }

    #[test]
    fn preference_auc_rewards_alignment() {
        // Owner mag verge (👍), nicht hn (👎).
        let rows = vec![fb("verge", "ai", "up"), fb("hn", "rust", "down")];

        // Config, die verge hoch und hn niedrig gewichtet → perfekte Übereinstimmung.
        let aligned = cfg_sources(&[("verge", 2.0), ("hn", 0.2)]);
        assert_eq!(preference_auc(&aligned, &rows, test_now()), Some(1.0));

        // Neutral → alle Scores gleich → unentschieden.
        assert_eq!(
            preference_auc(&Config::default(), &rows, test_now()),
            Some(0.5)
        );

        // Invertiert → widerspricht der Präferenz → 0.0 (das Eval-Gate würde verwerfen).
        let inverted = cfg_sources(&[("verge", 0.2), ("hn", 2.0)]);
        assert_eq!(preference_auc(&inverted, &rows, test_now()), Some(0.0));
    }

    #[test]
    fn preference_auc_none_without_both_signs() {
        // Nur positives Feedback → kein diskriminierendes Paar → nicht beurteilbar.
        let rows = vec![fb("verge", "ai", "up"), fb("hn", "rust", "up")];
        assert_eq!(preference_auc(&Config::default(), &rows, test_now()), None);
    }

    #[test]
    fn aggregate_counts_pos_neg() {
        let rows = vec![
            fb("verge", "ai", "up"),
            fb("verge", "ai", "up"),
            fb("hn", "rust", "down"),
        ];
        let (sources, topics) = aggregate(&rows, test_now());
        assert_eq!(sources["verge"].pos, 2.0);
        assert_eq!(sources["hn"].neg, 1.0);
        assert_eq!(topics["ai"].pos, 2.0);
    }

    #[test]
    fn aggregate_decays_old_evidence() {
        // Ein 👍 von vor 30 Tagen (= Halbwertszeit) zählt halb, eines von vor 60 Tagen viertel.
        let rows = vec![
            fb_at("verge", "ai", "up", "2026-07-01T12:00:00Z"), // frisch → 1.0
            fb_at("verge", "ai", "up", "2026-06-01T12:00:00Z"), // 30 Tage → 0.5
            fb_at("verge", "ai", "up", "2026-05-02T12:00:00Z"), // 60 Tage → 0.25
        ];
        let (sources, _) = aggregate(&rows, test_now());
        assert!((sources["verge"].pos - 1.75).abs() < 1e-6);
    }

    #[test]
    fn aggregate_counts_unparseable_timestamp_fully() {
        // Fail-open: kaputter Zeitstempel ⇒ volles Gewicht statt stiller Entwertung.
        let rows = vec![fb_at("verge", "ai", "up", "kein-datum")];
        let (sources, _) = aggregate(&rows, test_now());
        assert_eq!(sources["verge"].pos, 1.0);
    }

    #[test]
    fn preference_auc_weights_recent_over_stale() {
        // Konflikt auf derselben Quelle: altes 👍 (60 Tage) vs. frisches 👎 auf "hn",
        // dazu ein frisches 👍 auf "verge". Eine Config, die hn HOCH gewichtet, gewinnt
        // nur das alte (leichte) Paar und verliert das frische (schwere) → AUC < 0.5.
        let rows = vec![
            fb_at("verge", "ai", "up", "2026-07-01T12:00:00Z"),
            fb_at("hn", "rust", "up", "2026-05-02T12:00:00Z"),
            fb_at("hn", "rust", "down", "2026-07-01T12:00:00Z"),
        ];
        let hn_heavy = cfg_sources(&[("verge", 0.2), ("hn", 2.0)]);
        let auc = preference_auc(&hn_heavy, &rows, test_now()).unwrap();
        assert!(auc < 0.5, "frisches Negativ muss dominieren, AUC war {auc}");

        // Die zur gezeigten (frischen) Präferenz passende Config schlägt sie klar.
        let aligned = cfg_sources(&[("verge", 2.0), ("hn", 0.2)]);
        let auc_aligned = preference_auc(&aligned, &rows, test_now()).unwrap();
        assert!(auc_aligned > 0.5);
    }

    #[test]
    fn preference_auc_weights_explicit_over_open() {
        // "open" (0.3) darf ein explizites 👍/👎-Paar nicht überstimmen: verge bekommt
        // ein schwaches open-Positiv UND ein volles Down; hn ein volles Up.
        let rows = vec![
            fb("verge", "ai", "open"),
            fb("verge", "ai", "down"),
            fb("hn", "rust", "up"),
        ];
        // Config bevorzugt verge → stimmt nur mit dem schwachen open-Paar überein.
        let verge_heavy = cfg_sources(&[("verge", 2.0), ("hn", 0.2)]);
        let auc = preference_auc(&verge_heavy, &rows, test_now()).unwrap();
        assert!(
            auc < 0.5,
            "explizites Feedback muss dominieren, AUC war {auc}"
        );
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

    async fn temp_store(tag: &str) -> (Store, String) {
        let path = std::env::temp_dir()
            .join(format!("ibrief-learn-{tag}-{}.db", std::process::id()))
            .to_string_lossy()
            .to_string();
        let _ = std::fs::remove_file(&path);
        (Store::open(&path).await.unwrap(), path)
    }

    async fn seed_eval(store: &Store, date: &str, version: &str, total: f64) {
        store
            .save_eval(&ibrief_store::EvalRow {
                date: date.into(),
                config_version: version.into(),
                rubric_version: "r1".into(),
                behavior: total,
                judge: total,
                structure: total,
                total,
                notes: vec![],
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn outer_loop_demotes_config_that_evals_worse_than_parent() {
        let (store, path) = temp_store("demote").await;
        store
            .save_config("cfg-parent", None, "seed", "{}")
            .await
            .unwrap();
        store
            .save_config("cfg-child", Some("cfg-parent"), "learn", "{}")
            .await
            .unwrap();
        store.set_active_config("cfg-child").await.unwrap();

        // 3 Eval-Tage je Arm: Parent klar besser als das aktive Kind.
        for (i, (p, c)) in [(0.8, 0.5), (0.82, 0.48), (0.79, 0.52)].iter().enumerate() {
            seed_eval(&store, &format!("2026-06-2{i}"), "cfg-parent", *p).await;
            seed_eval(&store, &format!("2026-06-2{}", i + 5), "cfg-child", *c).await;
        }

        let out = outer_loop_check(&store).await.unwrap();
        assert_eq!(out.decision, AbDecision::Reject);
        assert!(out.rolled_back);
        assert_eq!(
            store.active_config_version().await.unwrap().as_deref(),
            Some("cfg-parent"),
            "Demotion muss den Parent reaktivieren"
        );
        // Demotion ist als Experiment protokolliert.
        let exps = store.recent_experiments(5).await.unwrap();
        assert!(
            exps.iter()
                .any(|e| e.kind == "config" && e.status == "demoted")
        );

        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn outer_loop_waits_without_enough_eval_days() {
        let (store, path) = temp_store("wait").await;
        store
            .save_config("cfg-parent", None, "seed", "{}")
            .await
            .unwrap();
        store
            .save_config("cfg-child", Some("cfg-parent"), "learn", "{}")
            .await
            .unwrap();
        store.set_active_config("cfg-child").await.unwrap();
        seed_eval(&store, "2026-06-29", "cfg-parent", 0.8).await;
        seed_eval(&store, "2026-06-30", "cfg-child", 0.5).await;

        let out = outer_loop_check(&store).await.unwrap();
        assert_eq!(out.decision, AbDecision::KeepRunning);
        assert!(!out.rolled_back);
        assert_eq!(
            store.active_config_version().await.unwrap().as_deref(),
            Some("cfg-child"),
            "ohne genug Evidenz bleibt die aktive Config"
        );

        std::fs::remove_file(&path).ok();
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
