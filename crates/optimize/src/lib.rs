//! Prompt-Optimierung (§6.3 B / §6.5): ein Optimizer-LLM erzeugt eine Prompt-Variante,
//! die im **Schatten-Test** gegen den aktiven Prompt antritt — und nur bei klarem
//! Vorsprung Default wird. Kein Risiko fürs Live-Briefing.
//!
//! Urteils-Design (§T2.4): Der Judge vergleicht beide Ausgaben **paarweise in einem
//! Prompt** statt sie einzeln absolut zu benoten — absolute 0-1-Scores kleiner lokaler
//! Modelle sind deutlich verrauschter als Vergleiche. Jedes Paar wird **zweimal mit
//! getauschten Positionen** bewertet (Positions-Bias kürzt sich raus: nur wer beide
//! Reihenfolgen gewinnt, gewinnt das Paar) und über **mehrere jüngste Briefing-Tage**
//! wiederholt. Übernommen wird nur, wer mindestens einen Tag gewinnt und keinen verliert.
//!
//! Aktuell für den `tldr`-Slot. Weitere Slots (Sektions-Kuratierung) folgen demselben Muster.

use anyhow::Result;
use ibrief_llm::{Completion, LanguageModel};
use ibrief_store::{ExperimentRow, Store};
use serde::Deserialize;

/// Slot-Bezeichner für den TL;DR-Prompt.
pub const SLOT_TLDR: &str = "tldr";

/// Über wie viele jüngste Briefing-Tage der Schatten-Test läuft (sofern vorhanden).
const SHADOW_DATES: usize = 3;

/// Standard-Prompt für den TL;DR-Slot (Seed). `{items}` wird durch die Meldungen ersetzt.
pub const TLDR_DEFAULT: &str = "Hier sind die heutigen Meldungen:\n{items}\n\n\
Wähle die 3 wichtigsten aus. Antworte mit genau 3 kurzen deutschen Stichpunkten, \
je einer pro Zeile, ohne Nummerierung.";

const OPTIMIZER_SYSTEM: &str = "Du optimierst Prompts. Gib AUSSCHLIESSLICH den verbesserten Prompt-Text zurück, \
ohne Erklärung. Der Platzhalter {items} MUSS erhalten bleiben.";

const JUDGE_SYSTEM: &str = "Du bist ein strenger Gutachter. Antworte ausschließlich mit JSON.";

/// Aktiver Prompt eines Slots.
#[derive(Debug, Clone)]
pub struct ActivePrompt {
    pub version: String,
    pub template: String,
}

/// Urteil über EIN Paar (aktiv vs. Kandidat) nach beiden Positions-Reihenfolgen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairVerdict {
    /// Kandidat gewinnt BEIDE Reihenfolgen.
    Candidate,
    /// Aktiver Prompt gewinnt BEIDE Reihenfolgen.
    Active,
    /// Widersprüchlich oder unentschieden (z.B. reiner Positions-Bias des Judges).
    Tie,
}

/// Bilanz des Schatten-Tests über alle bewerteten Tage.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShadowTally {
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
}

impl ShadowTally {
    fn dates(&self) -> usize {
        self.wins + self.losses + self.ties
    }
}

/// Übernahme-Regel (§T2.4): mindestens ein Tages-Sieg, keine Niederlage. Konservativ —
/// bei widersprüchlichem Judge bleibt der aktive Prompt.
pub fn should_adopt(tally: &ShadowTally) -> bool {
    tally.wins > 0 && tally.losses == 0
}

/// Ergebnis eines Optimierungslaufs.
#[derive(Debug, Clone)]
pub struct OptimizeOutcome {
    pub adopted: bool,
    pub active_version: String,
    pub candidate_version: String,
    pub tally: Option<ShadowTally>,
    pub reason: String,
}

/// Lädt den aktiven TL;DR-Prompt; legt beim ersten Mal den Default an.
pub async fn active_tldr(store: &Store) -> Result<ActivePrompt> {
    if let Some((version, template)) = store.active_prompt(SLOT_TLDR).await? {
        return Ok(ActivePrompt { version, template });
    }
    let template = TLDR_DEFAULT.to_string();
    let version = version_of(&template);
    store
        .save_prompt(SLOT_TLDR, &version, None, "Seed-Default", &template)
        .await?;
    store.set_active_prompt(SLOT_TLDR, &version).await?;
    Ok(ActivePrompt { version, template })
}

/// Ein Optimierungs-Schritt für den TL;DR-Slot. `date = Some(..)` beschränkt den
/// Schatten-Test auf diesen Tag; `None` nutzt bis zu [`SHADOW_DATES`] jüngste Briefings.
pub async fn optimize_tldr(
    store: &Store,
    optimizer: &dyn LanguageModel,
    synth: &dyn LanguageModel,
    judge: &dyn LanguageModel,
    date: Option<&str>,
) -> Result<OptimizeOutcome> {
    let active = active_tldr(store).await?;

    let dates: Vec<String> = match date {
        Some(d) => vec![d.to_string()],
        None => store.recent_briefing_dates(SHADOW_DATES as i64).await?,
    };
    // Beispiel-Inputs je Tag; Tage ohne (auffindbares) Briefing fallen heraus.
    let mut inputs: Vec<(String, String)> = Vec::new();
    for d in &dates {
        if let Some(b) = store.load_briefing(d).await? {
            let text = items_to_text(&b);
            if !text.is_empty() {
                inputs.push((d.clone(), text));
            }
        }
    }
    if inputs.is_empty() {
        anyhow::bail!("kein Briefing als Schatten-Input (erst `ibrief brief`)");
    }

    let candidate_template = propose_prompt(&active.template, optimizer).await?;
    let candidate_version = version_of(&candidate_template);

    if candidate_version == active.version {
        return Ok(OptimizeOutcome {
            adopted: false,
            active_version: active.version,
            candidate_version,
            tally: None,
            reason: "Kandidat identisch zum aktiven Prompt".into(),
        });
    }

    let mut tally = ShadowTally::default();
    for (d, items_text) in &inputs {
        let verdict = shadow_compare(
            items_text,
            &active.template,
            &candidate_template,
            synth,
            judge,
        )
        .await?;
        tracing::info!(date = %d, ?verdict, "Schatten-Vergleich (pairwise, beide Reihenfolgen)");
        match verdict {
            PairVerdict::Candidate => tally.wins += 1,
            PairVerdict::Active => tally.losses += 1,
            PairVerdict::Tie => tally.ties += 1,
        }
    }
    let adopted = should_adopt(&tally);

    let date_ids: Vec<&str> = inputs.iter().map(|(d, _)| d.as_str()).collect();
    let experiment_id = version_of(&format!(
        "{}|{}|{}",
        active.version,
        candidate_version,
        date_ids.join(",")
    ));
    store
        .save_experiment(&ExperimentRow {
            id: experiment_id,
            kind: "prompt".into(),
            slot: SLOT_TLDR.into(),
            control: active.version.clone(),
            candidate: candidate_version.clone(),
            status: if adopted { "promoted" } else { "rejected" }.into(),
            created_at: String::new(), // vom Store gesetzt
        })
        .await?;

    let reason = format!(
        "Pairwise-Judge über {} Tag(e): Kandidat {} Sieg(e), {} Niederlage(n), {} unentschieden",
        tally.dates(),
        tally.wins,
        tally.losses,
        tally.ties
    );

    if adopted {
        store
            .save_prompt(
                SLOT_TLDR,
                &candidate_version,
                Some(&active.version),
                &reason,
                &candidate_template,
            )
            .await?;
        store
            .set_active_prompt(SLOT_TLDR, &candidate_version)
            .await?;
    }

    Ok(OptimizeOutcome {
        adopted,
        active_version: active.version,
        candidate_version,
        tally: Some(tally),
        reason,
    })
}

/// Optimizer-LLM erzeugt eine Prompt-Variante.
pub async fn propose_prompt(active: &str, optimizer: &dyn LanguageModel) -> Result<String> {
    let prompt = format!(
        "Aktueller Prompt:\n---\n{active}\n---\n\nVerbessere ihn, damit das TL;DR prägnanter \
und anschlussfähiger für Entscheidungen/Gespräche wird. Behalte den Platzhalter {{items}}. \
Gib nur den neuen Prompt zurück."
    );
    let req = Completion::new(prompt)
        .with_system(OPTIMIZER_SYSTEM)
        .temperature(0.7);
    Ok(optimizer.complete(&req).await?.trim().to_string())
}

/// Schatten-Vergleich für EINEN Input: beide Prompts rendern lassen, dann Pairwise-Urteil
/// in beiden Positions-Reihenfolgen.
pub async fn shadow_compare(
    items_text: &str,
    active_template: &str,
    candidate_template: &str,
    synth: &dyn LanguageModel,
    judge: &dyn LanguageModel,
) -> Result<PairVerdict> {
    let active_out = synth
        .complete(&Completion::new(fill(active_template, items_text)))
        .await?;
    let candidate_out = synth
        .complete(&Completion::new(fill(candidate_template, items_text)))
        .await?;
    judge_pair(items_text, &active_out, &candidate_out, judge).await
}

/// Rohes Einzel-Urteil des Judges: welche der beiden Varianten (A oder B) ist besser?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawWinner {
    A,
    B,
    Tie,
}

#[derive(Deserialize)]
struct JudgeOut {
    winner: String,
}

async fn judge_once(
    items_text: &str,
    a: &str,
    b: &str,
    judge: &dyn LanguageModel,
) -> Result<RawWinner> {
    let prompt = format!(
        "Zwei TL;DR-Varianten derselben Meldungen. Welche ist besser nach Prägnanz, \
Relevanz und Anschlussfähigkeit für Entscheidungen?\n\nMELDUNGEN:\n{items_text}\n\n\
VARIANTE A:\n{a}\n\nVARIANTE B:\n{b}\n\n\
Antworte NUR mit JSON {{\"winner\":\"A\"}} oder {{\"winner\":\"B\"}} oder {{\"winner\":\"tie\"}}."
    );
    let req = Completion::new(prompt)
        .with_system(JUDGE_SYSTEM)
        .temperature(0.1)
        .json();
    let raw = judge.complete(&req).await?;
    let out: JudgeOut = serde_json::from_str(&extract_json(&raw))?;
    Ok(match out.winner.trim().to_uppercase().as_str() {
        "A" => RawWinner::A,
        "B" => RawWinner::B,
        _ => RawWinner::Tie,
    })
}

/// Pairwise-Urteil mit Positions-Swap: Runde 1 (A=aktiv, B=Kandidat), Runde 2 getauscht.
/// Nur ein Sieg in BEIDEN Reihenfolgen zählt — ein Judge, der stur Position A wählt,
/// produziert so ein Tie statt eines falschen Signals.
pub async fn judge_pair(
    items_text: &str,
    active_out: &str,
    candidate_out: &str,
    judge: &dyn LanguageModel,
) -> Result<PairVerdict> {
    let round1 = judge_once(items_text, active_out, candidate_out, judge).await?;
    let round2 = judge_once(items_text, candidate_out, active_out, judge).await?;

    // In Runde 1 ist der Kandidat B, in Runde 2 ist er A.
    let candidate_r1 = round1 == RawWinner::B;
    let candidate_r2 = round2 == RawWinner::A;
    let active_r1 = round1 == RawWinner::A;
    let active_r2 = round2 == RawWinner::B;

    Ok(if candidate_r1 && candidate_r2 {
        PairVerdict::Candidate
    } else if active_r1 && active_r2 {
        PairVerdict::Active
    } else {
        PairVerdict::Tie
    })
}

fn fill(template: &str, items_text: &str) -> String {
    if template.contains("{items}") {
        template.replace("{items}", items_text)
    } else {
        format!("{template}\n\n{items_text}")
    }
}

fn items_to_text(b: &ibrief_core::Briefing) -> String {
    let mut t = String::new();
    for s in &b.sections {
        for it in &s.items {
            let line = it.summary.clone().unwrap_or_else(|| it.title.clone());
            t.push_str(&format!("- {line}\n"));
        }
    }
    t
}

fn extract_json(s: &str) -> String {
    match (s.find('{'), s.rfind('}')) {
        (Some(a), Some(b)) if b >= a => s[a..=b].to_string(),
        _ => s.to_string(),
    }
}

/// Inhaltsbasierte Prompt-Versions-ID (FNV-1a).
pub fn version_of(template: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in template.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("tpl-{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Synth: Ausgabe enthält "candidate", wenn der Prompt den Marker "BETTER" trägt.
    struct MarkerSynth;
    #[async_trait]
    impl LanguageModel for MarkerSynth {
        async fn complete(&self, req: &Completion) -> Result<String, ibrief_llm::ModelError> {
            if req.prompt.contains("BETTER") {
                Ok("candidate tldr".into())
            } else {
                Ok("active tldr".into())
            }
        }
        fn label(&self) -> &str {
            "synth"
        }
    }

    /// Judge: wählt die Variante, deren Text "candidate" enthält — positions-unabhängig.
    struct ContentJudge;
    #[async_trait]
    impl LanguageModel for ContentJudge {
        async fn complete(&self, req: &Completion) -> Result<String, ibrief_llm::ModelError> {
            let a = section(&req.prompt, "VARIANTE A:", "VARIANTE B:");
            let winner = if a.contains("candidate") { "A" } else { "B" };
            Ok(format!("{{\"winner\":\"{winner}\"}}"))
        }
        fn label(&self) -> &str {
            "judge"
        }
    }

    /// Judge mit reinem Positions-Bias: wählt IMMER Variante A.
    struct PositionBiasedJudge;
    #[async_trait]
    impl LanguageModel for PositionBiasedJudge {
        async fn complete(&self, _req: &Completion) -> Result<String, ibrief_llm::ModelError> {
            Ok(r#"{"winner":"A"}"#.into())
        }
        fn label(&self) -> &str {
            "biased-judge"
        }
    }

    fn section<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
        let s = text.find(start).map(|i| i + start.len()).unwrap_or(0);
        let e = text.find(end).unwrap_or(text.len());
        &text[s..e.max(s)]
    }

    #[tokio::test]
    async fn shadow_prefers_better_candidate_in_both_orders() {
        let verdict = shadow_compare(
            "items",
            "active {items}",
            "BETTER {items}",
            &MarkerSynth,
            &ContentJudge,
        )
        .await
        .unwrap();
        assert_eq!(verdict, PairVerdict::Candidate);
    }

    #[tokio::test]
    async fn position_bias_cancels_to_tie() {
        // Der Kern von §T2.4: ein Judge, der stur "A" antwortet, erzeugt KEIN Signal.
        let verdict = shadow_compare(
            "items",
            "active {items}",
            "BETTER {items}",
            &MarkerSynth,
            &PositionBiasedJudge,
        )
        .await
        .unwrap();
        assert_eq!(verdict, PairVerdict::Tie);
    }

    #[test]
    fn adoption_requires_win_and_no_loss() {
        let t = |wins, losses, ties| ShadowTally { wins, losses, ties };
        assert!(should_adopt(&t(1, 0, 0)));
        assert!(should_adopt(&t(2, 0, 1)));
        assert!(!should_adopt(&t(0, 0, 3))); // nur Ties → kein Beleg
        assert!(!should_adopt(&t(2, 1, 0))); // eine Niederlage disqualifiziert
        assert!(!should_adopt(&t(0, 0, 0)));
    }

    #[test]
    fn version_is_stable() {
        assert_eq!(version_of("abc"), version_of("abc"));
        assert_ne!(version_of("abc"), version_of("abd"));
    }
}
