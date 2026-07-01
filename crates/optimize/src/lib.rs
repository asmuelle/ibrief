//! Prompt-Optimierung (§6.3 B / §6.5): ein Optimizer-LLM erzeugt eine Prompt-Variante,
//! die im **Schatten-Test** gegen den aktiven Prompt antritt (beide rendern, der Judge
//! bewertet) — und nur bei klarem Vorsprung Default wird. Kein Risiko fürs Live-Briefing.
//!
//! Aktuell für den `tldr`-Slot. Weitere Slots (Sektions-Kuratierung) folgen demselben Muster.

use anyhow::Result;
use ibrief_llm::{Completion, LanguageModel};
use ibrief_store::{ExperimentRow, Store};
use serde::Deserialize;

/// Slot-Bezeichner für den TL;DR-Prompt.
pub const SLOT_TLDR: &str = "tldr";

/// Mindest-Vorsprung im Judge-Score, damit ein Kandidat den aktiven Prompt ablöst.
const SHADOW_MARGIN: f64 = 0.05;

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

/// Ergebnis des Schatten-Vergleichs.
#[derive(Debug, Clone)]
pub struct Shadow {
    pub active_score: f64,
    pub candidate_score: f64,
}

/// Ergebnis eines Optimierungslaufs.
#[derive(Debug, Clone)]
pub struct OptimizeOutcome {
    pub adopted: bool,
    pub active_version: String,
    pub candidate_version: String,
    pub shadow: Option<Shadow>,
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

/// Ein Optimierungs-Schritt für den TL;DR-Slot anhand eines Beispiel-Briefings (`date`).
pub async fn optimize_tldr(
    store: &Store,
    optimizer: &dyn LanguageModel,
    synth: &dyn LanguageModel,
    judge: &dyn LanguageModel,
    date: &str,
) -> Result<OptimizeOutcome> {
    let active = active_tldr(store).await?;

    let Some(briefing) = store.load_briefing(date).await? else {
        anyhow::bail!("kein Briefing für {date} (Beispiel-Input für den Schatten-Test fehlt)");
    };
    let items_text = items_to_text(&briefing);

    let candidate_template = propose_prompt(&active.template, optimizer).await?;
    let candidate_version = version_of(&candidate_template);

    if candidate_version == active.version {
        return Ok(OptimizeOutcome {
            adopted: false,
            active_version: active.version,
            candidate_version,
            shadow: None,
            reason: "Kandidat identisch zum aktiven Prompt".into(),
        });
    }

    let shadow = shadow_compare(
        &items_text,
        &active.template,
        &candidate_template,
        synth,
        judge,
    )
    .await?;
    let adopted = shadow.candidate_score > shadow.active_score + SHADOW_MARGIN;

    let experiment_id = version_of(&format!("{}|{}|{date}", active.version, candidate_version));
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
        "Schatten-Judge: aktiv={:.2} vs. Kandidat={:.2} (Marge {SHADOW_MARGIN})",
        shadow.active_score, shadow.candidate_score
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
        shadow: Some(shadow),
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

/// Schatten-Vergleich: beide Prompts rendern lassen, beide Ausgaben vom Judge bewerten.
pub async fn shadow_compare(
    items_text: &str,
    active_template: &str,
    candidate_template: &str,
    synth: &dyn LanguageModel,
    judge: &dyn LanguageModel,
) -> Result<Shadow> {
    let active_out = synth
        .complete(&Completion::new(fill(active_template, items_text)))
        .await?;
    let candidate_out = synth
        .complete(&Completion::new(fill(candidate_template, items_text)))
        .await?;
    Ok(Shadow {
        active_score: judge_text(&active_out, judge).await?,
        candidate_score: judge_text(&candidate_out, judge).await?,
    })
}

fn fill(template: &str, items_text: &str) -> String {
    if template.contains("{items}") {
        template.replace("{items}", items_text)
    } else {
        format!("{template}\n\n{items_text}")
    }
}

#[derive(Deserialize)]
struct JudgeOut {
    overall: f64,
}

async fn judge_text(text: &str, judge: &dyn LanguageModel) -> Result<f64> {
    let prompt = format!(
        "Bewerte dieses TL;DR von 0.0 bis 1.0 nach Prägnanz, Relevanz und Anschlussfähigkeit.\n\
TL;DR:\n{text}\n\nAntworte nur mit JSON {{\"overall\": 0.0}}."
    );
    let req = Completion::new(prompt)
        .with_system(JUDGE_SYSTEM)
        .temperature(0.1);
    let raw = judge.complete(&req).await?;
    let out: JudgeOut = serde_json::from_str(&extract_json(&raw))?;
    Ok(out.overall.clamp(0.0, 1.0))
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

    /// Judge: 0.9 wenn der bewertete Text "candidate" enthält, sonst 0.5.
    struct MarkerJudge;
    #[async_trait]
    impl LanguageModel for MarkerJudge {
        async fn complete(&self, req: &Completion) -> Result<String, ibrief_llm::ModelError> {
            let overall = if req.prompt.contains("candidate") {
                0.9
            } else {
                0.5
            };
            Ok(format!("{{\"overall\": {overall}}}"))
        }
        fn label(&self) -> &str {
            "judge"
        }
    }

    #[tokio::test]
    async fn shadow_prefers_better_candidate() {
        let shadow = shadow_compare(
            "items",
            "active {items}",
            "BETTER {items}",
            &MarkerSynth,
            &MarkerJudge,
        )
        .await
        .unwrap();
        assert!(shadow.candidate_score > shadow.active_score);
        assert!((shadow.candidate_score - 0.9).abs() < 1e-9);
        assert!((shadow.active_score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn version_is_stable() {
        assert_eq!(version_of("abc"), version_of("abc"));
        assert_ne!(version_of("abc"), version_of("abd"));
    }
}
