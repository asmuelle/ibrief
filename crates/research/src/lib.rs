//! AutoResearch (§14): gated Deep-Research-Loop — PLAN→SEARCH→READ→REFLECT→SYNTHESIZE→VERIFY.
//!
//! Bewusst begrenzt: Stoppkriterien (max. Iterationen/Quellen), Beleg-Pflicht je Aussage,
//! unbelegte Claims werden NICHT ins Briefing übernommen. Die Such-Schicht ist über
//! [`ResearchSource`] abstrahiert — Default ist der lokale Content-Korpus
//! ([`StoreResearchSource`]); ein Web-Backend kann später dieselbe Trait erfüllen.

use anyhow::Result;
use async_trait::async_trait;
use ibrief_llm::{Completion, LanguageModel};
use ibrief_store::Store;
use serde::Deserialize;
use std::collections::HashSet;

/// Ein gefundenes Dokument (Quelle + Text).
#[derive(Debug, Clone)]
pub struct Doc {
    pub url: String,
    pub text: String,
}

/// Such-Backend. Implementierbar über lokalen Korpus, Web-API, etc.
#[async_trait]
pub trait ResearchSource: Send + Sync {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<Doc>>;
}

/// Lokaler Korpus aus dem Content-Store (kein externer Dienst nötig).
pub struct StoreResearchSource<'a> {
    store: &'a Store,
}

impl<'a> StoreResearchSource<'a> {
    pub fn new(store: &'a Store) -> Self {
        Self { store }
    }
}

#[async_trait]
impl ResearchSource for StoreResearchSource<'_> {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<Doc>> {
        let rows = self.store.search_items(query, limit as i64).await?;
        Ok(rows
            .into_iter()
            .map(|r| Doc {
                url: r.url,
                text: r.text,
            })
            .collect())
    }
}

/// Stoppkriterien (§14.3).
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub max_iterations: usize,
    pub max_sources: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            max_sources: 12,
        }
    }
}

/// Eine belegte (oder zu belegende) Aussage.
#[derive(Debug, Clone, Deserialize)]
pub struct Claim {
    pub text: String,
    #[serde(default)]
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Complete,
    Partial,
    Aborted,
}

/// Ergebnis-Vertrag (§14.4).
#[derive(Debug, Clone)]
pub struct ResearchResult {
    pub question: String,
    pub answer_md: String,
    pub claims: Vec<Claim>,
    pub unverified_claims: Vec<String>,
    pub sources_used: Vec<String>,
    pub iterations: usize,
    pub status: Status,
}

/// Verifikation (§14.5): nur Claims mit erreichbarer, zitierter Quelle gelten.
pub fn verify_claims(claims: Vec<Claim>, available: &HashSet<String>) -> (Vec<Claim>, Vec<String>) {
    let mut verified = Vec::new();
    let mut unverified = Vec::new();
    for c in claims {
        if !c.source.is_empty() && available.contains(&c.source) {
            verified.push(c);
        } else {
            unverified.push(c.text);
        }
    }
    (verified, unverified)
}

/// Der gated Research-Loop.
pub async fn research(
    question: &str,
    source: &dyn ResearchSource,
    model: &dyn LanguageModel,
    budget: &Budget,
) -> Result<ResearchResult> {
    let mut docs: Vec<Doc> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut query = question.to_string();
    let mut iterations = 0;

    while iterations < budget.max_iterations && docs.len() < budget.max_sources {
        iterations += 1;
        let found = source
            .search(&query, budget.max_sources - docs.len())
            .await?;
        let mut added = 0;
        for d in found {
            if seen.insert(d.url.clone()) {
                docs.push(d);
                added += 1;
            }
        }
        if added == 0 {
            break; // keine neuen Erkenntnisse → early stop (§14.3)
        }
        match reflect(model, question, &docs).await? {
            Some(next) => query = next,
            None => break, // Modell meldet "fertig"
        }
    }

    if docs.is_empty() {
        return Ok(ResearchResult {
            question: question.to_string(),
            answer_md: String::new(),
            claims: vec![],
            unverified_claims: vec![],
            sources_used: vec![],
            iterations,
            status: Status::Aborted,
        });
    }

    let synth = synthesize(model, question, &docs).await?;
    let available: HashSet<String> = docs.iter().map(|d| d.url.clone()).collect();
    let (verified, unverified) = verify_claims(synth.claims, &available);

    let status = if iterations >= budget.max_iterations && docs.len() >= budget.max_sources {
        Status::Partial
    } else {
        Status::Complete
    };

    Ok(ResearchResult {
        question: question.to_string(),
        answer_md: synth.answer,
        claims: verified,
        unverified_claims: unverified,
        sources_used: docs.into_iter().map(|d| d.url).collect(),
        iterations,
        status,
    })
}

async fn reflect(
    model: &dyn LanguageModel,
    question: &str,
    docs: &[Doc],
) -> Result<Option<String>> {
    let prompt = format!(
        "Forschungsfrage: {question}\nBisher {} Quellen gelesen.\n\n\
Ist die Frage ausreichend beantwortet? Antworte mit 'DONE', \
oder gib EINE präzisere Folge-Suchanfrage (eine Zeile).",
        docs.len()
    );
    let raw = model
        .complete(&Completion::new(prompt).temperature(0.3))
        .await?;
    let line = raw.trim();
    if line.is_empty() || line.to_uppercase().contains("DONE") {
        Ok(None)
    } else {
        Ok(Some(line.lines().next().unwrap_or(line).trim().to_string()))
    }
}

#[derive(Deserialize)]
struct Synthesis {
    #[serde(default)]
    answer: String,
    #[serde(default)]
    claims: Vec<Claim>,
}

async fn synthesize(model: &dyn LanguageModel, question: &str, docs: &[Doc]) -> Result<Synthesis> {
    let mut corpus = String::new();
    for d in docs {
        corpus.push_str(&format!("[{}] {}\n", d.url, truncate(&d.text, 500)));
    }
    let prompt = format!(
        "Frage: {question}\n\nQuellen:\n{corpus}\n\nFasse die Antwort zusammen. \
Belege JEDE Aussage mit genau einer Quellen-URL aus der Liste. \
Antworte NUR mit JSON: {{\"answer\":\"...\",\"claims\":[{{\"text\":\"...\",\"source\":\"<url>\"}}]}}.",
    );
    let req = Completion::new(prompt)
        .with_system("Du bist ein präziser Rechercheur. Nur belegte Aussagen. Nur JSON.")
        .temperature(0.2);
    let raw = model.complete(&req).await?;
    Ok(serde_json::from_str(&extract_json(&raw))?)
}

fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

fn extract_json(s: &str) -> String {
    match (s.find('{'), s.rfind('}')) {
        (Some(a), Some(b)) if b >= a => s[a..=b].to_string(),
        _ => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_excludes_unsourced_and_fake_claims() {
        let mut available = HashSet::new();
        available.insert("https://real".to_string());
        let claims = vec![
            Claim {
                text: "belegt".into(),
                source: "https://real".into(),
            },
            Claim {
                text: "erfunden".into(),
                source: "https://fake".into(),
            },
            Claim {
                text: "ohne quelle".into(),
                source: "".into(),
            },
        ];
        let (verified, unverified) = verify_claims(claims, &available);
        assert_eq!(verified.len(), 1);
        assert_eq!(verified[0].text, "belegt");
        assert_eq!(unverified.len(), 2);
    }

    struct MockSource(Vec<Doc>);
    #[async_trait]
    impl ResearchSource for MockSource {
        async fn search(&self, _q: &str, _limit: usize) -> Result<Vec<Doc>> {
            Ok(self.0.clone())
        }
    }

    struct MockModel;
    #[async_trait]
    impl LanguageModel for MockModel {
        async fn complete(&self, req: &Completion) -> Result<String> {
            if req.prompt.contains("Folge-Suchanfrage") {
                Ok("DONE".into()) // reflect → fertig nach erster Runde
            } else {
                // synthesize: ein belegter + ein erfundener Claim
                Ok(r#"{"answer":"Zusammenfassung","claims":[
                    {"text":"belegt","source":"https://a"},
                    {"text":"erfunden","source":"https://x"}]}"#
                    .into())
            }
        }
        fn label(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn loop_verifies_and_drops_unsourced() {
        let source = MockSource(vec![
            Doc {
                url: "https://a".into(),
                text: "text a".into(),
            },
            Doc {
                url: "https://b".into(),
                text: "text b".into(),
            },
        ]);
        let res = research("Frage?", &source, &MockModel, &Budget::default())
            .await
            .unwrap();
        assert_eq!(res.status, Status::Complete);
        assert_eq!(res.claims.len(), 1); // nur der belegte überlebt
        assert_eq!(res.unverified_claims.len(), 1);
        assert_eq!(res.iterations, 1);
    }

    #[tokio::test]
    async fn empty_corpus_aborts() {
        let res = research(
            "Frage?",
            &MockSource(vec![]),
            &MockModel,
            &Budget::default(),
        )
        .await
        .unwrap();
        assert_eq!(res.status, Status::Aborted);
    }
}
