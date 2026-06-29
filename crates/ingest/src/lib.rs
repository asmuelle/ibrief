//! INGEST-Stage: Quellen abrufen, zu [`ContentItem`] normalisieren.
//!
//! M1: RSS/Atom via `feed-rs`. Fehler einzelner Quellen werden geloggt und
//! übersprungen — ein kaputter Feed darf das Briefing nicht verhindern.

use anyhow::Result;
use ibrief_core::ContentItem;
use serde::Deserialize;

/// Eine Quelle aus `sources.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Source {
    pub id: String,
    pub url: String,
}

/// Alle Quellen abrufen; fehlerhafte Quellen werden übersprungen.
pub async fn fetch_all(sources: &[Source]) -> Vec<ContentItem> {
    let client = reqwest::Client::new();
    let mut items = Vec::new();
    for src in sources {
        match fetch_one(&client, src).await {
            Ok(mut got) => {
                tracing::info!(source = %src.id, count = got.len(), "fetched");
                items.append(&mut got);
            }
            Err(e) => {
                tracing::warn!(source = %src.id, error = %e, "ingest fehlgeschlagen, überspringe Quelle");
            }
        }
    }
    items
}

async fn fetch_one(client: &reqwest::Client, src: &Source) -> Result<Vec<ContentItem>> {
    let bytes = client
        .get(&src.url)
        .header(reqwest::header::USER_AGENT, "ibrief/0.1")
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;

    let feed = feed_rs::parser::parse(&bytes[..])?;

    let items = feed
        .entries
        .into_iter()
        .map(|e| {
            let title = e.title.map(|t| t.content).unwrap_or_default();
            let url = e.links.first().map(|l| l.href.clone()).unwrap_or_default();
            let raw_summary = e.summary.map(|t| t.content);
            let id = if url.is_empty() {
                e.id.clone()
            } else {
                url.clone()
            };
            ContentItem {
                id,
                source_id: src.id.clone(),
                title,
                url,
                published_at: e.published,
                raw_summary,
                summary: None,
                topics: Vec::new(),
                entities: Vec::new(),
                embedding: None,
            }
        })
        .collect();

    Ok(items)
}
