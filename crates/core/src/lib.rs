//! Domänentypen für ibrief. Reine Daten — keine Logik, keine I/O.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Ein einzelnes Inhalts-Item aus einer Quelle, durch die Pipeline angereichert.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentItem {
    /// Stabile ID (für M1: die URL; später Content-Hash).
    pub id: String,
    pub source_id: String,
    pub title: String,
    pub url: String,
    pub published_at: Option<DateTime<Utc>>,
    /// Roh-Zusammenfassung aus dem Feed (falls vorhanden).
    pub raw_summary: Option<String>,
    /// Von der ENRICH-Stage erzeugte Ein-Satz-Zusammenfassung.
    pub summary: Option<String>,
    /// Themen-Tags aus ENRICH.
    pub topics: Vec<String>,
}

/// Eine Sektion des Briefings (z.B. "KI & Tech").
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BriefingSection {
    pub id: String,
    pub title: String,
    pub items: Vec<ContentItem>,
}

/// Das fertige Briefing eines Tages.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Briefing {
    pub date: String,
    /// "Die 3 Dinge heute" — Executive Summary.
    pub tldr: Vec<String>,
    pub sections: Vec<BriefingSection>,
}

/// Art eines Feedback-Signals (§6.1 der SPEC).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackKind {
    /// Explizit positiv (👍).
    Up,
    /// Explizit negativ (👎).
    Down,
    /// "mehr davon" (Thema/Quelle).
    More,
    /// "weniger davon".
    Less,
    /// Link geöffnet (implizit).
    Open,
}

impl FeedbackKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
            Self::More => "more",
            Self::Less => "less",
            Self::Open => "open",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "up" => Self::Up,
            "down" => Self::Down,
            "more" => Self::More,
            "less" => Self::Less,
            "open" => Self::Open,
            _ => return None,
        })
    }
}

/// Lernbare Gewichte (das zentrale, versionierte Daten-Artefakt aus §6.3/§8).
/// Fehlende Einträge bedeuten neutrales Gewicht 1.0. BTreeMap → kanonische
/// Serialisierung für stabile Versions-Hashes.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub source_weights: BTreeMap<String, f64>,
    #[serde(default)]
    pub topic_weights: BTreeMap<String, f64>,
}

impl Config {
    pub fn source_weight(&self, id: &str) -> f64 {
        self.source_weights.get(id).copied().unwrap_or(1.0)
    }

    pub fn topic_weight(&self, topic: &str) -> f64 {
        self.topic_weights.get(topic).copied().unwrap_or(1.0)
    }
}
