//! Persistenz (SQLite via sqlx). Hält Content, Briefing-Records und Feedback —
//! die Datengrundlage für den Selbstverbesserungs-Loop (§5.3 / §6.1 der SPEC).
//!
//! Bewusst Runtime-Queries (kein `query!`-Makro) → kein DATABASE_URL zur Buildzeit nötig.

use anyhow::Result;
use ibrief_core::{Briefing, BriefingSection, ContentItem, FeedbackKind};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};

/// Aggregierte Feedback-Zählung für ein Briefing (Eingabe für den Verhaltens-Score, §6.2).
#[derive(Debug, Default, Clone, Copy)]
pub struct FeedbackCounts {
    pub up: i64,
    pub down: i64,
    pub more: i64,
    pub less: i64,
    pub open: i64,
}

/// Eine zu persistierende Eval-Zeile (Note pro Briefing × config_version).
#[derive(Debug, Clone)]
pub struct EvalRow {
    pub date: String,
    pub config_version: String,
    pub rubric_version: String,
    pub behavior: f64,
    pub judge: f64,
    pub structure: f64,
    pub total: f64,
    pub notes: Vec<String>,
}

/// Feedback-Zeile angereichert um Quelle/Themen (für Gewichts-Lernen, §6.3).
#[derive(Debug, Clone)]
pub struct FeedbackMeta {
    pub source_id: String,
    pub topics: Vec<String>,
    pub kind: String,
}

/// Metadaten einer Config-Version (für `config list` / Rollback).
#[derive(Debug, Clone)]
pub struct ConfigMeta {
    pub version: String,
    pub parent: Option<String>,
    pub reason: String,
    pub created_at: String,
}

/// Ein A/B-Experiment (Prompt- oder Config-Variante gegen die aktive).
#[derive(Debug, Clone)]
pub struct ExperimentRow {
    pub id: String,
    pub kind: String,
    pub slot: String,
    pub control: String,
    pub candidate: String,
    pub status: String,
    pub created_at: String,
}

/// Eine Quelle in der (lernenden) Registry.
#[derive(Debug, Clone)]
pub struct SourceRow {
    pub id: String,
    pub url: String,
    pub active: bool,
    pub quality: f64,
    pub reason: String,
}

/// Ein Korpus-Dokument für AutoResearch (§14) — lokal aus dem Content-Store.
#[derive(Debug, Clone)]
pub struct DocRow {
    pub url: String,
    pub text: String,
}

const SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS content_items (
        id           TEXT PRIMARY KEY,
        source_id    TEXT NOT NULL,
        title        TEXT NOT NULL,
        url          TEXT NOT NULL,
        published_at TEXT,
        raw_summary  TEXT,
        summary      TEXT,
        topics       TEXT NOT NULL DEFAULT '[]',
        first_seen   TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS briefings (
        date           TEXT PRIMARY KEY,
        config_version TEXT NOT NULL,
        tldr           TEXT NOT NULL DEFAULT '[]',
        created_at     TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS briefing_items (
        briefing_date TEXT NOT NULL,
        item_id       TEXT NOT NULL,
        section_id    TEXT NOT NULL,
        position      INTEGER NOT NULL,
        PRIMARY KEY (briefing_date, item_id)
    )",
    "CREATE TABLE IF NOT EXISTS feedback (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        briefing_date TEXT NOT NULL,
        item_id       TEXT NOT NULL,
        kind          TEXT NOT NULL,
        created_at    TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS evals (
        date           TEXT NOT NULL,
        config_version TEXT NOT NULL,
        rubric_version TEXT NOT NULL,
        behavior       REAL NOT NULL,
        judge          REAL NOT NULL,
        structure      REAL NOT NULL,
        total          REAL NOT NULL,
        notes          TEXT NOT NULL DEFAULT '[]',
        created_at     TEXT NOT NULL,
        PRIMARY KEY (date, config_version)
    )",
    "CREATE TABLE IF NOT EXISTS configs (
        version    TEXT PRIMARY KEY,
        parent     TEXT,
        reason     TEXT NOT NULL,
        payload    TEXT NOT NULL,
        created_at TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS config_state (
        id             INTEGER PRIMARY KEY CHECK (id = 1),
        active_version TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS prompts (
        slot       TEXT NOT NULL,
        version    TEXT NOT NULL,
        template   TEXT NOT NULL,
        parent     TEXT,
        reason     TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (slot, version)
    )",
    "CREATE TABLE IF NOT EXISTS prompt_active (
        slot    TEXT PRIMARY KEY,
        version TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS experiments (
        id         TEXT PRIMARY KEY,
        kind       TEXT NOT NULL,
        slot       TEXT NOT NULL,
        control    TEXT NOT NULL,
        candidate  TEXT NOT NULL,
        status     TEXT NOT NULL,
        created_at TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS sources (
        id         TEXT PRIMARY KEY,
        url        TEXT NOT NULL,
        active     INTEGER NOT NULL DEFAULT 1,
        quality    REAL NOT NULL DEFAULT 0.5,
        reason     TEXT NOT NULL DEFAULT 'seed',
        created_at TEXT NOT NULL
    )",
];

pub struct Store {
    pool: SqlitePool,
}

impl Store {
    /// Öffnet (oder erstellt) die DB und legt das Schema an.
    pub async fn open(path: &str) -> Result<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(opts)
            .await?;
        let store = Self { pool };
        store.init().await?;
        Ok(store)
    }

    async fn init(&self) -> Result<()> {
        for stmt in SCHEMA {
            sqlx::query(stmt).execute(&self.pool).await?;
        }
        Ok(())
    }

    /// True, wenn das Item schon einmal gesehen wurde (Dedup-Grundlage).
    pub async fn is_known(&self, id: &str) -> Result<bool> {
        let row = sqlx::query("SELECT 1 FROM content_items WHERE id = ?")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.is_some())
    }

    /// Behält nur Items, die noch nie gesehen wurden (Dedup über Tage hinweg).
    pub async fn filter_unseen(&self, items: Vec<ContentItem>) -> Result<Vec<ContentItem>> {
        let mut out = Vec::new();
        for it in items {
            if !self.is_known(&it.id).await? {
                out.push(it);
            }
        }
        Ok(out)
    }

    /// Fügt ein Item ein oder aktualisiert Summary/Topics (first_seen bleibt erhalten).
    pub async fn upsert_item(&self, it: &ContentItem) -> Result<()> {
        let topics = serde_json::to_string(&it.topics)?;
        let published = it.published_at.map(|d| d.to_rfc3339());
        sqlx::query(
            "INSERT INTO content_items
                (id, source_id, title, url, published_at, raw_summary, summary, topics, first_seen)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(id) DO UPDATE SET
                summary = excluded.summary,
                topics  = excluded.topics",
        )
        .bind(it.id.as_str())
        .bind(it.source_id.as_str())
        .bind(it.title.as_str())
        .bind(it.url.as_str())
        .bind(published)
        .bind(it.raw_summary.as_deref())
        .bind(it.summary.as_deref())
        .bind(topics)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Speichert einen Briefing-Record samt gezeigter Items (globale Position für Button-Mapping).
    pub async fn save_briefing(&self, b: &Briefing, config_version: &str) -> Result<()> {
        let tldr = serde_json::to_string(&b.tldr)?;
        sqlx::query(
            "INSERT OR REPLACE INTO briefings (date, config_version, tldr, created_at)
             VALUES (?, ?, ?, ?)",
        )
        .bind(b.date.as_str())
        .bind(config_version)
        .bind(tldr)
        .bind(now())
        .execute(&self.pool)
        .await?;

        let mut position: i64 = 0;
        for sec in &b.sections {
            for it in &sec.items {
                sqlx::query(
                    "INSERT OR REPLACE INTO briefing_items
                        (briefing_date, item_id, section_id, position)
                     VALUES (?, ?, ?, ?)",
                )
                .bind(b.date.as_str())
                .bind(it.id.as_str())
                .bind(sec.id.as_str())
                .bind(position)
                .execute(&self.pool)
                .await?;
                position += 1;
            }
        }
        Ok(())
    }

    /// Item-ID an globaler Position eines Briefings (für Telegram-Callback-Mapping).
    pub async fn item_at(&self, briefing_date: &str, position: i64) -> Result<Option<String>> {
        let row = sqlx::query(
            "SELECT item_id FROM briefing_items WHERE briefing_date = ? AND position = ?",
        )
        .bind(briefing_date)
        .bind(position)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| r.get::<String, _>("item_id")))
    }

    /// Schreibt ein Feedback-Ereignis.
    pub async fn record_feedback(
        &self,
        briefing_date: &str,
        item_id: &str,
        kind: FeedbackKind,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO feedback (briefing_date, item_id, kind, created_at)
             VALUES (?, ?, ?, ?)",
        )
        .bind(briefing_date)
        .bind(item_id)
        .bind(kind.as_str())
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Aggregiert das Feedback eines Briefings nach Art.
    pub async fn feedback_counts(&self, briefing_date: &str) -> Result<FeedbackCounts> {
        let rows = sqlx::query(
            "SELECT kind, COUNT(*) AS n FROM feedback WHERE briefing_date = ? GROUP BY kind",
        )
        .bind(briefing_date)
        .fetch_all(&self.pool)
        .await?;

        let mut c = FeedbackCounts::default();
        for r in rows {
            let kind: String = r.get("kind");
            let n: i64 = r.get("n");
            match kind.as_str() {
                "up" => c.up = n,
                "down" => c.down = n,
                "more" => c.more = n,
                "less" => c.less = n,
                "open" => c.open = n,
                _ => {}
            }
        }
        Ok(c)
    }

    /// Rekonstruiert ein gespeichertes Briefing aus der DB (für Eval/Judge).
    /// Die Sektions-Titel werden nicht persistiert → hier dient `section_id` als Titel.
    pub async fn load_briefing(&self, date: &str) -> Result<Option<Briefing>> {
        let Some(head) = sqlx::query("SELECT tldr FROM briefings WHERE date = ?")
            .bind(date)
            .fetch_optional(&self.pool)
            .await?
        else {
            return Ok(None);
        };
        let tldr: Vec<String> =
            serde_json::from_str(&head.get::<String, _>("tldr")).unwrap_or_default();

        let rows = sqlx::query(
            "SELECT bi.section_id, ci.id, ci.source_id, ci.title, ci.url,
                    ci.published_at, ci.raw_summary, ci.summary, ci.topics
             FROM briefing_items bi
             JOIN content_items ci ON ci.id = bi.item_id
             WHERE bi.briefing_date = ?
             ORDER BY bi.position",
        )
        .bind(date)
        .fetch_all(&self.pool)
        .await?;

        let mut sections: Vec<BriefingSection> = Vec::new();
        for row in rows {
            let section_id: String = row.get("section_id");
            let published_at = row
                .get::<Option<String>, _>("published_at")
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                .map(|d| d.with_timezone(&chrono::Utc));
            let topics: Vec<String> =
                serde_json::from_str(&row.get::<String, _>("topics")).unwrap_or_default();
            let item = ContentItem {
                id: row.get("id"),
                source_id: row.get("source_id"),
                title: row.get("title"),
                url: row.get("url"),
                published_at,
                raw_summary: row.get("raw_summary"),
                summary: row.get("summary"),
                topics,
            };
            match sections.iter_mut().find(|s| s.id == section_id) {
                Some(s) => s.items.push(item),
                None => sections.push(BriefingSection {
                    id: section_id.clone(),
                    title: section_id,
                    items: vec![item],
                }),
            }
        }

        Ok(Some(Briefing {
            date: date.to_string(),
            tldr,
            sections,
        }))
    }

    /// Speichert eine Eval-Note (idempotent pro date × config_version).
    pub async fn save_eval(&self, e: &EvalRow) -> Result<()> {
        let notes = serde_json::to_string(&e.notes)?;
        sqlx::query(
            "INSERT OR REPLACE INTO evals
                (date, config_version, rubric_version, behavior, judge, structure, total, notes, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(e.date.as_str())
        .bind(e.config_version.as_str())
        .bind(e.rubric_version.as_str())
        .bind(e.behavior)
        .bind(e.judge)
        .bind(e.structure)
        .bind(e.total)
        .bind(notes)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Alle Feedback-Ereignisse mit Quelle/Themen des betroffenen Items.
    pub async fn feedback_join_meta(&self) -> Result<Vec<FeedbackMeta>> {
        let rows = sqlx::query(
            "SELECT ci.source_id, ci.topics, f.kind
             FROM feedback f
             JOIN content_items ci ON ci.id = f.item_id",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let topics: Vec<String> =
                serde_json::from_str(&r.get::<String, _>("topics")).unwrap_or_default();
            out.push(FeedbackMeta {
                source_id: r.get("source_id"),
                topics,
                kind: r.get("kind"),
            });
        }
        Ok(out)
    }

    /// Speichert eine Config-Version (idempotent — gleicher Inhalt = gleiche Version).
    pub async fn save_config(
        &self,
        version: &str,
        parent: Option<&str>,
        reason: &str,
        payload: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT OR IGNORE INTO configs (version, parent, reason, payload, created_at)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(version)
        .bind(parent)
        .bind(reason)
        .bind(payload)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Setzt die aktive Config-Version (atomar).
    pub async fn set_active_config(&self, version: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO config_state (id, active_version) VALUES (1, ?)
             ON CONFLICT(id) DO UPDATE SET active_version = excluded.active_version",
        )
        .bind(version)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Aktuell aktive Config-Version (None, wenn noch nie gelernt wurde).
    pub async fn active_config_version(&self) -> Result<Option<String>> {
        let row = sqlx::query("SELECT active_version FROM config_state WHERE id = 1")
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.get::<String, _>("active_version")))
    }

    /// JSON-Payload einer Config-Version.
    pub async fn load_config_payload(&self, version: &str) -> Result<Option<String>> {
        let row = sqlx::query("SELECT payload FROM configs WHERE version = ?")
            .bind(version)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.get::<String, _>("payload")))
    }

    /// True, wenn die Version existiert (für Rollback-Validierung).
    pub async fn config_exists(&self, version: &str) -> Result<bool> {
        let row = sqlx::query("SELECT 1 FROM configs WHERE version = ?")
            .bind(version)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.is_some())
    }

    /// Jüngste Config-Versionen (für `config list`).
    pub async fn recent_configs(&self, limit: i64) -> Result<Vec<ConfigMeta>> {
        let rows = sqlx::query(
            "SELECT version, parent, reason, created_at
             FROM configs ORDER BY created_at DESC LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|r| ConfigMeta {
                version: r.get("version"),
                parent: r.get("parent"),
                reason: r.get("reason"),
                created_at: r.get("created_at"),
            })
            .collect())
    }

    /// Speichert eine Prompt-Version für einen Slot (idempotent).
    pub async fn save_prompt(
        &self,
        slot: &str,
        version: &str,
        parent: Option<&str>,
        reason: &str,
        template: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT OR IGNORE INTO prompts (slot, version, template, parent, reason, created_at)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(slot)
        .bind(version)
        .bind(template)
        .bind(parent)
        .bind(reason)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Setzt die aktive Prompt-Version eines Slots.
    pub async fn set_active_prompt(&self, slot: &str, version: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO prompt_active (slot, version) VALUES (?, ?)
             ON CONFLICT(slot) DO UPDATE SET version = excluded.version",
        )
        .bind(slot)
        .bind(version)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Aktive Prompt-Version + Template eines Slots.
    pub async fn active_prompt(&self, slot: &str) -> Result<Option<(String, String)>> {
        let row = sqlx::query(
            "SELECT p.version, p.template
             FROM prompt_active pa
             JOIN prompts p ON p.slot = pa.slot AND p.version = pa.version
             WHERE pa.slot = ?",
        )
        .bind(slot)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| {
            (
                r.get::<String, _>("version"),
                r.get::<String, _>("template"),
            )
        }))
    }

    /// Speichert ein Experiment (idempotent).
    pub async fn save_experiment(&self, e: &ExperimentRow) -> Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO experiments
                (id, kind, slot, control, candidate, status, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(e.id.as_str())
        .bind(e.kind.as_str())
        .bind(e.slot.as_str())
        .bind(e.control.as_str())
        .bind(e.candidate.as_str())
        .bind(e.status.as_str())
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Jüngste Experimente (für `experiment list`).
    pub async fn recent_experiments(&self, limit: i64) -> Result<Vec<ExperimentRow>> {
        let rows = sqlx::query(
            "SELECT id, kind, slot, control, candidate, status, created_at
             FROM experiments ORDER BY created_at DESC LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|r| ExperimentRow {
                id: r.get("id"),
                kind: r.get("kind"),
                slot: r.get("slot"),
                control: r.get("control"),
                candidate: r.get("candidate"),
                status: r.get("status"),
                created_at: r.get("created_at"),
            })
            .collect())
    }

    /// Legt eine Quelle an (idempotent — Seed aus sources.toml).
    pub async fn seed_source(&self, id: &str, url: &str, reason: &str) -> Result<()> {
        sqlx::query(
            "INSERT OR IGNORE INTO sources (id, url, active, quality, reason, created_at)
             VALUES (?, ?, 1, 0.5, ?, ?)",
        )
        .bind(id)
        .bind(url)
        .bind(reason)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    fn map_source(r: &sqlx::sqlite::SqliteRow) -> SourceRow {
        SourceRow {
            id: r.get("id"),
            url: r.get("url"),
            active: r.get::<i64, _>("active") != 0,
            quality: r.get("quality"),
            reason: r.get("reason"),
        }
    }

    /// Alle Quellen (auch deaktivierte).
    pub async fn all_sources(&self) -> Result<Vec<SourceRow>> {
        let rows = sqlx::query("SELECT id, url, active, quality, reason FROM sources ORDER BY id")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.iter().map(Self::map_source).collect())
    }

    /// Nur aktive Quellen (Eingabe fürs Ingest).
    pub async fn active_sources(&self) -> Result<Vec<SourceRow>> {
        let rows = sqlx::query(
            "SELECT id, url, active, quality, reason FROM sources WHERE active = 1 ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(Self::map_source).collect())
    }

    pub async fn set_source_active(&self, id: &str, active: bool) -> Result<()> {
        sqlx::query("UPDATE sources SET active = ? WHERE id = ?")
            .bind(active as i64)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn set_source_quality(&self, id: &str, quality: f64) -> Result<()> {
        sqlx::query("UPDATE sources SET quality = ? WHERE id = ?")
            .bind(quality)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Wie oft jede Quelle bislang in Briefings ausgewählt wurde.
    pub async fn selection_counts(&self) -> Result<Vec<(String, i64)>> {
        let rows = sqlx::query(
            "SELECT ci.source_id AS sid, COUNT(*) AS n
             FROM briefing_items bi
             JOIN content_items ci ON ci.id = bi.item_id
             GROUP BY ci.source_id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| (r.get::<String, _>("sid"), r.get::<i64, _>("n")))
            .collect())
    }

    /// Volltext-Suche über den lokalen Content-Korpus (AutoResearch §14, lokal).
    pub async fn search_items(&self, query: &str, limit: i64) -> Result<Vec<DocRow>> {
        let needle = format!("%{query}%");
        let rows = sqlx::query(
            "SELECT url, COALESCE(summary, title) AS text
             FROM content_items
             WHERE title LIKE ? OR summary LIKE ?
             LIMIT ?",
        )
        .bind(&needle)
        .bind(&needle)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .iter()
            .map(|r| DocRow {
                url: r.get("url"),
                text: r.get("text"),
            })
            .collect())
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ibrief_core::{Briefing, BriefingSection, ContentItem};

    fn item(id: &str) -> ContentItem {
        ContentItem {
            id: id.into(),
            source_id: "s".into(),
            title: "t".into(),
            url: format!("https://example.com/{id}"),
            published_at: None,
            raw_summary: None,
            summary: Some("sum".into()),
            topics: vec!["a".into()],
        }
    }

    #[tokio::test]
    async fn dedup_briefing_and_feedback() {
        let path = std::env::temp_dir()
            .join(format!("ibrief-test-{}.db", std::process::id()))
            .to_string_lossy()
            .to_string();
        let _ = std::fs::remove_file(&path);

        let store = Store::open(&path).await.unwrap();

        // Dedup: erst beide neu, nach Upsert nur noch unbekannte.
        let fresh = store
            .filter_unseen(vec![item("a"), item("b")])
            .await
            .unwrap();
        assert_eq!(fresh.len(), 2);
        for it in &fresh {
            store.upsert_item(it).await.unwrap();
        }
        let again = store
            .filter_unseen(vec![item("a"), item("c")])
            .await
            .unwrap();
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].id, "c");

        // Briefing-Record + Positions-Mapping.
        let b = Briefing {
            date: "2026-06-28".into(),
            tldr: vec!["x".into()],
            sections: vec![BriefingSection {
                id: "ai_tech".into(),
                title: "T".into(),
                items: vec![item("a"), item("b")],
            }],
        };
        store.save_briefing(&b, "m2-test").await.unwrap();
        assert_eq!(
            store.item_at("2026-06-28", 0).await.unwrap().as_deref(),
            Some("a")
        );
        assert_eq!(
            store.item_at("2026-06-28", 1).await.unwrap().as_deref(),
            Some("b")
        );

        store
            .record_feedback("2026-06-28", "a", FeedbackKind::Up)
            .await
            .unwrap();

        std::fs::remove_file(&path).ok();
    }
}
