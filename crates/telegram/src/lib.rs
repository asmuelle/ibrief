//! Feedback-Kanal: Telegram-Push mit Inline-Buttons (👍/👎), direkt über die Bot-API.
//!
//! Bewusst ohne Framework — nur `reqwest`. Zwei Funktionen:
//! - [`Telegram::send_briefing`]   — Briefing pushen, je Item ein 👍/👎-Button.
//! - [`Telegram::run_feedback_loop`] — Button-Klicks (Callback-Queries) abholen und im Store ablegen.
//!
//! Callback-Daten sind auf 64 Bytes begrenzt → wir kodieren `fb:<datum>:<position>:<kind>`
//! und lösen die Position über den Store zur Item-ID auf (URLs wären zu lang).

use anyhow::Result;
use ibrief_core::{Briefing, FeedbackKind};
use ibrief_store::Store;
use serde::Deserialize;

pub struct Telegram {
    token: String,
    chat_id: String,
    http: reqwest::Client,
}

impl Telegram {
    pub fn new(token: String, chat_id: String) -> Self {
        Self {
            token,
            chat_id,
            http: reqwest::Client::new(),
        }
    }

    fn url(&self, method: &str) -> String {
        format!("https://api.telegram.org/bot{}/{}", self.token, method)
    }

    /// Pusht das Briefing: eine Kopfnachricht mit TL;DR, danach je Item eine Nachricht mit Buttons.
    pub async fn send_briefing(&self, b: &Briefing) -> Result<()> {
        let mut head = format!("*Morning Briefing — {}*\n", b.date);
        for t in &b.tldr {
            head.push_str(&format!("• {}\n", strip_md(t)));
        }
        self.send_message(&head, None).await?;

        let mut position: i64 = 0;
        for sec in &b.sections {
            for it in &sec.items {
                let text = format!(
                    "*{}*\n{}\n{}",
                    strip_md(&it.title),
                    it.summary.clone().unwrap_or_default(),
                    it.url
                );
                self.send_message(&text, Some(thumbs_keyboard(&b.date, position)))
                    .await?;
                position += 1;
            }
        }
        Ok(())
    }

    async fn send_message(&self, text: &str, keyboard: Option<serde_json::Value>) -> Result<()> {
        let mut body = serde_json::json!({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": true,
        });
        if let Some(kb) = keyboard {
            body["reply_markup"] = kb;
        }
        self.http
            .post(self.url("sendMessage"))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Long-Polling-Schleife: holt Callback-Queries und schreibt Feedback in den Store.
    /// Läuft bis zum Prozessende (für `ibrief feedback`).
    pub async fn run_feedback_loop(&self, store: &Store) -> Result<()> {
        let mut offset: i64 = 0;
        tracing::info!("Telegram-Feedback-Loop gestartet");
        loop {
            let updates = match self.get_updates(offset).await {
                Ok(u) => u,
                Err(e) => {
                    tracing::warn!(error = %e, "getUpdates fehlgeschlagen, neuer Versuch");
                    continue;
                }
            };
            for up in updates {
                offset = up.update_id + 1;
                let Some(cb) = up.callback_query else {
                    continue;
                };
                if let Some(data) = &cb.data
                    && let Some((date, pos, kind)) = parse_callback(data)
                {
                    match store.item_at(&date, pos).await {
                        Ok(Some(item_id)) => {
                            if let Err(e) = store.record_feedback(&date, &item_id, kind).await {
                                tracing::warn!(error = %e, "Feedback speichern fehlgeschlagen");
                            } else {
                                tracing::info!(%date, %item_id, kind = kind.as_str(), "Feedback erfasst");
                            }
                        }
                        Ok(None) => tracing::warn!(%date, pos, "kein Item an Position"),
                        Err(e) => tracing::warn!(error = %e, "item_at fehlgeschlagen"),
                    }
                }
                self.answer_callback(&cb.id).await.ok();
            }
        }
    }

    async fn get_updates(&self, offset: i64) -> Result<Vec<Update>> {
        let body = serde_json::json!({
            "offset": offset,
            "timeout": 30,
            "allowed_updates": ["callback_query"],
        });
        let resp: TgResponse<Vec<Update>> = self
            .http
            .post(self.url("getUpdates"))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp.result)
    }

    async fn answer_callback(&self, callback_id: &str) -> Result<()> {
        let body = serde_json::json!({ "callback_query_id": callback_id, "text": "notiert ✓" });
        self.http
            .post(self.url("answerCallbackQuery"))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

fn thumbs_keyboard(date: &str, position: i64) -> serde_json::Value {
    serde_json::json!({
        "inline_keyboard": [[
            { "text": "👍", "callback_data": format!("fb:{date}:{position}:up") },
            { "text": "👎", "callback_data": format!("fb:{date}:{position}:down") },
        ]]
    })
}

fn parse_callback(data: &str) -> Option<(String, i64, FeedbackKind)> {
    let mut parts = data.split(':');
    if parts.next()? != "fb" {
        return None;
    }
    let date = parts.next()?.to_string();
    let pos = parts.next()?.parse().ok()?;
    let kind = FeedbackKind::parse(parts.next()?)?;
    Some((date, pos, kind))
}

/// Minimaler Schutz gegen Markdown-Konflikte in Telegram-Texten.
fn strip_md(s: &str) -> String {
    s.replace(['*', '_', '`', '['], "")
}

#[derive(Deserialize)]
struct TgResponse<T> {
    result: T,
}

#[derive(Deserialize)]
struct Update {
    update_id: i64,
    #[serde(default)]
    callback_query: Option<CallbackQuery>,
}

#[derive(Deserialize)]
struct CallbackQuery {
    id: String,
    #[serde(default)]
    data: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_callback_data() {
        let (date, pos, kind) = parse_callback("fb:2026-06-28:3:up").unwrap();
        assert_eq!(date, "2026-06-28");
        assert_eq!(pos, 3);
        assert_eq!(kind, FeedbackKind::Up);
        assert!(parse_callback("nope:x").is_none());
    }
}
