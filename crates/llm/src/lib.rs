//! LLM-Gateway: ein Trait, mehrere austauschbare Backends.
//!
//! - [`OllamaClient`]  — lokales Standard-Backend (kostenlos).
//! - [`ClaudeCodeModel`] — Abo-basierte Kalibrierung via `claude -p` (Claude Code CLI).
//!
//! Weitere Backends (Codex-CLI, direkte API) implementieren einfach [`LanguageModel`].

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Eine Anfrage an ein Sprachmodell.
#[derive(Debug, Clone)]
pub struct Completion {
    pub system: Option<String>,
    pub prompt: String,
    pub temperature: f32,
    /// Obergrenze der zu generierenden Tokens (Ollama: `num_predict`). `None` = Modell-Default.
    /// Deckelt ausuferndes Generieren — der größte Einzel-Hebel gegen die Enrich-Latenz.
    pub max_tokens: Option<u32>,
}

impl Completion {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            system: None,
            prompt: prompt.into(),
            temperature: 0.4,
            max_tokens: None,
        }
    }
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
    /// Setzt die Token-Obergrenze der Antwort (Ollama: `num_predict`).
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }
}

/// Gemeinsame Schnittstelle aller Modelle. Pipeline-Stages hängen nur hiervon ab.
#[async_trait]
pub trait LanguageModel: Send + Sync {
    async fn complete(&self, req: &Completion) -> Result<String>;
    fn label(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Ollama (lokal)
// ---------------------------------------------------------------------------

/// Wie lange Ollama das Modell nach einer Anfrage geladen hält. Vermeidet teures Neuladen
/// (~15-20 GB) zwischen Enrich-/Synth-Stage und über Prozessgrenzen (`brief`→`eval`→`learn`).
const DEFAULT_KEEP_ALIVE: &str = "30m";
/// Kontextfenster-Obergrenze. Verhindert, dass Ollama das (u.U. riesige) Modell-Maximum
/// vorhält; großzügig genug für Briefing-Prompts.
const DEFAULT_NUM_CTX: u32 = 8192;
/// Gesamt-Timeout je Anfrage. Bounded gegen hängende Inferenz im nächtlichen Batch
/// (sonst blockiert ein einziger Hänger den ganzen Lauf), aber großzügig für große Modelle.
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// Timeout für den reinen Verbindungsaufbau (Ollama nicht erreichbar → schnell scheitern).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

pub struct OllamaClient {
    base_url: String,
    model: String,
    label: String,
    keep_alive: String,
    num_ctx: u32,
    http: reqwest::Client,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(DEFAULT_REQUEST_TIMEOUT)
            .build()
            .expect("reqwest-Client bauen (nur TLS-Init kann scheitern)");
        Self {
            base_url: base_url.into(),
            label: format!("ollama:{model}"),
            model,
            keep_alive: DEFAULT_KEEP_ALIVE.to_string(),
            num_ctx: DEFAULT_NUM_CTX,
            http,
        }
    }

    /// Überschreibt, wie lange Ollama das Modell geladen hält (z.B. "60m", "0" = sofort entladen).
    pub fn with_keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = keep_alive.into();
        self
    }

    /// Überschreibt die Kontextfenster-Obergrenze (`num_ctx`).
    pub fn with_num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = num_ctx;
        self
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
    keep_alive: &'a str,
    options: ChatOptions,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatOptions {
    temperature: f32,
    num_ctx: u32,
    /// Max. zu generierende Tokens; `-1` = unbegrenzt (Ollama-Default), sonst die Deckelung.
    num_predict: i32,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[async_trait]
impl LanguageModel for OllamaClient {
    async fn complete(&self, req: &Completion) -> Result<String> {
        let mut messages = Vec::new();
        if let Some(sys) = &req.system {
            messages.push(ChatMessage {
                role: "system",
                content: sys.as_str(),
            });
        }
        messages.push(ChatMessage {
            role: "user",
            content: req.prompt.as_str(),
        });

        let body = ChatRequest {
            model: self.model.as_str(),
            messages,
            stream: false,
            keep_alive: self.keep_alive.as_str(),
            options: ChatOptions {
                temperature: req.temperature,
                num_ctx: self.num_ctx,
                // None → -1 (Ollama-Default: unbegrenzt), erhält das bisherige Verhalten.
                num_predict: req.max_tokens.map(|n| n as i32).unwrap_or(-1),
            },
        };

        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("Ollama-Request an {url} fehlgeschlagen"))?
            .error_for_status()
            .context("Ollama antwortete mit Fehlerstatus")?;

        let parsed: ChatResponse = resp.json().await.context("Ollama-Antwort dekodieren")?;
        Ok(parsed.message.content)
    }

    fn label(&self) -> &str {
        &self.label
    }
}

// ---------------------------------------------------------------------------
// Claude Code (Abo-basiert) — für periodische Kalibrierung des lokalen Judge.
//
// Ruft die Claude-Code-CLI im Headless-Modus auf:
//     claude -p "<prompt>" --output-format json [--model <model>]
// und rechnet damit gegen das vorhandene Abo statt per-Token-API.
//
// Bewusst niederfrequent einsetzen (z.B. wöchentlich) wegen Fair-Use/Rate-Limits.
// ---------------------------------------------------------------------------

pub struct ClaudeCodeModel {
    binary: String,
    model: Option<String>,
    label: String,
    timeout: Duration,
}

impl ClaudeCodeModel {
    /// `model` = optionaler Modell-Slug (z.B. "opus"); None nutzt den CLI-Default.
    pub fn new(model: Option<String>) -> Self {
        Self {
            binary: "claude".into(),
            model,
            label: "claude-code(abo)".into(),
            timeout: Duration::from_secs(120),
        }
    }
}

#[derive(Deserialize)]
struct ClaudeResult {
    result: String,
}

#[async_trait]
impl LanguageModel for ClaudeCodeModel {
    async fn complete(&self, req: &Completion) -> Result<String> {
        // System- und User-Prompt zusammenführen (CLI hat kein separates System-Flag).
        let prompt = match &req.system {
            Some(sys) => format!("{sys}\n\n{}", req.prompt),
            None => req.prompt.clone(),
        };

        let mut cmd = tokio::process::Command::new(&self.binary);
        cmd.arg("-p")
            .arg(&prompt)
            .arg("--output-format")
            .arg("json");
        if let Some(m) = &self.model {
            cmd.arg("--model").arg(m);
        }

        let out = tokio::time::timeout(self.timeout, cmd.output())
            .await
            .with_context(|| {
                format!(
                    "`{}` (Claude Code CLI) timed out after {:?}",
                    self.binary, self.timeout
                )
            })??;

        if !out.status.success() {
            anyhow::bail!(
                "claude beendet mit {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
        }

        let parsed: ClaudeResult = serde_json::from_slice(&out.stdout)
            .context("`claude --output-format json` parsen (Feld `result`)")?;
        Ok(parsed.result)
    }

    fn label(&self) -> &str {
        &self.label
    }
}
