//! LLM-Gateway: ein Trait, mehrere austauschbare Backends.
//!
//! - [`OllamaClient`]  — lokales Standard-Backend (kostenlos).
//! - [`ClaudeCodeModel`] — Abo-basierte Kalibrierung via `claude -p` (Claude Code CLI).
//!
//! Weitere Backends (Codex-CLI, direkte API) implementieren einfach [`LanguageModel`].

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Eine Anfrage an ein Sprachmodell.
#[derive(Debug, Clone)]
pub struct Completion {
    pub system: Option<String>,
    pub prompt: String,
    pub temperature: f32,
}

impl Completion {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            system: None,
            prompt: prompt.into(),
            temperature: 0.4,
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

pub struct OllamaClient {
    base_url: String,
    model: String,
    label: String,
    http: reqwest::Client,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        Self {
            base_url: base_url.into(),
            label: format!("ollama:{model}"),
            model,
            http: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
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
            options: ChatOptions {
                temperature: req.temperature,
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
}

impl ClaudeCodeModel {
    /// `model` = optionaler Modell-Slug (z.B. "opus"); None nutzt den CLI-Default.
    pub fn new(model: Option<String>) -> Self {
        Self {
            binary: "claude".into(),
            model,
            label: "claude-code(abo)".into(),
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

        let out = cmd.output().await.with_context(|| {
            format!(
                "`{}` (Claude Code CLI) konnte nicht gestartet werden",
                self.binary
            )
        })?;

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
