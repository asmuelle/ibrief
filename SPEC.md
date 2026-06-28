# ibrief — Spezifikation

> Ein selbstverbesserndes, personalisiertes Morning Briefing.
> Es lernt jeden Tag dazu: bessere Quellen, bessere Kuratierung, bessere Prompts.

**Version:** 0.1 (Entwurf)
**Datum:** 2026-06-28
**Status:** Spezifikation — noch keine Implementierung

---

## 1. Vision & Ziel

`ibrief` erstellt jede Nacht ein kuratiertes Briefing, das auf eine konkrete Person zugeschnitten ist (Rollen, Interessen, Ziele, Werte). Es ersetzt das morgendliche, ziellose Scrollen durch **5 Minuten Signal statt Stunden Rauschen**.

Der entscheidende Unterschied zu einer News-App: Das System **misst, was nützlich war, und passt sich an**. Es betreibt einen täglichen Lernzyklus über drei Hebel:

1. **Personalisierung** — Gewichtung von Themen/Quellen anhand echten Nutzungsfeedbacks.
2. **Prompt-Optimierung** — die Anweisungen an das LLM (Kuratierung, Synthese, Tonalität) werden selbst optimiert.
3. **Quellen-Evolution** — neue Quellen werden vorgeschlagen, schwache aussortiert.

### Nicht-Ziele (v1)

- Kein autonomes Umschreiben des eigenen Anwendungscodes (zu riskant; siehe §9).
- Kein Echtzeit-Newsfeed über den Tag — bewusst ein Batch-Produkt pro Tag.
- Keine Multi-User-Plattform — zunächst Single-User („Owner").

---

## 1.1 Designphilosophie (LLM-OS / gated AutoResearch)

`ibrief` folgt bewusst einer „LLM-OS"-Haltung (im Sinne von Karpathys Bild vom LLM als
austauschbarem Prozessor-Kern), nicht der eines vollautonomen Agenten:

1. **Das Gedächtnis ist das Produkt, nicht das Modell.**
   Der Wert liegt im persistenten, versionierten Zustand *außerhalb* des Kontextfensters —
   Profil, gelernte Gewichte, Prompt-Varianten, Eval-Historie (§6, §8). Das LLM ist ein
   austauschbarer „Rechenkern"; der kuratierte Zustand ist das eigentliche „zweite Gehirn".
   Konsequenz: kein Lock-in auf ein Modell; Config überlebt Modellwechsel.

2. **Autonomie auf der Leine (autonomy slider).**
   Selbstverbesserung heißt hier *vorschlagen → gegen reales Feedback verifizieren → absichern
   → übernehmen → zurückrollen können* — niemals blindes Selbst-Optimieren. Gelernt werden nur
   **Daten-Artefakte** (Gewichte, Prompts, Quellen), nie Anwendungscode (§9). Der Mensch bleibt
   in der Verifikationsschleife; wesentliche Änderungen sind transparent und per Klick reversibel.

3. **Schnelle Generierung–Verifikation-Loops.**
   Jede vorgeschlagene Änderung durchläuft eine billige Verifikation (Schatten-/Backtest +
   Safety Gate, §6.4/§6.5), bevor sie Default wird. Verifikation ist günstiger als Generierung —
   deshalb wird viel vorgeschlagen, aber wenig (und nur Geprüftes) übernommen.

4. **Engagement ist nicht das Optimierungsziel.**
   Ein naiver Engagement-Optimierer macht den Owner *bequemer, aber dümmer*. Das System opfert
   bewusst kurzfristigen Engagement-Score, um Diversität und Anti-Blase-Invarianten zu halten
   (§3, §9 Drift-Wächter). „Nützlich für Urteilskraft" schlägt „angenehm zu klicken".

5. **AutoResearch nur *gated*.**
   Tiefe, mehrstufige Recherche (Karpathy-Style Deep-Research-Loops) ist mächtig, aber teuer und
   halluzinationsanfällig. Sie ist daher ein **optionales, budgetiertes, verifiziertes Modul**
   (§14), kein durchgängig laufender autonomer Agent.

---

## 2. Nutzerprofil (Beispiel-Owner)

Das System ist profil-getrieben. Das Profil ist Daten, kein Code.

```yaml
owner:
  rollen: [softwareentwickler, ki-unternehmer, vater, buerger]
  werte: { politisch: links-demokratisch, anti-filterblase: true }
  ziele:
    - weiterentwicklung
    - am-ball-bleiben (KI/Tech)
    - gute-entscheidungen
    - interessante-gespraeche
  sprache: de
  briefing_zeit: "06:30 Europe/Zurich"
  lesezeit_budget_min: 5
```

---

## 3. Inhaltsmodell — Sektionen des Briefings

Jede Sektion ist ein Plugin mit definiertem Vertrag (§5.2). Default-Set:

| ID | Sektion | Zweck | Default-Gewicht |
|----|---------|-------|-----------------|
| `tldr` | Die 3 Dinge heute | Executive Summary, immer oben | fix |
| `ai_tech` | KI & Tech | Capability-Shifts, Releases, Paper des Tages | hoch |
| `business` | Unternehmer-Radar | Markt, Wettbewerb, strategische Frage | mittel |
| `world` | Welt & Politik | Nachrichten mit Einordnung | mittel |
| `counterpoint` | Gegenperspektive | seriöse Stimme gegen die eigene Position | hoch (anti-blase) |
| `culture` | Kultur | Buch/Film/Essay/Album als Gesprächsstoff | niedrig |
| `growth` | 1%-Block | ein Lern-Happen + Reflexionsfrage | mittel |
| `personal` | Heute persönlich | Kalender, Beziehungs-Nudges | hoch |
| `conversation` | Gesprächs-Köder | 3 Gesprächseinstiege | mittel |
| `wildcard` | Wildcard | bewusst überraschend, außerhalb der Interessen | fix-niedrig |

**Invarianten:**
- Jeder Eintrag trägt ein Feld `warum_relevant` (1 Satz, Personalisierungsbezug).
- Gesamtlänge ≤ Lesezeit-Budget (Token-Schätzung → Wörter → Minuten).
- `counterpoint` und `wildcard` sind **nicht** durch Personalisierung abschaltbar (Schutz gegen Echo-Kammer).

---

## 4. Architektur

### 4.1 Pipeline (nächtlicher Lauf)

```
                ┌─────────────┐
   Quellen ──▶  │  INGEST     │  Fetch, Dedup, Normalisierung → ContentPool
                └─────┬───────┘
                      ▼
                ┌─────────────┐
                │  ENRICH     │  Zusammenfassen, Themen-Tags, Entitäten, Embeddings
                └─────┬───────┘
                      ▼
                ┌─────────────┐
                │  SCORE      │  Relevanz × Owner-Gewichte × Neuheit × Diversität
                └─────┬───────┘
                      ▼
                ┌─────────────┐
                │  CURATE     │  Pro Sektion Top-N wählen, Synthese-LLM, warum_relevant
                └─────┬───────┘
                      ▼
                ┌─────────────┐
                │  RENDER     │  Markdown/HTML/E-Mail/Push, „Die 3 Dinge"
                └─────┬───────┘
                      ▼
                   Auslieferung  ──▶  Owner
                      │
                      ▼ (über den Tag)
                ┌─────────────┐
                │  FEEDBACK   │  Öffnungen, Klicks, 👍/👎, Verweildauer, „mehr/weniger davon"
                └─────┬───────┘
                      ▼
                ┌─────────────┐
                │  LEARN      │  Eval → Vorschläge → Gate → Anwenden (siehe §6)
                └─────────────┘
```

### 4.2 Komponenten

- **Scheduler** — triggert die Pipeline (z.B. 02:00) und den Lernlauf (z.B. nach Auslieferung + Feedbackfenster).
- **Source Registry** — versionierte Liste aktiver Quellen mit Metadaten & Qualitätsscore.
- **Content Store** — normalisierte Items + Embeddings (für Dedup/Diversität/Neuheit).
- **Config Store** — *die lernbaren Artefakte*: Gewichte, Prompts, Quellenliste. Versioniert (§8).
- **LLM Gateway** — modell-agnostisch, kostenbewusstes Routing (§7).
- **Eval Engine** — bewertet Briefing-Qualität (§6.2).
- **Feedback Collector** — sammelt explizite + implizite Signale.

---

## 5. Datenverträge

### 5.1 ContentItem

```ts
interface ContentItem {
  id: string;              // stabiler Hash aus url+title
  source_id: string;
  url: string;
  title: string;
  published_at: string;    // ISO
  raw_text: string;
  summary?: string;        // von ENRICH
  topics: string[];        // Tags
  entities: string[];
  embedding?: number[];
  novelty: number;         // 0..1, vs. letzte N Tage
  scores: Record<string,number>; // pro Sektion
}
```

### 5.2 Section-Plugin-Vertrag

```ts
interface SectionPlugin {
  id: string;
  selectCandidates(pool: ContentItem[], ctx: OwnerContext): ContentItem[];
  curate(candidates: ContentItem[], ctx: OwnerContext): Promise<SectionOutput>;
  // Lernbare Parameter werden aus dem Config Store injiziert:
  params: { weight: number; prompt_id: string; max_items: number };
}
```

### 5.3 BriefingRecord (für Lernen wichtig)

```ts
interface BriefingRecord {
  date: string;
  config_version: string;     // welche Gewichte/Prompts/Quellen aktiv waren
  sections: SectionOutput[];
  items_shown: string[];      // ContentItem-IDs
  feedback: FeedbackEvent[];  // füllt sich über den Tag
  eval?: EvalResult;          // §6.2
}
```

---

## 6. Der Selbstverbesserungs-Mechanismus (Kern)

> Prinzip: **Vorschlagen → Bewerten → Absichern → Anwenden → Zurückrollen können.**
> Niemals blind optimieren. Jede Änderung ist versioniert und reversibel.

### 6.1 Feedback-Signale

| Signal | Typ | Gewicht |
|--------|-----|---------|
| 👍 / 👎 pro Eintrag | explizit | hoch |
| „mehr/weniger davon" (Thema/Quelle) | explizit | hoch |
| Link geöffnet | implizit | mittel |
| Verweildauer pro Sektion | implizit | mittel |
| Eintrag in Gesprächs-Köder genutzt (manuell markiert) | explizit | hoch |
| Briefing gar nicht geöffnet | implizit | negativ global |

Implizite Signale werden konservativ gewichtet (Klick ≠ Wert). Explizites Feedback dominiert.

### 6.2 Eval Engine — „Was ist ein gutes Briefing?"

Tägliche Bewertung als Zahl + Diagnose. Drei Quellen kombiniert:

1. **Verhaltens-Score** — aggregiertes Feedback (§6.1), normalisiert.
2. **LLM-as-Judge** — ein separater Modell-Lauf bewertet das Briefing gegen eine Rubrik (Relevanz, Neuheit, Diversität, Anti-Blase, Prägnanz, Handlungsnähe). Rubrik liegt versioniert im Config Store.
3. **Strukturelle Checks (deterministisch)** — Lesezeit eingehalten? `counterpoint`/`wildcard` vorhanden? Dedup ok? Keine Quelle dominiert (>X%)?

```
eval_score = w1·verhalten + w2·judge + w3·struktur   // w aus Config, default 0.5/0.3/0.2
```

Wichtig: Der **Verhaltens-Score ist die Ground Truth**; LLM-Judge dient nur, wenn Feedback dünn ist (Cold-Start), und wird gegen reales Feedback kalibriert.

### 6.3 Optimizer — was wird gelernt

Drei lernbare Artefakte, jeweils mit eigener Strategie:

**A) Gewichte (Bandit-Ansatz)**
- Themen-/Quellen-/Sektionsgewichte als Multi-Armed-Bandit (Thompson Sampling).
- Exploration eingebaut → verhindert vorzeitiges Festfahren auf wenige Quellen (Anti-Blase auf Mechanismus-Ebene).
- Update: täglich, kleine Lernrate, mit Decay (alte Vorlieben verblassen).

**B) Prompts (Meta-Prompt-Optimierung)**
- Bei wiederholt schwachen Sektionen erzeugt ein „Optimizer-LLM" Prompt-Varianten (vgl. Ansatz `prompt-optimizer`).
- Neue Variante läuft als **Schatten-/A-B-Test** (§6.5), bevor sie Default wird.

**C) Quellen-Evolution**
- Vorschlag neuer Quellen: aus oft geöffneten Links extrahierte Domains, aus Owner-„mehr davon", aus LLM-Recherche.
- Aussortieren: Quellen mit dauerhaft niedrigem Beitrag (selten ausgewählt / oft 👎) werden deaktiviert (nicht gelöscht).
- Jede aktive Quelle hat einen rollierenden Qualitätsscore.

### 6.4 Safety Gate (verpflichtend vor jedem Übernehmen)

Eine vorgeschlagene Änderung wird nur Default, wenn **alle** Gates passen:

1. **Eval nicht verschlechtert** — neue Config ≥ alte Config (mit Signifikanz-/Mindestbeobachtungsschwelle).
2. **Invarianten intakt** — counterpoint/wildcard vorhanden, Quellen-Diversität ≥ Schwelle, Lesezeit ok.
3. **Kosten im Budget** — §7.
4. **Kein Quellen-Monokultur-Drift** — Herfindahl-Index der Quellenverteilung unter Grenzwert.
5. **Owner-Veto möglich** — wesentliche Änderungen werden im Briefing transparent gemacht („ibrief hat X angepasst, weil …"); Owner kann mit einem Klick zurückrollen.

Schlägt ein Gate fehl → Rollback auf vorherige Config-Version, Vorfall wird geloggt.

### 6.5 A/B- & Schatten-Tests

- **Schattenlauf:** Variante wird parallel berechnet, aber nicht ausgeliefert; nur vom LLM-Judge bewertet (kein Risiko, kostet Tokens).
- **A/B über Zeit:** an Tag-Slices alternierend ausspielen, Verhaltensfeedback vergleichen. Single-User → Tage als Stichprobe, daher langsame, vorsichtige Übernahme (Mindest-N Tage).

### 6.6 Lern-Lebenszyklus (täglich)

```
1. Briefing von gestern + Feedbackfenster abgeschlossen
2. Eval Engine bewertet → EvalResult
3. Optimizer erzeugt Kandidaten-Änderungen (Gewichte/Prompts/Quellen)
4. Kandidaten im Schatten-/Backtest bewerten
5. Safety Gate prüfen
6a. PASS → neue Config-Version aktivieren, Changelog schreiben
6b. FAIL → verwerfen/zurückrollen, Grund loggen
7. Wochen-Report an Owner: was wurde gelernt, was verworfen
```

---

## 7. Kosten & Modell-Routing

- **Routing nach Aufgabe:** günstiges/schnelles Modell (z.B. Haiku-Klasse) für Zusammenfassen/Tagging im Mengengeschäft; starkes Modell (Opus/Sonnet-Klasse) für Synthese, LLM-Judge und Prompt-Optimierung.
- **Prompt-Caching** für stabile Profil-/Rubrik-Teile.
- **Tägliches Token-Budget** als harte Grenze; Schattenläufe zählen mit. Bei Überschreitung wird Optimierung (nicht das Briefing) gedrosselt.
- Embeddings lokal/günstig; Dedup deterministisch vor LLM-Aufrufen (spart Tokens).
- **Lokal-first:** Massen- *und* Synthese-Tier laufen lokal (Ollama auf M4 Max). Laufender Betrieb ~0 € (nur Strom) → der gesamte nächtliche Lernzyklus ist kostenfrei.
- **Abo-basierte Kalibrierung:** Der periodische Frontier-Judge (§6.2) läuft über vorhandene Abos via CLI — `claude -p` (Claude Code) bzw. Codex-CLI — als Subprozess statt per-Token-API. Bei Flatrate günstiger; bewusst niederfrequent (z.B. wöchentlich) wegen Fair-Use/Rate-Limits. Backend ist über den `LanguageModel`-Trait (§11) austauschbar.

---

## 8. Persistenz & Versionierung

- **Config als versioniertes Artefakt** (Git-artig oder Tabelle mit `version`, `parent`, `diff`, `created_by`, `reason`).
- Jeder `BriefingRecord` referenziert seine `config_version` → volle Nachvollziehbarkeit, welche Einstellung welches Ergebnis brachte.
- **Rollback** = Aktivieren einer früheren Version (atomar).
- Datenhaltung: lokal-first (SQLite + Files) genügt für Single-User; Quellen-Rohdaten nach N Tagen prunen, Embeddings/Eval-Historie behalten.

---

## 9. Sicherheit & Leitplanken

- **Kein Self-Code-Rewrite in v1.** Gelernt werden ausschließlich *Daten-Artefakte* (Gewichte, Prompts, Quellenliste), nie der Anwendungscode. Das hält das System vorhersehbar und auditierbar.
- **Prompt-Injection-Schutz:** Quelleninhalte sind *Daten*, nie Instruktionen. ENRICH/CURATE behandeln Fremdtext sandboxed; klare System/User/Content-Trennung im LLM Gateway.
- **Quellen-Whitelist & SSRF-Schutz** beim Fetch; Ratelimits, Timeouts.
- **Datenschutz:** Kalender/persönliche Daten bleiben lokal; keine Weitergabe an Dritt-Quellen. Secrets nur via Env/Secret-Manager.
- **Transparenz:** Owner sieht jederzeit, *warum* ein Eintrag erschien und *was* das System geändert hat.
- **Drift-Wächter:** Wenn die Quellen-Diversität über Wochen sinkt, erzwingt das System Exploration (Schutz vor selbstverstärkender Blase) — auch wenn das den kurzfristigen Eval-Score senkt.

---

## 10. Auslieferung & Schnittstellen

- **Kanäle:** E-Mail (HTML), Markdown-Datei, optional Push/Telegram. Ein Web-View mit Feedback-Buttons (👍/👎, „mehr/weniger").
- **Feedback-API:** jeder Eintrag hat verlinkte Feedback-Aktionen (auch aus der E-Mail per One-Click-Link).
- **Owner-Kommandos:** „mehr KI-Hardware", „weniger Krypto", „Quelle X raus", „rollback gestern".

---

## 11. Tech-Stack (Rust, local-first)

Ausgelegt auf lokalen Betrieb (M4 Max, 128 GB) mit lokalen LLMs; Cloud nur optional zur Kalibrierung.

| Schicht | Wahl | Alternative / später |
|---|---|---|
| Sprache/Runtime | **Rust + Tokio** | — |
| LLM-Bedienung | **Ollama** (OpenAI-kompatibel, Metal) | mistral.rs / Candle (in-process, reines Rust-Binary) |
| LLM-Framework | dünner `reqwest`-Client hinter `LanguageModel`-Trait | `rig` (Agents/RAG/Tools) ab M6/AutoResearch |
| Lokale Modelle | Massen-Tier: Qwen2.5 14B / Llama 3.1 8B · Synthese/Judge: Llama 3.3 70B / Qwen2.5 72B / Mistral Large 123B (Q4) | — |
| Embeddings | `fastembed-rs` (rein Rust, ONNX) | Ollama-Embeddings (nomic/mxbai) |
| Daten + Vektor | **SQLite + sqlite-vec** (`sqlx`/`rusqlite`) | LanceDB (embedded) |
| Ingest | `reqwest` + `feed-rs` (RSS/Atom) + `scraper` + `readability` | — |
| Pipeline | plain async Rust Stages | Graph-Lib erst bei Bedarf |
| Web/API + Feedback | `axum` (+ Maud/Askama) | Leptos/Dioxus (Rust→WASM) |
| Scheduling | macOS `launchd` (`StartCalendarInterval`) | `tokio-cron-scheduler` |
| Delivery | `lettre` (SMTP) / Resend · `teloxide` (Telegram: Push + Feedback-Buttons) | — |
| Config-Versionierung | `serde` + TOML im Git-Repo (`config_version` = Commit-Hash) | DB-Tabelle |
| Observability | `tracing` (+ OpenTelemetry) | Langfuse via HTTP |

### LLM-Gateway als Trait

Alle Modelle liegen hinter einem `LanguageModel`-Trait → Backends sind ohne Pipeline-Änderung austauschbar:

- `OllamaClient` — lokales Standard-Backend (kostenlos, Massen- und Synthese-Tier).
- `ClaudeCodeModel` / `CodexModel` — **Abo-basierte Kalibrierung** (§7): ruft `claude -p --output-format json` bzw. das Codex-CLI als Subprozess auf und rechnet gegen das vorhandene Abo statt per-Token-API.
- `ApiModel` — direkter Anthropic/OpenAI-API-Call (optional, falls je nötig).

### Hosting / Betrieb

Single-Binary, lokal. Da der Mac nachts schläft: Briefing per `launchd` beim morgendlichen Aufwachen erzeugen (verpasste Jobs laufen beim Wake nach), Lernlauf direkt danach. Kein Server nötig.

---

## 12. Meilensteine

| Phase | Inhalt | Ergebnis |
|-------|--------|----------|
| **M1 — Statisches Briefing** | Ingest → Enrich → Curate → Render, feste Gewichte/Prompts | Tägliches Briefing, manuell justiert |
| **M2 — Feedback** | Feedback Collector + BriefingRecord + Web-View | Signale werden gesammelt |
| **M3 — Eval Engine** | Verhaltens-Score + LLM-Judge + Strukturchecks | Briefing bekommt täglich eine Note |
| **M4 — Lernen (Gewichte)** | Bandit-Optimierung + Safety Gate + Versionierung | System personalisiert sich messbar |
| **M5 — Prompt-Optimierung** | Schatten-/A-B-Tests für Prompts | Kuratierungsqualität steigt selbsttätig |
| **M6 — Quellen-Evolution + AutoResearch** | Quellen-Vorschlag/-Pruning + Drift-Wächter + gated Deep-Research-Modul (§14) | System erweitert seinen eigenen Horizont, recherchiert Deep Dives selbst |

Jeder Meilenstein ist eigenständig nützlich — kein Big-Bang.

---

## 13. Erfolgskriterien

- **Engagement:** Öffnungsrate und 👍-Quote steigen über die ersten ~4 Wochen messbar.
- **Eval-Trend:** gleitender Eval-Score steigt, ohne dass Diversität fällt.
- **Subjektiv:** der Owner findet pro Briefing ≥1 Sache, die ein Gespräch oder eine Entscheidung beeinflusst.
- **Selbstverbesserung verifiziert:** mind. eine vom System übernommene Änderung pro Woche, mit dokumentiertem Vorher/Nachher — und nachweislich funktionierendem Rollback.

---

## 14. AutoResearch-Modul (M6, optional & gated)

> Karpathy-Style Deep-Research-Loop — aber budgetiert, mit Stoppkriterien und Pflicht-Verifikation.
> Niemals durchgängig laufender autonomer Agent; immer durch Budget und Gate begrenzt.

### 14.1 Einsatzzwecke

Genau zwei Trigger, sonst läuft das Modul nicht:

1. **Deep Dive des Tages** — *ein* Thema (aus `ai_tech`/`growth`, höchstgerankt oder vom Owner
   angefragt) wird mehrstufig durchdrungen statt nur angerissen.
2. **Quellen-Discovery** — Suche nach neuen hochwertigen Quellen für die Quellen-Evolution
   (§6.3C), z.B. wenn der Drift-Wächter Exploration verlangt.

### 14.2 Loop-Architektur

```
PLAN ──▶ SEARCH ──▶ READ ──▶ EXTRACT ──▶ REFLECT ──┐
  ▲                                                 │
  └──────────  offene Frage übrig? ja  ◀────────────┘
                       │ nein / Stoppkriterium
                       ▼
                  SYNTHESIZE ──▶ VERIFY ──▶ Ergebnis (oder Verwerfen)
```

- **PLAN** — zerlegt die Forschungsfrage in Teilfragen (vom Modell), legt max. Tiefe fest.
- **SEARCH** — Web-/API-Suche je Teilfrage (Whitelist + SSRF-Schutz wie §9).
- **READ/EXTRACT** — Quelltext laden, relevante Claims + Belegstellen extrahieren (als *Daten*,
  nie als Instruktionen behandeln — Prompt-Injection-Schutz §9).
- **REFLECT** — „Was ist beantwortet? Welche Lücke bleibt? Lohnt eine weitere Iteration?"
- **SYNTHESIZE** — Zusammenfassung **mit Quellenangaben pro Aussage** (Pflicht).
- **VERIFY** — separater Modell-/Regel-Lauf: Deckt jede Kernaussage eine zitierte Quelle?
  Widersprüche? Unbelegte Behauptungen werden markiert oder entfernt.

### 14.3 Stoppkriterien (ODER-verknüpft — was zuerst greift)

| Kriterium | Default |
|-----------|---------|
| Max. Iterationen (REFLECT-Runden) | 3 |
| Max. gelesene Quellen | 12 |
| Max. Wall-Clock | 5 min |
| Max. Token-Budget für diesen Lauf | aus Tagesbudget reserviertes Kontingent (§7) |
| Keine *neuen* Erkenntnisse mehr (Novelty der letzten Runde < Schwelle) | early stop |
| Teilfragen alle beantwortet | early stop |

Kein Kriterium erfüllt + Budget erschöpft → **kontrollierter Abbruch**, liefert Teilergebnis
mit Hinweis „unvollständig", statt weiterzulaufen.

### 14.4 Output-Vertrag

```ts
interface ResearchResult {
  question: string;
  answer_md: string;            // Synthese, kurz, lesezeit-konform
  claims: { text: string; sources: string[] }[]; // jede Aussage belegt
  sources_used: string[];
  unverified_claims: string[];  // von VERIFY markiert, NICHT ins Briefing
  iterations: number;
  tokens_spent: number;
  status: "complete" | "partial" | "aborted";
}
```

- Nur **verifizierte** Claims dürfen ins Briefing. `unverified_claims` werden geloggt, nicht ausgespielt.
- Bei `status != complete` wird im Briefing transparent „vorläufig" gekennzeichnet.

### 14.5 Verifikation & Gate (zusätzlich zu §6.4)

- **Beleg-Pflicht:** Jede ins Briefing übernommene Aussage hat ≥1 zitierte, erreichbare Quelle.
- **Cross-Check:** Bei strittigen/politischen Themen ≥2 unabhängige Quellen, möglichst aus
  unterschiedlichen Lagern (verstärkt das Anti-Blase-Ziel statt es zu untergraben).
- **Kostendeckel:** Überschreitet ein Lauf sein reserviertes Token-Kontingent, wird er beendet —
  das *Briefing selbst* hat immer Vorrang vor AutoResearch (§7).
- **Owner-Override:** Owner kann „grab dich tiefer rein zu X" anstoßen (höheres Budget,
  explizit autorisiert) oder AutoResearch ganz deaktivieren.

### 14.6 Bewusste Grenzen

- Kein Schreibzugriff irgendwohin; AutoResearch **liest und synthetisiert nur**.
- Keine Aktionen in der Welt (keine Mails, keine Posts) — reines Erkenntnis-Modul.
- Discovery schlägt Quellen nur **vor**; Aufnahme läuft durch das normale Safety Gate (§6.4).
```
