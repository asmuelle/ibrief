//! Kontrolliertes Topic-Vokabular (§T2.3).
//!
//! Freiform-Tags fragmentieren die Bandit-Arme: "KI", "Künstliche Intelligenz" und "AI"
//! lernen als getrennte Arme und akkumulieren nie gemeinsame Evidenz — bei Single-User-
//! Feedback-Volumen konvergiert das Topic-Lernen so nicht. Deshalb: feste Taxonomie im
//! Enrich-Prompt + Normalisierung der Modell-Ausgabe auf kanonische Topics. Tags außerhalb
//! des Vokabulars werden verworfen (leere Topics ⇒ neutrales Gewicht 1.0 in `rank`).

/// Kanonisches Topic-Vokabular — die einzigen gültigen Bandit-Arme fürs Topic-Lernen.
/// Bewusst klein (~20), damit Evidenz sich konzentriert. Erweiterungen hier UND im
/// Alias-Mapping ergänzen; bestehende Arme nie umbenennen (Config-Versionen referenzieren sie).
pub const TOPICS: &[&str] = &[
    "ki",
    "llm",
    "agents",
    "ml-forschung",
    "chips",
    "hardware",
    "software-engineering",
    "programmiersprachen",
    "open-source",
    "cloud",
    "sicherheit",
    "datenschutz",
    "politik-regulierung",
    "big-tech",
    "startups",
    "wirtschaft",
    "robotik",
    "raumfahrt",
    "wissenschaft",
    "energie",
    "medien",
    "gaming",
    "krypto",
    "gesellschaft",
];

/// Häufige Synonyme/Sprachvarianten → kanonisches Topic. Lowercase-Schlüssel.
const ALIASES: &[(&str, &str)] = &[
    ("ai", "ki"),
    ("künstliche intelligenz", "ki"),
    ("artificial intelligence", "ki"),
    ("genai", "ki"),
    ("generative ki", "ki"),
    ("large language model", "llm"),
    ("large language models", "llm"),
    ("sprachmodell", "llm"),
    ("sprachmodelle", "llm"),
    ("chatbot", "llm"),
    ("agent", "agents"),
    ("agenten", "agents"),
    ("ki-agenten", "agents"),
    ("machine learning", "ml-forschung"),
    ("maschinelles lernen", "ml-forschung"),
    ("deep learning", "ml-forschung"),
    ("halbleiter", "chips"),
    ("semiconductor", "chips"),
    ("gpu", "chips"),
    ("gpus", "chips"),
    ("prozessor", "chips"),
    ("software", "software-engineering"),
    ("softwareentwicklung", "software-engineering"),
    ("entwicklung", "software-engineering"),
    ("programmierung", "programmiersprachen"),
    ("rust", "programmiersprachen"),
    ("python", "programmiersprachen"),
    ("opensource", "open-source"),
    ("quelloffen", "open-source"),
    ("security", "sicherheit"),
    ("cybersecurity", "sicherheit"),
    ("cybersicherheit", "sicherheit"),
    ("it-sicherheit", "sicherheit"),
    ("privacy", "datenschutz"),
    ("privatsphäre", "datenschutz"),
    ("politik", "politik-regulierung"),
    ("regulierung", "politik-regulierung"),
    ("eu", "politik-regulierung"),
    ("gesetz", "politik-regulierung"),
    ("openai", "big-tech"),
    ("anthropic", "big-tech"),
    ("google", "big-tech"),
    ("microsoft", "big-tech"),
    ("apple", "big-tech"),
    ("meta", "big-tech"),
    ("amazon", "big-tech"),
    ("nvidia", "big-tech"),
    ("startup", "startups"),
    ("gründer", "startups"),
    ("venture capital", "startups"),
    ("ökonomie", "wirtschaft"),
    ("economy", "wirtschaft"),
    ("finanzen", "wirtschaft"),
    ("roboter", "robotik"),
    ("robotics", "robotik"),
    ("weltraum", "raumfahrt"),
    ("space", "raumfahrt"),
    ("astronomie", "raumfahrt"),
    ("forschung", "wissenschaft"),
    ("science", "wissenschaft"),
    ("strom", "energie"),
    ("energy", "energie"),
    ("klima", "energie"),
    ("presse", "medien"),
    ("journalismus", "medien"),
    ("spiele", "gaming"),
    ("videospiele", "gaming"),
    ("kryptowährung", "krypto"),
    ("bitcoin", "krypto"),
    ("blockchain", "krypto"),
    ("kultur", "gesellschaft"),
    ("arbeit", "gesellschaft"),
];

/// Vokabular als kommaseparierte Liste (für den Enrich-Prompt).
pub fn vocabulary_list() -> String {
    TOPICS.join(", ")
}

/// Normalisiert EIN Tag auf sein kanonisches Topic. `None` = außerhalb des Vokabulars.
pub fn normalize_topic(raw: &str) -> Option<&'static str> {
    let t = raw.trim().to_lowercase();
    let t = t.trim_matches(|c: char| !c.is_alphanumeric());
    if t.is_empty() {
        return None;
    }
    if let Some(canonical) = TOPICS.iter().find(|k| **k == t) {
        return Some(canonical);
    }
    ALIASES
        .iter()
        .find(|(alias, _)| *alias == t)
        .map(|(_, canonical)| *canonical)
}

/// Normalisiert die Tag-Liste eines Items: kanonisch, dedupliziert (Reihenfolge stabil),
/// max. 3. Unbekannte Tags fallen heraus — lieber kein Topic als ein toter Arm.
pub fn normalize_topics(raw: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(3);
    for r in raw {
        if let Some(c) = normalize_topic(r)
            && !out.iter().any(|x| x == c)
        {
            out.push(c.to_string());
            if out.len() == 3 {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_collapse_to_canonical_arm() {
        // Der Kern von T2.3: Sprachvarianten landen im SELBEN Bandit-Arm.
        assert_eq!(normalize_topic("KI"), Some("ki"));
        assert_eq!(normalize_topic("Künstliche Intelligenz"), Some("ki"));
        assert_eq!(normalize_topic("AI"), Some("ki"));
        assert_eq!(normalize_topic(" Sprachmodelle "), Some("llm"));
        assert_eq!(normalize_topic("Cybersecurity"), Some("sicherheit"));
    }

    #[test]
    fn unknown_tags_are_dropped() {
        assert_eq!(normalize_topic("Quantencomputer-Origami"), None);
        assert_eq!(normalize_topic(""), None);
        assert_eq!(normalize_topic("###"), None);
    }

    #[test]
    fn normalize_topics_dedups_and_caps() {
        let raw = vec![
            "AI".to_string(),
            "KI".to_string(), // Duplikat nach Normalisierung
            "GPU".to_string(),
            "Roboter".to_string(),
            "Startup".to_string(), // über der Kappe von 3
        ];
        let out = normalize_topics(&raw);
        assert_eq!(out, vec!["ki", "chips", "robotik"]);
    }

    #[test]
    fn every_alias_targets_vocabulary() {
        for (alias, canonical) in ALIASES {
            assert!(
                TOPICS.contains(canonical),
                "Alias '{alias}' zeigt auf unbekanntes Topic '{canonical}'"
            );
        }
    }
}
