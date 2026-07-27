import { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

import Phishing from "./Phishing";
import Deepfake from "./Deepfake";
import TextExtractor from "./TextExtractor";

function UserDashboard() {

  const [activeTool, setActiveTool] = useState("phishing-link");
  const [url, setUrl] = useState("");
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [explanationAvailable, setExplanationAvailable] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ── Text Extractor state ──────────────────────────────────────────────────
  // Kept entirely separate from the Phishing tool's `url`/`result`/etc.
  // state above (different variable names, own useState calls) so the two
  // tools' data never collide - switching tabs doesn't clear or overwrite
  // either tool's last result, and each tool's request lifecycle
  // (loading/error) is tracked independently.
  const [message, setMessage] = useState("");                 // raw textarea contents (controlled input)
  const [textResult, setTextResult] = useState("");            // "Scam Message" | "Legitimate Message" | ""
  const [textConfidence, setTextConfidence] = useState(null);  // 0-1 float from the model, or null
  const [textExplanation, setTextExplanation] = useState("");  // Gemini's "why this result" text
  const [textExplanationAvailable, setTextExplanationAvailable] = useState(true);
  // Translation-related state - only meaningful when the backend detected
  // non-English input. `textTranslated` gates whether TextExtractor.jsx
  // renders the translation box at all.
  const [textTranslated, setTextTranslated] = useState(false);
  const [textTranslatedText, setTextTranslatedText] = useState("");
  const [textDetectedLanguage, setTextDetectedLanguage] = useState(null);
  const [textLoading, setTextLoading] = useState(false);  // true while /detect-text is in flight
  const [textError, setTextError] = useState("");         // network/request-level error only

  const navigate = useNavigate();

  const checkURL = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await axios.post("/detect", { url });

      setResult(response.data.result);
      setConfidence(response.data.confidence);
      setExplanation(response.data.explanation);
      setExplanationAvailable(response.data.explanation_available);
    } catch {
      setError("Detection failed. Check backend and try again.");
    } finally {
      setLoading(false);
    }
  };

  // Handler for TextExtractor's "Check Message" button. Sends the raw
  // message to the backend, which internally: (1) translates it to English
  // if needed, (2) classifies the English version with the trained
  // ensemble, (3) asks Gemini to explain the verdict - all three results
  // come back in one response, which is why this single handler fans out
  // into six separate setState calls below rather than making three
  // separate requests.
  const checkMessage = async () => {
    setTextLoading(true);
    setTextError("");  // clear any previous request's error before trying again

    try {
      const response = await axios.post("/detect-text", { message });

      setTextResult(response.data.result);
      setTextConfidence(response.data.confidence);
      setTextExplanation(response.data.explanation);
      setTextExplanationAvailable(response.data.explanation_available);
      // These three stay at their defaults (false / "" / null) for English
      // input, since the backend only populates them when translation
      // actually happened - see translate_to_english() in
      // gemini_explainer.py.
      setTextTranslated(response.data.translated);
      setTextTranslatedText(response.data.translated_text);
      setTextDetectedLanguage(response.data.detected_language);
    } catch {
      // Covers network failures, the backend being down, or a non-2xx
      // response - deliberately generic rather than surfacing raw axios
      // error internals to the user.
      setTextError("Detection failed. Check backend and try again.");
    } finally {
      setTextLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("scamsenseAuth");
    localStorage.removeItem("scamsenseRole");
    navigate("/", { replace: true });
  };

  return(

    <div className="page-shell dashboard-shell">
      <div className="bg-grid" aria-hidden="true" />

      <main className={"card dashboard-card reveal-up" + (activeTool === "deepfake" ? " deepfake-active" : "")}>
        <header className="dashboard-head">
          <div>
            <p className="eyebrow">ScamSense</p>
            <h1>User Dashboard</h1>
          </div>
          <button type="button" className="ghost-link" onClick={logout}>Log Out</button>
        </header>

        <nav className="tool-nav" aria-label="User tools">
          <button
            type="button"
            className={activeTool === "text-extractor" ? "tool-tab active" : "tool-tab"}
            onClick={() => setActiveTool("text-extractor")}
          >
            Text Extractor
          </button>

          <button
            type="button"
            className={activeTool === "deepfake" ? "tool-tab active" : "tool-tab"}
            onClick={() => setActiveTool("deepfake")}
          >
            Deepfake
          </button>

          <button
            type="button"
            className={activeTool === "phishing-link" ? "tool-tab active" : "tool-tab"}
            onClick={() => setActiveTool("phishing-link")}
          >
            Phishing Link
          </button>
        </nav>

        {activeTool === "phishing-link" && (
          <Phishing
            url={url}
            setUrl={setUrl}
            result={result}
            confidence={confidence}
            explanation={explanation}
            explanationAvailable={explanationAvailable}
            loading={loading}
            error={error}
            onCheckURL={checkURL}
          />
        )}

        {/* Only mounted while its tab is active - all of TextExtractor's
            state lives here in the parent, so switching away to another
            tab and back preserves whatever message/result was last shown,
            it just isn't rendered while a different tool is selected. */}
        {activeTool === "text-extractor" && (
          <TextExtractor
            message={message}
            setMessage={setMessage}
            result={textResult}
            confidence={textConfidence}
            explanation={textExplanation}
            explanationAvailable={textExplanationAvailable}
            translated={textTranslated}
            translatedText={textTranslatedText}
            detectedLanguage={textDetectedLanguage}
            loading={textLoading}
            error={textError}
            onCheckMessage={checkMessage}
          />
        )}

        {activeTool === "deepfake" && (
          <Deepfake />
        )}
      </main>
    </div>

  );
}

export default UserDashboard;