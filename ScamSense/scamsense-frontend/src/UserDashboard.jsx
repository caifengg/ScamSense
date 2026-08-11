import { useEffect, useState } from "react";
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

  // Screenshot upload state - kept separate from the /detect-text loading/error
  // state above since extraction and classification are two independent
  // requests the user can trigger at different times.
  const [textExtracting, setTextExtracting] = useState(false);  // true while /extract-text-from-image is in flight
  const [textExtractError, setTextExtractError] = useState(""); // error from the screenshot step only

  // Flag-as-wrong state for the currently-shown result only. `textCheckId` is
  // the text_checks row id the backend returned for the last /detect-text
  // call (null if it couldn't be saved, e.g. not logged in or DB down) -
  // the flag button is hidden whenever this is null. `textFlagged` flips to
  // true after a successful flag and disables the button, since each check
  // can only be flagged once (also enforced server-side).
  const [textCheckId, setTextCheckId] = useState(null);
  const [textFlagged, setTextFlagged] = useState(false);
  const [textFlagging, setTextFlagging] = useState(false);
  const [textFlagError, setTextFlagError] = useState("");

  // Recent-checks history (Text Extractor only).
  const [textHistory, setTextHistory] = useState([]);
  const [textHistoryLoading, setTextHistoryLoading] = useState(false);
  const [textHistoryError, setTextHistoryError] = useState("");
  const [textHistoryDeletingId, setTextHistoryDeletingId] = useState(null);

  const userId = localStorage.getItem("scamsenseUserId");

  const navigate = useNavigate();

  // Loads (or reloads) the current user's Text Extractor history. Called on
  // mount and after every successful check/flag/delete so the list always
  // reflects the latest state without a manual refresh.
  const fetchTextHistory = async () => {
    if (!userId) return;

    setTextHistoryLoading(true);
    setTextHistoryError("");

    try {
      const response = await axios.get("/text-checks", { params: { user_id: userId } });
      setTextHistory(response.data.checks);
    } catch {
      setTextHistoryError("Could not load history. Check backend and try again.");
    } finally {
      setTextHistoryLoading(false);
    }
  };

  useEffect(() => {
    fetchTextHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      const response = await axios.post("/detect-text", { message, user_id: userId });

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
      // New check → reset flag state for it (a fresh check_id has never
      // been flagged yet) and refresh the history list to include it.
      setTextCheckId(response.data.check_id);
      setTextFlagged(false);
      setTextFlagError("");
      fetchTextHistory();
    } catch {
      // Covers network failures, the backend being down, or a non-2xx
      // response - deliberately generic rather than surfacing raw axios
      // error internals to the user.
      setTextError("Detection failed. Check backend and try again.");
    } finally {
      setTextLoading(false);
    }
  };

  // Handler for TextExtractor's "Flag as wrong" button. Only shown/enabled
  // for a check that was actually saved (textCheckId != null) and hasn't
  // already been flagged. The backend still enforces "one flag per check"
  // itself (see flag_text_check() in app.py), so this frontend check is
  // just to keep the button from being clickable twice in normal use.
  const flagTextCheck = async () => {
    if (!textCheckId || textFlagged) return;

    setTextFlagging(true);
    setTextFlagError("");

    try {
      await axios.post(`/text-checks/${textCheckId}/flag`, { user_id: userId });
      setTextFlagged(true);
      fetchTextHistory();
    } catch (err) {
      const backendMessage = err?.response?.data?.error;
      // A 409 here almost always means it was already flagged (e.g. a
      // duplicate click that raced ahead of the button disabling) - treat
      // that the same as success rather than showing a scary error.
      if (err?.response?.status === 409) {
        setTextFlagged(true);
      } else {
        setTextFlagError(backendMessage || "Could not flag this result. Check backend and try again.");
      }
    } finally {
      setTextFlagging(false);
    }
  };

  // Handler for deleting one row from the Text Extractor's history list.
  // Only removes the history entry - it does not clear whatever result is
  // currently shown above, even if that result happens to be the same check.
  const deleteTextHistoryEntry = async (id) => {
    setTextHistoryDeletingId(id);
    setTextHistoryError("");

    try {
      await axios.delete(`/text-checks/${id}`, { params: { user_id: userId } });
      setTextHistory((prev) => prev.filter((entry) => entry.id !== id));
    } catch {
      setTextHistoryError("Could not delete that entry. Check backend and try again.");
    } finally {
      setTextHistoryDeletingId(null);
    }
  };

  // Handler for TextExtractor's "Upload Screenshot" input. Reads the file as
  // a base64 data URL, sends it to /extract-text-from-image (Gemini vision
  // transcription only - no classification happens here), then drops the
  // transcribed text into the same `message` state the textarea uses, so the
  // user can review/edit it before clicking "Check Message" as normal.
  const uploadScreenshot = async (file) => {
    setTextExtracting(true);
    setTextExtractError("");

    try {
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });

      const response = await axios.post("/extract-text-from-image", {
        image: dataUrl,
        mime_type: file.type || "image/jpeg",
      });

      setMessage(response.data.text);
    } catch (err) {
      // A 422 from the backend means the screenshot itself couldn't be read
      // (unreadable image, no text found) - surface that specific message
      // when available, otherwise fall back to a generic one.
      const backendMessage = err?.response?.data?.error;
      setTextExtractError(backendMessage || "Could not read that screenshot. Check backend and try again.");
    } finally {
      setTextExtracting(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("scamsenseAuth");
    localStorage.removeItem("scamsenseRole");
    localStorage.removeItem("scamsenseUserId");
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
            onUploadScreenshot={uploadScreenshot}
            extracting={textExtracting}
            extractError={textExtractError}
            checkId={textCheckId}
            flagged={textFlagged}
            flagging={textFlagging}
            flagError={textFlagError}
            onFlagWrong={flagTextCheck}
            history={textHistory}
            historyLoading={textHistoryLoading}
            historyError={textHistoryError}
            historyDeletingId={textHistoryDeletingId}
            onDeleteHistoryEntry={deleteTextHistoryEntry}
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