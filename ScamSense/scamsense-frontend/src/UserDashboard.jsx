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

        {activeTool === "text-extractor" && (
          <TextExtractor />
        )}

        {activeTool === "deepfake" && (
          <Deepfake />
        )}
      </main>
    </div>

  );
}

export default UserDashboard;