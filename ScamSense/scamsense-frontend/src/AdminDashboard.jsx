import { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

function AdminDashboard(){

  const [activeTool, setActiveTool] = useState("text-extractor");
  const [users,setUsers] = useState(0);
  const [scams,setScams] = useState(0);
  const [textChecks, setTextChecks] = useState([]);
  const [deepfakeChecks, setDeepfakeChecks] = useState([]);
  const [phishingDetections, setPhishingDetections] = useState([]);
  const [textLoading, setTextLoading] = useState(false);
  const [deepfakeLoading, setDeepfakeLoading] = useState(false);
  const [phishingLoading, setPhishingLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(()=>{

    const loadStats = ()=>{

      axios.get(
        "/admin/stats"
      )
      .then((response)=>{

        setUsers(response.data.total_users);
        setScams(response.data.total_scams);

      })
      .catch(()=>{
        // Silently fail - stats are optional
      });

    };

    const loadTextChecks = ()=>{
      setTextLoading(true);

      axios.get("/admin/text-checks", { params: { limit: 50 } })
        .then((response)=>{
          setTextChecks(response.data.checks || []);
        })
        .catch(()=>{
          setError("Could not load admin text extractor results.");
        })
        .finally(()=>{
          setTextLoading(false);
        });
    };

    const loadDeepfakeChecks = ()=>{
      setDeepfakeLoading(true);

      axios.get("/admin/deepfake-results", { params: { limit: 50 } })
        .then((response)=>{
          setDeepfakeChecks(response.data.checks || []);
        })
        .catch(()=>{
          setError("Could not load admin deepfake results.");
        })
        .finally(()=>{
          setDeepfakeLoading(false);
        });
    };

    const loadPhishingDetections = ()=>{
      setPhishingLoading(true);

      axios.get("/admin/phishing-detections", { params: { limit: 50 } })
        .then((response)=>{
          setPhishingDetections(response.data.detections || []);
        })
        .catch(()=>{
          setError("Could not load admin phishing results.");
        })
        .finally(()=>{
          setPhishingLoading(false);
        });
    };

    loadStats();
    loadTextChecks();
    loadDeepfakeChecks();
    loadPhishingDetections();

    const statsInterval = setInterval(loadStats, 5000);
    const textInterval = setInterval(loadTextChecks, 5000);
    const deepfakeInterval = setInterval(loadDeepfakeChecks, 5000);
    const phishingInterval = setInterval(loadPhishingDetections, 5000);

    return ()=>{
      clearInterval(statsInterval);
      clearInterval(textInterval);
      clearInterval(deepfakeInterval);
      clearInterval(phishingInterval);
    };

  },[]);

  const logout = () => {
    localStorage.removeItem("scamsenseAuth");
    localStorage.removeItem("scamsenseRole");
    navigate("/", { replace: true });
  };

  return(

    <div className="page-shell dashboard-shell">
      <div className="bg-grid" aria-hidden="true" />

      <main className="card dashboard-card reveal-up">
        <header className="dashboard-head">
          <div>
            <p className="eyebrow">ScamSense</p>
            <h1>Admin Dashboard</h1>
          </div>
          <button type="button" className="ghost-link" onClick={logout}>Log Out</button>
        </header>

        {error && <p className="error-text">{error}</p>}

        <section className="stats-grid">
          <article className="stat-card">
            <p className="stat-label">Total Users</p>
            <h2>{users}</h2>
          </article>

          <article className="stat-card warning">
            <p className="stat-label">Scams Detected</p>
            <h2>{scams}</h2>
          </article>
        </section>

        <nav className="tool-nav" aria-label="Admin tools" style={{ marginTop: "20px" }}>
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

        {activeTool === "text-extractor" && (
          <section className="admin-list-card">
            <p className="subtle">Text Extractor results (message, result, confidence)</p>

            {textLoading && <p className="subtle">Loading text extractor results...</p>}
            {!textLoading && textChecks.length === 0 && <p className="subtle">No text extractor results yet.</p>}

            {textChecks.length > 0 && (
              <div className="admin-table-wrap">
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>Message</th>
                      <th>Result</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {textChecks.map((entry, index) => (
                      <tr key={entry.created_at ? `${entry.created_at}-${index}` : index}>
                        <td>{entry.message}</td>
                        <td>{entry.result}</td>
                        <td>{entry.confidence !== null && entry.confidence !== undefined ? `${(entry.confidence * 100).toFixed(0)}%` : "N/A"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {activeTool === "deepfake" && (
          <section className="admin-list-card">
            <p className="subtle">Deepfake results (real or fake)</p>

            {deepfakeLoading && <p className="subtle">Loading deepfake results...</p>}
            {!deepfakeLoading && deepfakeChecks.length === 0 && <p className="subtle">No deepfake results yet.</p>}

            {deepfakeChecks.length > 0 && (
              <div className="admin-table-wrap">
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>Session</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {deepfakeChecks.map((entry, index) => (
                      <tr key={entry.session_id ? `${entry.session_id}-${index}` : index}>
                        <td>{entry.session_id}</td>
                        <td>{entry.result === "DEEPFAKE" ? "Fake" : entry.result === "REAL" ? "Real" : entry.result}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {activeTool === "phishing-link" && (
          <section className="admin-list-card">
            <p className="subtle">Phishing link results (URL and result)</p>

            {phishingLoading && <p className="subtle">Loading phishing results...</p>}
            {!phishingLoading && phishingDetections.length === 0 && <p className="subtle">No phishing link results yet.</p>}

            {phishingDetections.length > 0 && (
              <div className="admin-table-wrap">
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>URL</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {phishingDetections.map((entry, index) => (
                      <tr key={entry.created_at ? `${entry.created_at}-${index}` : index}>
                        <td>{entry.url}</td>
                        <td>{entry.result}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}
      </main>
    </div>

  );
}

export default AdminDashboard;