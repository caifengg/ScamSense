import { useState } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";

function Login() {

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const login = async () => {
    setError("");
    setLoading(true);

    try {

      const response = await axios.post(
        "/login",
        {
          email,
          password
        }
      );

      if (response.data.success && response.data.role === "admin") {
        localStorage.setItem("scamsenseAuth", "true");
        localStorage.setItem("scamsenseRole", "admin");
        navigate("/admin", { replace: true });
        return;
      }

      if (response.data.success) {
        localStorage.setItem("scamsenseAuth", "true");
        localStorage.setItem("scamsenseRole", "user");
        navigate("/user", { replace: true });
        return;
      }

      setError("Invalid email or password.");
    } catch {
      setError("Unable to connect to backend. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-shell auth-shell">
      <div className="bg-orb orb-a" aria-hidden="true" />
      <div className="bg-orb orb-b" aria-hidden="true" />

      <main className="card auth-card reveal-up">
        <p className="eyebrow">ScamSense</p>
        <h1>Welcome Back</h1>
        <p className="subtle">Sign in to run phishing checks and monitor activity.</p>

        <div className="form-grid">
          <label>
            Email
            <input
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>

          <label>
            Password
            <input
              type="password"
              placeholder="Your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>

          {error && <p className="error-text">{error}</p>}

          <button type="button" onClick={login} disabled={loading}>
            {loading ? "Signing In..." : "Login"}
          </button>
        </div>

        <p className="switch-link">
          New here? <Link to="/signup">Create an account</Link>
        </p>
      </main>
    </div>
  );
}

export default Login;