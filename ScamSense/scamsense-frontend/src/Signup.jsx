import { useState } from "react";
import axios from "axios";
import { Link } from "react-router-dom";

function Signup() {

  const [username,setUsername] = useState("");
  const [email,setEmail] = useState("");
  const [password,setPassword] = useState("");
  const [role,setRole] = useState("user");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const signup = async ()=>{
    setMessage("");
    setError("");
    setLoading(true);

    try {
      await axios.post(
        "/signup",
        {
          username,
          email,
          password,
          role
        }
      );

      setMessage("Account created successfully. You can log in now.");
    } catch (err) {
      const serverError = err?.response?.data?.error;
      setError(serverError || "Signup failed. Please check your details and try again.");
    } finally {
      setLoading(false);
    }
  };

  return(
    <div className="page-shell auth-shell">
      <div className="bg-orb orb-c" aria-hidden="true" />
      <div className="bg-orb orb-d" aria-hidden="true" />

      <main className="card auth-card reveal-up">
        <p className="eyebrow">ScamSense</p>
        <h1>Create Account</h1>
        <p className="subtle">Get started with scam detection in under a minute.</p>

        <div className="form-grid">
          <label>
            Username
            <input
              placeholder="Your name"
              value={username}
              onChange={(e)=>setUsername(e.target.value)}
            />
          </label>

          <label>
            Email
            <input
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e)=>setEmail(e.target.value)}
            />
          </label>

          <label>
            Password
            <input
              type="password"
              placeholder="Choose a strong password"
              value={password}
              onChange={(e)=>setPassword(e.target.value)}
            />
          </label>

          <label>
            Role
            <select
              value={role}
              onChange={(e)=>setRole(e.target.value)}
            >
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </select>
          </label>

          {message && <p className="success-text">{message}</p>}
          {error && <p className="error-text">{error}</p>}

          <button type="button" onClick={signup} disabled={loading}>
            {loading ? "Creating..." : "Sign Up"}
          </button>
        </div>

        <p className="switch-link">
          Already have an account? <Link to="/">Back to login</Link>
        </p>
      </main>
    </div>
  );
}

export default Signup;