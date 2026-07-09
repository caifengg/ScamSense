import { useEffect, useState } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";

function AdminDashboard(){

  const [users,setUsers] = useState(0);
  const [scams,setScams] = useState(0);
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
        setError("Could not load admin stats.");
      });

    };

    loadStats();

    const statsInterval = setInterval(loadStats, 5000);

    return ()=>clearInterval(statsInterval);

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
      </main>
    </div>

  );
}

export default AdminDashboard;