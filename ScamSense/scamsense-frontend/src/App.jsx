import { BrowserRouter, Navigate, Routes, Route } from "react-router-dom";

import Login from "./Login";
import Signup from "./Signup";
import UserDashboard from "./UserDashboard";
import AdminDashboard from "./AdminDashboard";

function RequireAuth({ children, role }) {
  const isAuthenticated = localStorage.getItem("scamsenseAuth") === "true";
  const userRole = localStorage.getItem("scamsenseRole");

  if (!isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  if (role && userRole !== role) {
    return <Navigate to={userRole === "admin" ? "/admin" : "/user"} replace />;
  }

  return children;
}

function LoginRoute() {
  const isAuthenticated = localStorage.getItem("scamsenseAuth") === "true";
  const userRole = localStorage.getItem("scamsenseRole");

  if (!isAuthenticated) {
    return <Login />;
  }

  return <Navigate to={userRole === "admin" ? "/admin" : "/user"} replace />;
}

function App(){

  return(

    <BrowserRouter>

      <Routes>

        <Route
          path="/"
          element={<LoginRoute />}
        />

        <Route
          path="/signup"
          element={<Signup />}
        />

        <Route
          path="/user"
          element={
            <RequireAuth role="user">
              <UserDashboard />
            </RequireAuth>
          }
        />

        <Route
          path="/admin"
          element={
            <RequireAuth role="admin">
              <AdminDashboard />
            </RequireAuth>
          }
        />

      </Routes>

    </BrowserRouter>

  );
}

export default App;