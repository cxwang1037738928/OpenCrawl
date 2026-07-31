/**
 * AuthPage.jsx — combined login/register landing page (the app requires a
 * session). Register creates the account and logs straight in (no email
 * verification, no password resets for now). Shows the demo credentials.
 *
 * On a DEMO deployment registration is closed and everyone shares the demo
 * account, so the Register control is hidden. The backend refuses the route
 * regardless — this only keeps the UI honest about what is possible.
 */

import { useEffect, useState } from 'react';
import { getDeploymentConfig, login, register, setToken } from '../api.js';

export default function AuthPage({ onAuth }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [demo, setDemo] = useState(false);

  useEffect(() => {
    let live = true;
    getDeploymentConfig().then((config) => { if (live) setDemo(Boolean(config.demo)); });
    return () => { live = false; };
  }, []);

  async function submit(action) {
    if (busy) return;
    setError(null);
    setBusy(true);
    try {
      const { token, user } = await action(email.trim(), password);
      setToken(token);
      onAuth(user);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-page">
      <form className="auth-card" onSubmit={(event) => { event.preventDefault(); submit(login); }}>
        <h1 className="wordmark">OpenCrawl</h1>
        <p className="auth-sub">
          {demo ? 'Log in with the demo account below.' : 'Log in, or register a new account.'}
        </p>
        <p className="auth-demo-hint">
          Try the demo with the username demo@gmail.com and password: demo123
        </p>

        <label className="auth-field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            autoComplete="email"
            onChange={(event) => setEmail(event.target.value)}
            required
          />
        </label>

        <label className="auth-field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            autoComplete="current-password"
            minLength={6}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
        </label>

        {error && <div className="auth-error">{error}</div>}

        <div className="auth-actions">
          <button className="btn" type="submit" disabled={busy}>
            Log in
          </button>
          {!demo && (
            <button
              className="btn btn-secondary"
              type="button"
              disabled={busy}
              onClick={() => submit(register)}
            >
              Register
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
