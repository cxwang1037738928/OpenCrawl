/**
 * demo.js — read-only guard for the public demo deployment
 *
 * The hosted demo serves one fixed collection with one pre-built graph. Visitors
 * share a single account and may log in, browse, open chats and use the
 * embedding/graph visualizers; everything that would change the corpus or the
 * server's configuration is refused.
 *
 * The guard lives in the REQUEST path, not only in the UI. Hiding a button stops
 * the honest path but not a direct API call, and one blocked route missed in the
 * frontend would otherwise let a visitor delete the demo's only document.
 *
 * DEMO=true in .env turns it on. Anything else — unset, empty, "false", "0" —
 * leaves the app fully writable, so a local checkout behaves normally.
 */

export const DEMO = ['true', '1', 'yes'].includes(
  (process.env.DEMO || '').trim().toLowerCase());

/**
 * Refuse a mutating route in demo mode.
 *
 * 403 with a stated reason rather than 404: the frontend already hides these
 * controls, so anything reaching here is either a direct API call or a control
 * we failed to hide — and in the second case a silent no-op would look like the
 * feature is broken rather than disabled.
 */
export function blockInDemo(what) {
  return (req, res, next) => {
    if (!DEMO) return next();
    res.status(403).json({
      error: `${what} is disabled in this demo. The corpus and knowledge graph are fixed.`,
    });
  };
}
