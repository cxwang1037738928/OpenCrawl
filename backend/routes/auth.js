/**
 * routes/auth.js — /api/auth
 *
 *   POST /register {email, password} — create account, return {token, user}
 *   POST /login    {email, password} — verify credentials, return {token, user}
 *   GET  /me                         — current user (requires Bearer token)
 *
 * Passwords are bcrypt-hashed; no email verification or resets for now.
 */

import { Router } from 'express';
import bcrypt from 'bcryptjs';
import { prisma } from '../db.js';
import { signToken, userFromRequest } from '../middleware/auth.js';

const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const publicUser = (user) => ({ id: user.id, email: user.email, isAdmin: user.isAdmin });

/** Returns the normalized email, or throws a 400 with the reason. */
function validateCredentials(body) {
  const { email, password } = body ?? {};
  const err = (message) => Object.assign(new Error(message), { status: 400 });
  if (typeof email !== 'string' || !EMAIL_RE.test(email.trim())) {
    throw err('A valid email is required');
  }
  if (typeof password !== 'string' || password.length < 6) {
    throw err('Password must be at least 6 characters');
  }
  return { email: email.trim().toLowerCase(), password };
}

export const authRouter = Router();

authRouter.post('/register', wrap(async (req, res) => {
  const { email, password } = validateCredentials(req.body);
  const existing = await prisma.user.findUnique({ where: { email } });
  if (existing) return res.status(409).json({ error: 'That email is already registered' });
  const user = await prisma.user.create({
    data: { email, password: await bcrypt.hash(password, 10) },
  });
  res.status(201).json({ token: signToken(user), user: publicUser(user) });
}));

authRouter.post('/login', wrap(async (req, res) => {
  const { email, password } = validateCredentials(req.body);
  const user = await prisma.user.findUnique({ where: { email } });
  if (!user || !(await bcrypt.compare(password, user.password))) {
    return res.status(401).json({ error: 'Invalid email or password' });
  }
  res.json({ token: signToken(user), user: publicUser(user) });
}));

// Soft check: {user: null} for guests/expired tokens — never a 401, since
// the app works without an account.
authRouter.get('/me', wrap(async (req, res) => {
  const tokenUser = userFromRequest(req);
  if (!tokenUser) return res.json({ user: null });
  const user = await prisma.user.findUnique({ where: { id: tokenUser.id } });
  res.json({ user: user ? publicUser(user) : null });
}));
