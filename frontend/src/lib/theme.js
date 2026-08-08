import { useState, useCallback } from 'react';

/**
 * Categorical series palette (cluster identity) for the night sky.
 *
 * Slot ORDER is the colorblind-safety mechanism, not cosmetics: chosen by
 * enumerating all 8! orderings and keeping the one that maximizes the minimum
 * adjacent-pair CVD ΔE on the night surface (worst adjacent ΔE 41.3 — CVD,
 * lightness, and contrast all pass).
 *
 * Clusters past slot 8 fold into a muted tone — beyond 8 hues identity isn't
 * distinguishable anyway; the legend + hover labels carry it instead.
 */
export const SERIES = {
  colors: ['#199e70', '#c98500', '#d55181', '#d95926', '#3987e5', '#008300', '#9085e9', '#e66767'],
  other:  '#5b6a84',
};

export function seriesColor(slot) {
  return slot < SERIES.colors.length ? SERIES.colors[slot] : SERIES.other;
}

/** Read a CSS custom property off :root (for canvas renderers, which can't
 * resolve var() themselves). */
export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---------------------------------------------------------------------------
// Crawlers
// ---------------------------------------------------------------------------

/**
 * The three crawler personas. Sapphire is the live academic pipeline; ruby
 * and topaz are planned. `accent` mirrors the CSS tokens so the switcher can
 * paint each gem in its own color regardless of the active crawler.
 */
export const CRAWLERS = {
  sapphire: {
    name: 'Sapphire',
    accent: '#58b0e8',
    tagline: 'academic papers',
    ready: true,
  },
  ruby: {
    name: 'Ruby',
    accent: '#e8586b',
    tagline: 'medical documents',
    ready: false,
  },
  topaz: {
    name: 'Topaz',
    accent: '#e8b058',
    tagline: 'general documents',
    ready: true,
  },
};

function currentCrawler() {
  const crawlerName = document.documentElement.dataset.crawler;
  return crawlerName in CRAWLERS ? crawlerName : 'sapphire';
}

/** Active crawler + a setter that stamps data-crawler on <html> (persisted),
 * which the per-crawler CSS token overrides key off. */
export function useCrawler() {
  const [crawler, setCrawlerState] = useState(currentCrawler);

  const setCrawler = useCallback((next) => {
    if (!(next in CRAWLERS)) return;
    if (next === 'sapphire') delete document.documentElement.dataset.crawler;
    else document.documentElement.dataset.crawler = next;
    localStorage.setItem('opencrawl-crawler', next);
    setCrawlerState(next);
  }, []);

  return { crawler, setCrawler };
}
