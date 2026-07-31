/**
 * Chat.jsx — RAG chat over the selected chat's corpus.
 *
 * The question is embedded HERE, in the browser, with the same MiniLM model
 * the corpus was embedded with (transformers.js; browser cache deliberately
 * off, so the model re-downloads each session). The vector goes to
 * /api/chats/:chatId/chat, which retrieves chunks (cosine × category keyword
 * boost) and answers with the Ollama reasoning model.
 *
 * The conversation lives on the Chat row in Postgres: it is loaded on mount
 * (App remounts this component per chat) and the server appends each
 * question/answer pair; deleting a pair PATCHes the pruned conversation back.
 */

import { useEffect, useRef, useState } from 'react';
import { postChat, getChat, updateChat } from '../api.js';
import {
  embedQuery, SEGMENT_RE, markerCitingSentences, citeInline, inlineMd,
  citedMarkerNumbers, toConversation,
} from '../utils/Chat_utils.jsx';

/**
 * Minimal markdown renderer for assistant replies — headings, bullet and
 * numbered lists, blockquotes, paragraphs, plus inline emphasis. The models
 * emit markdown whether or not we ask, and raw ### / ** in the bubble reads
 * as garbage. Citation markers stay clickable inside all of it.
 */
function CitedText({ text, sources, query, onCitation }) {
  // Each block gets its own marker→claim cursor, so a [n] in one paragraph or
  // list item is matched to that block's citing sentence, not a neighbour's.
  const makeCite = (blockText) => {
    const claimCursor = { claims: markerCitingSentences(blockText), next: 0 };
    return (segment, key) => citeInline(segment, sources, query, onCitation, key, claimCursor);
  };
  const blocks = [];

  // A marker on its own line cites the line above it (a favourite small-model
  // format, especially after list items). Merge such lines upward so the
  // marker lands in its claim's block — as its own block it would carry no
  // claim, and after a list it WOULD become its own block.
  const lines = [];
  let lastContentIdx = -1;
  for (const rawLine of (text || '').split('\n')) {
    const markerOnly = (rawLine.match(SEGMENT_RE) || []).length > 0
      && rawLine.replace(SEGMENT_RE, '').trim() === '';
    if (markerOnly && lastContentIdx >= 0) {
      lines[lastContentIdx] += ` ${rawLine.trim()}`;
    } else {
      lines.push(rawLine);
      if (rawLine.trim()) lastContentIdx = lines.length - 1;
    }
  }

  for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
    const line = lines[lineIdx];
    if (!line.trim()) continue;

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      const [, , headingText] = headingMatch;
      blocks.push(
        <div className="md-h" key={lineIdx}>{inlineMd(headingText, `h${lineIdx}`, makeCite(headingText))}</div>
      );
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quoteLines = [];
      while (lineIdx < lines.length && /^\s*>\s?/.test(lines[lineIdx])) {
        quoteLines.push(lines[lineIdx].replace(/^\s*>\s?/, ''));
        lineIdx++;
      }
      lineIdx--;
      const quoteText = quoteLines.join(' ');
      blocks.push(
        <blockquote key={lineIdx}>{inlineMd(quoteText, `q${lineIdx}`, makeCite(quoteText))}</blockquote>
      );
      continue;
    }

    if (/^\s*(?:[-*+]|\d+[.)])\s+/.test(line)) {
      const listItems = [];
      const isOrdered = /^\s*\d+[.)]\s+/.test(line);
      while (lineIdx < lines.length && /^\s*(?:[-*+]|\d+[.)])\s+/.test(lines[lineIdx])) {
        listItems.push(lines[lineIdx].replace(/^\s*(?:[-*+]|\d+[.)])\s+/, ''));
        lineIdx++;
      }
      lineIdx--;
      const List = isOrdered ? 'ol' : 'ul';
      blocks.push(
        <List key={lineIdx}>
          {listItems.map((listItem, itemIdx) => (
            <li key={itemIdx}>{inlineMd(listItem, `l${lineIdx}_${itemIdx}`, makeCite(listItem))}</li>
          ))}
        </List>
      );
      continue;
    }

    // Paragraph: absorb following non-blank, non-structural lines.
    const paragraphLines = [line];
    while (lineIdx + 1 < lines.length && lines[lineIdx + 1].trim() &&
           !/^\s*(?:[-*+]|\d+[.)])\s+|^\s*>|^#{1,6}\s/.test(lines[lineIdx + 1])) {
      paragraphLines.push(lines[++lineIdx]);
    }
    const paragraphText = paragraphLines.join(' ');
    blocks.push(<p key={lineIdx}>{inlineMd(paragraphText, `p${lineIdx}`, makeCite(paragraphText))}</p>);
  }

  return blocks;
}

function Sources({ sources, reply, query, onCitation }) {
  if (!sources?.length) return null;
  // Chips are the documents the answer cites — a retrieved but uncited chunk
  // is not a source. Both verified [n] and flagged [n!] markers earn a chip;
  // the flag belongs on the CLAIM, in the message text, not on the document,
  // which is the same document either way.
  const { verified, flagged } = citedMarkerNumbers(reply);

  // One chip per document, best-scoring cited chunk first (sources arrive in
  // score order).
  const chipByDoc = new Map();   // docId -> source
  sources.forEach((source, sourceIdx) => {
    const excerptNumber = sourceIdx + 1;
    if (!verified.has(excerptNumber) && !flagged.has(excerptNumber)) return;
    if (!chipByDoc.has(source.docId)) chipByDoc.set(source.docId, source);
  });
  if (!chipByDoc.size) return null;

  return (
    <div className="msg-sources">
      {[...chipByDoc.values()].map((source) => (
        <button
          className="source-chip"
          key={source.docId}
          title={`${source.heading || 'document'} · score ${source.score} — open in Documents`}
          onClick={() => onCitation?.(source, query)}
        >
          {source.filename.replace(/\.pdf$/i, '')}
        </button>
      ))}
    </div>
  );
}

/**
 * Delete control on a reply: a small × that widens on hover, then asks to
 * confirm before removing the question/answer pair.
 */
function DeletePair({ confirming, onAsk, onConfirm, onCancel }) {
  return (
    <div className="msg-actions">
      {confirming ? (
        <div className="delete-confirm" role="dialog" aria-label="Confirm delete">
          <span>Delete this response?</span>
          <button className="delete-yes" onClick={onConfirm}>Yes</button>
          <button className="delete-no" onClick={onCancel}>Cancel</button>
        </div>
      ) : (
        <button className="delete-pair" onClick={onAsk} aria-label="Delete this pair">
          <span className="delete-mark">×</span>
          <span className="delete-label">delete this pair?</span>
        </button>
      )}
    </div>
  );
}

export default function Chat({ chatId, active, onCitation, onFirstMessage }) {
  const [messages, setMessages] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState(null);   // busy string | null
  const [confirmingIdx, setConfirmingIdx] = useState(null);   // reply awaiting delete confirmation
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  // Restore the persisted conversation. Replies regain `query` as {text} from
  // the question above them (its embedding was never stored — the viewer
  // re-embeds the text if a citation needs it).
  useEffect(() => {
    let cancelled = false;
    getChat(chatId)
      .then(({ chat }) => {
        if (cancelled) return;
        let lastQuestion = null;
        setMessages((chat.conversation || []).map((message) => {
          if (message.role === 'user') { lastQuestion = message.content; return message; }
          return { ...message, query: lastQuestion ? { text: lastQuestion } : null };
        }));
      })
      .catch((err) => console.warn('[chat] could not load conversation:', err.message))
      .finally(() => { if (!cancelled) setLoaded(true); });
    return () => { cancelled = true; };
  }, [chatId]);

  // Drop a reply and the question above it — from the thread, from the
  // history the model sees, and from the conversation persisted server-side.
  function deletePair(replyIdx) {
    setMessages((thread) => {
      const pruned = thread.filter((message, messageIdx) => {
        if (messageIdx === replyIdx) return false;
        return !(messageIdx === replyIdx - 1 && message.role === 'user');
      });
      updateChat(chatId, { conversation: toConversation(pruned) })
        .catch((err) => console.warn('[chat] could not persist deletion:', err.message));
      return pruned;
    });
    setConfirmingIdx(null);
  }

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, status]);

  useEffect(() => {
    if (active) inputRef.current?.focus();
  }, [active]);

  async function send() {
    const question = input.trim();
    if (!question || status) return;

    const isFirstMessage = messages.length === 0;
    setMessages((thread) => [...thread, { role: 'user', content: question }]);
    setInput('');

    try {
      const queryEmbedding = await embedQuery(question, setStatus);
      setStatus('retrieving + reasoning…');
      // The server holds the history — only the new question travels.
      const { reply, sources, model } = await postChat(chatId, {
        content: question,
        queryEmbedding,
      });
      // The first exchange retitles the chat server-side; mirror it in the list.
      if (isFirstMessage) onFirstMessage?.(chatId, question.slice(0, 60));
      // Keep the query with the reply: citation clicks pass it to the PDF
      // viewer, which scores the chunk's sentences against it to decide
      // what to highlight.
      const query = { text: question, embedding: queryEmbedding };
      setMessages((thread) => [...thread, { role: 'assistant', content: reply, sources, model, query }]);
    } catch (err) {
      setMessages((thread) => [...thread, { role: 'assistant', error: err.message }]);
    } finally {
      setStatus(null);
    }
  }

  function onKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  }

  return (
    <div className="chat-wrap">
      <div className="chat-scroll" ref={scrollRef}>
        {loaded && messages.length === 0 && !status && (
          <div className="chat-hero">
            <h2>Ask your corpus</h2>
            <p>
              Answers are retrieved from this chat's indexed documents and grounded
              with citations. The question is embedded locally in your browser.
            </p>
          </div>
        )}

        <div className="chat-thread">
          {messages.map((message, messageIdx) =>
            message.role === 'user' ? (
              <div className="msg user" key={messageIdx}>{message.content}</div>
            ) : message.error ? (
              <div className="msg assistant error" key={messageIdx}>
                {message.error}
                <DeletePair
                  confirming={confirmingIdx === messageIdx}
                  onAsk={() => setConfirmingIdx(messageIdx)}
                  onConfirm={() => deletePair(messageIdx)}
                  onCancel={() => setConfirmingIdx(null)}
                />
              </div>
            ) : (
              <div className="msg assistant" key={messageIdx}>
                <div className="msg-body">
                  <CitedText
                    text={message.content}
                    sources={message.sources}
                    query={message.query}
                    onCitation={onCitation}
                  />
                </div>
                <Sources
                  sources={message.sources}
                  reply={message.content}
                  query={message.query}
                  onCitation={onCitation}
                />
                <DeletePair
                  confirming={confirmingIdx === messageIdx}
                  onAsk={() => setConfirmingIdx(messageIdx)}
                  onConfirm={() => deletePair(messageIdx)}
                  onCancel={() => setConfirmingIdx(null)}
                />
              </div>
            )
          )}
          {status && <div className="chat-status">{status}</div>}
        </div>
      </div>

      <div className="composer">
        <textarea
          ref={inputRef}
          rows={1}
          placeholder="Ask about your documents…"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={onKeyDown}
          disabled={!!status}
        />
        <button className="btn" onClick={send} disabled={!!status || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
