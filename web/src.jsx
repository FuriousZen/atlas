import React, {useEffect, useState} from "react";
import {createRoot} from "react-dom/client";
import "./style.css";

const TOPICS = ["reranking", "retrieval", "rag", "agents"];

function AskTab() {
  const [question, setQuestion] = useState("new science today?");
  const [answer, setAnswer] = useState(null);
  const [error, setError] = useState("");
  const ask = async (event) => {
    event.preventDefault();
    setError("");
    try {
      const response = await fetch("/api/ask", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({question})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Request failed");
      setAnswer(data);
    } catch (err) { setError(err.message); }
  };
  return <><form onSubmit={ask}><input value={question} onChange={e => setQuestion(e.target.value)} aria-label="Research question"/><button>Ask Atlas</button></form>{error && <p className="error">{error}</p>}{answer && <section className="answer"><p className="eyebrow">{answer.model}{answer.stub ? " / stub provider" : ""}</p><p>{answer.answer}</p><h2>Sources</h2>{(answer.citations || []).map(c => <article key={c.chunk_id}><strong>{c.title || c.paper_id}</strong><span>{c.section} · {Number(c.score || 0).toFixed(3)}</span><a href={c.url} target="_blank" rel="noreferrer">Open paper</a></article>)}</section>}</>;
}

function BriefTab() {
  const [topic, setTopic] = useState("reranking");
  const [minCitations, setMinCitations] = useState(80);
  const [brief, setBrief] = useState(null);
  const [error, setError] = useState("");
  const load = async (selectedTopic, threshold) => {
    setError("");
    try {
      const response = await fetch(`/api/brief?topic=${encodeURIComponent(selectedTopic)}&min_citations=${threshold}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Request failed");
      setBrief(data);
    } catch (err) { setError(err.message); }
  };
  useEffect(() => { load(topic, minCitations); }, []);
  const pick = (next) => { setTopic(next); load(next, minCitations); };
  const slide = (next) => { setMinCitations(next); load(topic, next); };
  return <section><div>{TOPICS.map(t => <button key={t} type="button" onClick={() => pick(t)} disabled={t === topic}>{t}</button>)}</div><label>min_citations: {minCitations}<input type="range" min="0" max="200" value={minCitations} onChange={e => slide(Number(e.target.value))} aria-label="Minimum citations"/></label>{error && <p className="error">{error}</p>}{brief && <section className="answer"><p className="eyebrow">kept {brief.kept}</p>{(brief.papers || []).map(p => <article key={p.id}><strong>{p.title}</strong><span>{p.citations} citations · {p.year || "n/a"} · {p.venue || "n/a"}</span><span>{p.why_kept || (p.summary && p.summary.why_kept) || ""}</span><a href={p.url} target="_blank" rel="noreferrer">Open paper</a></article>)}</section>}</section>;
}

function ManageTab() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    (async () => {
      try {
        const response = await fetch("/api/health");
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.error || "Request failed");
        setHealth(data);
      } catch (err) { setError(err.message); }
    })();
  }, []);
  if (error) return <p className="error">{error}</p>;
  if (!health) return <p className="lede">Loading capacity…</p>;
  const capacity = health.capacity || {};
  const store = health.store || {};
  return <section className="answer"><p className="eyebrow">capacity</p><p>Papers {store.papers ?? "?"} / {capacity.papers} · chunks {store.chunks ?? "?"} / {capacity.chunks} · bytes {store.bytes ?? "?"} / {capacity.bytes}</p><p>Eviction policy: {capacity.eviction} (oldest documents go first). Grace window: {capacity.min_keep_days} days.</p><p>LLM: {(health.llm && health.llm.effective_backend) || "unknown"}</p></section>;
}

function App() {
  const [tab, setTab] = useState("ask");
  return <main><header><p className="eyebrow">ATLAS / RESEARCH SIGNAL</p><h1>Ask what changed.</h1><p className="lede">A bounded, citation-first research library that polls the literature and keeps the signal.</p></header><nav>{["ask", "brief", "manage"].map(t => <button key={t} type="button" onClick={() => setTab(t)} disabled={t === tab}>{t[0].toUpperCase() + t.slice(1)}</button>)}</nav>{tab === "ask" && <AskTab />}{tab === "brief" && <BriefTab />}{tab === "manage" && <ManageTab />}<footer><a href="http://127.0.0.1:3000">Open Grafana</a><span>Provider is optional. Configure one in .env when ready.</span></footer></main>;
}

createRoot(document.getElementById("root")).render(<App />);
