import { useState } from "react";

const API = "http://127.0.0.1:8001";

export default function App() {
  const [brief, setBrief] = useState("");
  const [file, setFile] = useState(null);
  const [demoName, setDemoName] = useState("");
  const [createResult, setCreateResult] = useState(null);
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleCreate(e) {
    e.preventDefault();
    if (!brief.trim()) return;
    setLoading(true);
    setError("");
    setCreateResult(null);
    try {
      const form = new FormData();
      form.append("brief", brief);
      if (file) form.append("files", file);
      const res = await fetch(`${API}/create-agent`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || JSON.stringify(data));
      setCreateResult(data);
      setDemoName(data.demo_name || "");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleChat(e) {
    e.preventDefault();
    if (!chatInput.trim() || !demoName) return;
    setLoading(true);
    setError("");
    const userMsg = chatInput;
    setChatInput("");
    try {
      const form = new FormData();
      form.append("user_input", userMsg);
      const res = await fetch(`${API}/chat/${encodeURIComponent(demoName)}`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || JSON.stringify(data));
      setChatHistory(h => [...h, { q: userMsg, a: data.response }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 800, margin: "40px auto", fontFamily: "system-ui, sans-serif", padding: "0 16px" }}>
      <h1>🔧 ForgeAI – The Agent That Builds Agents</h1>

      <section style={{ background: "#f5f5f5", padding: 20, borderRadius: 8, marginBottom: 24 }}>
        <h2>Step 1: Create an Agent</h2>
        <form onSubmit={handleCreate}>
          <div style={{ marginBottom: 12 }}>
            <label><strong>Brief</strong><br />
              <textarea
                value={brief}
                onChange={e => setBrief(e.target.value)}
                rows={3}
                style={{ width: "100%", padding: 8, marginTop: 4 }}
                placeholder="Describe what your agent should do..."
              />
            </label>
          </div>
          <div style={{ marginBottom: 12 }}>
            <label><strong>Upload PDF (optional)</strong><br />
              <input type="file" accept=".pdf" onChange={e => setFile(e.target.files[0])} style={{ marginTop: 4 }} />
            </label>
          </div>
          <button type="submit" disabled={loading} style={{ padding: "8px 20px", cursor: "pointer" }}>
            {loading ? "Creating…" : "Create Agent"}
          </button>
        </form>
        {createResult && (
          <div style={{ marginTop: 16, background: "#e6ffe6", padding: 12, borderRadius: 6 }}>
            <strong>✅ Agent created!</strong>
            <pre style={{ fontSize: 12, whiteSpace: "pre-wrap", marginTop: 8 }}>
              {JSON.stringify(createResult, null, 2)}
            </pre>
          </div>
        )}
      </section>

      {createResult && (
        <section style={{ background: "#f0f4ff", padding: 20, borderRadius: 8 }}>
          <h2>Step 2: Chat with "{demoName}"</h2>
          <div style={{ minHeight: 100, maxHeight: 300, overflowY: "auto", marginBottom: 12, background: "#fff", padding: 12, borderRadius: 6, border: "1px solid #ddd" }}>
            {chatHistory.length === 0 && <em style={{ color: "#888" }}>Ask a question…</em>}
            {chatHistory.map((m, i) => (
              <div key={i} style={{ marginBottom: 12 }}>
                <div><strong>You:</strong> {m.q}</div>
                <div style={{ marginTop: 4, color: "#333" }}><strong>Agent:</strong> {m.a}</div>
              </div>
            ))}
          </div>
          <form onSubmit={handleChat} style={{ display: "flex", gap: 8 }}>
            <input
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              placeholder="Type your question…"
              style={{ flex: 1, padding: 8 }}
            />
            <button type="submit" disabled={loading} style={{ padding: "8px 16px" }}>
              {loading ? "…" : "Send"}
            </button>
          </form>
        </section>
      )}

      {error && (
        <div style={{ marginTop: 16, background: "#ffe6e6", padding: 12, borderRadius: 6, color: "#c00" }}>
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
  );
}
