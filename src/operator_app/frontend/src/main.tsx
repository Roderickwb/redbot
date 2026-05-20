import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  Bot,
  Check,
  CirclePause,
  Clock3,
  FileText,
  Lock,
  MessageSquare,
  RefreshCw,
  Shield,
  Snowflake,
  X
} from "lucide-react";
import "./styles.css";

type AnyRecord = Record<string, any>;

type Bundle = {
  snapshot: AnyRecord;
  cockpit: AnyRecord;
  recommendations: AnyRecord;
  quality: AnyRecord;
  positions: AnyRecord;
  trades: AnyRecord;
  safety: AnyRecord;
};

const tabs = ["Cockpit", "Recommendations", "Positions", "Safety"] as const;
type Tab = (typeof tabs)[number];

async function fetchJson(path: string) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function asNumber(value: any, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function Badge({ value }: { value: string | number | undefined }) {
  const text = String(value ?? "unknown");
  const kind = text.toLowerCase();
  return <span className={`badge ${kind}`}>{text}</span>;
}

function Metric({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {sub ? <small>{sub}</small> : null}
    </div>
  );
}

function App() {
  const [data, setData] = useState<Bundle | null>(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<Tab>("Cockpit");
  const [busy, setBusy] = useState(false);

  async function load() {
    setBusy(true);
    setError("");
    try {
      const [snapshot, cockpit, recommendations, quality, positions, trades, safety] = await Promise.all([
        fetchJson("/api/snapshot"),
        fetchJson("/api/cockpit"),
        fetchJson("/api/recommendations"),
        fetchJson("/api/recommendation-quality"),
        fetchJson("/api/positions"),
        fetchJson("/api/trades?limit=80"),
        fetchJson("/api/safety")
      ]);
      setData({ snapshot, cockpit, recommendations, quality, positions, trades, safety });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 60_000);
    return () => window.clearInterval(timer);
  }, []);

  const decision = data?.cockpit?.daily_decision ?? {};
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Red Bot Operator</div>
          <h1>{decision.label ?? "Loading cockpit"}</h1>
        </div>
        <button className="icon-btn" onClick={load} disabled={busy} title="Refresh">
          <RefreshCw size={19} />
        </button>
      </header>

      {error ? <div className="alert"><AlertTriangle size={17} />{error}</div> : null}

      <nav className="tabs">
        {tabs.map((item) => (
          <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>
            {item}
          </button>
        ))}
      </nav>

      {!data ? <div className="loading">Loading operator data...</div> : null}
      {data && tab === "Cockpit" ? <Cockpit data={data} /> : null}
      {data && tab === "Recommendations" ? <Recommendations data={data} reload={load} /> : null}
      {data && tab === "Positions" ? <Positions data={data} /> : null}
      {data && tab === "Safety" ? <Safety data={data} /> : null}
    </div>
  );
}

function Cockpit({ data }: { data: Bundle }) {
  const cockpit = data.cockpit;
  const learning = cockpit.learning ?? {};
  const risk = cockpit.risk ?? {};
  const live = cockpit.live_changes ?? {};
  const safety = cockpit.safety ?? {};
  const rec = cockpit.recommendations ?? {};
  return (
    <main className="stack">
      <section className="panel status-panel">
        <div>
          <span className="eyebrow">Status</span>
          <h2>{cockpit.status}</h2>
          <p>{cockpit.daily_decision?.reason}</p>
        </div>
        <Badge value={live.status} />
      </section>

      <section className="grid">
        <Metric label="ML" value={`${learning.ml_status ?? "?"}`} sub={`AUC ${learning.ml_auc ?? "-"} | rows ${learning.ml_rows ?? 0}`} />
        <Metric label="Market" value={learning.market_regime ?? "-"} sub={learning.market_bias ?? ""} />
        <Metric label="Recommendations" value={`${rec.needs_operator_review ?? 0} review`} sub={`${rec.quality_tracked ?? 0} tracked`} />
        <Metric label="Safety" value={safety.status ?? "-"} sub={`live entries ${String(safety.live_entry_orders_allowed)}`} />
      </section>

      <section className="panel">
        <h3>Risk & Learning</h3>
        <div className="list">
          <Row label="Risk advice" value={`${risk.advice_verdict ?? "-"} | stable ${risk.advice_stable_data_down ?? 0}`} />
          <Row label="Risk history" value={`${risk.history_verdict ?? "-"} | net R ${risk.history_net_saved_r ?? 0}`} />
          <Row label="Guards" value={`${risk.guard_verdict ?? "-"} | issue ${risk.guard_primary_issue ?? "-"}`} />
          <Row label="Exits" value={`${risk.exit_verdict ?? "-"} | pnl ${risk.exit_total_pnl_eur ?? 0}`} />
          <Row label="Lifecycle" value={`${risk.lifecycle_verdict ?? "-"} | issues ${risk.lifecycle_issues ?? 0}`} />
        </div>
      </section>

      <section className="panel">
        <h3>Next</h3>
        {(cockpit.next_actions ?? []).slice(0, 5).map((item: string) => <p className="next" key={item}>{item}</p>)}
      </section>
    </main>
  );
}

function Recommendations({ data, reload }: { data: Bundle; reload: () => Promise<void> }) {
  const items = data.recommendations.items ?? [];
  const quality = data.quality.summary ?? {};
  const grouped = useMemo(() => ({
    review: items.filter((i: AnyRecord) => i.status === "needs_operator_review"),
    wait: items.filter((i: AnyRecord) => i.status === "wait_more_evidence"),
    context: items.filter((i: AnyRecord) => i.status === "auto_accept_as_context"),
    blocked: items.filter((i: AnyRecord) => i.status === "blocked")
  }), [items]);
  return (
    <main className="stack">
      <section className="grid">
        <Metric label="Tracked" value={quality.tracked_items ?? 0} sub={`${quality.days_observed ?? 0} days`} />
        <Metric label="Attention" value={quality.needs_attention ?? 0} sub={`${quality.unstable ?? 0} unstable`} />
        <Metric label="Review" value={grouped.review.length} sub={`${grouped.wait.length} waiting`} />
        <Metric label="Context" value={grouped.context.length} sub={`${grouped.blocked.length} blocked`} />
      </section>
      {items.map((item: AnyRecord) => <RecommendationCard key={item.id} item={item} reload={reload} />)}
    </main>
  );
}

function RecommendationCard({ item, reload }: { item: AnyRecord; reload: () => Promise<void> }) {
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState("");
  async function decide(action: string) {
    setSaving(action);
    try {
      await fetch("/api/decisions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_id: item.id,
          source_type: "recommendation",
          action,
          reason: note,
          source_path: "analysis/recommendations/latest_recommendation_aggregator.json"
        })
      });
      setNote("");
      await reload();
    } finally {
      setSaving("");
    }
  }
  return (
    <article className="panel recommendation">
      <div className="card-head">
        <div>
          <h3>{item.title}</h3>
          <p>{item.headline}</p>
        </div>
        <Badge value={item.status} />
      </div>
      <p className="why">{item.why}</p>
      <textarea value={note} onChange={(e) => setNote(e.target.value)} placeholder="Note" />
      <div className="actions">
        <button onClick={() => decide("approve")} disabled={!!saving} title="Approve intent"><Check size={16} />Approve</button>
        <button onClick={() => decide("reject")} disabled={!!saving} title="Reject intent"><X size={16} />Reject</button>
        <button onClick={() => decide("wait")} disabled={!!saving} title="Wait"><Clock3 size={16} />Wait</button>
        <button onClick={() => decide("freeze")} disabled={!!saving} title="Freeze"><Snowflake size={16} />Freeze</button>
        <button onClick={() => decide("note")} disabled={!!saving} title="Save note"><MessageSquare size={16} />Note</button>
      </div>
    </article>
  );
}

function Positions({ data }: { data: Bundle }) {
  const positions = data.positions.summary ?? {};
  const lifecycles = data.positions.lifecycles ?? [];
  const trades = data.trades.rows ?? [];
  return (
    <main className="stack">
      <section className="grid">
        <Metric label="Lifecycle" value={positions.verdict ?? "-"} sub={`${positions.issue_count ?? 0} issues`} />
        <Metric label="Open" value={positions.open_masters ?? 0} sub={`${positions.closed_masters ?? 0} closed`} />
        <Metric label="Child trades" value={positions.child_trades ?? 0} sub={`${positions.master_trades ?? 0} masters`} />
        <Metric label="High issues" value={positions.high_issues ?? 0} sub={`${positions.medium_issues ?? 0} medium`} />
      </section>
      <section className="panel">
        <h3>Open Positions</h3>
        {lifecycles.filter((p: AnyRecord) => p.status !== "closed").slice(0, 10).map((p: AnyRecord) => (
          <Row key={p.master_trade_id} label={`${p.symbol} ${p.position_id ?? ""}`} value={`${p.status} | amount ${p.master_amount} | pnl ${p.realized_pnl_eur}`} />
        ))}
      </section>
      <section className="panel">
        <h3>Recent Trades</h3>
        <div className="trade-list">
          {trades.slice(0, 30).map((t: AnyRecord) => (
            <div className="trade" key={t.id}>
              <strong>{t.symbol}</strong>
              <span>{t.side} {t.status}</span>
              <span>{t.price}</span>
              <span>{t.pnl_eur ?? 0}</span>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

function Safety({ data }: { data: Bundle }) {
  const safety = data.cockpit.safety ?? data.safety ?? {};
  const live = data.cockpit.live_changes ?? {};
  return (
    <main className="stack">
      <section className="panel status-panel">
        <Shield size={28} />
        <div>
          <h2>{safety.status ?? "UNKNOWN"}</h2>
          <p>{safety.reason ?? "Safety report loaded."}</p>
        </div>
      </section>
      <section className="grid">
        <Metric label="Kill" value={String(safety.kill_switch_active)} />
        <Metric label="Meltdown" value={String(safety.meltdown_active)} />
        <Metric label="Live entries" value={String(safety.live_entry_orders_allowed)} />
        <Metric label="Live enforcement" value={String(live.live_enforcement)} />
      </section>
      <section className="panel">
        <h3>Blocked In App V1</h3>
        <div className="list">
          <Row label="Enable live enforcement" value="forbidden" />
          <Row label="Risk-up live" value="forbidden" />
          <Row label="Entry-rule live" value="forbidden" />
          <Row label="Clear kill-switch" value="admin flow only" />
        </div>
      </section>
    </main>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
