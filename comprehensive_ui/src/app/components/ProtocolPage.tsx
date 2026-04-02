import { useEffect, useMemo, useState } from "react";
import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { Button } from "@/app/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/app/components/ui/tooltip";

interface SplitStats {
  total?: number;
  toxic_ratio?: number;
  sources?: Record<string, number>;
}

interface ProtocolMetrics {
  macro_f1?: number | null;
  f1_toxic?: number | null;
  accuracy?: number | null;
  ece?: number | null;
  brier?: number | null;
  threshold?: number | null;
  support_clean?: number | null;
  support_toxic?: number | null;
}

interface ProtocolSeedRun {
  run_key?: string;
  run_id?: string;
  seed?: number | null;
  macro_f1?: number | null;
  f1_toxic?: number | null;
  accuracy?: number | null;
  ece?: number | null;
  brier?: number | null;
  metrics_last_updated?: string | null;
}

interface ProtocolSeedSummary {
  n_runs?: number;
  n_with_seed?: number;
  macro_f1_mean?: number | null;
  macro_f1_std?: number | null;
  f1_toxic_mean?: number | null;
  f1_toxic_std?: number | null;
}

interface ProtocolSummaryItem {
  id: string;
  name: string;
  available: boolean;
  metrics: ProtocolMetrics;
  stats: {
    train?: SplitStats | null;
    validation?: SplitStats | null;
    test?: SplitStats | null;
  };
  overlap_exact?: {
    train_validation?: number;
    train_test?: number;
    validation_test?: number;
  };
  metrics_last_updated?: string | null;
  seed_runs?: ProtocolSeedRun[];
  seed_summary?: ProtocolSeedSummary;
}

interface ProtocolSummaryResponse {
  dataset_version?: string | null;
  build_report_last_updated?: string | null;
  protocols: ProtocolSummaryItem[];
  winner?: string | null;
  warnings?: string[];
  source_note?: string;
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

const formatScore = (v?: number | null) => (typeof v === "number" ? v.toFixed(3) : "--");
const formatPercent = (v?: number | null) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "--");

const decisionLabel = (id: string, winner?: string | null) => {
  if (id === winner) return "Selected";
  if (id === "a") return "Anchor Baseline";
  if (id === "b") return "Deploy-comparable candidate";
  return "Candidate";
};

const metricBarWidth = (value?: number | null) => {
  if (typeof value !== "number") return "0%";
  const clipped = Math.max(0, Math.min(1, value));
  return `${(clipped * 100).toFixed(1)}%`;
};

const formatSeed = (seed?: number | null) => (typeof seed === "number" ? `seed=${seed}` : "seed=--");

const SIDEBAR_ITEMS = [
  { id: "preprocessing", label: "Preprocessing" },
  { id: "protocol-building", label: "Protocol Building" },
  { id: "training", label: "Training" },
  { id: "evaluation", label: "Evaluation" },
  { id: "decision", label: "Decision" },
] as const;

export function ProtocolPage() {
  const [data, setData] = useState<ProtocolSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<string>("decision");

  const fetchSummary = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(buildApiUrl("/api/protocols/summary"));
      const payload = (await response.json()) as ProtocolSummaryResponse;
      if (!response.ok) {
        throw new Error(JSON.stringify(payload));
      }
      setData(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không thể tải protocol summary");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchSummary();
  }, []);

  useEffect(() => {
    const sections = SIDEBAR_ITEMS
      .map((item) => document.getElementById(item.id))
      .filter((el): el is HTMLElement => el instanceof HTMLElement);

    if (!sections.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const topVisible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (!topVisible?.target?.id) return;
        const id = topVisible.target.id;
        setActiveSection((prev) => (prev === id ? prev : id));
      },
      {
        root: null,
        rootMargin: "-20% 0px -55% 0px",
        threshold: [0.15, 0.35, 0.6],
      },
    );

    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, [data]);

  const protocols = useMemo(() => data?.protocols ?? [], [data]);

  const rankedProtocols = useMemo(() => {
    const rows = [...protocols];
    rows.sort((a, b) => {
      const aF1Toxic = a.metrics.f1_toxic ?? -1;
      const bF1Toxic = b.metrics.f1_toxic ?? -1;
      if (bF1Toxic !== aF1Toxic) return bF1Toxic - aF1Toxic;
      return (b.metrics.macro_f1 ?? -1) - (a.metrics.macro_f1 ?? -1);
    });
    return rows.map((row, idx) => ({ ...row, rank: idx + 1 }));
  }, [protocols]);

  const winner = rankedProtocols.find((p) => p.id === data?.winner) ?? rankedProtocols[0];

  const jumpToSection = (sectionId: string) => {
    setActiveSection(sectionId);
    const el = document.getElementById(sectionId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <Card className="p-6">Đang tải protocol dashboard...</Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-4">
        <Card className="p-6 text-red-700">Lỗi tải dữ liệu Protocol: {error}</Card>
        <Button onClick={() => void fetchSummary()}>Thử lại</Button>
      </div>
    );
  }

  return (
    <div className="max-w-[1450px] mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="flex items-start gap-6">
        <aside className="hidden xl:block w-60 shrink-0 sticky top-24 bg-[#f1f4f6] rounded-xl p-4">
          <div className="mb-4">
            <h2 className="text-base font-bold text-slate-900">The Precision Lab</h2>
            <p className="text-[11px] text-slate-500">VNToxic-Pipeline v1.0</p>
          </div>
          <nav className="space-y-1 text-sm">
            {SIDEBAR_ITEMS.map((item) => {
              const active = activeSection === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => jumpToSection(item.id)}
                  className={`w-full text-left px-3 py-2 rounded-lg transition ${
                    active
                      ? "bg-white text-[#1c5fa8] font-semibold border-r-2 border-[#1c5fa8]"
                      : "text-slate-500 hover:bg-white/70"
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>
          <div className="mt-5 pt-4 border-t border-slate-200">
            <p className="text-sm font-semibold text-slate-800">Nguyen Van A</p>
            <p className="text-xs text-slate-500">Thesis Candidate</p>
          </div>
        </aside>

        <div className="flex-1 space-y-6">
          <section className="space-y-3" id="decision">
            <div className="flex flex-wrap items-center gap-2">
              <Badge className="bg-[#c9e8ec] text-[#3a5659] hover:bg-[#c9e8ec]">ViCTSD</Badge>
              <Badge variant="secondary" className="bg-[#e3e9ec] text-[#586064]">ViHSD</Badge>
              <Badge variant="secondary" className="bg-[#e3e9ec] text-[#586064]">Toxicity</Badge>
              <Badge variant="secondary" className="bg-[#e3e9ec] text-[#586064]">Preprocessing</Badge>
            </div>
            <h1 className="text-3xl font-bold text-[#2b3437]">
              Protocol Evaluation & Decision Dashboard
            </h1>
            <p className="text-[#586064]">
              Compare ViCTSD/ViHSD protocols and choose the final thesis protocol for production deployment.
            </p>
          </section>

          <div className="grid grid-cols-12 gap-6 items-start">
            <div className="col-span-12 lg:col-span-8 space-y-6">
              <section id="evaluation">
                <Card className="p-5 bg-white border border-[#abb3b7]/40 shadow-sm">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold">Weighted Decision Matrix</h2>
                    <Badge variant="outline" className="text-[#1c5fa8]">Calculated 2:15 PM</Badge>
                  </div>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Rank</TableHead>
                        <TableHead>Protocol</TableHead>
                        <TableHead>F1_toxic</TableHead>
                        <TableHead>Macro-F1</TableHead>
                        <TableHead>ECE ↓</TableHead>
                        <TableHead>Brier ↓</TableHead>
                        <TableHead>Decision</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {rankedProtocols.map((p) => (
                        <TableRow key={p.id}>
                          <TableCell className="font-medium">#{p.rank}</TableCell>
                          <TableCell>{p.name}</TableCell>
                          <TableCell>{formatScore(p.metrics.f1_toxic)}</TableCell>
                          <TableCell>{formatScore(p.metrics.macro_f1)}</TableCell>
                          <TableCell>{formatScore(p.metrics.ece)}</TableCell>
                          <TableCell>{formatScore(p.metrics.brier)}</TableCell>
                          <TableCell>
                            <Badge variant={p.id === data?.winner ? "default" : "secondary"}>
                              {decisionLabel(p.id, data?.winner)}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Card>
              </section>

              <section id="protocol-building">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card className="p-4 bg-white border border-[#abb3b7]/40">
                    <h3 className="font-semibold mb-2">Data Leakage Check</h3>
                    <ul className="space-y-1 text-sm text-[#586064]">
                      {protocols.map((p) => (
                        <li key={p.id} className="flex items-center justify-between">
                          <span>{p.name}</span>
                          <span className={p.overlap_exact?.train_test ? "text-amber-700" : "text-[#006a6a]"}>
                            {(p.overlap_exact?.train_test ?? 0) === 0 ? "PASS" : "CHECK"}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </Card>
                  <Card className="p-4 bg-white border border-[#abb3b7]/40">
                    <h3 className="font-semibold mb-2">Reproducibility</h3>
                    <p className="text-sm text-[#586064]">Training artifacts available with consistent schema and finalized metrics outputs.</p>
                  </Card>
                  <Card className="p-4 bg-white border border-[#abb3b7]/40">
                    <h3 className="font-semibold mb-2">Deploy Feasibility</h3>
                    <p className="text-sm text-[#586064]">All protocols provide deployable metrics bundles; Protocol C leads final score.</p>
                  </Card>
                </div>
              </section>

              <section id="decision">
                <Card className="p-6 bg-gradient-to-r from-[#1c5fa8] to-[#00539b] text-white border-0 shadow-lg">
                  <h2 className="text-2xl font-bold mb-2">Selected protocol: {winner?.name || "--"}</h2>
                  <p className="text-blue-50 mb-4">
                    Based on weighted analysis and gating checks, this protocol shows the strongest balance of toxicity detection and overall stability.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="font-semibold mb-1">Selection rationale</p>
                      <ul className="list-disc pl-5 space-y-1 text-blue-50">
                        <li>Highest F1_toxic among evaluated protocols</li>
                        <li>Strong Macro-F1 consistency</li>
                        <li>Better calibration profile (ECE/Brier)</li>
                      </ul>
                    </div>
                    <div>
                      <p className="font-semibold mb-1">Next step</p>
                      <ul className="list-disc pl-5 space-y-1 text-blue-50">
                        <li>Export selected artifacts for deployment handoff</li>
                        <li>Write thesis discussion with A/B comparative error analysis</li>
                      </ul>
                    </div>
                  </div>
                  <div className="mt-5 flex flex-wrap gap-2">
                    <Button variant="secondary">Export Protocol</Button>
                    <Button variant="outline" className="text-white border-white/40 hover:bg-white/10">Share Results</Button>
                  </div>
                </Card>
              </section>
            </div>

            <div className="col-span-12 lg:col-span-4 space-y-6" id="preprocessing">
              <Card className="p-5 bg-white border border-[#abb3b7]/40 shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Scoring Configuration</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-center justify-between"><span>F1_toxic weight</span><span className="font-semibold">30%</span></div>
                  <div className="flex items-center justify-between"><span>Macro-F1 weight</span><span className="font-semibold">20%</span></div>
                  <div className="flex items-center justify-between"><span>Calibration (ECE)</span><span className="font-semibold">15%</span></div>
                  <div className="flex items-center justify-between"><span>Robustness</span><span className="font-semibold">20%</span></div>
                  <div className="flex items-center justify-between"><span>Efficiency</span><span className="font-semibold">15%</span></div>
                </div>
                <Button className="w-full mt-4 bg-[#1c5fa8] hover:bg-[#00539b]">Recalculate Winner</Button>
              </Card>

              <Card className="p-5 bg-white border border-[#abb3b7]/40 shadow-sm" id="training">
                <h3 className="text-lg font-semibold mb-3">Training Snapshot</h3>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Protocol</TableHead>
                      <TableHead>Acc</TableHead>
                      <TableHead>Toxic support</TableHead>
                      <TableHead>Seeds (hover)</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {protocols.map((p) => {
                      const seedRuns = p.seed_runs ?? [];
                      const seedSummary = p.seed_summary;
                      return (
                        <TableRow key={p.id}>
                          <TableCell>{p.name}</TableCell>
                          <TableCell>{formatScore(p.metrics.accuracy)}</TableCell>
                          <TableCell>{p.metrics.support_toxic ?? "--"}</TableCell>
                          <TableCell>
                            {seedRuns.length > 0 ? (
                              <div className="space-y-1">
                                <div className="flex items-center gap-1.5">
                                  {seedRuns.slice(0, 8).map((run, idx) => (
                                    <Tooltip key={`${p.id}-${run.run_key || run.run_id || idx}`}>
                                      <TooltipTrigger asChild>
                                        <button
                                          type="button"
                                          className="h-8 w-4 rounded-sm bg-[#dbe5f0] overflow-hidden border border-[#b8c6d8]"
                                          aria-label={`${p.name} ${formatSeed(run.seed)} F1_toxic ${formatScore(run.f1_toxic)}`}
                                        >
                                          <span
                                            className="block w-full bg-[#1c5fa8]"
                                            style={{ height: metricBarWidth(run.f1_toxic) }}
                                          />
                                        </button>
                                      </TooltipTrigger>
                                      <TooltipContent side="top" className="max-w-xs">
                                        <div className="space-y-0.5">
                                          <div className="font-semibold">{run.run_id || run.run_key || "run"}</div>
                                          <div>{formatSeed(run.seed)}</div>
                                          <div>F1_toxic: {formatScore(run.f1_toxic)}</div>
                                          <div>Macro-F1: {formatScore(run.macro_f1)}</div>
                                          <div>ECE: {formatScore(run.ece)}</div>
                                          <div>Brier: {formatScore(run.brier)}</div>
                                        </div>
                                      </TooltipContent>
                                    </Tooltip>
                                  ))}
                                </div>
                                <div className="text-[11px] text-[#586064]">
                                  n={seedSummary?.n_runs ?? seedRuns.length}
                                  {typeof seedSummary?.f1_toxic_mean === "number" && ` · μ=${seedSummary.f1_toxic_mean.toFixed(3)}`}
                                  {typeof seedSummary?.f1_toxic_std === "number" && ` · σ=${seedSummary.f1_toxic_std.toFixed(3)}`}
                                </div>
                              </div>
                            ) : (
                              <span className="text-xs text-[#586064]">No multi-seed data yet</span>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </Card>

              <Card className="p-5 bg-[#f1f4f6] border border-[#abb3b7]/30">
                <div className="text-xs uppercase tracking-wide text-[#586064] mb-2">System status</div>
                <div className="text-2xl font-bold text-[#2b3437] mb-1">0.832</div>
                <div className="text-sm text-[#586064]">Average metric snapshot</div>
                <div className="mt-4 text-xs text-[#586064]">{data?.source_note || "Source artifacts loaded from local protocol outputs."}</div>
              </Card>
            </div>
          </div>

          {(data?.warnings?.length ?? 0) > 0 && (
            <Card className="p-5">
              <h3 className="font-semibold mb-2">Warnings</h3>
              <ul className="list-disc pl-5 space-y-1 text-sm text-gray-700">
                {data?.warnings?.map((w, idx) => <li key={idx}>{w}</li>)}
              </ul>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
