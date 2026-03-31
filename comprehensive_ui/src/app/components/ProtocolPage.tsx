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

const formatScore = (v?: number | null) => (typeof v === "number" ? v.toFixed(4) : "--");
const formatPercent = (v?: number | null) => (typeof v === "number" ? `${(v * 100).toFixed(2)}%` : "--");

const decisionLabel = (id: string, winner?: string | null) => {
  if (id === winner) return "Selected";
  if (id === "a") return "Anchor Baseline";
  if (id === "b") return "Deploy-comparable candidate";
  return "Candidate";
};

const SIDEBAR_ITEMS = [
  { id: "raw-data", label: "Raw Data" },
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

  const winnerName = rankedProtocols.find((p) => p.id === data?.winner)?.name || "--";

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
    <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="flex items-start gap-6">
        <aside className="hidden xl:block w-64 shrink-0 sticky top-24 bg-slate-50 border border-slate-200 rounded-xl p-4">
          <div className="mb-4">
            <h2 className="text-lg font-bold text-slate-900">The Precision Lab</h2>
            <p className="text-xs text-slate-500">VNToxic-Pipeline v1.0</p>
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
                      ? "bg-blue-50 text-blue-700 font-semibold border border-blue-200"
                      : "text-slate-500 hover:bg-slate-100"
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

        <div className="flex-1 space-y-8">
          <section className="space-y-3" id="decision">
            <div className="flex flex-wrap items-center gap-2">
              <Badge>ViCTSD</Badge>
              <Badge variant="secondary">ViHSD</Badge>
              <Badge variant="outline">Protocol Plan</Badge>
            </div>
            <h1 className="text-3xl font-bold" style={{ color: "var(--viet-primary)" }}>
              Protocol Evaluation & Decision Dashboard
            </h1>
            <p className="text-gray-600">
              So sánh Protocol A/B/C bằng metrics thực nghiệm và bằng chứng build để chốt protocol cuối cho thesis.
            </p>
          </section>

          <section id="raw-data">
            <Card className="p-6 space-y-4 bg-white border border-slate-200">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Import artifacts from Colab (Simulated)</h2>
                <Badge variant="outline">Ready for scoring</Badge>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-gray-500 mb-2">Google Drive Artifact URL</p>
                    <div className="flex gap-2">
                      <input
                        className="flex-1 px-3 py-2 rounded-md border border-gray-300 bg-gray-50 text-sm"
                        value="https://drive.google.com/drive/folders/1z9X-k9..."
                        readOnly
                      />
                      <Button variant="outline" size="sm">Validate</Button>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" className="flex-1">Load sample artifacts</Button>
                    <Button size="sm" className="flex-1">Import and validate</Button>
                  </div>
                </div>
                <div className="rounded-lg bg-slate-50 border border-slate-200 p-4">
                  <p className="text-xs uppercase tracking-wide text-gray-500 mb-3">Artifact Manifest</p>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center justify-between"><span>metrics_a.json</span><span className="text-teal-700">Verified</span></li>
                    <li className="flex items-center justify-between"><span>metrics_b.json</span><span className="text-teal-700">Verified</span></li>
                    <li className="flex items-center justify-between"><span>metrics_c.json</span><span className="text-teal-700">Verified</span></li>
                    <li className="flex items-center justify-between"><span>victsd_v1_protocol_build_report.json</span><span className="text-teal-700">Verified</span></li>
                  </ul>
                </div>
              </div>
            </Card>
          </section>

          <section id="preprocessing">
            <Card className="p-5 space-y-3">
              <h2 className="text-xl font-semibold">Preprocessing</h2>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-gray-700">
                <li className="rounded-md bg-slate-50 px-3 py-2">Trim leading/trailing whitespace</li>
                <li className="rounded-md bg-slate-50 px-3 py-2">Normalize Unicode to NFC</li>
                <li className="rounded-md bg-slate-50 px-3 py-2">Normalize whitespace (collapse multiple spaces)</li>
                <li className="rounded-md bg-slate-50 px-3 py-2">Preserve case, punctuation, emoji for toxicity signals</li>
              </ul>
            </Card>
          </section>

          <section id="protocol-building">
            <Card className="p-5 space-y-4">
              <h2 className="text-xl font-semibold">Protocol Building (Leakage Evidence)</h2>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Protocol</TableHead>
                    <TableHead>Train total</TableHead>
                    <TableHead>Train toxic ratio</TableHead>
                    <TableHead>Val total</TableHead>
                    <TableHead>Test total</TableHead>
                    <TableHead>Overlap train-val</TableHead>
                    <TableHead>Overlap train-test</TableHead>
                    <TableHead>Overlap val-test</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {protocols.map((p) => (
                    <TableRow key={p.id}>
                      <TableCell className="font-medium">{p.name}</TableCell>
                      <TableCell>{p.stats.train?.total ?? "--"}</TableCell>
                      <TableCell>{formatPercent(p.stats.train?.toxic_ratio)}</TableCell>
                      <TableCell>{p.stats.validation?.total ?? "--"}</TableCell>
                      <TableCell>{p.stats.test?.total ?? "--"}</TableCell>
                      <TableCell>{p.overlap_exact?.train_validation ?? "--"}</TableCell>
                      <TableCell>{p.overlap_exact?.train_test ?? "--"}</TableCell>
                      <TableCell>{p.overlap_exact?.validation_test ?? "--"}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          </section>

          <section id="training">
            <Card className="p-5 space-y-4">
              <h2 className="text-xl font-semibold">Training Artifacts</h2>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Protocol</TableHead>
                    <TableHead>Accuracy</TableHead>
                    <TableHead>Support clean</TableHead>
                    <TableHead>Support toxic</TableHead>
                    <TableHead>Metrics updated</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {protocols.map((p) => (
                    <TableRow key={p.id}>
                      <TableCell className="font-medium">{p.name}</TableCell>
                      <TableCell>{formatScore(p.metrics.accuracy)}</TableCell>
                      <TableCell>{p.metrics.support_clean ?? "--"}</TableCell>
                      <TableCell>{p.metrics.support_toxic ?? "--"}</TableCell>
                      <TableCell>{p.metrics_last_updated || "--"}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          </section>

          <section id="evaluation">
            <Card className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Weighted Decision Matrix</h2>
                <Badge variant="outline">Calculated from latest artifacts</Badge>
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

          <section id="decision" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="p-5">
                <div className="text-sm text-gray-500">Winner</div>
                <div className="text-lg font-semibold mt-1">{winnerName}</div>
              </Card>
              <Card className="p-5">
                <div className="text-sm text-gray-500">Dataset version</div>
                <div className="text-lg font-semibold mt-1">{data?.dataset_version || "--"}</div>
              </Card>
              <Card className="p-5">
                <div className="text-sm text-gray-500">Build report updated</div>
                <div className="text-lg font-semibold mt-1">{data?.build_report_last_updated || "--"}</div>
              </Card>
            </div>

            {(data?.warnings?.length ?? 0) > 0 && (
              <Card className="p-5">
                <h3 className="font-semibold mb-2">Warnings</h3>
                <ul className="list-disc pl-5 space-y-1 text-sm text-gray-700">
                  {data?.warnings?.map((w, idx) => <li key={idx}>{w}</li>)}
                </ul>
              </Card>
            )}

            <Card className="p-4 text-sm text-gray-600">
              {data?.source_note || "Source artifacts loaded from local protocol outputs."}
            </Card>
          </section>
        </div>
      </div>
    </div>
  );
}
