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
import { useI18n } from "@/app/i18n/context";

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

interface ArtifactVersions {
  dataset_version?: string;
  model_version?: string;
  policy_version?: string;
}

interface LeakageEvidence {
  train_validation?: number;
  train_test?: number;
  validation_test?: number;
  has_train_test_leakage?: boolean;
  has_any_overlap?: boolean;
}

interface DomainMismatchEvidence {
  risk_level?: string;
  vihsd_train_ratio?: number;
  train_source_mix?: Record<string, number>;
  summary?: string;
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
  source_mix_by_split?: {
    train?: Record<string, number>;
    validation?: Record<string, number>;
    test?: Record<string, number>;
  };
  overlap_exact?: {
    train_validation?: number;
    train_test?: number;
    validation_test?: number;
  };
  leakage_evidence?: LeakageEvidence;
  domain_mismatch?: DomainMismatchEvidence;
  artifact_versions?: ArtifactVersions;
  metrics_last_updated?: string | null;
  seed_runs?: ProtocolSeedRun[];
  seed_summary?: ProtocolSeedSummary;
}

interface ProtocolSummaryResponse {
  dataset_version?: string | null;
  model_version?: string | null;
  policy_version?: string | null;
  artifact_versions?: ArtifactVersions;
  missing_required_versions?: string[];
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
const formatSeed = (seed?: number | null) => (typeof seed === "number" ? `seed=${seed}` : "seed=?");

const metricBarWidth = (value?: number | null) => {
  if (typeof value !== "number") return "0%";
  const clipped = Math.max(0, Math.min(1, value));
  return `${(clipped * 100).toFixed(1)}%`;
};

const decisionLabel = (id: string, winner: string | null | undefined, t: (key: string) => string) => {
  if (id === winner) return t("protocol.labels.selected");
  if (id === "a") return t("protocol.labels.anchor");
  if (id === "b") return t("protocol.labels.deployCandidate");
  return t("protocol.labels.candidate");
};

const renderSourceMix = (sources?: Record<string, number>) => {
  const entries = Object.entries(sources || {});
  if (!entries.length) return "--";
  return entries.map(([name, count]) => `${name}: ${count}`).join(" · ");
};

export function ProtocolPage() {
  const { t } = useI18n();
  const [data, setData] = useState<ProtocolSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
      setError(err instanceof Error ? err.message : t("protocol.cannotLoadSummary"));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchSummary();
  }, [t]);

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

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <Card className="p-6">{t("protocol.loadingDashboard")}</Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-4">
        <Card className="p-6 text-text-danger">{t("protocol.loadErrorPrefix")} {error}</Card>
        <Button onClick={() => void fetchSummary()}>{t("protocol.retry")}</Button>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-6">
      <section className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge className="bg-background-info text-text-info">{t("protocol.tags.victsd")}</Badge>
          <Badge variant="secondary" className="bg-muted text-muted-foreground">{t("protocol.tags.vihsd")}</Badge>
          <Badge variant="secondary" className="bg-muted text-muted-foreground">{t("protocol.tags.toxicity")}</Badge>
          <Badge variant="secondary" className="bg-muted text-muted-foreground">{t("protocol.tags.leakage")}</Badge>
        </div>
        <h1 className="text-3xl font-bold text-foreground">{t("protocol.title")}</h1>
        <p className="text-muted-foreground">{t("protocol.subtitle")}</p>
      </section>

      <Card className="p-4 bg-background-secondary border border-border/40">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-sm">
          <div>
            <p className="text-muted-foreground">{t("protocol.version.dataset")}</p>
            <p className="font-medium">{data?.artifact_versions?.dataset_version || data?.dataset_version || "--"}</p>
          </div>
          <div>
            <p className="text-muted-foreground">{t("protocol.version.model")}</p>
            <p className="font-medium">{data?.artifact_versions?.model_version || data?.model_version || "--"}</p>
          </div>
          <div>
            <p className="text-muted-foreground">{t("protocol.version.policy")}</p>
            <p className="font-medium">{data?.artifact_versions?.policy_version || data?.policy_version || "--"}</p>
          </div>
          <div>
            <p className="text-muted-foreground">{t("protocol.version.updated")}</p>
            <p className="font-medium">{data?.build_report_last_updated || "--"}</p>
          </div>
        </div>
        {(data?.missing_required_versions?.length ?? 0) > 0 && (
          <p className="text-xs text-text-warning mt-3">
            {t("protocol.version.missing")}: {data?.missing_required_versions?.join(", ")}
          </p>
        )}
      </Card>

      <Card className="p-5 bg-card border border-border/40 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">{t("protocol.matrix.title")}</h2>
          <Badge variant="outline" className="text-text-info">{t("protocol.matrix.rankedBy")}</Badge>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("protocol.matrix.columns.rank")}</TableHead>
              <TableHead>{t("protocol.matrix.columns.protocol")}</TableHead>
              <TableHead>{t("protocol.matrix.columns.f1Toxic")}</TableHead>
              <TableHead>{t("protocol.matrix.columns.macroF1")}</TableHead>
              <TableHead>{t("protocol.matrix.columns.ece")}</TableHead>
              <TableHead>{t("protocol.matrix.columns.brier")}</TableHead>
              <TableHead>{t("protocol.matrix.columns.decision")}</TableHead>
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
                    {decisionLabel(p.id, data?.winner, t)}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {protocols.map((p) => (
          <Card key={p.id} className="p-5 bg-card border border-border/40">
            <div className="flex items-center justify-between gap-3 mb-3">
              <h3 className="text-lg font-semibold">{p.name}</h3>
              <Badge variant={(p.leakage_evidence?.has_train_test_leakage ?? false) ? "destructive" : "secondary"}>
                {(p.leakage_evidence?.has_train_test_leakage ?? false)
                  ? t("protocol.evidence.leakageCheck")
                  : t("protocol.evidence.leakagePass")}
              </Badge>
            </div>

            <div className="space-y-2 text-sm">
              <p>
                <span className="text-muted-foreground">{t("protocol.evidence.trainSourceMix")}: </span>
                {renderSourceMix(p.source_mix_by_split?.train)}
              </p>
              <p>
                <span className="text-muted-foreground">{t("protocol.evidence.overlapTrainTest")}: </span>
                {p.leakage_evidence?.train_test ?? p.overlap_exact?.train_test ?? 0}
              </p>
              <p>
                <span className="text-muted-foreground">{t("protocol.evidence.vihsdRatio")}: </span>
                {formatPercent(p.domain_mismatch?.vihsd_train_ratio ?? null)}
              </p>
              <p>
                <span className="text-muted-foreground">{t("protocol.evidence.risk")}: </span>
                {(p.domain_mismatch?.risk_level || "--").toUpperCase()}
              </p>
            </div>

            <div className="mt-3 rounded-lg border-l-4 border-l-border-info bg-background-info p-3 text-sm text-text-info">
              {p.domain_mismatch?.summary || t("protocol.evidence.noSummary")}
            </div>
          </Card>
        ))}
      </div>

      <Card className="p-5 bg-card border border-border/40 shadow-sm">
        <h2 className="text-lg font-semibold mb-3">{t("protocol.seed.title")}</h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{t("protocol.seed.protocol")}</TableHead>
              <TableHead>{t("protocol.seed.samples")}</TableHead>
              <TableHead>{t("protocol.seed.f1ToxicMean")}</TableHead>
              <TableHead>{t("protocol.seed.seedRuns")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {protocols.map((p) => {
              const seedRuns = p.seed_runs ?? [];
              const seedSummary = p.seed_summary;
              return (
                <TableRow key={`${p.id}-seed`}>
                  <TableCell>{p.name}</TableCell>
                  <TableCell>{seedSummary?.n_runs ?? 0}</TableCell>
                  <TableCell>{formatScore(seedSummary?.f1_toxic_mean)}</TableCell>
                  <TableCell>
                    {seedRuns.length ? (
                      <div className="flex items-center gap-1.5">
                        {seedRuns.slice(0, 8).map((run, idx) => (
                          <Tooltip key={`${p.id}-${run.run_key || run.run_id || idx}`}>
                            <TooltipTrigger asChild>
                              <button
                                type="button"
                                className="h-8 w-4 rounded-sm bg-background-info overflow-hidden border border-border-info"
                                aria-label={`${p.name} ${formatSeed(run.seed)} F1_toxic ${formatScore(run.f1_toxic)}`}
                              >
                                <span
                                  className="block w-full bg-primary"
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
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    ) : (
                      <span className="text-xs text-muted-foreground">{t("protocol.seed.noData")}</span>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Card>

      <Card className="p-6 bg-gradient-to-r from-text-info to-primary text-white border-0 shadow-lg">
        <h2 className="text-2xl font-bold mb-2">
          {t("protocol.selection.title", { protocol: winner?.name || "--" })}
        </h2>
        <p className="text-background-info mb-3">{t("protocol.selection.subtitle")}</p>
        <p className="text-sm text-background-info">{data?.source_note || t("protocol.selection.sourceFallback")}</p>
      </Card>

      {(data?.warnings?.length ?? 0) > 0 && (
        <Card className="p-5">
          <h3 className="font-semibold mb-2">{t("protocol.warnings.title")}</h3>
          <ul className="list-disc pl-5 space-y-1 text-sm text-foreground">
            {data?.warnings?.map((w, idx) => <li key={idx}>{w}</li>)}
          </ul>
        </Card>
      )}
    </div>
  );
}
