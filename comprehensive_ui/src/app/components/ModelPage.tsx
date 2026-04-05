import { useEffect, useMemo, useState } from "react";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/components/ui/table";
import { Badge } from "@/app/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/app/components/ui/dialog";
import { Label } from "@/app/components/ui/label";
import {
  ArrowRight,
  CheckCircle,
  XCircle,
  TrendingUp,
  Database,
  Cpu,
  LineChart,
  FileText,
  Download,
} from "lucide-react";
import {
  LineChart as RechartsLine,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { useI18n } from "@/app/i18n/context";

interface ModelPageProps {
  onTryNow: () => void;
}

interface RegistryRun {
  run_id: string;
  model_name: string;
  dataset_version: string;
  created_at: string;
  checkpoint_path: string;
  hyperparameters: Record<string, unknown>;
  metrics: Record<string, number>;
  is_baseline?: boolean;
}

interface RegistryResponse {
  runs: RegistryRun[];
  last_updated?: string | null;
}

interface EvalPolicyResponse {
  policy: {
    split?: { train?: number; val?: number; test?: number };
    test_set_fixed?: boolean;
    fixed_since?: string;
    hard_case_subsets_evaluated?: boolean;
    note?: string;
  };
  last_updated?: string | null;
}

interface ErrorRow {
  text: string;
  true_label: number;
  predicted_label: number;
  confidence?: number;
  source_dataset?: string;
  subset_tag?: string;
}

interface ErrorResponse {
  items: ErrorRow[];
  last_updated?: string | null;
}

interface HardCaseRow extends ErrorRow {
  candidate_reason?: string;
}

interface HardCaseResponse {
  items: HardCaseRow[];
  last_updated?: string | null;
}

interface PreprocessStep {
  id: string;
  label: string;
  active: boolean;
}

interface PreprocessResponse {
  steps: PreprocessStep[];
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

const formatRatio = (value?: number) => (value === undefined ? "--" : `${(value * 100).toFixed(1)}%`);

const formatScore = (value: number | null | undefined, t: (key: string) => string) =>
  (value == null ? t("model.common.noMetrics") : value.toFixed(3));


const toVisibleWhitespace = (text: string) =>
  text
    .replace(/ /g, "·")
    .replace(/\t/g, "↹")
    .replace(/\n/g, "↵\n");

const applyActiveSteps = (text: string, steps: PreprocessStep[]) => {
  let output = text;
  steps.forEach((step) => {
    if (!step.active) return;
    if (step.id === "trim") {
      output = output.trim();
    }
    if (step.id === "normalize_unicode") {
      output = output.normalize("NFC");
    }
    if (step.id === "normalize_whitespace") {
      output = output.replace(/\s+/g, " ").trim();
    }
  });
  return output;
};

export function ModelPage({ onTryNow }: ModelPageProps) {
  const { t } = useI18n();
  const [registry, setRegistry] = useState<RegistryResponse>({ runs: [] });
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [preprocessSteps, setPreprocessSteps] = useState<PreprocessStep[]>([]);
  const [policy, setPolicy] = useState<EvalPolicyResponse | null>(null);
  const [errorRows, setErrorRows] = useState<ErrorRow[]>([]);
  const [errorLastUpdated, setErrorLastUpdated] = useState<string | null>(null);
  const [hardCases, setHardCases] = useState<HardCaseRow[]>([]);
  const [hardCaseLastUpdated, setHardCaseLastUpdated] = useState<string | null>(null);
  const [errorFilter, setErrorFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [subsetFilter, setSubsetFilter] = useState("all");
  const [demoStepIndex, setDemoStepIndex] = useState(0);
  const [demoExampleIndex, setDemoExampleIndex] = useState(0);

  const demoExamples = useMemo(
    () => [
      t("model.pipeline.demoExample1"),
      t("model.pipeline.demoExample2"),
      t("model.pipeline.demoExample3"),
    ],
    [t],
  );

  const fetchRegistry = async (refresh = false) => {
    try {
      const endpoint = refresh ? "/api/experiments/registry?refresh=true" : "/api/experiments/registry";
      const response = await fetch(buildApiUrl(endpoint));
      const data = (await response.json()) as RegistryResponse;
      if (!response.ok) throw new Error(JSON.stringify(data));
      setRegistry(data);
      setRegistryError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("model.status.cannotLoadRegistry");
      setRegistryError(message);
    }
  };

  useEffect(() => {

    const fetchSteps = async () => {
      try {
        const response = await fetch(buildApiUrl("/api/preprocessing/steps"));
        const data = (await response.json()) as PreprocessResponse;
        if (!response.ok) throw new Error(JSON.stringify(data));
        setPreprocessSteps(data.steps || []);
      } catch {
        setPreprocessSteps([]);
      }
    };

    const fetchPolicy = async () => {
      try {
        const response = await fetch(buildApiUrl("/api/eval/policy"));
        const data = (await response.json()) as EvalPolicyResponse;
        if (!response.ok) throw new Error(JSON.stringify(data));
        setPolicy(data);
      } catch {
        setPolicy(null);
      }
    };

    const fetchErrors = async () => {
      try {
        const response = await fetch(buildApiUrl("/api/eval/errors"));
        const data = (await response.json()) as ErrorResponse;
        if (!response.ok) throw new Error(JSON.stringify(data));
        setErrorRows(data.items || []);
        setErrorLastUpdated(data.last_updated ?? null);
      } catch {
        setErrorRows([]);
        setErrorLastUpdated(null);
      }
    };

    const fetchHardCases = async () => {
      try {
        const response = await fetch(buildApiUrl("/api/eval/hard-cases"));
        const data = (await response.json()) as HardCaseResponse;
        if (!response.ok) throw new Error(JSON.stringify(data));
        setHardCases(data.items || []);
        setHardCaseLastUpdated(data.last_updated ?? null);
      } catch {
        setHardCases([]);
        setHardCaseLastUpdated(null);
      }
    };

    void fetchRegistry();
    void fetchSteps();
    void fetchPolicy();
    void fetchErrors();
    void fetchHardCases();
  }, []);

  useEffect(() => {
    if (!preprocessSteps.length) return;
    const totalSteps = preprocessSteps.length;
    const interval = window.setInterval(() => {
      setDemoStepIndex((prev) => (prev + 1) % totalSteps);
      setDemoExampleIndex((prev) => (prev + 1) % demoExamples.length);
    }, 2500);
    return () => window.clearInterval(interval);
  }, [preprocessSteps.length]);

  const baselineRun = useMemo(
    () => registry.runs.find((run) => run.is_baseline) || registry.runs[0],
    [registry.runs],
  );

  const comparisonData = useMemo(
    () =>
      registry.runs.map((run) => ({
        model: run.model_name,
        macroF1: run.metrics?.f1 ?? 0,
        toxicF1: (run.metrics as Record<string, number> | undefined)?.f1_toxic ?? run.metrics?.f1 ?? 0,
      })),
    [registry.runs],
  );

  const hasRealTrainingCurve = Array.isArray(baselineRun?.hyperparameters?.training_curve);
  const trainingData = useMemo(() => {
    const curve = baselineRun?.hyperparameters?.training_curve;
    if (Array.isArray(curve)) return curve as { epoch: number; loss: number; f1: number }[];
    return [
      { epoch: 1, loss: 0.68, f1: 0.42 },
      { epoch: 2, loss: 0.55, f1: 0.52 },
      { epoch: 3, loss: 0.48, f1: 0.61 },
      { epoch: 4, loss: 0.42, f1: 0.68 },
      { epoch: 5, loss: 0.38, f1: 0.71 },
    ];
  }, [baselineRun]);

  const pipelineSteps = preprocessSteps.length
    ? preprocessSteps
    : [
        { id: "trim", label: t("model.pipeline.stepTrim"), active: true },
        { id: "normalize_unicode", label: t("model.pipeline.stepUnicode"), active: true },
        { id: "normalize_whitespace", label: t("model.pipeline.stepWhitespace"), active: true },
        { id: "lowercase", label: t("model.pipeline.stepLowercase"), active: false },
        { id: "remove_emoji", label: t("model.pipeline.stepEmoji"), active: false },
        { id: "strip_punctuation", label: t("model.pipeline.stepPunctuation"), active: false },
        { id: "teencode", label: t("model.pipeline.stepTeencode"), active: false },
      ];

  const activeSteps = pipelineSteps.filter((step) => step.active);
  const demoInput = demoExamples[demoExampleIndex];
  const demoOutput = applyActiveSteps(demoInput, pipelineSteps);

  const filteredErrors = useMemo(() => {
    return errorRows.filter((row) => {
      const mismatch = row.true_label === 1 && row.predicted_label === 0 ? "fn" : "fp";
      const matchesType = errorFilter === "all" || mismatch === errorFilter;
      const matchesSource = sourceFilter === "all" || row.source_dataset === sourceFilter;
      const matchesSubset = subsetFilter === "all" || row.subset_tag === subsetFilter;
      return matchesType && matchesSource && matchesSubset;
    });
  }, [errorRows, errorFilter, sourceFilter, subsetFilter]);

  const errorStats = useMemo(() => {
    let fp = 0;
    let fn = 0;
    errorRows.forEach((row) => {
      if (row.true_label === 1 && row.predicted_label === 0) fn += 1;
      if (row.true_label === 0 && row.predicted_label === 1) fp += 1;
    });
    return { fp, fn };
  }, [errorRows]);

  const errorSources = useMemo(() => {
    const sources = new Set<string>();
    errorRows.forEach((row) => row.source_dataset && sources.add(row.source_dataset));
    return ["all", ...Array.from(sources).sort((a, b) => a.localeCompare(b))];
  }, [errorRows]);

  const errorSubsets = useMemo(() => {
    const subsets = new Set<string>();
    errorRows.forEach((row) => row.subset_tag && subsets.add(row.subset_tag));
    return ["all", ...Array.from(subsets).sort((a, b) => a.localeCompare(b))];
  }, [errorRows]);

  const hardCaseBreakdown = useMemo(() => {
    const breakdown: Record<string, number> = {};
    hardCases.forEach((row) => {
      const reason = row.candidate_reason || "unknown";
      breakdown[reason] = (breakdown[reason] ?? 0) + 1;
    });
    return breakdown;
  }, [hardCases]);

  const downloadHardCases = () => {
    const blob = new Blob([JSON.stringify(hardCases, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "hard_case_candidates.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ backgroundColor: "var(--background)" }} className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <h1 className="text-4xl mb-4" style={{ color: "var(--primary)" }}>
            {t("model.hero.title")}
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            {t("model.hero.subtitle")}
          </p>
        </div>

        {/* Model Description */}
        <Card className="bg-card p-8 mb-8 shadow-lg">
          <div className="flex items-start gap-4 mb-6">
            <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 bg-background-info">
              <Cpu className="w-6 h-6" style={{ color: "var(--primary)" }} />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl mb-3" style={{ color: "var(--primary)" }}>
                {t("model.about.title")}
              </h2>
              <p className="text-foreground leading-relaxed mb-4">
                {t("model.about.description")}
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--success)" }} />
                  <div>
                    <h4 className="mb-1">{t("model.about.preprocessTitle")}</h4>
                    <p className="text-sm text-muted-foreground">{t("model.about.preprocessDesc")}</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--success)" }} />
                  <div>
                    <h4 className="mb-1">{t("model.about.keepCaseTitle")}</h4>
                    <p className="text-sm text-muted-foreground">{t("model.about.keepCaseDesc")}</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Database className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--primary)" }} />
                  <div>
                    <h4 className="mb-1">{t("model.about.datasetTitle")}</h4>
                    <p className="text-sm text-muted-foreground">{t("model.about.datasetDesc")}</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <TrendingUp className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--primary)" }} />
                  <div>
                    <h4 className="mb-1">{t("model.about.transferTitle")}</h4>
                    <p className="text-sm text-muted-foreground">{t("model.about.transferDesc")}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Experiment Registry */}
        <Card className="bg-card p-8 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <h2 className="text-2xl mb-1" style={{ color: "var(--primary)" }}>
                {t("model.registry.title")}
              </h2>
              <p className="text-sm text-muted-foreground">{t("model.registry.subtitle")}</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge className="bg-background-info text-text-info">{t("model.common.lastUpdated", { value: registry.last_updated ?? t("model.common.na") })}</Badge>
              <Button variant="outline" onClick={() => void fetchRegistry(true)}>
                {t("model.registry.refresh")}
              </Button>
            </div>
          </div>
          {registryError && <p className="text-sm text-destructive mb-4">{registryError}</p>}
          {baselineRun && (
            <div className="border rounded-lg p-5 mb-6">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-lg" style={{ color: "var(--primary)" }}>
                    {t("model.registry.baselineLabel", { name: baselineRun.model_name })}
                  </h3>
                  <p className="text-sm text-muted-foreground">{t("model.registry.dataset", { value: baselineRun.dataset_version })}</p>
                </div>
                <Badge className="bg-background-success text-text-success">{baselineRun.created_at}</Badge>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="bg-card border p-4">
                  <p className="text-xs text-muted-foreground">{t("model.metrics.f1")}</p>
                  <p className="text-xl text-foreground">{formatScore(baselineRun.metrics?.f1, t)}</p>
                </Card>
                <Card className="bg-card border p-4">
                  <p className="text-xs text-muted-foreground">{t("model.metrics.precision")}</p>
                  <p className="text-xl text-foreground">{formatScore(baselineRun.metrics?.precision, t)}</p>
                </Card>
                <Card className="bg-card border p-4">
                  <p className="text-xs text-muted-foreground">{t("model.metrics.recall")}</p>
                  <p className="text-xl text-foreground">{formatScore(baselineRun.metrics?.recall, t)}</p>
                </Card>
                <Card className="bg-card border p-4">
                  <p className="text-xs text-muted-foreground">{t("model.metrics.accuracy")}</p>
                  <p className="text-xl text-foreground">{formatScore(baselineRun.metrics?.accuracy, t)}</p>
                </Card>
              </div>
              <p className="text-sm text-muted-foreground mt-3">{t("model.registry.checkpoint", { value: baselineRun.checkpoint_path })}</p>
            </div>
          )}

          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("model.registry.table.runId")}</TableHead>
                  <TableHead>{t("model.registry.table.model")}</TableHead>
                  <TableHead>{t("model.registry.table.dataset")}</TableHead>
                  <TableHead className="text-right">{t("model.metrics.f1")}</TableHead>
                  <TableHead className="text-right">{t("model.metrics.precision")}</TableHead>
                  <TableHead className="text-right">{t("model.metrics.recall")}</TableHead>
                  <TableHead className="text-right">{t("model.metrics.accuracy")}</TableHead>
                  <TableHead>{t("model.registry.table.created")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {registry.runs.map((run) => (
                  <TableRow key={run.run_id}>
                    <TableCell>{run.run_id}</TableCell>
                    <TableCell>{run.model_name}</TableCell>
                    <TableCell>{run.dataset_version}</TableCell>
                    <TableCell className="text-right">{formatScore(run.metrics?.f1, t)}</TableCell>
                    <TableCell className="text-right">{formatScore(run.metrics?.precision, t)}</TableCell>
                    <TableCell className="text-right">{formatScore(run.metrics?.recall, t)}</TableCell>
                    <TableCell className="text-right">{formatScore(run.metrics?.accuracy, t)}</TableCell>
                    <TableCell>{run.created_at}</TableCell>
                  </TableRow>
                ))}
                {!registry.runs.length && (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center text-sm text-muted-foreground">
                      {t("model.registry.noExperiments")}
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </Card>

        {/* Pipeline Demo */}
        <Card className="bg-card p-8 mb-8 shadow-lg">
          <div className="flex items-start gap-4 mb-6">
            <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 bg-background-info">
              <FileText className="w-6 h-6" style={{ color: "var(--primary)" }} />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl mb-2" style={{ color: "var(--primary)" }}>
                {t("model.pipeline.title")}
              </h2>
              <p className="text-sm text-muted-foreground">
                {t("model.pipeline.subtitle")}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="border rounded-lg p-4 bg-background-secondary">
              <p className="text-xs text-muted-foreground mb-2">{t("model.pipeline.inputWhitespace")}</p>
              <p className="text-foreground text-sm font-mono whitespace-pre-wrap">{toVisibleWhitespace(demoInput)}</p>
              <div className="mt-4">
                <p className="text-xs text-muted-foreground mb-2">{t("model.pipeline.outputWhitespaceAfterSteps", { count: activeSteps.length })}</p>
                <p className="text-foreground text-sm font-mono whitespace-pre-wrap">{toVisibleWhitespace(demoOutput)}</p>
              </div>
            </div>

            <div className="border rounded-lg p-4">
              <p className="text-xs text-muted-foreground mb-3">{t("model.pipeline.steps")}</p>
              <div className="space-y-2">
                {pipelineSteps.map((step, idx) => {
                  const isActive = step.active;
                  const isCurrent = idx === demoStepIndex;
                  return (
                    <div
                      key={step.id}
                      className={`flex items-center justify-between rounded-lg border px-3 py-2 text-sm ${
                        isActive ? "bg-card" : "bg-background-secondary text-muted-foreground"
                      } ${isCurrent ? "border-border-info" : "border-border"}`}
                    >
                      <span>{step.label}</span>
                      <Badge className={isActive ? "bg-background-success text-text-success" : "bg-muted text-muted-foreground"}>
                        {isActive ? t("model.pipeline.active") : t("model.pipeline.plannedNotActive")}
                      </Badge>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="mt-6 border rounded-lg p-4 bg-background-info/40">
            <h3 className="text-sm mb-3" style={{ color: "var(--primary)" }}>
              {t("model.pipeline.whyOnlyThreeActive")}
            </h3>
            <ul className="list-disc pl-5 space-y-2 text-sm text-foreground">
              <li>
                {t("model.pipeline.reason1")}
              </li>
              <li>{t("model.pipeline.reason2")}</li>
              <li>{t("model.pipeline.reason3")}</li>
              <li>{t("model.pipeline.reason4")}</li>
              <li>{t("model.pipeline.reason5")}</li>
            </ul>
          </div>
        </Card>

        {/* Performance Metrics */}
        <Card className="bg-card p-8 mb-8 shadow-lg">
          <div className="flex items-start gap-4 mb-6">
            <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 bg-background-info">
              <LineChart className="w-6 h-6" style={{ color: "var(--primary)" }} />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl mb-3" style={{ color: "var(--primary)" }}>
                {t("model.metrics.title")}
              </h2>
              <p className="text-foreground mb-6">{t("model.metrics.qualityByRegistry")}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-background-info p-6 rounded-xl">
              <h4 className="text-sm text-muted-foreground mb-2">{t("model.metrics.macroF1")}</h4>
              <div className="flex items-end gap-2">
                <span className="text-4xl" style={{ color: "var(--primary)" }}>
                  {formatScore(baselineRun?.metrics?.f1, t)}
                </span>
                <Badge className="mb-2" style={{ backgroundColor: "var(--success)" }}>
                  {t("model.common.baseline")}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mt-2">{t("model.metrics.avgF1AllClasses")}</p>
            </div>

            <div className="bg-background-danger p-6 rounded-xl">
              <h4 className="text-sm text-muted-foreground mb-2">{t("model.metrics.precision")}</h4>
              <div className="flex items-end gap-2">
                <span className="text-4xl" style={{ color: "var(--destructive)" }}>
                  {formatScore(baselineRun?.metrics?.precision, t)}
                </span>
                <Badge className="mb-2 bg-text-warning">{t("model.common.baseline")}</Badge>
              </div>
              <p className="text-sm text-muted-foreground mt-2">{t("model.metrics.precisionDesc")}</p>
            </div>

            <div className="bg-background-success p-6 rounded-xl">
              <h4 className="text-sm text-muted-foreground mb-2">{t("model.metrics.recall")}</h4>
              <div className="flex items-end gap-2">
                <span className="text-4xl" style={{ color: "var(--success)" }}>
                  {formatScore(baselineRun?.metrics?.recall, t)}
                </span>
                <Badge className="mb-2" style={{ backgroundColor: "var(--success)" }}>
                  {t("model.common.baseline")}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mt-2">{t("model.metrics.recallDesc")}</p>
            </div>
          </div>

          {/* Detailed Metrics Table */}
          <div className="mb-6">
            <h3 className="text-xl mb-4" style={{ color: "var(--primary)" }}>
              {t("model.metrics.detailsTitle")}
            </h3>
            <div className="border rounded-lg overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("model.metrics.metric")}</TableHead>
                    <TableHead className="text-right">{t("model.common.baseline")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell>{t("model.metrics.precision")}</TableCell>
                    <TableCell className="text-right">{formatScore(baselineRun?.metrics?.precision, t)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>{t("model.metrics.recall")}</TableCell>
                    <TableCell className="text-right">{formatScore(baselineRun?.metrics?.recall, t)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>{t("model.metrics.f1Score")}</TableCell>
                    <TableCell className="text-right font-medium">{formatScore(baselineRun?.metrics?.f1, t)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>{t("model.metrics.accuracy")}</TableCell>
                    <TableCell className="text-right">{formatScore(baselineRun?.metrics?.accuracy, t)}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </div>
          </div>
        </Card>

        {/* Model Comparison */}
        <Card className="bg-card p-8 mb-8 shadow-lg">
          <h2 className="text-2xl mb-6" style={{ color: "var(--primary)" }}>
            {t("model.comparison.title")}
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="macroF1" fill="var(--primary)" name={t("model.comparison.macroF1")} />
              <Bar dataKey="toxicF1" fill="var(--destructive)" name={t("model.comparison.toxicClassF1")} />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-sm text-muted-foreground mt-4 text-center">
            {t("model.comparison.subtitle")}
          </p>
        </Card>

        {/* Evaluation Policy */}
        <Card className="bg-card p-8 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-2xl mb-2" style={{ color: "var(--primary)" }}>
                {t("model.evaluation.title")}
              </h2>
              <p className="text-sm text-muted-foreground">{t("model.evaluation.subtitle")}</p>
            </div>
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline">{t("model.evaluation.viewDetails")}</Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>{t("model.evaluation.title")}</DialogTitle>                  <DialogDescription>
                    {t("model.evaluation.description")}
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 text-sm text-foreground">
                  <div>
                    <p className="font-medium">{t("model.evaluation.splitRatio")}</p>
                    <p>
                      {t("model.evaluation.train")}: {formatRatio(policy?.policy.split?.train)} | {t("model.evaluation.val")}: {formatRatio(policy?.policy.split?.val)} | {t("model.evaluation.test")}:{" "}
                      {formatRatio(policy?.policy.split?.test)}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">{t("model.evaluation.fixedTestSet")}</p>
                    <p>
                      {policy?.policy.test_set_fixed
                        ? t("model.common.yes")
                        : t("model.common.no")} ({t("model.evaluation.since", { value: policy?.policy.fixed_since ?? t("model.common.na") })})
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">{t("model.evaluation.hardCaseSubsets")}</p>
                    <p>
                      {policy?.policy.hard_case_subsets_evaluated
                        ? t("model.evaluation.evaluated")
                        : t("model.evaluation.notApplied")}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">{t("model.evaluation.notes")}</p>
                    <p>{policy?.policy.note ?? t("model.common.na")}</p>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </Card>

        {/* Training Visualization */}
        <Card className="bg-card p-8 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
            <h2 className="text-2xl" style={{ color: "var(--primary)" }}>
              {t("model.training.title")}
            </h2>
            <Badge className={hasRealTrainingCurve ? "bg-background-success text-text-success" : "bg-background-warning text-text-warning"}>
              {hasRealTrainingCurve ? t("model.training.curveSourceReal") : t("model.training.curveSourceIllustrative")}
            </Badge>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Training Loss Curve */}
            <div>
              <h3 className="mb-4">{t("model.training.trainingLoss")}</h3>
              <ResponsiveContainer width="100%" height={250}>
                <RechartsLine data={trainingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: t("model.training.epoch"), position: "insideBottom", offset: -5 }} />
                  <YAxis label={{ value: t("model.training.loss"), angle: -90, position: "insideLeft" }} />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="var(--destructive)" 
                    strokeWidth={2}
                    name={t("model.training.validationLoss")}
                  />
                </RechartsLine>
              </ResponsiveContainer>
              <p className="text-sm text-muted-foreground mt-2 text-center">
                {t("model.training.lossTrend")}
              </p>
            </div>

            {/* F1 Score Curve */}
            <div>
              <h3 className="mb-4">{t("model.training.validationF1")}</h3>
              <ResponsiveContainer width="100%" height={250}>
                <RechartsLine data={trainingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: t("model.training.epoch"), position: "insideBottom", offset: -5 }} />
                  <YAxis domain={[0, 1]} label={{ value: t("model.training.f1Score"), angle: -90, position: "insideLeft" }} />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="f1" 
                    stroke="var(--success)" 
                    strokeWidth={2}
                    name={t("model.training.macroF1")}
                  />
                </RechartsLine>
              </ResponsiveContainer>
              <p className="text-sm text-muted-foreground mt-2 text-center">
                {t("model.training.f1Trend")}
              </p>
            </div>
          </div>
        </Card>

        {/* Error Analysis */}
        <Card className="bg-card p-8 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <h2 className="text-2xl mb-1" style={{ color: "var(--primary)" }}>
                {t("model.error.title")}
              </h2>
              <p className="text-sm text-muted-foreground">{t("model.error.subtitle")}</p>
            </div>
            <Badge className="bg-background-info text-text-info">{t("model.common.lastUpdated", { value: errorLastUpdated ?? t("model.common.na") })}</Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <Card className="border p-5">
              <div className="flex items-center gap-2 mb-2">
                <XCircle className="w-5 h-5" style={{ color: "var(--destructive)" }} />
                <h3 style={{ color: "var(--primary)" }}>{t("model.error.falsePositives")}</h3>
              </div>
              <p className="text-3xl" style={{ color: "var(--primary)" }}>{errorStats.fp}</p>
            </Card>
            <Card className="border p-5">
              <div className="flex items-center gap-2 mb-2">
                <XCircle className="w-5 h-5" style={{ color: "var(--success)" }} />
                <h3 style={{ color: "var(--primary)" }}>{t("model.error.falseNegatives")}</h3>
              </div>
              <p className="text-3xl" style={{ color: "var(--primary)" }}>{errorStats.fn}</p>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div>
              <Label className="text-sm text-muted-foreground">{t("model.error.mismatchType")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={errorFilter}
                onChange={(event) => setErrorFilter(event.target.value)}
              >
                <option value="all">{t("model.common.all")}</option>
                <option value="fp">{t("model.error.falsePositive")}</option>
                <option value="fn">{t("model.error.falseNegative")}</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">{t("model.error.sourceDataset")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={sourceFilter}
                onChange={(event) => setSourceFilter(event.target.value)}
              >
                {errorSources.map((source) => (
                  <option key={source} value={source}>
                    {source}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">{t("model.error.subsetTag")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={subsetFilter}
                onChange={(event) => setSubsetFilter(event.target.value)}
              >
                {errorSubsets.map((subset) => (
                  <option key={subset} value={subset}>
                    {subset}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("model.table.text")}</TableHead>
                  <TableHead className="text-right">{t("model.error.true")}</TableHead>
                  <TableHead className="text-right">{t("model.error.pred")}</TableHead>
                  <TableHead className="text-right">{t("model.error.confidence")}</TableHead>
                  <TableHead>{t("model.table.source")}</TableHead>
                  <TableHead>{t("model.table.subset")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredErrors.map((row, idx) => (
                  <TableRow key={`${row.text.slice(0, 16)}-${idx}`}>
                    <TableCell className="max-w-[360px]">
                      <details>
                        <summary className="cursor-pointer truncate">{row.text}</summary>
                        <p className="mt-2 text-sm text-muted-foreground">{row.text}</p>
                      </details>
                    </TableCell>
                    <TableCell className="text-right">{row.true_label}</TableCell>
                    <TableCell className="text-right">{row.predicted_label}</TableCell>
                    <TableCell className="text-right">{row.confidence?.toFixed(2) ?? t("model.common.na")}</TableCell>
                    <TableCell>{row.source_dataset ?? t("model.common.na")}</TableCell>
                    <TableCell>{row.subset_tag ?? t("model.common.na")}</TableCell>
                  </TableRow>
                ))}
                {!filteredErrors.length && (
                  <TableRow>
                    <TableCell colSpan={6} className="text-center text-sm text-muted-foreground">
                      {t("model.error.noRows")}
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </Card>

        {/* Hard-case Candidates */}
        <Card className="bg-card p-8 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <h2 className="text-2xl mb-1" style={{ color: "var(--primary)" }}>
                {t("model.hardCase.title")}
              </h2>
              <p className="text-sm text-muted-foreground">{t("model.hardCase.subtitle")}</p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Badge className="bg-background-info text-text-info">{t("model.common.lastUpdated", { value: hardCaseLastUpdated ?? t("model.common.na") })}</Badge>
              <Button variant="outline" onClick={downloadHardCases} disabled={!hardCases.length}>
                <Download className="w-4 h-4 mr-2" />
                {t("model.hardCase.exportForAnnotation")}
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Card className="border p-4">
              <p className="text-sm text-muted-foreground">{t("model.hardCase.candidateCount")}</p>
              <p className="text-2xl" style={{ color: "var(--primary)" }}>{hardCases.length}</p>
            </Card>
            <Card className="border p-4 md:col-span-2">
              <p className="text-sm text-muted-foreground mb-2">{t("model.hardCase.breakdownByReason")}</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(hardCaseBreakdown).map(([reason, count]) => (
                  <Badge key={reason} className="bg-background-info text-text-info">
                    {reason}: {count}
                  </Badge>
                ))}
                {!Object.keys(hardCaseBreakdown).length && <span className="text-sm text-muted-foreground">{t("model.common.na")}</span>}
              </div>
            </Card>
          </div>

          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("model.table.text")}</TableHead>
                  <TableHead>{t("model.hardCase.reason")}</TableHead>
                  <TableHead>{t("model.table.source")}</TableHead>
                  <TableHead>{t("model.table.subset")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {hardCases.slice(0, 6).map((row, idx) => (
                  <TableRow key={`${row.text.slice(0, 16)}-${idx}`}>
                    <TableCell className="max-w-[320px] truncate" title={row.text}>
                      {row.text}
                    </TableCell>
                    <TableCell>{row.candidate_reason ?? t("model.common.na")}</TableCell>
                    <TableCell>{row.source_dataset ?? t("model.common.na")}</TableCell>
                    <TableCell>{row.subset_tag ?? t("model.common.na")}</TableCell>
                  </TableRow>
                ))}
                {!hardCases.length && (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center text-sm text-muted-foreground">
                      {t("model.hardCase.noCandidates")}
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </Card>

        {/* MLOps & Reliability */}
        <Card className="bg-card p-8 mb-8 shadow-lg">
          <h2 className="text-2xl mb-6" style={{ color: "var(--primary)" }}>
            {t("model.mlops.title")}
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="border rounded-lg p-5">
              <h4 className="mb-3" style={{ color: "var(--primary)" }}>
                {t("model.mlops.versioningTitle")}
              </h4>
              <p className="text-sm text-muted-foreground mb-3">
                {t("model.mlops.versioningDesc")}
              </p>
              <Badge className="bg-background-info text-text-info">{t("model.mlops.versionCurrent")}</Badge>
            </div>

            <div className="border rounded-lg p-5">
              <h4 className="mb-3" style={{ color: "var(--primary)" }}>
                {t("model.mlops.experimentTrackingTitle")}
              </h4>
              <p className="text-sm text-muted-foreground mb-3">
                {t("model.mlops.experimentTrackingDesc")}
              </p>
              <Badge className="bg-background-success text-text-success">{t("model.mlops.statusActive")}</Badge>
            </div>

            <div className="border rounded-lg p-5">
              <h4 className="mb-3" style={{ color: "var(--primary)" }}>
                {t("model.mlops.monitoringTitle")}
              </h4>
              <p className="text-sm text-muted-foreground mb-3">
                {t("model.mlops.monitoringDesc")}
              </p>
              <Badge className="bg-background-info text-text-info">{t("model.mlops.statusRealtime")}</Badge>
            </div>
          </div>

          <div className="border-l-4 border-l-primary p-5 rounded bg-background-info/40">
            <h4 className="mb-2" style={{ color: "var(--primary)" }}>
              {t("model.disclaimer.title")}
            </h4>
            <p className="text-foreground">
              {t("model.disclaimer.text")}
            </p>
          </div>
        </Card>

        {/* CTA */}
        <div className="text-center">
          <div className="inline-block bg-card rounded-2xl shadow-lg p-8">
            <h3 className="text-2xl mb-4" style={{ color: "var(--primary)" }}>
              {t("model.cta.title")}
            </h3>
            <p className="text-muted-foreground mb-6 max-w-md">
              {t("model.cta.subtitle")}
            </p>
            <Button
              onClick={onTryNow}
              className="h-12 px-8"
              style={{ backgroundColor: "var(--primary)" }}
            >
              {t("model.cta.button")}
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
