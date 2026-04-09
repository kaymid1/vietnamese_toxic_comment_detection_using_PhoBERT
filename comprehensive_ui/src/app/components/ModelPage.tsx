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
import {
  ArrowRight,
  CheckCircle,
  TrendingUp,
  Database,
  Cpu,
  FileText,
} from "lucide-react";
import {
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { useI18n } from "@/app/i18n/context";
import { TrainingChecklist } from "../../components/TrainingChecklist";
import { TrainingResults } from "../../components/TrainingResults";
import { TrainingStoreProvider } from "../../hooks/useTrainingStore";

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

    void fetchRegistry();
    void fetchSteps();
    void fetchPolicy();
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

        {/* Training Checklist & Results */}
        <TrainingStoreProvider>
          <Card className="bg-card p-8 mb-8 shadow-lg">
            <div className="mb-6">
              <h2 className="text-2xl mb-2" style={{ color: "var(--primary)" }}>
                Training Checklist & Results
              </h2>
              <p className="text-sm text-muted-foreground">
                Theo dõi tiến độ checklist thí nghiệm và lưu kết quả metric cho từng kịch bản train.
              </p>
            </div>

            <div className="space-y-5">
              <TrainingChecklist />
              <TrainingResults />
            </div>
          </Card>
        </TrainingStoreProvider>

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
