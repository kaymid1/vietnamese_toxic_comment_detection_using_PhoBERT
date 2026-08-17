import { useEffect, useMemo, useRef, useState, type ChangeEvent, type ComponentProps, type MouseEvent, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, BarChart3, Check, CircleHelp, EyeOff, GripHorizontal, History, Lock, MessageCircle, MoreHorizontal, Plus, RefreshCw, RotateCcw, Sparkles, ThumbsUp, Unlock } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip as RechartTooltip, XAxis, YAxis } from "recharts";
import { toast } from "sonner";
import { Card } from "@/app/components/ui/card";
import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { Badge } from "@/app/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { Progress } from "@/app/components/ui/progress";
import { useProgressNotification } from "@/app/components/ProgressNotification";
import { Checkbox } from "@/app/components/ui/checkbox";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/app/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/app/components/ui/dropdown-menu";
import { getModelLabel } from "@/app/modelCatalog";
import {
  DEFAULT_MLFLOW_GATE_THRESHOLDS,
  MlflowBadge,
  MlflowTooltipBody,
  formatGeminiAction,
  formatMlflowConfidence,
  getConstructivenessPresentation,
  getDataSourcePresentation,
  getGateBucketPresentation,
  getLockPresentation,
  getReviewStatusPresentation,
  getScorePresentation,
  getToxicityPresentation,
  getTrainingSelectionPresentation,
  getVerificationStatusPresentation,
  makeMlflowTooltip,
} from "@/app/mlflowPresentation";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/app/components/ui/dialog";
import {
  buildApiUrl,
  useMlflowStore,
  type MlflowCandidate,
  type MlflowGeminiReviewSuggestion,
  type MlflowModelReEvaluationResponse,
  type MlflowPrediction,
  type MlflowUnusedScope,
} from "../../hooks/useMlflowStore";


interface MLFlowPageProps {
  availableModels: string[];
  onModelsChanged?: () => Promise<void> | void;
  adminToken: string;
  onAdminUnauthorized: () => void;
}

const MLFLOW_URLS_DRAFT_KEY = "viettoxic:mlflow:urlsText";
const MLFLOW_MODEL_DRAFT_KEY = "viettoxic:mlflow:selectedModel";
const MLFLOW_ACTIVE_TAB_KEY = "viettoxic:mlflow:activeTab";
const MLFLOW_CLEAR_ALL_CONFIRM_TOKEN = "DELETE_ALL_MLFLOW_DATA";
const KAGGLE_TERMINAL_STATUSES = new Set(["completed", "failed", "dry_run", "placeholder"]);

type IconButtonWithTooltipProps = ComponentProps<typeof Button> & {
  label: string;
  tooltip?: string;
  children: ReactNode;
};

function IconButtonWithTooltip({ label, tooltip, children, ...buttonProps }: IconButtonWithTooltipProps) {
  return (
    <Tooltip delayDuration={1000}>
      <TooltipTrigger asChild>
        <span className="inline-flex">
          <Button aria-label={label} title={label} {...buttonProps}>
            {children}
            <span className="sr-only">{label}</span>
          </Button>
        </span>
      </TooltipTrigger>
      <TooltipContent>
        <MlflowTooltipBody text={tooltip || makeMlflowTooltip(label, "Bấm để thực hiện thao tác này.")} />
      </TooltipContent>
    </Tooltip>
  );
}

function SectionInfoTooltip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Tooltip delayDuration={300}>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex h-5 w-5 items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={label}
        >
          <CircleHelp className="h-4 w-4" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" align="start" sideOffset={8} className="max-w-md p-3">
        <div className="space-y-2 text-xs leading-relaxed">{children}</div>
      </TooltipContent>
    </Tooltip>
  );
}

function formatInferenceTime(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return timestamp;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function PredictionBadges({ prediction }: { prediction: MlflowPrediction }) {
  const hasPredictedLabel = prediction.predicted_label === 0 || prediction.predicted_label === 1;
  const isLegacyBackfill = prediction.record_origin === "legacy_backfill";
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <Badge variant="outline">{getModelLabel(prediction.model_id)}</Badge>
      {hasPredictedLabel ? (
        <MlflowBadge presentation={getToxicityPresentation(prediction.predicted_label)} />
      ) : (
        <Badge variant="outline">Label unavailable</Badge>
      )}
      {prediction.raw_toxicity_score != null && isLegacyBackfill && (
        <Badge variant="outline">Toxicity score {prediction.raw_toxicity_score.toFixed(3)}</Badge>
      )}
      {prediction.raw_toxicity_score != null && !isLegacyBackfill && (
        <MlflowBadge presentation={getScorePresentation(prediction.raw_toxicity_score, DEFAULT_MLFLOW_GATE_THRESHOLDS)} />
      )}
      {prediction.adjusted_toxicity_score != null && (
        <Badge variant="outline">Adjusted {prediction.adjusted_toxicity_score.toFixed(3)}</Badge>
      )}
      {prediction.seg_threshold_used != null && (
        <Badge variant="outline">Threshold {prediction.seg_threshold_used.toFixed(3)}</Badge>
      )}
      {prediction.created_at && <Badge variant="outline">Inference time: {formatInferenceTime(prediction.created_at)}</Badge>}
      {isLegacyBackfill && <Badge variant="secondary">Legacy record</Badge>}
      {prediction.record_origin === "model_re_evaluation" && <Badge variant="secondary">Model re-evaluation</Badge>}
      {prediction.constructiveness_label != null && (
        <MlflowBadge presentation={getConstructivenessPresentation(prediction.constructiveness_label)} />
      )}
      {prediction.agreement_with_human === true && <Badge variant="default">Agreement with human</Badge>}
      {prediction.agreement_with_human === false && <Badge variant="destructive">Disagreement with human</Badge>}
    </div>
  );
}

function PredictionSummary({ prediction }: { prediction: MlflowPrediction }) {
  const hasPredictedLabel = prediction.predicted_label === 0 || prediction.predicted_label === 1;
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
      <span className="font-medium text-foreground">{getModelLabel(prediction.model_id)}</span>
      {hasPredictedLabel ? (
        <MlflowBadge presentation={getToxicityPresentation(prediction.predicted_label)} />
      ) : (
        <Badge variant="outline">Label unavailable</Badge>
      )}
      {prediction.raw_toxicity_score != null && (
        <span className="text-muted-foreground">Toxicity score {prediction.raw_toxicity_score.toFixed(3)}</span>
      )}
      {prediction.constructiveness_label != null && (
        <MlflowBadge presentation={getConstructivenessPresentation(prediction.constructiveness_label)} />
      )}
      {prediction.agreement_with_human === true && <Badge variant="default">Agreement with human</Badge>}
      {prediction.agreement_with_human === false && <Badge variant="destructive">Disagreement with human</Badge>}
    </div>
  );
}

function PredictionDetails({ prediction }: { prediction: MlflowPrediction }) {
  const isLegacyBackfill = prediction.record_origin === "legacy_backfill";
  const origin = isLegacyBackfill
    ? "Legacy record"
    : prediction.record_origin === "model_re_evaluation"
      ? "Model re-evaluation"
      : prediction.record_origin;
  return (
    <details className="text-xs text-muted-foreground">
      <summary className="cursor-pointer font-medium text-foreground">Prediction details</summary>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-md bg-muted/35 p-2">
        {prediction.raw_toxicity_score != null && <><dt>Raw toxicity score</dt><dd>{prediction.raw_toxicity_score.toFixed(3)}</dd></>}
        {prediction.adjusted_toxicity_score != null && <><dt>Adjusted toxicity score</dt><dd>{prediction.adjusted_toxicity_score.toFixed(3)}</dd></>}
        {prediction.seg_threshold_used != null && <><dt>Threshold</dt><dd>{prediction.seg_threshold_used.toFixed(3)}</dd></>}
        {prediction.created_at && <><dt>Inference time</dt><dd>{formatInferenceTime(prediction.created_at)}</dd></>}
        {origin && <><dt>Origin</dt><dd>{origin}</dd></>}
      </dl>
    </details>
  );
}

function PredictionEvidence({ item, compact = false }: { item: MlflowCandidate; compact?: boolean }) {
  if (!item.latest_prediction) return null;
  if (compact) {
    const previousPredictions = item.previous_predictions ?? [];
    return (
      <div className="space-y-2 rounded-md border bg-muted/15 p-3">
        <div className="space-y-1">
          <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Latest Prediction</p>
          <PredictionSummary prediction={item.latest_prediction} />
          <PredictionDetails prediction={item.latest_prediction} />
        </div>
        {previousPredictions.length > 0 && (
          <details className="border-t pt-2 text-xs">
            <summary className="cursor-pointer font-medium text-foreground">Previous predictions ({previousPredictions.length})</summary>
            <div className="mt-2 space-y-3">
              {previousPredictions.map((prediction) => (
                <div key={prediction.id} className="space-y-1 rounded-md bg-muted/35 p-2">
                  <PredictionSummary prediction={prediction} />
                  <PredictionDetails prediction={prediction} />
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    );
  }
  return (
    <div className="space-y-2 rounded-md border bg-muted/20 p-2">
      <div className="space-y-1">
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Latest Prediction</p>
        <PredictionBadges prediction={item.latest_prediction} />
      </div>
      {(item.previous_predictions?.length ?? 0) > 0 && (
        <div className="space-y-1">
          <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Previous Predictions</p>
          {item.previous_predictions?.map((prediction) => (
            <PredictionBadges key={prediction.id} prediction={prediction} />
          ))}
        </div>
      )}
    </div>
  );
}

function HumanReviewEvidence({ item, embedded = false }: { item: MlflowCandidate; embedded?: boolean }) {
  const hasVerifiedHumanLabel =
    item.verification_status === "manual_accepted" && (item.human_label === 0 || item.human_label === 1);
  if (!hasVerifiedHumanLabel && !item.requires_human_review) return null;

  const label = item.human_label === 1 ? "Toxic" : "Clean";
  const escalationReason = item.review_reason === "model_conflict" ? "Model conflict" : "Model uncertain";
  const containerClassName = embedded
    ? "flex min-w-0 items-center gap-1.5 text-xs font-medium"
    : hasVerifiedHumanLabel
      ? "flex items-center gap-1.5 rounded-md border border-emerald-200/70 bg-emerald-50/40 p-2 text-xs font-medium dark:border-emerald-900/50 dark:bg-emerald-950/15"
      : "flex items-center gap-1.5 rounded-md border border-amber-200/70 bg-amber-50/40 p-2 text-xs font-medium dark:border-amber-900/50 dark:bg-amber-950/15";
  return (
    <div className={containerClassName}>
      {hasVerifiedHumanLabel ? (
        <>
          <Check className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" aria-hidden="true" />
          <span className="text-emerald-800 dark:text-emerald-200">Human verified · {label}</span>
        </>
      ) : (
        <>
          <AlertTriangle className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400" aria-hidden="true" />
          <span className="text-amber-800 dark:text-amber-200">Human review required</span>
          <span className="text-muted-foreground">· {escalationReason}</span>
        </>
      )}
    </div>
  );
}

const safeReadLocalStorageString = (key: string, fallback = "") => {
  try {
    const raw = window.localStorage.getItem(key);
    if (raw == null) return fallback;
    try {
      const parsed = JSON.parse(raw);
      if (typeof parsed === "string") return parsed;
    } catch {
      // backward compatibility: previously stored raw/plain string
    }
    return raw;
  } catch {
    return fallback;
  }
};

const safeWriteLocalStorageString = (key: string, value: string) => {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // ignore quota / private mode errors
  }
};

export function MLFlowPage({ availableModels, onModelsChanged, adminToken, onAdminUnauthorized }: MLFlowPageProps) {
  const showLegacyIngest = false;
  const { start: startProgress, update: updateProgress, succeed: succeedProgress, fail: failProgress } = useProgressNotification();
  const isDeprecatedModel = (model: string) => model.toLowerCase().includes("deprecated");
  const {
    loading,
    error,
    hasNoBatch,
    ingestStage,
    ingestProgress,
    ingestStageMessage,
    activeBatchId,
    overview,
    candidates,
    trainingPreview,
    trainingPlan,
    candidateTotal,
    candidatePage,
    candidatePageSize,
    thresholdStatus,
    reviewHistory,
    reviewHistoryTotal,
    reviewHistoryPage,
    crawlHistory,
    crawlHistoryTotal,
    crawlHistoryPage,
    comparePayload,
    registryModels,
    lastBundlePath,
    doStatus,
    doPreflight,
    automationStatus,
    automationStatusError,
    ingest,
    refreshOverview,
    refreshCandidates,
    refreshTrainingPreview,
    refreshTrainingPlan,
    reviewTrainingPreview,
    geminiReviewTrainingPreview,
    geminiReviewCandidates,
    reEvaluateWithModel,
    refreshReviewHistory,
    refreshCrawlHistory,
    reviewCandidates,
    clearMlflowAll,
    refreshThresholdStatus,
    exportBundle,
    importModelZip,
    triggerDO,
    refreshDOPreflight,
    refreshDOStatus,
    refreshAutomationStatus,
    openDORun,
    geminiEvaluateKaggleRun,
    clearDOSession,
    refreshCompare,
    refreshModelRegistry,
    updateModelRegistryLifecycle,
    promote,
    rollback,
  } = useMlflowStore({ adminToken, onUnauthorized: onAdminUnauthorized });

  const [urlsText, setUrlsText] = useState(() => {
    if (typeof window === "undefined") return "";
    return safeReadLocalStorageString(MLFLOW_URLS_DRAFT_KEY, "");
  });
  const [selectedModel, setSelectedModel] = useState<string>(() => {
    if (typeof window === "undefined") return availableModels[0] || "";
    return safeReadLocalStorageString(MLFLOW_MODEL_DRAFT_KEY, availableModels[0] || "");
  });
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<number[]>([]);
  const [selectedPreviewIds, setSelectedPreviewIds] = useState<number[]>([]);
  const [candidateGeminiSuggestions, setCandidateGeminiSuggestions] = useState<Record<number, MlflowGeminiReviewSuggestion>>({});
  const [candidateGeminiReviewing, setCandidateGeminiReviewing] = useState(false);
  const [candidateGeminiApplying, setCandidateGeminiApplying] = useState(false);
  const [geminiSuggestions, setGeminiSuggestions] = useState<Record<number, MlflowGeminiReviewSuggestion>>({});
  const [geminiReviewing, setGeminiReviewing] = useState(false);
  const [geminiApplying, setGeminiApplying] = useState(false);
  const [bulkPreviewUpdating, setBulkPreviewUpdating] = useState(false);
  const [modelReEvaluating, setModelReEvaluating] = useState(false);
  const [reEvaluationModel, setReEvaluationModel] = useState(() => availableModels[0] || "");
  const [reEvaluationScope, setReEvaluationScope] = useState<"selected" | "all_auto_eligible">("selected");
  const [lastReEvaluation, setLastReEvaluation] = useState<MlflowModelReEvaluationResponse | null>(null);
  const [crawlHistoryOpen, setCrawlHistoryOpen] = useState(false);
  const [importModelName, setImportModelName] = useState("");
  const [importModelZipFile, setImportModelZipFile] = useState<File | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);
  const [includeUnusedInExport, setIncludeUnusedInExport] = useState(false);
  const [unusedScope, setUnusedScope] = useState<MlflowUnusedScope>("all");
  const [historyDecision, setHistoryDecision] = useState<"all" | "accepted" | "rejected" | "discarded">("all");
  const [reviewHistoryOpen, setReviewHistoryOpen] = useState(false);
  const [crawlSummary, setCrawlSummary] = useState<{
    status_counts?: Record<string, number>;
    timeout_count?: number;
    total_urls?: number;
  } | null>(null);
  const [activeTab, setActiveTab] = useState(() => {
    const stored = safeReadLocalStorageString(MLFLOW_ACTIVE_TAB_KEY, "step1");
    return ["step1", "step4", "step5"].includes(stored) ? stored : "step1";
  });
  const [selectedModelKind, setSelectedModelKind] = useState<"phobert" | "lr_smoke">("phobert");
  const [selectedTrainingMode, setSelectedTrainingMode] = useState<"retrain" | "finetune">("finetune");
  const [balanceStrategy, setBalanceStrategy] = useState<"balanced_50_50" | "all">("balanced_50_50");
  const [finetuneBaseModel, setFinetuneBaseModel] = useState("");
  const [promotionDialogOpen, setPromotionDialogOpen] = useState(false);
  const [trainingPreviewListHeight, setTrainingPreviewListHeight] = useState(320);
  const [kaggleTriggerPending, setKaggleTriggerPending] = useState(false);
  const [geminiEvaluating, setGeminiEvaluating] = useState(false);
  const prevDoStatusRef = useRef<string>("idle");
  const announcedAutomationEventRef = useRef<number | null>(null);
  const kaggleTriggerPendingRef = useRef(false);
  const trainingPreviewResizeRef = useRef<{ pointerId: number; startY: number; startHeight: number } | null>(null);

  useEffect(() => {
    void refreshOverview();
    void refreshCandidates(undefined, 1, "all_batches");
    void refreshThresholdStatus(activeBatchId);
    void refreshTrainingPreview(1, "all_batches");
    void refreshCompare();
    void refreshModelRegistry();
    void refreshDOPreflight();
    void refreshAutomationStatus();
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => void refreshAutomationStatus(), 15000);
    return () => window.clearInterval(timer);
  }, [refreshAutomationStatus]);

  useEffect(() => {
    const firstSelectable = availableModels.find((model) => !isDeprecatedModel(model)) || availableModels[0] || "";
    if (!selectedModel && firstSelectable) {
      setSelectedModel(firstSelectable);
      return;
    }
    if (selectedModel && isDeprecatedModel(selectedModel) && firstSelectable && selectedModel !== firstSelectable) {
      setSelectedModel(firstSelectable);
    }
  }, [availableModels, selectedModel]);

  useEffect(() => {
    if (!reEvaluationModel || !availableModels.includes(reEvaluationModel)) {
      setReEvaluationModel(availableModels.find((model) => !isDeprecatedModel(model)) || availableModels[0] || "");
    }
  }, [availableModels, reEvaluationModel]);

  useEffect(() => {
    if (reviewHistoryOpen) void refreshReviewHistory(undefined, historyDecision, 1, "all_batches");
  }, [historyDecision, reviewHistoryOpen]);

  useEffect(() => {
    void refreshTrainingPlan(balanceStrategy, "all_batches");
  }, [balanceStrategy, refreshTrainingPlan, trainingPreview?.items]);

  useEffect(() => {
    safeWriteLocalStorageString(MLFLOW_URLS_DRAFT_KEY, urlsText);
  }, [urlsText]);

  useEffect(() => {
    if (!selectedModel) return;
    safeWriteLocalStorageString(MLFLOW_MODEL_DRAFT_KEY, selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    safeWriteLocalStorageString(MLFLOW_ACTIVE_TAB_KEY, activeTab);
  }, [activeTab]);

  useEffect(() => {
    const availableIds = new Set(candidates.map((item) => item.id));
    setSelectedCandidateIds((prev) => prev.filter((id) => availableIds.has(id)));
    setCandidateGeminiSuggestions((prev) => {
      const next: Record<number, MlflowGeminiReviewSuggestion> = {};
      for (const [key, value] of Object.entries(prev)) {
        const id = Number(key);
        if (availableIds.has(id)) next[id] = value;
      }
      return next;
    });
  }, [candidates]);

  useEffect(() => {
    const availableIds = new Set((trainingPreview?.items || []).map((item) => item.id));
    setSelectedPreviewIds((prev) => prev.filter((id) => availableIds.has(id)));
    setGeminiSuggestions((prev) => {
      const next: Record<number, MlflowGeminiReviewSuggestion> = {};
      for (const [key, value] of Object.entries(prev)) {
        const id = Number(key);
        if (availableIds.has(id)) next[id] = value;
      }
      return next;
    });
  }, [trainingPreview?.items]);

  const parsedUrls = useMemo(
    () =>
      urlsText
        .split(/\r?\n/)
        .map((u) => u.trim())
        .filter(Boolean),
    [urlsText],
  );

  const thresholdProgress = useMemo(() => {
    if (!thresholdStatus || !trainingPlan) return 0;
    const max = Math.max(1, thresholdStatus.target_max_test_stage || 10);
    return Math.min(100, (trainingPlan.summary.mlflow_added / max) * 100);
  }, [thresholdStatus, trainingPlan]);
  const bundleIncludedCount = trainingPlan?.summary.mlflow_added ?? 0;
  const bundleTargetCount = Math.max(1, thresholdStatus?.target_max_test_stage ?? 10);
  const bundleReady = Boolean(trainingPlan && bundleIncludedCount >= bundleTargetCount);
  const availableGeminiSuggestions = Object.values(geminiSuggestions);
  const availableCandidateGeminiSuggestions = Object.values(candidateGeminiSuggestions);
  const visibleTrainingPreviewItems = trainingPreview?.items || [];
  const selectedAutoEligibleCount = visibleTrainingPreviewItems.filter(
    (item) =>
      selectedPreviewIds.includes(item.id) &&
      item.verification_status === "auto_accepted" &&
      item.gate_bucket === "accepted" &&
      Boolean(item.selected_for_training),
  ).length;
  const toxicityDistribution = useMemo(
    () => [
      { name: "Độc hại", value: trainingPreview?.counts.selected_toxic ?? 0, color: "#ef4444" },
      { name: "Sạch", value: trainingPreview?.counts.selected_clean ?? 0, color: "#22c55e" },
    ],
    [trainingPreview?.counts.selected_clean, trainingPreview?.counts.selected_toxic],
  );
  const constructivenessDistribution = useMemo(
    () => [
      { name: "Có tính xây dựng", value: trainingPreview?.constructiveness.constructive ?? 0, color: "#2563eb" },
      { name: "Không xây dựng", value: trainingPreview?.constructiveness.non_constructive ?? 0, color: "#f59e0b" },
      { name: "Ẩn/chưa có nhãn", value: trainingPreview?.constructiveness.masked ?? 0, color: "#94a3b8" },
    ],
    [
      trainingPreview?.constructiveness.constructive,
      trainingPreview?.constructiveness.masked,
      trainingPreview?.constructiveness.non_constructive,
    ],
  );

  const ingestStageMeta = useMemo(() => {
    if (ingestStage === "crawl") return { label: "Crawl", variant: "default" as const };
    if (ingestStage === "inference") return { label: "Inference", variant: "default" as const };
    if (ingestStage === "finalize") return { label: "Finalize", variant: "default" as const };
    if (ingestStage === "completed") return { label: "Completed", variant: "secondary" as const };
    if (ingestStage === "error") return { label: "Error", variant: "destructive" as const };
    return { label: "Idle", variant: "secondary" as const };
  }, [ingestStage]);

  const inferDomainFromUrl = (url: string) => {
    try {
      const host = new URL(url).hostname.toLowerCase();
      if (
        host.includes("vnexpress.net") ||
        host.includes("dantri.com.vn") ||
        host.includes("tuoitre.vn") ||
        host.includes("thanhnien.vn") ||
        host.includes("vietnamnet.vn")
      ) {
        return "news";
      }
      if (
        host.includes("facebook.com") ||
        host.includes("fb.com") ||
        host.includes("instagram.com") ||
        host.includes("tiktok.com") ||
        host.includes("youtube.com")
      ) {
        return "social";
      }
      if (host.includes("voz.vn") || host.includes("reddit.com") || host.includes("webtretho.com")) {
        return "forum";
      }
      return "unknown";
    } catch {
      return "unknown";
    }
  };

  const resolveDomainTag = (item: { domain_category?: string | null; url: string }) => {
    const domain = (item.domain_category || "").trim().toLowerCase();
    if (domain) return domain;
    return inferDomainFromUrl(item.url);
  };

  const toggleCandidate = (id: number) => {
    setSelectedCandidateIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  const handleCandidateRowToggle = (event: MouseEvent<HTMLDivElement>, id: number) => {
    const target = event.target as HTMLElement;
    if (target.closest("button, input, textarea, select, option, a, details, summary, [role='button'], [role='link'], [data-row-interactive]")) {
      return;
    }
    toggleCandidate(id);
  };

  const handleSelectAllCandidates = () => {
    const ids = candidates.map((item) => item.id);
    setSelectedCandidateIds(ids);
  };

  const handleUnselectAllCandidates = () => {
    setSelectedCandidateIds([]);
  };

  const togglePreviewSelection = (id: number) => {
    setSelectedPreviewIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  const handleSelectAllPreviewRows = () => {
    setSelectedPreviewIds(visibleTrainingPreviewItems.map((item) => item.id));
  };

  const handleUnselectAllPreviewRows = () => {
    setSelectedPreviewIds([]);
  };

  const handlePreviewRowToggle = (event: MouseEvent<HTMLDivElement>, id: number) => {
    const target = event.target as HTMLElement;
    if (target.closest("button, input, textarea, select, option, a")) {
      return;
    }
    togglePreviewSelection(id);
  };

  const clampTrainingPreviewHeight = (height: number) => Math.min(960, Math.max(240, height));

  const handleTrainingPreviewResizeStart: NonNullable<ComponentProps<"div">["onPointerDown"]> = (event) => {
    event.preventDefault();
    trainingPreviewResizeRef.current = {
      pointerId: event.pointerId,
      startY: event.clientY,
      startHeight: trainingPreviewListHeight,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handleTrainingPreviewResizeMove: NonNullable<ComponentProps<"div">["onPointerMove"]> = (event) => {
    const resize = trainingPreviewResizeRef.current;
    if (!resize || resize.pointerId !== event.pointerId) return;
    setTrainingPreviewListHeight(clampTrainingPreviewHeight(resize.startHeight + event.clientY - resize.startY));
  };

  const handleTrainingPreviewResizeEnd: NonNullable<ComponentProps<"div">["onPointerUp"]> = (event) => {
    if (trainingPreviewResizeRef.current?.pointerId !== event.pointerId) return;
    trainingPreviewResizeRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const handleIngest = async () => {
    if (parsedUrls.length === 0) {
      setStatusText("Nhập ít nhất 1 URL.");
      toast.warning("Nhập ít nhất 1 URL trước khi ingest.");
      return;
    }
    setStatusText(null);
    startProgress("mlflow-ingest", { title: "MLflow ingest", message: "Đang crawl và tạo Training Preview...", value: 8 });
    try {
      const result = await ingest(parsedUrls, selectedModel || undefined);
      const counts = result.counts || {};
      const summary = result.crawl_summary || null;
      const total = Number(counts.total || 0);
      const candidateCount = Number(counts.candidate || 0);
      setStatusText(`Đã ingest batch ${result.batch_id}`);
      setCrawlSummary(summary);
      succeedProgress("mlflow-ingest", {
        message: total > 0 ? `Hoàn tất: ${total} segments, ${candidateCount} candidates.` : "Crawl hoàn tất nhưng chưa tìm thấy comment.",
      });
      setSelectedCandidateIds([]);
      void refreshTrainingPreview(1, "all_batches");

      if (total <= 0) {
        toast.warning("Crawl hoàn tất nhưng không tìm thấy comment.");
      } else {
        toast.success(`Ingest thành công: ${total} segments, ${candidateCount} candidates.`);
      }
      return true;
    } catch {
      failProgress("mlflow-ingest", { message: "Ingest thất bại. Kiểm tra URL hoặc log backend." });
      setStatusText("Ingest thất bại.");
      toast.error("Ingest thất bại.");
    }
  };

  const reviewCandidateItems = async (
    selectedItems: MlflowCandidate[],
    action: "include_toxic" | "include_clean" | "drop",
  ) => {
    if (selectedItems.length === 0) return;

    try {
      const payload = await reviewCandidates(
        selectedItems.map((item) => ({
          id: item.id,
          action,
          decision: action === "drop" ? "reject" : "accept",
          pseudo_label:
            action === "include_toxic"
              ? 1
              : action === "include_clean"
                ? 0
                : item.pseudo_label === 0
                  ? 0
                  : item.pseudo_label === 1
                    ? 1
                    : undefined,
        })),
      );
      void refreshTrainingPreview(1, "all_batches");
      const skippedLocked = payload.skipped_locked || 0;

      if (action === "include_toxic") {
        setStatusText(`Đã lưu ${payload.updated} mẫu Toxic vào DB.`);
        toast.success(`Đã lưu ${payload.updated} mẫu Toxic vào DB.`);
      } else if (action === "include_clean") {
        setStatusText(`Đã lưu ${payload.updated} mẫu Clean vào DB.`);
        toast.success(`Đã lưu ${payload.updated} mẫu Clean vào DB.`);
      } else {
        setStatusText(`Đã Remove ${payload.updated} mẫu khỏi train set.`);
        toast.success(`Đã Remove ${payload.updated} mẫu khỏi train set.`);
      }
      if (skippedLocked > 0) {
        toast.error(`Bỏ qua ${skippedLocked} mẫu đang lock.`);
      }
    } catch {
      setStatusText("Lưu review vào DB thất bại.");
      toast.error("Lưu review vào DB thất bại.");
    }
  };

  const handleBulkReview = async (action: "include_toxic" | "include_clean" | "drop") => {
    if (selectedCandidateIds.length === 0) return;
    const selectedItems = candidates.filter((item) => selectedCandidateIds.includes(item.id));
    if (await reviewCandidateItems(selectedItems, action)) setSelectedCandidateIds([]);
  };

  const handleModelReEvaluation = async (
    selection: "selected" | "all_auto_eligible",
    sampleIds: number[] = [],
  ) => {
    if (!reEvaluationModel) {
      toast.warning("Chọn project model trước khi re-evaluate.");
      return;
    }
    if (selection === "selected" && sampleIds.length === 0) {
      toast.warning("Chọn ít nhất một sample để re-evaluate.");
      return;
    }
    const requestedCount = selection === "selected" ? sampleIds.length : trainingPreview?.counts.auto_eligible ?? 0;
    if (selection === "all_auto_eligible") {
      const confirmed = window.confirm(
        `Re-evaluate ${requestedCount} auto-labelled training-eligible samples với ${getModelLabel(reEvaluationModel)}?`,
      );
      if (!confirmed) return;
    }

    setModelReEvaluating(true);
    startProgress("mlflow-model-reevaluation", {
      title: "Re-evaluate with Model",
      message: `Đang chạy ${getModelLabel(reEvaluationModel)} trên ${requestedCount} sample...`,
      value: 12,
    });
    try {
      const payload = await reEvaluateWithModel({
        modelId: reEvaluationModel,
        selection,
        sampleIds,
        trainingScope: "all_batches",
      });
      setLastReEvaluation(payload);
      const summary = payload.summary;
      const message = `Evaluated ${summary.evaluated} · Agreement ${summary.agreement} · Conflict ${summary.conflict} · Uncertain ${summary.uncertain} · Skipped ${summary.skipped} · Failed ${summary.failed}`;
      setStatusText(message);
      if (summary.failed > 0) {
        failProgress("mlflow-model-reevaluation", { message });
        toast.error(message);
      } else {
        succeedProgress("mlflow-model-reevaluation", { message });
        toast.success(message);
      }
      setSelectedPreviewIds([]);
      setSelectedCandidateIds([]);
    } catch {
      failProgress("mlflow-model-reevaluation", { message: "Model re-evaluation thất bại." });
      toast.error("Model re-evaluation thất bại.");
    } finally {
      setModelReEvaluating(false);
    }
  };

  const handleBulkLock = async (lockState: boolean) => {
    if (selectedCandidateIds.length === 0) return;
    const selectedItems = candidates.filter((item) => selectedCandidateIds.includes(item.id));
    if (selectedItems.length === 0) return;
    try {
      const payload = await reviewCandidates(
        selectedItems.map((item) => ({
          id: item.id,
          lock_state: lockState,
        })),
      );
      const changed = payload.locked_updated ?? payload.updated;
      const message = lockState ? `Đã lock ${changed} mẫu.` : `Đã unlock ${changed} mẫu.`;
      setStatusText(message);
      toast.success(message);
      void refreshTrainingPreview(trainingPreview?.page || 1, "all_batches");
    } catch {
      toast.error(lockState ? "Lock mẫu thất bại." : "Unlock mẫu thất bại.");
    }
  };

  const handleGeminiReviewCandidates = async () => {
    if (selectedCandidateIds.length === 0) {
      toast.warning("Chọn ít nhất 1 dòng Manual Verify để Gemini review.");
      return;
    }
    setCandidateGeminiReviewing(true);
    startProgress("gemini-candidate-review", { title: "Gemini Review", message: `Đang đánh giá ${selectedCandidateIds.length} dòng Manual Verify...` });
    try {
      setCandidateGeminiSuggestions((prev) => {
        const next = { ...prev };
        selectedCandidateIds.forEach((id) => delete next[id]);
        return next;
      });
      const payload = await geminiReviewCandidates(selectedCandidateIds);
      const next = Object.fromEntries(payload.suggestions.map((item) => [item.id, item]));
      setCandidateGeminiSuggestions((prev) => ({ ...prev, ...next }));
      succeedProgress("gemini-candidate-review", { message: `Đã review ${payload.reviewed}/${payload.requested} dòng.` });
      if (payload.failed_ids?.length) {
        toast.warning(`Gemini đã review ${payload.reviewed}/${payload.requested} dòng Manual Verify. ${payload.failed_ids.length} dòng chưa có JSON hợp lệ, hãy thử lại các dòng đó.`);
      } else {
        toast.success(`Gemini đã review ${payload.reviewed}/${payload.requested} dòng Manual Verify.`);
      }
    } catch (error) {
      const detail = error instanceof Error && error.message ? error.message : "Gemini review thất bại.";
      failProgress("gemini-candidate-review", { message: detail });
      toast.error(detail);
    } finally {
      setCandidateGeminiReviewing(false);
    }
  };

  const buildCandidateGeminiReviewUpdate = (suggestion: MlflowGeminiReviewSuggestion) => ({
    id: suggestion.id,
    action: suggestion.toxicity_label === 1 ? ("include_toxic" as const) : ("include_clean" as const),
    decision: "accept" as const,
    pseudo_label: suggestion.toxicity_label,
    ...(suggestion.constructiveness_label === 0 || suggestion.constructiveness_label === 1
      ? { constructiveness_label: suggestion.constructiveness_label }
      : { clear_constructiveness: true }),
    label_source: "gemini_assist",
    label_confidence: suggestion.confidence,
    reviewed_by_gemini: true,
    review_provider: suggestion.provider,
    review_model_name: suggestion.model,
  });

  const handleApplyCandidateGeminiSuggestions = async (suggestions: MlflowGeminiReviewSuggestion[]) => {
    if (suggestions.length === 0) return;
    setCandidateGeminiApplying(true);
    try {
      await reviewCandidates(suggestions.map(buildCandidateGeminiReviewUpdate));
      const appliedIds = new Set(suggestions.map((item) => item.id));
      setCandidateGeminiSuggestions((prev) =>
        Object.fromEntries(Object.entries(prev).filter(([id]) => !appliedIds.has(Number(id)))),
      );
      setSelectedCandidateIds((prev) => prev.filter((id) => !appliedIds.has(id)));
      await refreshTrainingPreview(1, "all_batches");
      toast.success(`Đã áp dụng và chuyển ${suggestions.length} mẫu sang Training Preview.`);
    } catch {
      toast.error("Áp dụng gợi ý Gemini trong Manual Verify thất bại.");
    } finally {
      setCandidateGeminiApplying(false);
    }
  };

  const handlePreviewSelection = async (id: number, selected: boolean, isLocked: boolean) => {
    if (!selected && isLocked) {
      toast.error("Mẫu đang lock, hãy unlock trước khi bỏ khỏi training.");
      return;
    }
    try {
      const payload = await reviewTrainingPreview([{ id, selected_for_training: selected }]);
      if ((payload.skipped_locked || 0) > 0) {
        toast.error("Mẫu đang lock nên không thể bỏ khỏi training.");
        return;
      }
      toast.success(selected ? "Đã chọn mẫu cho training." : "Đã loại mẫu khỏi training.");
    } catch {
      toast.error("Cập nhật preview thất bại.");
    }
  };

  const handlePreviewConstructiveness = async (id: number, label: 0 | 1 | null) => {
    try {
      await reviewTrainingPreview([
        label === null
          ? { id, clear_constructiveness: true }
          : { id, constructiveness_label: label },
      ]);
      toast.success("Đã cập nhật constructiveness.");
    } catch {
      toast.error("Cập nhật constructiveness thất bại.");
    }
  };

  const handlePreviewToxicity = async (id: number, label: 0 | 1) => {
    try {
      await reviewTrainingPreview([
        {
          id,
          pseudo_label: label,
          label_source: "manual_override",
          label_confidence: "high",
        },
      ]);
      toast.success(label === 1 ? "Đã sửa nhãn thành Độc hại." : "Đã sửa nhãn thành Sạch.");
    } catch {
      toast.error("Cập nhật nhãn độc hại thất bại.");
    }
  };

  const handleBulkPreviewReview = async (
    updates: Parameters<typeof reviewTrainingPreview>[0],
    successMessage: string,
    failureMessage: string,
  ) => {
    if (updates.length === 0) {
      toast.warning("Chọn ít nhất 1 dòng Training Preview để thao tác.");
      return;
    }
    setBulkPreviewUpdating(true);
    try {
      const payload = await reviewTrainingPreview(updates);
      setSelectedPreviewIds([]);
      if ((payload.skipped_locked || 0) > 0) {
        toast.warning(`${successMessage} ${payload.skipped_locked} mẫu lock đã được giữ nguyên.`);
      } else {
        toast.success(successMessage);
      }
    } catch {
      toast.error(failureMessage);
    } finally {
      setBulkPreviewUpdating(false);
    }
  };

  const handleBulkPreviewSelection = (selected: boolean) => {
    const ids = [...selectedPreviewIds];
    void handleBulkPreviewReview(
      ids.map((id) => ({ id, selected_for_training: selected })),
      selected ? `Đã chọn ${ids.length} mẫu cho training.` : `Đã bỏ ${ids.length} mẫu khỏi training.`,
      selected ? "Chọn các mẫu cho training thất bại." : "Bỏ các mẫu khỏi training thất bại.",
    );
  };

  const handleBulkPreviewToxicity = (label: 0 | 1) => {
    const ids = [...selectedPreviewIds];
    void handleBulkPreviewReview(
      ids.map((id) => ({ id, pseudo_label: label, label_source: "manual_override", label_confidence: "high" })),
      label === 1 ? `Đã gán nhãn Độc hại cho ${ids.length} mẫu.` : `Đã gán nhãn Sạch cho ${ids.length} mẫu.`,
      "Cập nhật nhãn độc hại cho các mẫu đã chọn thất bại.",
    );
  };

  const handleBulkPreviewConstructiveness = (label: 0 | 1 | null) => {
    const ids = [...selectedPreviewIds];
    void handleBulkPreviewReview(
      ids.map((id) => (label === null ? { id, clear_constructiveness: true } : { id, constructiveness_label: label })),
      label === 1
        ? `Đã gán nhãn Có tính xây dựng cho ${ids.length} mẫu.`
        : label === 0
          ? `Đã gán nhãn Không xây dựng cho ${ids.length} mẫu.`
          : `Đã ẩn/xóa nhãn tính xây dựng của ${ids.length} mẫu.`,
      "Cập nhật tính xây dựng cho các mẫu đã chọn thất bại.",
    );
  };

  const handleBulkPreviewLock = (lockState: boolean) => {
    const ids = [...selectedPreviewIds];
    void handleBulkPreviewReview(
      ids.map((id) => ({ id, lock_state: lockState })),
      lockState ? `Đã khóa ${ids.length} mẫu.` : `Đã mở khóa ${ids.length} mẫu.`,
      lockState ? "Khóa các mẫu đã chọn thất bại." : "Mở khóa các mẫu đã chọn thất bại.",
    );
  };

  const handleGeminiReviewPreview = async () => {
    if (selectedPreviewIds.length === 0) {
      toast.warning("Chọn ít nhất 1 dòng preview để Gemini review.");
      return;
    }
    const reviewIds = [...selectedPreviewIds];
    setGeminiReviewing(true);
    startProgress("gemini-preview-review", { title: "Gemini Review", message: `Đang đánh giá ${reviewIds.length} dòng Training Preview...` });
    try {
      setGeminiSuggestions((prev) => {
        const next = { ...prev };
        reviewIds.forEach((id) => delete next[id]);
        return next;
      });
      const payload = await geminiReviewTrainingPreview(reviewIds);
      const next = Object.fromEntries(payload.suggestions.map((item) => [item.id, item]));
      setGeminiSuggestions((prev) => ({ ...prev, ...next }));
      setSelectedPreviewIds([]);
      succeedProgress("gemini-preview-review", { message: `Đã review ${payload.reviewed}/${payload.requested} dòng.` });
      if (payload.failed_ids?.length) {
        toast.warning(`Gemini đã review ${payload.reviewed}/${payload.requested} dòng. ${payload.failed_ids.length} dòng chưa có JSON hợp lệ, hãy thử lại các dòng đó.`);
      } else {
        toast.success(`Gemini đã review ${payload.reviewed}/${payload.requested} dòng.`);
      }
    } catch (error) {
      const detail = error instanceof Error && error.message ? error.message : "Gemini review thất bại.";
      failProgress("gemini-preview-review", { message: detail });
      toast.error(detail);
    } finally {
      setGeminiReviewing(false);
    }
  };

  const buildGeminiReviewUpdate = (suggestion: MlflowGeminiReviewSuggestion) => ({
    id: suggestion.id,
    pseudo_label: suggestion.toxicity_label,
    ...(suggestion.constructiveness_label === 0 || suggestion.constructiveness_label === 1
      ? { constructiveness_label: suggestion.constructiveness_label }
      : { clear_constructiveness: true }),
    label_source: "gemini_assist",
    label_confidence: suggestion.confidence,
    reviewed_by_gemini: true,
    review_provider: suggestion.provider,
    review_model_name: suggestion.model,
  });

  const clearAppliedGeminiSuggestions = (suggestions: MlflowGeminiReviewSuggestion[]) => {
    const appliedIds = new Set(suggestions.map((suggestion) => suggestion.id));
    setGeminiSuggestions((prev) => Object.fromEntries(Object.entries(prev).filter(([id]) => !appliedIds.has(Number(id)))));
  };

  const dismissGeminiSuggestion = (suggestion: MlflowGeminiReviewSuggestion) => {
    setGeminiSuggestions((prev) => {
      const next = { ...prev };
      delete next[suggestion.id];
      return next;
    });
  };

  const dismissCandidateGeminiSuggestion = (suggestion: MlflowGeminiReviewSuggestion) => {
    setCandidateGeminiSuggestions((prev) => {
      const next = { ...prev };
      delete next[suggestion.id];
      return next;
    });
  };

  const handleApplyGeminiSuggestions = async (suggestions: MlflowGeminiReviewSuggestion[]) => {
    if (suggestions.length === 0) return;
    setGeminiApplying(true);
    try {
      await reviewTrainingPreview(suggestions.map(buildGeminiReviewUpdate));
      clearAppliedGeminiSuggestions(suggestions);
      setSelectedPreviewIds([]);
      toast.success(`Đã áp dụng ${suggestions.length} gợi ý Gemini.`);
    } catch {
      toast.error("Áp dụng gợi ý Gemini thất bại.");
    } finally {
      setGeminiApplying(false);
    }
  };

  const handleToggleCrawlHistory = () => {
    const nextOpen = !crawlHistoryOpen;
    setCrawlHistoryOpen(nextOpen);
    if (nextOpen) void refreshCrawlHistory(1);
  };
  const handlePreviewLock = async (id: number, lockState: boolean) => {
    try {
      await reviewTrainingPreview([{ id, lock_state: lockState }]);
      toast.success(lockState ? "Đã lock mẫu." : "Đã unlock mẫu.");
    } catch {
      toast.error(lockState ? "Lock mẫu thất bại." : "Unlock mẫu thất bại.");
    }
  };

  const handleClearAllMlflow = async () => {
    const firstConfirm = window.confirm("Xóa toàn bộ dữ liệu MLFlow? Hành động này không thể hoàn tác.");
    if (!firstConfirm) return;

    const token = window.prompt(`Nhập ${MLFLOW_CLEAR_ALL_CONFIRM_TOKEN} để xác nhận clear all:`);
    if (token === null) return;
    if (token.trim() !== MLFLOW_CLEAR_ALL_CONFIRM_TOKEN) {
      toast.error("Sai confirm token. Đã hủy clear all.");
      return;
    }

    try {
      const payload = await clearMlflowAll(token.trim());
      setSelectedCandidateIds([]);
      void refreshTrainingPreview(1, "all_batches");
      const rows = payload.deleted_rows;
      toast.success(
        `Đã clear MLFlow: do_run=${rows.mlflow_do_run}, artifacts=${rows.mlflow_training_artifact}, predictions=${rows.mlflow_comment_prediction ?? 0}, items=${rows.mlflow_comment_item}, batches=${rows.mlflow_crawl_batch}.`,
      );
    } catch {
      toast.error("Clear all MLFlow thất bại.");
    }
  };

  const downloadAdminFile = async (url: string, filename: string) => {
    const response = await fetch(buildApiUrl(url), {
      headers: { Authorization: `Bearer ${adminToken}` },
    });
    if (response.status === 401) {
      onAdminUnauthorized();
      throw new Error("Admin session expired");
    }
    if (!response.ok) {
      const raw = await response.text();
      throw new Error(raw || "Download failed");
    }

    const blob = await response.blob();
    const objectUrl = window.URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.rel = "noopener noreferrer";
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.URL.revokeObjectURL(objectUrl);
  };

  const handleExportBundle = async () => {
    try {
      const payload = await exportBundle(activeBatchId, {
        scope: "all_batches",
        modelKind: selectedModelKind,
        trainingMode: selectedTrainingMode,
        balanceStrategy,
        bundleProfile: "clean_victsd_gold",
        includeBaseModel: false,
        baseModel: selectedTrainingMode === "finetune" ? finetuneBaseModel.trim() || selectedModel : selectedModel,
        includeUnused: includeUnusedInExport,
        unusedScope,
      });

      await downloadAdminFile(payload.download_url, payload.bundle_path.split("/").pop() || "mlflow_bundle.zip");

      const merge = payload.merge_stats;
      const mergeText = merge
        ? ` | merge train +${merge.added_to_train}, dup ${merge.skipped_duplicate}, final ${merge.final_train_count}`
        : "";
      setStatusText(
        `Đã tạo và tải bundle (${payload.scope}). accepted ${payload.count}, candidate ${payload.candidate_count}, unused ${payload.unused_count}${mergeText}.`,
      );
    } catch {
      setStatusText("Export bundle thất bại.");
    }
  };

  const handleImportModelZip = async () => {
    const modelName = importModelName.trim();
    if (!modelName) {
      setStatusText("Nhập model name trước khi import.");
      return;
    }
    if (!importModelZipFile) {
      setStatusText("Chọn file ZIP model trước khi import.");
      return;
    }
    try {
      const payload = await importModelZip(modelName, importModelZipFile);
      if (typeof onModelsChanged === "function") {
        await onModelsChanged();
      }
      setSelectedModel(payload.model_id);
      setImportModelName("");
      setImportModelZipFile(null);
      setStatusText(`Đã import model ${payload.model_id}.`);
      toast.success(`Đã import model ${payload.model_id}.`);
    } catch {
      setStatusText("Import model ZIP thất bại.");
      toast.error("Import model ZIP thất bại.");
    }
  };

  const handleTriggerDO = async () => {
    if (kaggleTriggerPendingRef.current) return;
    const currentRunId = typeof doStatus?.run_id === "string" ? doStatus.run_id : "";
    const currentStatus = typeof doStatus?.status === "string" ? doStatus.status.toLowerCase() : "";
    if (currentRunId && !KAGGLE_TERMINAL_STATUSES.has(currentStatus)) {
      toast.message(`Kaggle run ${currentRunId} đang hoạt động; không tạo thêm run mới.`);
      return;
    }

    const resolvedBaseModel =
      selectedTrainingMode === "retrain" ? selectedModel.trim() : finetuneBaseModel.trim() || selectedModel.trim();
    if (selectedTrainingMode === "retrain" && !resolvedBaseModel) {
      setStatusText("Retrain yêu cầu base model. Hãy chọn model ở bước ingest trước khi trigger.");
      toast.error("Thiếu base model cho retrain.");
      return;
    }

    kaggleTriggerPendingRef.current = true;
    setKaggleTriggerPending(true);
    startProgress("kaggle-pipeline", { title: "Kaggle pipeline", message: "Đang gửi training job lên Kaggle...", value: 8 });
    try {
      const payload = await triggerDO({
        modelKind: selectedModelKind,
        trainingMode: selectedTrainingMode,
        baseModel: selectedModelKind === "lr_smoke" ? undefined : resolvedBaseModel || undefined,
        balanceStrategy,
        bundleScope: "all_batches",
      });
      const trainingLabel = selectedModelKind === "lr_smoke" ? "LR SMOKE" : selectedTrainingMode === "finetune" ? "FINETUNE" : "RETRAIN";
      setStatusText(`Đã trigger Kaggle run ${payload.run_id} (${payload.status}) - ${trainingLabel}.`);
      updateProgress("kaggle-pipeline", { message: `Run ${payload.run_id} đã được tạo, đang theo dõi tiến trình...`, value: 15 });
      toast.success(`Đã trigger Kaggle run ${payload.run_id} (${trainingLabel}).`);
    } catch (error) {
      const detail = error instanceof Error && error.message ? error.message : "Không rõ nguyên nhân.";
      setStatusText(`Trigger Kaggle pipeline thất bại: ${detail}`);
      failProgress("kaggle-pipeline", { message: detail });
      toast.error(`Trigger Kaggle pipeline thất bại: ${detail}`);
    } finally {
      kaggleTriggerPendingRef.current = false;
      setKaggleTriggerPending(false);
    }
  };

  const handleRefreshDOStatus = async () => {
    const targetRunId = typeof doStatus?.run_id === "string" ? doStatus.run_id : "";
    if (!targetRunId) {
      setStatusText("Chưa có run_id để refresh status.");
      return;
    }
    try {
      await refreshDOStatus(targetRunId);
      setStatusText(`Đã refresh trạng thái run ${targetRunId}.`);
    } catch {
      setStatusText("Refresh status thất bại.");
      toast.error("Refresh status thất bại.");
    }
  };

  const handleGeminiEvaluate = async (force = false) => {
    const runId = typeof doStatus?.run_id === "string" ? doStatus.run_id : "";
    if (!runId) return;
    setGeminiEvaluating(true);
    startProgress("gemini-evaluate", { title: "Gemini Evaluate", message: "Đang so sánh kết quả train và production..." });
    try {
      const payload = await geminiEvaluateKaggleRun(runId, force);
      setStatusText(payload.status === "cached" ? "Đã tải nhận định Gemini đã lưu." : "Gemini đã đánh giá kết quả train mới.");
      succeedProgress("gemini-evaluate", { message: payload.status === "cached" ? "Đã tải nhận định đã lưu." : "Đánh giá đã hoàn tất." });
      toast.success(payload.status === "cached" ? "Đã tải nhận định Gemini." : "Gemini Evaluate hoàn tất.");
    } catch (error) {
      const detail = error instanceof Error && error.message ? error.message : "Gemini Evaluate thất bại.";
      setStatusText(detail);
      failProgress("gemini-evaluate", { message: detail });
      toast.error(detail);
    } finally {
      setGeminiEvaluating(false);
    }
  };

  const handlePromote = async () => {
    const sourceRunId = comparePayload?.candidate?.source_run_id;
    if (!sourceRunId) {
      setStatusText("Chưa có candidate model để promote.");
      return;
    }
    try {
      const payload = await promote(
        sourceRunId,
        comparePayload?.candidate?.artifact_checksum,
        comparePayload?.current?.model,
      );
      setStatusText(payload.message || payload.status);
      setPromotionDialogOpen(false);
      await onModelsChanged?.();
      await refreshCompare(sourceRunId);
      toast.success(payload.message || "Promotion hoàn tất.");
    } catch (error) {
      setStatusText("Promote thất bại.");
      toast.error(error instanceof Error ? error.message : "Promote thất bại.");
    }
  };

  const handleRollback = async () => {
    const modelFamily = comparePayload?.model_family;
    if (!modelFamily) return;
    try {
      const payload = await rollback(modelFamily, comparePayload?.current?.model);
      setStatusText(payload.message || payload.status);
      await onModelsChanged?.();
      if (comparePayload?.candidate?.source_run_id) {
        await refreshCompare(comparePayload.candidate.source_run_id);
      }
      toast.success(payload.message || "Rollback hoàn tất.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Rollback thất bại.");
    }
  };

  const defaultDoStages = [
    "trigger_vm_gpu",
    "upload_data_and_train_files",
    "train",
    "save_artifact",
    "destroy_vm",
  ];
  const doStageLabels: Record<string, string> = {
    trigger_vm_gpu: "Provision VM (CPU/GPU)",
    upload_data_and_train_files: "Upload data + train files",
    train: "Train trên VM",
    save_artifact: "Lưu artifact",
    destroy_vm: "Destroy VM",
    prepare_local_bundle: "Prepare local fresh bundle",
    train_local_m1: "Train trên Kaggle",
    finalize_local_run: "Finalize local run",
  };

  const doStages =
    Array.isArray(doStatus?.stages) && (doStatus.stages as unknown[]).every((stage) => typeof stage === "string")
      ? (doStatus.stages as string[])
      : defaultDoStages;
  const doStatusValue = (doStatus?.status as string | undefined) || "idle";
  const doHasActiveRun = Boolean(
    doStatus?.run_id && !KAGGLE_TERMINAL_STATUSES.has(doStatusValue.toLowerCase()),
  );
  const doCurrentStage = (doStatus?.current_stage as string | undefined) || "";
  const doLogs = Array.isArray(doStatus?.logs) ? (doStatus?.logs as string[]) : [];
  const doRunId = (doStatus?.run_id as string | undefined) || "-";
  const doProvider = (doStatus?.provider as string | undefined) || "-";
  const doBatchId = (doStatus?.batch_id as string | undefined) || "-";
  const doGpuProfile = (doStatus?.gpu_profile as string | undefined) || "-";
  const doComputeMode = ((doStatus?.compute_mode as string | undefined) || "kaggle").toLowerCase();
  const doTrainingMode = ((doStatus?.training_mode as string | undefined) || "unknown").toLowerCase();
  const doModelKind = String(doStatus?.model_kind || "").toLowerCase();
  const doModelLabel = doModelKind === "lr_smoke" ? "TF-IDF + LR" : doModelKind === "phobert" ? "PhoBERT" : "Unknown model";
  const doAutomationEvent = automationStatus?.events.find((event) => event.source_run_id === doRunId);
  const doAutomationMode = doAutomationEvent?.detail?.match(/mode=([^;]+)/)?.[1] || null;
  const doBaseModel = (doStatus?.base_model as string | undefined) || "";
  const doDropletProfile = (doStatus?.droplet_profile as string | undefined) || doGpuProfile;
  const doEtaEstimate = Number(doStatus?.eta_estimate_minutes);
  const doTrainDuration = Number(doStatus?.train_duration_minutes);
  const doCpuPercent = Number(doStatus?.cpu_percent);
  const doMemoryPercent = Number(doStatus?.memory_percent);
  const doTelemetryLastSampleAt = (doStatus?.telemetry_last_sample_at as string | undefined) || "";
  const doDropletId = (doStatus?.droplet_id as string | undefined) || "-";
  const doArtifactUri = (doStatus?.artifact_uri as string | undefined) || "";
  const doArtifactKind = ((doStatus?.artifact_kind as string | undefined) || "none").toLowerCase();
  const doChecksum = (doStatus?.artifact_checksum as string | undefined) || "";
  const doArtifactDownloadUrl = doStatus?.artifact_download_url || "";
  const doMetrics = doStatus?.metrics || null;
  const doPreviousRun = doStatus?.previous_run || null;
  const doPreviousMetrics = doPreviousRun?.metrics || null;
  const doEvidence = doMetrics?.dataset_evidence || null;
  const doPreviousEvidence = doPreviousMetrics?.dataset_evidence || null;
  const doEvidenceDurationSeconds = Number(doEvidence?.duration_seconds);
  const doHasEvidenceDuration = Number.isFinite(doEvidenceDurationSeconds) && doEvidenceDurationSeconds >= 0;
  const doBundleEvidenceVerified = Boolean(
    doEvidence?.bundle_sha256 && doStatus?.bundle_checksum && doEvidence.bundle_sha256 === doStatus.bundle_checksum,
  );
  const doMetricChartData = ["accuracy", "macro_f1", "f1_toxic", "precision", "recall"].map((metric) => ({
    metric,
    validation: doMetrics?.splits?.validation?.[metric] ?? null,
    test: doMetrics?.splits?.test?.[metric] ?? null,
  }));
  const doTestConfusion = doMetrics?.confusion_matrix?.test || null;
  const doRunComparisonData = [
    { metric: "accuracy", label: "Accuracy", current: doMetrics?.accuracy ?? null, previous: doPreviousMetrics?.accuracy ?? null },
    { metric: "macro_f1", label: "Macro F1", current: doMetrics?.macro_f1 ?? null, previous: doPreviousMetrics?.macro_f1 ?? null },
    { metric: "f1_toxic", label: "F1 Toxic", current: doMetrics?.f1_toxic ?? null, previous: doPreviousMetrics?.f1_toxic ?? null },
    { metric: "precision", label: "Precision", current: doMetrics?.precision ?? null, previous: doPreviousMetrics?.precision ?? null },
    { metric: "recall", label: "Recall", current: doMetrics?.recall ?? null, previous: doPreviousMetrics?.recall ?? null },
  ].map((item) => ({
    ...item,
    delta:
      typeof item.current === "number" && typeof item.previous === "number"
        ? item.current - item.previous
        : null,
  }));
  const productionComparisonData = [
    { metric: "accuracy", label: "Accuracy" },
    { metric: "macro_f1", label: "Macro F1" },
    { metric: "f1_toxic", label: "F1 Toxic" },
    { metric: "precision", label: "Precision" },
    { metric: "recall", label: "Recall" },
  ].map((item) => ({
    ...item,
    current: comparePayload?.current?.metrics?.[item.metric] ?? null,
    candidate: comparePayload?.candidate?.metrics?.[item.metric] ?? null,
    delta: comparePayload?.deltas?.[item.metric] ?? null,
  }));
  const doPreviousTestSize = Number(doPreviousMetrics?.sizes?.test);
  const doCurrentTestSize = Number(doMetrics?.sizes?.test);
  const doComparableTestSet =
    Number.isFinite(doPreviousTestSize) &&
    Number.isFinite(doCurrentTestSize) &&
    doPreviousTestSize === doCurrentTestSize;
  const doF1Comparison = doRunComparisonData.filter((item) => item.metric === "f1_toxic" || item.metric === "macro_f1");
  const doF1Deltas = doF1Comparison.map((item) => item.delta).filter((value): value is number => typeof value === "number");
  const doComparisonSummary =
    doF1Deltas.length !== 2
      ? "CHƯA ĐỦ DỮ LIỆU"
      : doF1Deltas.every((value) => Math.abs(value) < 1e-9)
        ? "KHÔNG ĐỔI"
        : doF1Deltas.every((value) => value > 0)
        ? "HAI F1 CÙNG TĂNG"
        : doF1Deltas.every((value) => value >= 0)
          ? "F1 KHÔNG GIẢM"
        : doF1Deltas.every((value) => value < 0)
          ? "HAI F1 CÙNG GIẢM"
          : "CÓ TRADE-OFF";
  const doErrorMessage = (doStatus?.error_message as string | undefined) || "";
  const doRunMode = ((doStatus?.run_mode as string | undefined) || "unknown").toLowerCase();
  const doIsDryRun = doStatusValue === "dry_run";
  const doStatusSource = ((doStatus?.status_source as string | undefined) || "local_db").toLowerCase();
  const doStageTimestamps =
    doStatus?.stage_timestamps && typeof doStatus.stage_timestamps === "object"
      ? (doStatus.stage_timestamps as Record<string, string | null>)
      : {};
  const doLogEvents = Array.isArray(doStatus?.log_events) ? (doStatus.log_events as Array<Record<string, unknown>>) : [];
  const doApiCallEvidence = "";
  const doIsMockRun = doRunMode === "mock" || doDropletId.toLowerCase().startsWith("mock_");
  const doIsPlaceholder =
    doStatusValue === "placeholder" ||
    doLogs.some((line) => line.toLowerCase().includes("placeholder flow only")) ||
    doIsMockRun;
  const doIsRestricted = /restricted|account tier|increase your account tier/i.test(doErrorMessage);
  const doIsMockArtifact = doArtifactKind === "mock" || doArtifactUri.toLowerCase().startsWith("mock://");
  const doHasRealArtifact = doArtifactKind === "real" && !!doArtifactUri;
  const formatIsoTs = (value: string | null | undefined) => {
    if (!value) return "-";
    const d = new Date(value);
    return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
  };
  const hasDoEtaEstimate = Number.isFinite(doEtaEstimate) && doEtaEstimate > 0;
  const hasDoTrainDuration = Number.isFinite(doTrainDuration) && doTrainDuration >= 0;
  const hasDoCpuPercent = Number.isFinite(doCpuPercent) && doCpuPercent >= 0;
  const hasDoMemoryPercent = Number.isFinite(doMemoryPercent) && doMemoryPercent >= 0;
  const hasDoTelemetrySample = doTelemetryLastSampleAt.length > 0;
  const formatMetric = (value: number | null | undefined) =>
    typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
  const formatMetricDelta = (value: number | null | undefined) =>
    typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(3)}` : "-";
  const metricDeltaClass = (value: number | null | undefined) =>
    typeof value !== "number" || !Number.isFinite(value)
      ? "text-muted-foreground"
      : value > 0
        ? "text-emerald-600 dark:text-emerald-400"
        : value < 0
          ? "text-rose-600 dark:text-rose-400"
          : "text-muted-foreground";
  const handleDownloadKaggleArtifact = () => {
    if (!doArtifactDownloadUrl) return;
    void downloadAdminFile(doArtifactDownloadUrl, doArtifactUri.split("/").pop() || "kaggle_exported_model.zip");
  };

  // Terminal/status-refresh entries are bookkeeping. Prefer the event that
  // actually represents the most recent automation attempt in the summary.
  const latestAutomationEvent = automationStatus?.events?.find(
    (event) => event.action === "train_started" || event.action === "train_start",
  ) || automationStatus?.events?.[0];
  const latestAutomationFamily = automationStatus?.families?.find((family) => family.model_family === (latestAutomationEvent?.model_family || "tfidf_lr"));
  useEffect(() => {
    if (!latestAutomationEvent || announcedAutomationEventRef.current === latestAutomationEvent.id) return;
    announcedAutomationEventRef.current = latestAutomationEvent.id;
    if (latestAutomationEvent.action === "train_started" && latestAutomationEvent.status === "running" && latestAutomationEvent.source_run_id) {
      toast.success(`Automatic ${latestAutomationEvent.model_family === "tfidf_lr" ? "TF-IDF training" : "PhoBERT training"} started`, {
        description: latestAutomationEvent.detail || "Automation threshold reached.",
        action: { label: "View run", onClick: () => void openDORun(latestAutomationEvent.source_run_id!) },
      });
    } else if (latestAutomationEvent.action === "train_started" && latestAutomationEvent.status === "dry_run") {
      toast.info("Automation dry run completed", { description: latestAutomationEvent.detail || "Kaggle submission was skipped." });
    } else if (latestAutomationEvent.status === "failed") {
      toast.error("Automatic training failed to start", { description: latestAutomationEvent.detail || "See automation details." });
    }
  }, [latestAutomationEvent, openDORun]);

  useEffect(() => {
    const prev = prevDoStatusRef.current;
    if (prev === doStatusValue) return;

    if (doStatusValue === "running") {
      startProgress("kaggle-pipeline", { title: "Kaggle pipeline", message: "Kaggle đang train mô hình..." });
      toast.message("Kaggle pipeline đang chạy.");
    } else if (doStatusValue === "completed") {
      succeedProgress("kaggle-pipeline", { message: "Kaggle pipeline đã hoàn tất." });
      toast.success("Kaggle pipeline hoàn tất.");
      if (doRunId) void refreshCompare(doRunId);
    } else if (doStatusValue === "failed") {
      failProgress("kaggle-pipeline", { message: doIsRestricted ? "GPU bị restricted; hãy chuyển CPU hoặc mở ticket tăng tier." : "Kaggle pipeline thất bại." });
      if (doIsRestricted) {
        toast.error("GPU bị restricted. Hãy chuyển CPU hoặc mở ticket tăng tier.");
      } else {
        toast.error("Kaggle pipeline thất bại.");
      }
    }

    prevDoStatusRef.current = doStatusValue;
  }, [doIsRestricted, doRunId, doStatusValue, failProgress, refreshCompare, startProgress, succeedProgress]);

  useEffect(() => {
    if (ingestStage) {
      updateProgress("mlflow-ingest", { message: ingestStageMessage || "Đang xử lý ingest...", value: Math.max(8, ingestProgress) });
    }
  }, [ingestProgress, ingestStage, ingestStageMessage, updateProgress]);

  const doCompletedIndex = doStages.findIndex((s) => s === doCurrentStage);
  const doHasStageProgress = ["running", "failed", "completed", "dry_run"].includes(doStatusValue);
  const doProgress =
    doStatusValue === "completed" || doStatusValue === "dry_run"
      ? 100
      : doStatusValue === "queued" || doStatusValue === "placeholder"
        ? 0
        : !doHasStageProgress || doCompletedIndex < 0
          ? 0
          : Math.min(95, Math.round(((doCompletedIndex + 1) / doStages.length) * 100));

  const doBadgeVariant =
    doStatusValue === "failed"
      ? "destructive"
      : doStatusValue === "completed"
        ? "secondary"
        : doStatusValue === "running" || doStatusValue === "queued"
          ? "default"
          : "outline";

  return (
    <div className="dashboard-page max-w-7xl mx-auto space-y-6">
      <Card className="p-5 border-border/80 bg-gradient-to-br from-background to-muted/30">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">Admin / Kaggle Retrain</p>
            <h1 className="text-2xl font-semibold">VietComment Analyzer Kaggle Retrain Console</h1>
            <p className="text-sm text-muted-foreground">Collect data → review & bundle → retrain on Kaggle → inspect metrics</p>
          </div>
          {showLegacyIngest && <div className="flex items-center gap-2">
            <Badge variant={ingestStageMeta.variant}>{ingestStageMeta.label}</Badge>
          </div>}
        </div>
      </Card>

      {error && (
        <Card className="p-4 border-destructive/40 bg-destructive/5">
          <p className="text-sm text-destructive">{error}</p>
        </Card>
      )}

      {statusText && (
        <Card className="p-4 border-border/70 bg-muted/40">
          <p className="text-sm">{statusText}</p>
        </Card>
      )}

      {showLegacyIngest && ingestStage !== "idle" && (
        <Card className="p-4 border-border/70 bg-muted/30 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-medium">Tiến trình ingest pipeline</p>
            <Badge variant={ingestStageMeta.variant}>{ingestStageMeta.label}</Badge>
          </div>
          <Progress value={ingestProgress} className="h-2" />
          <p className="text-xs text-muted-foreground">
            {ingestStageMessage || "Đang xử lý..."} ({Math.round(ingestProgress)}%)
          </p>
        </Card>
      )}

      {showLegacyIngest && hasNoBatch && (
        <Card className="p-5 border-primary/20 bg-primary/5 space-y-2">
          <h2 className="font-semibold">Chưa có dữ liệu</h2>
          <p className="text-sm text-muted-foreground">
            Đây là trạng thái bình thường khi mới vào hệ thống. Hãy nhập URL ở bước 1 và bấm <b>Ingest + Infer + Gate</b> để bắt đầu.
          </p>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="w-full grid grid-cols-3 h-auto">
          <TabsTrigger value="step1">Data & Review</TabsTrigger>
          <TabsTrigger value="step4">Kaggle Retrain</TabsTrigger>
          <TabsTrigger value="step5">Results, Gate & Registry</TabsTrigger>
        </TabsList>

        <TabsContent value="step1" className="space-y-4">
          {showLegacyIngest && (
          <Card className="p-4 space-y-3">
            <div className="grid gap-3 md:grid-cols-4">
              <div>
                <label className="text-sm">Model</label>
                <select
                  className="w-full mt-1 rounded-md border bg-background px-3 py-2 text-sm"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {availableModels.map((model) => {
                    const deprecated = isDeprecatedModel(model);
                    return (
                      <option key={model} value={model} disabled={deprecated} className={deprecated ? "text-muted-foreground" : undefined}>
                        {getModelLabel(model)}
                      </option>
                    );
                  })}
                </select>
              </div>
              <div className="md:col-span-3">
                <label className="text-sm">URLs (mỗi dòng 1 URL)</label>
                <textarea
                  className="w-full mt-1 min-h-24 rounded-md border bg-background px-3 py-2 text-sm"
                  value={urlsText}
                  onChange={(e) => setUrlsText(e.target.value)}
                  placeholder="https://vnexpress.net/..."
                />
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button onClick={handleIngest}>Ingest + Infer + Gate</Button>
              <Button
                variant="outline"
                onClick={() => {
                  void refreshOverview();
                  void refreshCandidates(undefined, 1, "all_batches");
                  void refreshTrainingPreview(1, "all_batches");
                  void refreshThresholdStatus(activeBatchId);
                  void refreshReviewHistory(undefined, historyDecision, 1, "all_batches");
                  if (crawlHistoryOpen) void refreshCrawlHistory(1);
                  void refreshCompare();
                }}
              >
                Refresh
              </Button>
              <Button variant={crawlHistoryOpen ? "secondary" : "outline"} onClick={handleToggleCrawlHistory}>
                <History className="h-4 w-4" />
                History
              </Button>
              <Button variant="destructive" onClick={handleClearAllMlflow}>
                Clear all MLFlow
              </Button>
            </div>
            {crawlSummary && (
              <Card className="border-border/70 bg-muted/30 p-3">
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <Badge variant="outline">URLs: {crawlSummary.total_urls ?? 0}</Badge>
                  <Badge variant="secondary">ok: {crawlSummary.status_counts?.ok ?? 0}</Badge>
                  <Badge variant="secondary">no_comments: {crawlSummary.status_counts?.no_comments ?? 0}</Badge>
                  <Badge variant="secondary">blocked: {crawlSummary.status_counts?.blocked ?? 0}</Badge>
                  <Badge variant={((crawlSummary.timeout_count ?? 0) > 0 ? "destructive" : "outline") as "destructive" | "outline"}>
                    timeout: {crawlSummary.timeout_count ?? 0}
                  </Badge>
                  <Badge variant="outline">retry: {crawlSummary.status_counts?.retried ?? 0}</Badge>
                  <Badge variant="outline">cache: {crawlSummary.status_counts?.from_cache ?? 0}</Badge>
                </div>
              </Card>
            )}
            {crawlHistoryOpen && (
              <div className="space-y-2 rounded-md border border-border/70 bg-muted/20 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-sm font-medium">Lịch sử URL đã crawl</p>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      {crawlHistoryTotal} URL · trang {crawlHistoryPage}
                    </span>
                    <Button size="sm" variant="ghost" onClick={() => refreshCrawlHistory(crawlHistoryPage)}>
                      <RotateCcw className="h-4 w-4" />
                      Tải lại
                    </Button>
                  </div>
                </div>
                <div className="max-h-56 space-y-1.5 overflow-auto pr-1">
                  {crawlHistory.map((item) => (
                    <div key={`${item.batch_id}:${item.url_hash}`} className="rounded-md border bg-background p-2">
                      <p className="break-all text-sm">{item.url}</p>
                      <p className="text-xs text-muted-foreground">
                        batch={item.batch_id} · segments={item.segment_count} · accepted={item.accepted_count} · candidate={item.candidate_count} · discarded={item.discarded_count}
                      </p>
                    </div>
                  ))}
                  {crawlHistory.length === 0 && <p className="text-sm text-muted-foreground">Chưa có lịch sử crawl.</p>}
                </div>
              </div>
            )}
          </Card>
          )}

          {showLegacyIngest && (
          <Card className="border-border/70 p-3">
            <div className="grid gap-3 divide-y md:grid-cols-[0.8fr_1.2fr_2fr] md:divide-x md:divide-y-0">
              <div className="flex items-center justify-between gap-3 md:pr-3">
                <div>
                  <p className="text-xs text-muted-foreground">Crawl mới</p>
                  <p className="text-xs text-muted-foreground">segments</p>
                </div>
                <p className="text-2xl font-semibold">{overview?.pipeline_counts?.crawled ?? 0}</p>
              </div>
              <div className="flex items-center justify-between gap-3 pt-3 md:px-3 md:pt-0">
                <div className="min-w-0">
                  <p className="text-xs text-muted-foreground">Infer + Pseudo-label</p>
                  <p className="truncate text-xs text-muted-foreground">{getModelLabel(overview?.model_name || selectedModel) || "-"}</p>
                </div>
                <p className="text-2xl font-semibold">{overview?.pipeline_counts?.inferred ?? 0}</p>
              </div>
              <div className="flex flex-wrap items-center gap-2 pt-3 md:pl-3 md:pt-0">
                <span className="mr-auto text-xs text-muted-foreground">Gate 0.8 / 0.2</span>
                <Badge variant="secondary">Accepted {overview?.pipeline_counts?.accepted ?? 0}</Badge>
                <Badge variant="outline">Candidate {overview?.pipeline_counts?.candidate ?? 0}</Badge>
                <Badge variant="outline">Discarded {overview?.pipeline_counts?.discarded ?? 0}</Badge>
              </div>
            </div>
          </Card>
          )}

          <Card className="border-border/80 bg-gradient-to-r from-background to-muted/25 p-3">
            <div className="grid gap-3 md:grid-cols-[auto_minmax(12rem,1fr)_auto] md:items-center">
              <div className="flex items-center gap-3">
                <div>
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Dataset Bundle</p>
                  <p className="font-medium">Snapshot sẽ được xuất</p>
                </div>
                <Badge variant={bundleReady ? "secondary" : "outline"}>
                  {!trainingPlan ? "Đang tính" : bundleReady ? "Ready" : "Not ready"}
                </Badge>
              </div>
              <div className="space-y-1">
                <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
                  <span>MLflow thực tế thêm vào bundle</span>
                  <span className="font-medium text-foreground">{bundleIncludedCount} / {bundleTargetCount}</span>
                </div>
                <Progress value={thresholdProgress} className="h-1.5" />
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                  <span>Đủ điều kiện: <b className="text-foreground">{trainingPlan?.summary.eligible_mlflow ?? "-"}</b></span>
                  <span>Sau cân bằng: <b className="text-foreground">{trainingPlan?.summary.after_balance ?? "-"}</b></span>
                  <span>Trùng loại: <b className="text-foreground">{trainingPlan?.summary.duplicates_skipped ?? "-"}</b></span>
                  <span>Tổng train: <b className="text-foreground">{trainingPlan?.summary.final_train ?? "-"}</b></span>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 md:justify-end">
                <Button size="sm" onClick={handleExportBundle} disabled={!trainingPlan}>Download bundle</Button>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button size="sm" variant="outline">Advanced</Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-2xl">
                    <DialogHeader>
                      <DialogTitle>Advanced bundle tools</DialogTitle>
                      <DialogDescription>Export options, model ZIP import, và refresh compare source.</DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div className="rounded-md border p-3 space-y-3 bg-muted/20">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="text-sm font-medium">Export bundle</p>
                            <p className="text-xs text-muted-foreground">
                              Mặc định: accepted + candidate. Có thể thêm unused/discarded.
                            </p>
                          </div>
                          <Button size="sm" onClick={handleExportBundle}>
                            Download
                          </Button>
                        </div>
                        <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border bg-background p-3">
                          <div>
                            <p className="text-sm font-medium">Include unused/discarded</p>
                            <p className="text-xs text-muted-foreground">Bật để export thêm discarded theo scope.</p>
                          </div>
                          <label className="inline-flex items-center gap-2 text-sm">
                            <Checkbox
                              checked={includeUnusedInExport}
                              onCheckedChange={(checked) => setIncludeUnusedInExport(checked === true)}
                            />
                            Include unused
                          </label>
                        </div>
                        {includeUnusedInExport && (
                          <div>
                            <label className="text-xs text-muted-foreground">Unused scope</label>
                            <select
                              className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
                              value={unusedScope}
                              onChange={(e) => setUnusedScope(e.target.value as MlflowUnusedScope)}
                            >
                              <option value="all">Tất cả discarded</option>
                              <option value="auto_discarded">Auto discarded (theo ngưỡng)</option>
                              <option value="manual_rejected">Manual rejected (do reviewer)</option>
                            </select>
                          </div>
                        )}
                        <p className="text-xs text-muted-foreground break-all">Last bundle path: {lastBundlePath || "-"}</p>
                      </div>

                      <div className="rounded-md border p-3 space-y-3 bg-muted/20">
                        <p className="text-sm font-medium">Model ZIP import</p>
                        <div className="grid gap-3 md:grid-cols-2">
                          <Input
                            placeholder="my_phobert_v3"
                            value={importModelName}
                            onChange={(e: ChangeEvent<HTMLInputElement>) => setImportModelName(e.target.value)}
                          />
                          <Input
                            type="file"
                            accept=".zip,application/zip"
                            onChange={(e: ChangeEvent<HTMLInputElement>) => {
                              const file = e.target.files?.[0] || null;
                              setImportModelZipFile(file);
                            }}
                          />
                        </div>
                        <p className="text-xs text-muted-foreground">
                          ZIP sẽ được giải nén vào models/options/phobert/&lt;model_name&gt; và tự refresh danh sách model.
                        </p>
                        <div className="flex flex-wrap gap-2">
                          <Button onClick={handleImportModelZip}>Import ZIP model</Button>
                          <Button variant="outline" onClick={() => refreshCompare()}>
                            Refresh compare source
                          </Button>
                        </div>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            </div>
          </Card>

          <Card className="p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-1.5">
                <h3 className="font-medium">Training Preview</h3>
                {(trainingPreview?.counts.requires_human_review ?? 0) > 0 && (
                  <Badge variant="destructive">
                    <AlertTriangle className="mr-1 h-3.5 w-3.5" />
                    {trainingPreview?.counts.requires_human_review} require human review
                  </Badge>
                )}
                <SectionInfoTooltip label="Thông tin chi tiết Training Preview">
                  <p>
                    Danh sách hiển thị comment accepted đã qua gate và được chọn cho training; các candidate chưa xác minh chỉ xuất hiện ở đây khi là ngoại lệ model_conflict hoặc model_uncertain cần human review.
                    Mẫu accepted phải được chọn cho training và có nhãn Độc hại hoặc Sạch hợp lệ mới đủ điều kiện vào accepted export set;
                    balanced export có thể lấy ít hơn. Checkbox đầu hàng chỉ chọn tạm thời cho thao tác trên màn hình.
                  </p>
                  <p>
                    Nhãn tính xây dựng hiển thị khi DB có giá trị 0/1; NULL nghĩa là ẩn hoặc chưa có nhãn, không mặc định là độ tin cậy thấp.
                    Màu Điểm độc hại dùng gate mặc định 0.20/0.80 vì Preview API chưa trả threshold theo từng batch.
                  </p>
                </SectionInfoTooltip>
              </div>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className="h-9 px-3">
                  {selectedPreviewIds.length} đã chọn
                </Badge>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleSelectAllPreviewRows}
                  disabled={visibleTrainingPreviewItems.length === 0}
                >
                  Chọn tất cả
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleUnselectAllPreviewRows}
                  disabled={selectedPreviewIds.length === 0}
                >
                  Bỏ chọn
                </Button>
                <Button
                  size="sm"
                  onClick={() => handleBulkPreviewSelection(true)}
                  disabled={bulkPreviewUpdating || selectedPreviewIds.length === 0}
                >
                  <Check className="h-4 w-4" />
                  {bulkPreviewUpdating ? "Đang cập nhật..." : `Chọn cho training (${selectedPreviewIds.length})`}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleBulkPreviewSelection(false)}
                  disabled={bulkPreviewUpdating || selectedPreviewIds.length === 0}
                >
                  Bỏ khỏi training
                </Button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={bulkPreviewUpdating || selectedPreviewIds.length === 0}
                    >
                      Thao tác đã chọn
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-64">
                    <DropdownMenuLabel>Cập nhật nhãn độc hại</DropdownMenuLabel>
                    <DropdownMenuItem onSelect={() => handleBulkPreviewToxicity(1)}>
                      <MessageCircle className="text-rose-600 dark:text-rose-400" />
                      Gán Độc hại
                    </DropdownMenuItem>
                    <DropdownMenuItem onSelect={() => handleBulkPreviewToxicity(0)}>
                      <Check className="text-emerald-600 dark:text-emerald-400" />
                      Gán Sạch
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuLabel>Tính xây dựng</DropdownMenuLabel>
                    <DropdownMenuItem onSelect={() => handleBulkPreviewConstructiveness(1)}>
                      <ThumbsUp className="text-teal-600 dark:text-teal-400" />
                      Gán Có tính xây dựng
                    </DropdownMenuItem>
                    <DropdownMenuItem onSelect={() => handleBulkPreviewConstructiveness(0)}>
                      <MessageCircle className="text-amber-600 dark:text-amber-400" />
                      Gán Không xây dựng
                    </DropdownMenuItem>
                    <DropdownMenuItem onSelect={() => handleBulkPreviewConstructiveness(null)}>
                      <EyeOff />
                      Ẩn hoặc xóa nhãn
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onSelect={() => handleBulkPreviewLock(true)}>
                      <Lock />
                      Khóa các mẫu đã chọn
                    </DropdownMenuItem>
                    <DropdownMenuItem onSelect={() => handleBulkPreviewLock(false)}>
                      <Unlock />
                      Mở khóa các mẫu đã chọn
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="inline-flex">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={handleGeminiReviewPreview}
                        disabled={geminiReviewing || selectedPreviewIds.length === 0}
                        aria-label="Gửi các hàng đang chọn tạm thời cho Gemini review"
                        title="Gửi các hàng đang chọn tạm thời cho Gemini review"
                      >
                        <Sparkles className="h-4 w-4" />
                        {geminiReviewing ? "Đang review..." : `Gemini review (${selectedPreviewIds.length})`}
                      </Button>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    <MlflowTooltipBody
                      text={makeMlflowTooltip(
                        "Gửi Gemini review",
                        "Gửi các mẫu đang chọn tạm thời cho trợ lý review. Việc này không tự thay đổi danh sách training.",
                      )}
                    />
                  </TooltipContent>
                </Tooltip>
                {availableGeminiSuggestions.length > 0 && (
                  <Button
                    size="sm"
                    onClick={() => void handleApplyGeminiSuggestions(availableGeminiSuggestions)}
                    disabled={geminiApplying || geminiReviewing}
                  >
                    <Check className="h-4 w-4" />
                    {geminiApplying ? "Đang áp dụng..." : `Áp dụng Gemini (${availableGeminiSuggestions.length})`}
                  </Button>
                )}
                <Dialog>
                  <DialogTrigger asChild>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={(trainingPreview?.counts.selected ?? 0) === 0}
                      onClick={() => void refreshTrainingPreview(trainingPreview?.page || 1, "all_batches")}
                    >
                      <BarChart3 className="h-4 w-4" />
                      Biểu đồ phân bố
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-4xl">
                    <DialogHeader>
                      <DialogTitle>Phân bố dữ liệu Training Preview</DialogTitle>
                      <DialogDescription>
                        Thống kê trên toàn bộ comment hiện còn được chọn cho training, không chỉ các hàng đang hiển thị.
                      </DialogDescription>
                    </DialogHeader>
                    <div className="grid gap-6 md:grid-cols-2">
                      <div className="rounded-md border p-3">
                        <h4 className="text-center text-sm font-medium">Độc hại / Sạch</h4>
                        <ResponsiveContainer width="100%" height={280}>
                          <PieChart>
                            <Pie
                              data={toxicityDistribution}
                              dataKey="value"
                              nameKey="name"
                              cx="50%"
                              cy="46%"
                              outerRadius={82}
                              labelLine={false}
                              label={({ percent }) => `${((percent || 0) * 100).toFixed(0)}%`}
                            >
                              {toxicityDistribution.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                            </Pie>
                            <RechartTooltip />
                            <Legend />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="rounded-md border p-3">
                        <h4 className="text-center text-sm font-medium">Tính xây dựng</h4>
                        <ResponsiveContainer width="100%" height={280}>
                          <PieChart>
                            <Pie
                              data={constructivenessDistribution}
                              dataKey="value"
                              nameKey="name"
                              cx="50%"
                              cy="46%"
                              outerRadius={82}
                              labelLine={false}
                              label={({ percent }) => `${((percent || 0) * 100).toFixed(0)}%`}
                            >
                              {constructivenessDistribution.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                            </Pie>
                            <RechartTooltip />
                            <Legend />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
                <IconButtonWithTooltip
                  label="Tải lại preview"
                  size="icon"
                  variant="outline"
                  onClick={() => refreshTrainingPreview(trainingPreview?.page || 1, "all_batches")}
                >
                  <RotateCcw className="h-4 w-4" />
                </IconButtonWithTooltip>
              </div>
            </div>
            <div className="rounded-xl border border-sky-200/70 bg-sky-50/40 p-3 dark:border-sky-900/40 dark:bg-sky-950/15">
              <div className="flex flex-wrap items-end gap-3">
                <div className="min-w-64 flex-1">
                  <label className="text-xs font-medium text-muted-foreground">Re-evaluate with Model</label>
                  <select
                    className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
                    value={reEvaluationModel}
                    onChange={(event) => setReEvaluationModel(event.target.value)}
                    disabled={modelReEvaluating}
                  >
                    {availableModels.filter((model) => !isDeprecatedModel(model)).map((model) => (
                      <option key={`reevaluate-${model}`} value={model}>{getModelLabel(model)}</option>
                    ))}
                  </select>
                </div>
                <div className="min-w-56">
                  <label className="text-xs font-medium text-muted-foreground">Scope</label>
                  <select
                    className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
                    value={reEvaluationScope}
                    onChange={(event) => setReEvaluationScope(event.target.value as "selected" | "all_auto_eligible")}
                    disabled={modelReEvaluating}
                  >
                    <option value="selected">Selected auto labels ({selectedAutoEligibleCount})</option>
                    <option value="all_auto_eligible">All auto training-eligible ({trainingPreview?.counts.auto_eligible ?? 0})</option>
                  </select>
                </div>
                <Button
                  onClick={() => void handleModelReEvaluation(reEvaluationScope, selectedPreviewIds)}
                  disabled={
                    modelReEvaluating ||
                    !reEvaluationModel ||
                    (reEvaluationScope === "selected"
                      ? selectedAutoEligibleCount === 0
                      : (trainingPreview?.counts.auto_eligible ?? 0) === 0)
                  }
                >
                  <RefreshCw className={`h-4 w-4 ${modelReEvaluating ? "animate-spin" : ""}`} />
                  {modelReEvaluating ? "Đang re-evaluate..." : "Re-evaluate Auto Labels"}
                </Button>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                Chỉ auto-labelled samples đang training-eligible được bulk evaluate. Human-reviewed labels không được thay đổi.
              </p>
              <div className="mt-2 grid gap-2 text-xs sm:grid-cols-3">
                <Badge variant="secondary">Ready for training: {trainingPreview?.counts.selected ?? 0}</Badge>
                <Badge variant={(trainingPreview?.counts.requires_human_review ?? 0) > 0 ? "destructive" : "outline"}>
                  Requires human review: {trainingPreview?.counts.requires_human_review ?? 0}
                </Badge>
                <Badge variant="outline">Excluded / removed: {trainingPreview?.counts.removed ?? 0}</Badge>
              </div>
              {lastReEvaluation && (
                <p className="mt-2 text-xs font-medium">
                  Last run: Evaluated {lastReEvaluation.summary.evaluated} · Agreement {lastReEvaluation.summary.agreement} · Conflict {lastReEvaluation.summary.conflict} · Needs review {lastReEvaluation.summary.needs_review} · Skipped {lastReEvaluation.summary.skipped} · Failed {lastReEvaluation.summary.failed}
                </p>
              )}
            </div>

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <div className="relative overflow-hidden rounded-xl border border-primary/20 bg-gradient-to-br from-primary/10 via-background to-background p-4 shadow-sm">
                <div className="absolute -right-5 -top-5 h-20 w-20 rounded-full bg-primary/10" aria-hidden="true" />
                <div className="relative flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Đã chọn cho training</p>
                    <p className="mt-2 text-3xl font-semibold tracking-tight">{trainingPreview?.counts.selected ?? 0}</p>
                    <p className="mt-1 text-xs text-muted-foreground">Mẫu đủ điều kiện để đưa vào bundle</p>
                  </div>
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
                    <Check className="h-5 w-5" />
                  </span>
                </div>
              </div>

              <div className="rounded-xl border border-rose-200/70 bg-gradient-to-br from-rose-50/70 via-background to-emerald-50/50 p-4 shadow-sm dark:border-rose-900/40 dark:from-rose-950/20 dark:to-emerald-950/15">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Độc hại / Sạch</p>
                  <MessageCircle className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                </div>
                <div className="mt-3 grid grid-cols-2 divide-x rounded-lg border bg-background/70">
                  <div className="px-3 py-2">
                    <p className="text-[11px] font-medium text-rose-600 dark:text-rose-400">Độc hại</p>
                    <p className="text-2xl font-semibold text-rose-700 dark:text-rose-300">{trainingPreview?.counts.selected_toxic ?? 0}</p>
                  </div>
                  <div className="px-3 py-2">
                    <p className="text-[11px] font-medium text-emerald-600 dark:text-emerald-400">Sạch</p>
                    <p className="text-2xl font-semibold text-emerald-700 dark:text-emerald-300">{trainingPreview?.counts.selected_clean ?? 0}</p>
                  </div>
                </div>
                <div className="mt-3 flex h-1.5 overflow-hidden rounded-full bg-emerald-200/70 dark:bg-emerald-950/70" aria-label="Tỷ lệ Độc hại và Sạch">
                  <span
                    className="bg-rose-500"
                    style={{
                      width: `${((trainingPreview?.counts.selected_toxic ?? 0) / Math.max(trainingPreview?.counts.selected ?? 0, 1)) * 100}%`,
                    }}
                  />
                </div>
              </div>

              <div className="rounded-xl border border-sky-200/70 bg-gradient-to-br from-sky-50/80 via-background to-background p-4 shadow-sm dark:border-sky-900/40 dark:from-sky-950/20">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Xuất cân bằng</p>
                  <BarChart3 className="h-4 w-4 text-sky-600 dark:text-sky-400" aria-hidden="true" />
                </div>
                <p className="mt-2 text-3xl font-semibold tracking-tight">{trainingPreview?.balance.balanced_count ?? 0}</p>
                <Progress
                  className="mt-3 h-2 bg-sky-100 dark:bg-sky-950"
                  value={((trainingPreview?.balance.balanced_count ?? 0) / Math.max(trainingPreview?.counts.selected ?? 0, 1)) * 100}
                />
                <p className="mt-2 text-xs text-muted-foreground">
                  {balanceStrategy === "balanced_50_50" ? "Theo chiến lược cân bằng 50 / 50" : "Theo chiến lược giữ toàn bộ dữ liệu"}
                </p>
              </div>

              <div className="rounded-xl border border-amber-200/70 bg-gradient-to-br from-amber-50/80 via-background to-background p-4 shadow-sm dark:border-amber-900/40 dark:from-amber-950/20">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Tính xây dựng</p>
                  <EyeOff className="h-4 w-4 text-amber-600 dark:text-amber-400" aria-hidden="true" />
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                  <div className="rounded-lg border border-amber-200/80 bg-background/70 px-3 py-2 dark:border-amber-900/50">
                    <p className="text-[11px] text-muted-foreground">Có nhãn</p>
                    <p className="text-2xl font-semibold">{trainingPreview?.constructiveness.included ?? 0}</p>
                  </div>
                  <div className="rounded-lg border border-dashed border-muted-foreground/30 bg-muted/30 px-3 py-2">
                    <p className="text-[11px] text-muted-foreground">Ẩn/chưa có nhãn</p>
                    <p className="text-2xl font-semibold text-muted-foreground">{trainingPreview?.constructiveness.masked ?? 0}</p>
                  </div>
                </div>
              </div>
            </div>
            <div
              className="space-y-1.5 overflow-auto pr-1"
              style={{ height: trainingPreviewListHeight }}
              aria-label="Danh sách Training Preview có thể thay đổi chiều cao"
            >
              <AnimatePresence initial={false}>
                {visibleTrainingPreviewItems.map((item, index) => {
                  const suggestion = geminiSuggestions[item.id];
                  const exportEligible =
                    item.gate_bucket === "accepted" && (item.pseudo_label === 0 || item.pseudo_label === 1);
                  const trainingSelection = getTrainingSelectionPresentation(item.selected_for_training, exportEligible);
                  const lockPresentation = getLockPresentation(item.is_locked);
                  const finetuneStatus = trainingPlan?.row_statuses[String(item.id)];
                  const requiresHumanReview = Boolean(item.requires_human_review);
                  return (
                    <motion.div
                      key={`preview-${item.id}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.18, delay: Math.min(index * 0.015, 0.12) }}
                      className={`cursor-pointer rounded-md border p-2.5 transition-colors hover:bg-muted/30 ${requiresHumanReview ? "border-amber-500/70 bg-amber-50/50 dark:bg-amber-950/15" : "border-border/70 hover:border-primary/35"}`}
                      onClick={(event) => handlePreviewRowToggle(event, item.id)}
                    >
                      <div className="flex flex-wrap items-start justify-between gap-2">
                        <Checkbox
                          checked={selectedPreviewIds.includes(item.id)}
                          onCheckedChange={() => togglePreviewSelection(item.id)}
                          aria-label="Chọn tạm thời hàng này để thao tác; không thay đổi selection training trong DB"
                        />
                        <div className="min-w-0 flex-1 space-y-2">
                          <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">{item.text}</p>
                          {requiresHumanReview && (
                            <div className="rounded-md border border-amber-500/50 bg-amber-100/60 p-2 text-xs dark:bg-amber-950/30">
                              <div className="flex flex-wrap items-center gap-2 font-medium text-amber-900 dark:text-amber-200">
                                <AlertTriangle className="h-4 w-4" />
                                {item.review_reason === "model_conflict" ? "Model Conflict" : "Uncertain Re-evaluation"}
                                <Badge variant="destructive">Human review required</Badge>
                              </div>
                              <p className="mt-1">Temporarily excluded from the next fine-tuning run.</p>
                              <Button
                                className="mt-2"
                                size="sm"
                                variant="outline"
                                onClick={() => document.getElementById("mlflow-manual-verify")?.scrollIntoView({ behavior: "smooth" })}
                              >
                                Review in Manual Verify
                              </Button>
                            </div>
                          )}
                          <PredictionEvidence item={item} />
                          <div className="flex flex-wrap gap-1 text-xs">
                            <MlflowBadge presentation={getToxicityPresentation(item.pseudo_label)} />
                            <MlflowBadge
                              presentation={getScorePresentation(item.score, DEFAULT_MLFLOW_GATE_THRESHOLDS)}
                            />
                            <MlflowBadge presentation={getConstructivenessPresentation(item.constructiveness_label)} />
                            <MlflowBadge presentation={lockPresentation} />
                            <MlflowBadge presentation={getReviewStatusPresentation(item.training_review_status)} prefix="Review" />
                            <MlflowBadge presentation={getDataSourcePresentation(item.source_type)} prefix="Nguồn" />
                            {item.label_source === "gemini_assist" && <Badge variant="secondary">Gemini assisted</Badge>}
                            <Badge variant={finetuneStatus?.will_finetune ? "secondary" : "outline"}>
                              {!finetuneStatus
                                ? "Đang tính trạng thái"
                                : finetuneStatus.will_finetune
                                  ? "Sẽ finetune"
                                  : "Không finetune"}
                            </Badge>
                          </div>
                          {finetuneStatus?.reason && <p className="text-xs text-muted-foreground">{finetuneStatus.reason}</p>}
                        </div>
                        <div className="flex shrink-0 items-center gap-1.5" aria-label="Thao tác cho comment này">
                          <div className="flex overflow-hidden rounded-md border shadow-sm" role="group" aria-label="Nhãn độc hại">
                            <Button
                              size="sm"
                              className="rounded-none border-0"
                              variant={item.pseudo_label === 1 ? "destructive" : "ghost"}
                              onClick={() => void handlePreviewToxicity(item.id, 1)}
                            >
                              Độc hại
                            </Button>
                            <Button
                              size="sm"
                              className={`rounded-none border-0 ${item.pseudo_label === 0 ? "bg-emerald-600 text-white hover:bg-emerald-700 hover:text-white dark:bg-emerald-600 dark:hover:bg-emerald-500" : ""}`}
                              variant={item.pseudo_label === 0 ? "default" : "ghost"}
                              onClick={() => void handlePreviewToxicity(item.id, 0)}
                            >
                              Sạch
                            </Button>
                          </div>
                          <IconButtonWithTooltip
                            label={trainingSelection.label}
                            tooltip={trainingSelection.tooltip}
                            size="icon"
                            variant={item.selected_for_training ? "default" : "outline"}
                            disabled={Boolean(item.is_locked) && Boolean(item.selected_for_training)}
                            onClick={() => handlePreviewSelection(item.id, !item.selected_for_training, Boolean(item.is_locked))}
                          >
                            {item.selected_for_training ? <Check className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
                          </IconButtonWithTooltip>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button size="icon" variant="outline" aria-label="Thêm thao tác" title="Thêm thao tác">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end" className="w-60">
                              <DropdownMenuLabel>Thao tác bổ sung</DropdownMenuLabel>
                              <DropdownMenuItem onSelect={() => void handlePreviewLock(item.id, !Boolean(item.is_locked))}>
                                {item.is_locked ? <Unlock /> : <Lock />}
                                {item.is_locked ? "Mở khóa mẫu" : "Khóa mẫu"}
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuLabel>Tính xây dựng</DropdownMenuLabel>
                              <DropdownMenuItem onSelect={() => void handlePreviewConstructiveness(item.id, 1)}>
                                <ThumbsUp className="text-teal-600 dark:text-teal-400" />
                                {item.constructiveness_label === 1 ? "Có tính xây dựng (đang chọn)" : "Có tính xây dựng"}
                              </DropdownMenuItem>
                              <DropdownMenuItem onSelect={() => void handlePreviewConstructiveness(item.id, 0)}>
                                <MessageCircle className="text-amber-600 dark:text-amber-400" />
                                {item.constructiveness_label === 0 ? "Không xây dựng (đang chọn)" : "Không xây dựng"}
                              </DropdownMenuItem>
                              <DropdownMenuItem onSelect={() => void handlePreviewConstructiveness(item.id, null)}>
                                <EyeOff />
                                {item.constructiveness_label == null ? "Ẩn/chưa có nhãn (đang chọn)" : "Ẩn hoặc xóa nhãn"}
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </div>
                      {suggestion && (
                        <div className="mt-2 rounded-md border border-primary/25 bg-primary/5 p-2 text-xs">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <Badge variant="outline">Gemini · {suggestion.model}</Badge>
                            <MlflowBadge presentation={getToxicityPresentation(suggestion.toxicity_label)} prefix="Đề xuất" />
                            <MlflowBadge presentation={getConstructivenessPresentation(suggestion.constructiveness_label)} />
                            <Badge variant="outline">{formatMlflowConfidence(suggestion.confidence)}</Badge>
                            <Badge variant="outline">{formatGeminiAction(suggestion.action)}</Badge>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => void handleApplyGeminiSuggestions([suggestion])}
                              disabled={geminiApplying}
                            >
                              <Sparkles className="h-4 w-4" />
                              Áp dụng đề xuất
                            </Button>
                            <Button size="sm" variant="outline" onClick={() => dismissGeminiSuggestion(suggestion)} disabled={geminiApplying}>
                              Không áp dụng
                            </Button>
                          </div>
                          {suggestion.reason && <p className="mt-1 text-muted-foreground">{suggestion.reason}</p>}
                        </div>
                      )}
                      <details className="mt-2 text-xs text-muted-foreground">
                        <summary className="cursor-pointer select-none">Chi tiết kỹ thuật</summary>
                        <div className="mt-1 flex flex-wrap gap-1">
                          <MlflowBadge presentation={getGateBucketPresentation(item.gate_bucket)} prefix="Bucket" />
                          <MlflowBadge presentation={getVerificationStatusPresentation(item.verification_status)} prefix="Verification" />
                          <Badge variant="outline">pseudo_label={item.pseudo_label ?? "NULL"}</Badge>
                          <MlflowBadge presentation={getScorePresentation(item.score, DEFAULT_MLFLOW_GATE_THRESHOLDS)} />
                          <Badge variant="outline">constructiveness_label={item.constructiveness_label ?? "NULL"}</Badge>
                          <Badge variant="outline">constructiveness_score={item.constructiveness_score?.toFixed(3) ?? "NULL"}</Badge>
                          <Badge variant="outline">is_locked={item.is_locked ? 1 : 0}</Badge>
                          <Badge variant="outline">training_review_status={item.training_review_status ?? "NULL"}</Badge>
                          {item.review_model_name && <Badge variant="outline">review_model={item.review_model_name}</Badge>}
                          <Badge variant="outline">source_type={item.source_type ?? "crawl"}</Badge>
                          {item.source_row_id != null && <Badge variant="outline">source_row_id={item.source_row_id}</Badge>}
                        </div>
                      </details>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
              {(!trainingPreview || trainingPreview.items.length === 0) && (
                <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  No training preview rows yet.
                </p>
              )}
            </div>
            <div
              role="separator"
              aria-label="Kéo để thay đổi chiều cao danh sách Training Preview"
              aria-orientation="horizontal"
              aria-valuemin={240}
              aria-valuemax={960}
              aria-valuenow={trainingPreviewListHeight}
              tabIndex={0}
              className="group flex h-7 touch-none cursor-row-resize select-none items-center justify-center rounded-md border border-dashed text-muted-foreground transition-colors hover:border-primary/40 hover:bg-muted/40 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onPointerDown={handleTrainingPreviewResizeStart}
              onPointerMove={handleTrainingPreviewResizeMove}
              onPointerUp={handleTrainingPreviewResizeEnd}
              onPointerCancel={handleTrainingPreviewResizeEnd}
              onKeyDown={(event) => {
                if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
                event.preventDefault();
                setTrainingPreviewListHeight((height) =>
                  clampTrainingPreviewHeight(height + (event.key === "ArrowDown" ? 48 : -48)),
                );
              }}
            >
              <GripHorizontal className="h-4 w-4" />
              <span className="ml-2 text-xs">Kéo để xem thêm hoặc thu gọn danh sách</span>
            </div>
          </Card>

          <Card id="mlflow-manual-verify" className="p-4 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5">
                <h3 className="font-medium">Manual Verify (DB persisted pool)</h3>
                {(trainingPreview?.counts.model_conflicts ?? 0) > 0 && (
                  <Badge variant="destructive">Model Conflicts: {trainingPreview?.counts.model_conflicts}</Badge>
                )}
                <SectionInfoTooltip label="Thông tin chi tiết Manual Verify">
                  <p>
                    Danh sách chỉ hiển thị candidate/unverified chưa qua gate; comment accepted trong Training Preview không xuất hiện lại ở đây.
                    Checkbox chỉ chọn hàng tạm thời để thao tác; Toxic, Clean và Remove mới cập nhật trực tiếp trạng thái DB trước export/retrain.
                  </p>
                </SectionInfoTooltip>
              </div>
              <div className="text-sm text-muted-foreground">
                {candidateTotal} items · page {candidatePage} · size {candidatePageSize}
              </div>
            </div>

            {(trainingPreview?.counts.requires_human_review ?? 0) > 0 && (
              <div className="flex items-center gap-2 rounded-md border border-amber-500/50 bg-amber-50 p-3 text-sm dark:bg-amber-950/20">
                <AlertTriangle className="h-5 w-5 text-amber-600" />
                <span><b>{trainingPreview?.counts.requires_human_review}</b> samples require human review after cross-model re-evaluation.</span>
              </div>
            )}

            <div className="flex flex-wrap gap-2">
              <select
                aria-label="Model dùng để re-evaluate Manual Verify sample"
                className="rounded-md border bg-background px-3 py-2 text-sm"
                value={reEvaluationModel}
                onChange={(event) => setReEvaluationModel(event.target.value)}
                disabled={modelReEvaluating}
              >
                {availableModels.filter((model) => !isDeprecatedModel(model)).map((model) => (
                  <option key={`manual-reevaluate-${model}`} value={model}>{getModelLabel(model)}</option>
                ))}
              </select>
              <Button size="sm" variant="outline" onClick={handleSelectAllCandidates} disabled={candidates.length === 0}>
                Chọn tạm thời tất cả
              </Button>
              <Button size="sm" variant="outline" onClick={handleUnselectAllCandidates} disabled={selectedCandidateIds.length === 0}>
                Bỏ chọn tạm thời
              </Button>
              <Button size="sm" variant="outline" onClick={() => void handleBulkLock(true)} disabled={selectedCandidateIds.length === 0}>
                Lock selected
              </Button>
              <Button size="sm" variant="outline" onClick={() => void handleBulkLock(false)} disabled={selectedCandidateIds.length === 0}>
                Unlock selected
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => void handleGeminiReviewCandidates()}
                disabled={candidateGeminiReviewing || selectedCandidateIds.length === 0}
              >
                <Sparkles className="mr-1 h-4 w-4" />
                {candidateGeminiReviewing ? "Đang review..." : `Gemini review (${selectedCandidateIds.length})`}
              </Button>
              {availableCandidateGeminiSuggestions.length > 0 && (
                <Button
                  size="sm"
                  onClick={() => void handleApplyCandidateGeminiSuggestions(availableCandidateGeminiSuggestions)}
                  disabled={candidateGeminiApplying || candidateGeminiReviewing}
                >
                  {candidateGeminiApplying
                    ? "Đang áp dụng..."
                    : `Áp dụng Gemini (${availableCandidateGeminiSuggestions.length})`}
                </Button>
              )}
            </div>

            <div className="space-y-1.5 max-h-[34rem] overflow-auto pr-1">
              {candidates.map((item) => {
                const suggestion = candidateGeminiSuggestions[item.id];
                const isSelected = selectedCandidateIds.includes(item.id);
                return (
                  <div
                    key={item.id}
                    className={`flex cursor-pointer items-start gap-2 rounded-md border p-2.5 transition-colors ${
                      isSelected
                        ? "border-primary/60 bg-primary/5"
                        : "border-border/70 hover:border-primary/35 hover:bg-muted/30"
                    }`}
                    onClick={(event) => handleCandidateRowToggle(event, item.id)}
                  >
                    <div data-row-interactive onClick={(event) => event.stopPropagation()}>
                      <Checkbox
                        checked={selectedCandidateIds.includes(item.id)}
                        onCheckedChange={() => toggleCandidate(item.id)}
                        aria-label="Chọn tạm thời hàng này để thao tác Manual Verify"
                      />
                    </div>
                    <div className="min-w-0 flex-1 space-y-3">
                      <p className="whitespace-pre-wrap break-words py-1 text-[15px] font-medium leading-6 text-foreground">{item.text}</p>
                      {item.requires_human_review && (
                        <div data-row-interactive onClick={(event) => event.stopPropagation()}>
                          <HumanReviewEvidence item={item} embedded />
                        </div>
                      )}
                      {item.latest_prediction ? (
                        <div data-row-interactive onClick={(event) => event.stopPropagation()}>
                          <PredictionEvidence item={item} compact />
                        </div>
                      ) : (
                        <p className="text-xs text-muted-foreground">No model prediction history is available.</p>
                      )}
                      <div data-row-interactive onClick={(event) => event.stopPropagation()}>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => void handleModelReEvaluation("selected", [item.id])}
                          disabled={modelReEvaluating || !reEvaluationModel}
                        >
                          <RefreshCw className={`h-4 w-4 ${modelReEvaluating ? "animate-spin" : ""}`} />
                          Re-evaluate with Model
                        </Button>
                      </div>
                      <details data-row-interactive className="text-xs text-muted-foreground" onClick={(event) => event.stopPropagation()}>
                        <summary className="cursor-pointer">Secondary metadata</summary>
                        <div className="mt-2 space-y-1 rounded-md bg-muted/35 p-2">
                          <p className="break-all">{item.url}</p>
                          <p>Domain: {resolveDomainTag(item)}</p>
                          {item.label_confidence && <p>Gate confidence: {formatMlflowConfidence(item.label_confidence)}</p>}
                          {item.label_source && <p>Gate source: {item.label_source === "auto_gate" ? "Automatic gate" : item.label_source}</p>}
                          {item.is_locked && <p>Locked</p>}
                        </div>
                      </details>
                      {suggestion && (
                        <div data-row-interactive className="mt-2 space-y-2 rounded-md border border-primary/25 bg-primary/5 p-2 text-xs" onClick={(event) => event.stopPropagation()}>
                          <div className="flex flex-wrap items-center gap-1.5">
                            <Badge variant="outline">Gemini · {suggestion.model}</Badge>
                            <MlflowBadge presentation={getToxicityPresentation(suggestion.toxicity_label)} />
                            <MlflowBadge presentation={getConstructivenessPresentation(suggestion.constructiveness_label)} />
                            <Badge variant="outline">{formatMlflowConfidence(suggestion.confidence)}</Badge>
                            <Badge variant="outline">{formatGeminiAction(suggestion.action)}</Badge>
                          </div>
                          {suggestion.reason && <p className="text-muted-foreground">{suggestion.reason}</p>}
                          <Button
                            size="sm"
                            onClick={() => void handleApplyCandidateGeminiSuggestions([suggestion])}
                            disabled={candidateGeminiApplying}
                          >
                            Áp dụng dòng này
                          </Button>
                          <Button size="sm" variant="outline" onClick={() => dismissCandidateGeminiSuggestion(suggestion)} disabled={candidateGeminiApplying}>
                            Không áp dụng
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              {candidates.length === 0 && (
                <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  Không có item để verify trong DB hiện tại.
                </p>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-2 rounded-md border bg-muted/20 p-2">
              <span className="px-1 text-xs font-medium text-muted-foreground" aria-live="polite">
                {selectedCandidateIds.length} selected
              </span>
              <Button disabled={selectedCandidateIds.length === 0} onClick={() => void handleBulkReview("include_toxic")}>
                Toxic
              </Button>
              <Button disabled={selectedCandidateIds.length === 0} variant="secondary" onClick={() => void handleBulkReview("include_clean")}>
                Clean
              </Button>
              <Button
                disabled={selectedCandidateIds.length === 0}
                variant="destructive"
                onClick={() => void handleBulkReview("drop")}
              >
                Remove
              </Button>
              <Button size="icon" variant="outline" onClick={() => refreshCandidates(undefined, candidatePage, "all_batches")}>
                <RotateCcw />
              </Button>
            </div>
          </Card>

          <Card className="p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-medium">Review history (persisted in DB)</h3>
              <div className="flex items-center gap-2 text-sm">
                <Button
                  size="sm"
                  variant={reviewHistoryOpen ? "secondary" : "outline"}
                  onClick={() => setReviewHistoryOpen((open) => !open)}
                  aria-expanded={reviewHistoryOpen}
                >
                  {reviewHistoryOpen ? "Hide history" : "Show history"}
                </Button>
                {reviewHistoryOpen && <>
                <span className="text-muted-foreground">Filter</span>
                <select
                  className="rounded-md border bg-background px-3 py-2 text-sm"
                  value={historyDecision}
                  onChange={(e) => setHistoryDecision(e.target.value as "all" | "accepted" | "rejected" | "discarded")}
                >
                  <option value="all">All</option>
                  <option value="accepted">Accepted</option>
                  <option value="rejected">Rejected</option>
                  <option value="discarded">Discarded</option>
                </select>
                <Button size="sm" variant="outline" onClick={() => refreshReviewHistory(undefined, historyDecision, reviewHistoryPage, "all_batches")}>
                  Refresh history
                </Button>
                </>}
              </div>
            </div>
            {reviewHistoryOpen && <>
            <p className="text-xs text-muted-foreground">
              Total: <b>{reviewHistoryTotal}</b> · page <b>{reviewHistoryPage}</b>
            </p>
            <div className="space-y-1.5 max-h-72 overflow-auto pr-1">
              {reviewHistory.map((item) => {
                const hasPredictionEvidence = Boolean(
                  item.latest_prediction || (item.prediction_history?.length ?? 0) > 0,
                );
                return (
                  <div key={`history-${item.id}`} className="rounded-md border border-border/70 p-2.5 transition-colors hover:border-border hover:bg-muted/20">
                    <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">{item.text}</p>
                    <div className="mt-2 space-y-2">
                      <PredictionEvidence item={item} />
                      <HumanReviewEvidence item={item} />
                    </div>
                    <div className="mt-1 flex flex-wrap gap-1.5 text-xs text-muted-foreground">
                      <MlflowBadge presentation={getVerificationStatusPresentation(item.verification_status)} />
                      <MlflowBadge presentation={getGateBucketPresentation(item.gate_bucket)} />
                      <Badge variant="outline">domain={resolveDomainTag(item)}</Badge>
                      {!hasPredictionEvidence && (
                        <>
                          <MlflowBadge presentation={getScorePresentation(item.score, DEFAULT_MLFLOW_GATE_THRESHOLDS)} />
                          <MlflowBadge presentation={getToxicityPresentation(item.pseudo_label)} />
                        </>
                      )}
                      <Badge variant="outline">source={item.label_source ?? "-"}</Badge>
                      <Badge variant="outline">conf={item.label_confidence ?? "-"}</Badge>
                    </div>
                  </div>
                );
              })}
              {reviewHistory.length === 0 && (
                <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  Chưa có history cho filter hiện tại.
                </p>
              )}
            </div>
            </>}
          </Card>
        </TabsContent>

        <TabsContent value="step4" className="space-y-4">
          <Card className="p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h3 className="font-medium">Automation</h3>
                {latestAutomationFamily ? (
                  <p className="mt-1 text-xs text-muted-foreground">
                    {latestAutomationFamily.policy.enabled ? "Enabled" : "Blocked: global automation disabled"} · {latestAutomationFamily.model_family === "tfidf_lr" ? "TF-IDF + LR" : "PhoBERT"} · {latestAutomationFamily.policy.mode}
                  </p>
                ) : automationStatusError ? <div className="mt-1 flex items-center gap-2 text-xs text-destructive"><span>Unable to load automation status</span><Button size="sm" variant="outline" onClick={() => void refreshAutomationStatus()}>Retry</Button></div> : <p className="mt-1 text-xs text-muted-foreground">Loading automation state…</p>}
              </div>
              {latestAutomationFamily && <Badge variant={latestAutomationFamily.ready ? "secondary" : "outline"}>{latestAutomationFamily.ready ? "Ready to trigger" : latestAutomationFamily.blocked_reason || "Blocked"}</Badge>}
            </div>
            {latestAutomationFamily && (
              <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
                <span>New eligible: <b>{latestAutomationFamily.new_eligible_rows} / {latestAutomationFamily.policy.min_new_rows}</b></span>
                <span>Cooldown: <b>{latestAutomationFamily.policy.cooldown_minutes ? `${latestAutomationFamily.policy.cooldown_minutes} min` : "Ready"}</b></span>
                <span>Dry run: <b>{latestAutomationFamily.policy.dry_run ? "On" : "Off"}</b></span>
              </div>
            )}
            {latestAutomationEvent && (
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-md border bg-muted/20 p-2 text-xs">
                <span><b>Latest automation</b> · {latestAutomationEvent.model_family === "tfidf_lr" ? "TF-IDF + LR" : "PhoBERT"} · {latestAutomationEvent.status} · {formatIsoTs(latestAutomationEvent.created_at)}</span>
                {latestAutomationEvent.source_run_id ? <Button size="sm" variant="outline" onClick={() => void openDORun(latestAutomationEvent.source_run_id!)}>{latestAutomationEvent.status === "dry_run" ? "View details" : "View run"}</Button> : <span className="text-muted-foreground">{latestAutomationEvent.detail || "No Kaggle run was created."}</span>}
              </div>
            )}
          </Card>
          <Card className="p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="font-medium">Huấn luyện trên Google Kaggle</h3>
                  <Badge variant={doBadgeVariant as "default" | "secondary" | "destructive" | "outline"}>{doStatusValue}</Badge>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">Tạo bundle mới, chạy train qua Kaggle API và lưu bằng chứng cho từng run.</p>
              </div>
              <div className="flex items-center gap-2">
                <IconButtonWithTooltip label="Tải lại trạng thái Kaggle" size="icon" variant="outline" onClick={handleRefreshDOStatus}>
                  <RotateCcw className="h-4 w-4" />
                </IconButtonWithTooltip>
                <Button
                  variant="secondary"
                  onClick={handleDownloadKaggleArtifact}
                  disabled={doStatusValue !== "completed" || !doHasRealArtifact || !doArtifactDownloadUrl}
                >
                  Download exported model
                </Button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button size="icon" variant="outline" aria-label="Thêm thao tác Kaggle" title="Thêm thao tác Kaggle">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      onSelect={() => {
                        clearDOSession();
                        setStatusText("Đã clear Kaggle session hiện tại. Sẵn sàng trigger run mới.");
                      }}
                    >
                      Clear session
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>

            <div className="rounded-xl border bg-muted/20 p-4 space-y-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-medium">Cấu hình run</p>
                  <p className="text-xs text-muted-foreground">Chỉ chọn model, chế độ train và chính sách dữ liệu trước khi kích hoạt.</p>
                </div>
                <Badge variant="outline">Kaggle API</Badge>
              </div>

              <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(20rem,0.9fr)]">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground">MODEL</p>
                    <div className="flex overflow-hidden rounded-md border shadow-sm" role="group" aria-label="Loại model">
                      <Button
                        type="button"
                        className="rounded-none border-0"
                        variant={selectedModelKind === "phobert" ? "default" : "ghost"}
                        onClick={() => setSelectedModelKind("phobert")}
                      >
                        PhoBERT
                      </Button>
                      <Button
                        type="button"
                        className="rounded-none border-0"
                        variant={selectedModelKind === "lr_smoke" ? "default" : "ghost"}
                        onClick={() => {
                          setSelectedModelKind("lr_smoke");
                          setSelectedTrainingMode("retrain");
                        }}
                      >
                        TF-IDF + LR
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground">CHẾ ĐỘ</p>
                    <div className="flex overflow-hidden rounded-md border shadow-sm" role="group" aria-label="Chế độ huấn luyện">
                      <Button
                        type="button"
                        className="rounded-none border-0"
                        variant={selectedTrainingMode === "retrain" ? "default" : "ghost"}
                        onClick={() => setSelectedTrainingMode("retrain")}
                      >
                        Retrain
                      </Button>
                      <Button
                        type="button"
                        className="rounded-none border-0"
                        variant={selectedTrainingMode === "finetune" ? "default" : "ghost"}
                        onClick={() => setSelectedTrainingMode("finetune")}
                        disabled={selectedModelKind === "lr_smoke"}
                      >
                        Finetune
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2 sm:col-span-2">
                    <p className="text-xs font-medium text-muted-foreground">DỮ LIỆU</p>
                    <div className="flex overflow-hidden rounded-md border shadow-sm" role="group" aria-label="Chính sách dữ liệu">
                      <Button
                        type="button"
                        className="rounded-none border-0"
                        variant={balanceStrategy === "balanced_50_50" ? "default" : "ghost"}
                        onClick={() => setBalanceStrategy("balanced_50_50")}
                      >
                        Cân bằng 50 / 50
                      </Button>
                      <Button
                        type="button"
                        className="rounded-none border-0"
                        variant={balanceStrategy === "all" ? "default" : "ghost"}
                        onClick={() => setBalanceStrategy("all")}
                      >
                        Dùng toàn bộ approved
                      </Button>
                    </div>
                  </div>
                </div>

                <div className="rounded-lg border bg-background/80 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium">Dataset sẽ đưa vào run</p>
                      <p className="text-xs text-muted-foreground">Bundle được tạo mới khi bắt đầu chạy.</p>
                    </div>
                    <Badge variant={bundleReady ? "secondary" : "outline"}>{bundleReady ? "Ready" : "Chưa đủ"}</Badge>
                  </div>
                  <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                    <div className="rounded-md bg-muted/60 px-2 py-2">
                      <p className="text-[11px] text-muted-foreground">MLflow thêm</p>
                      <p className="text-lg font-semibold">{trainingPlan?.summary.mlflow_added ?? "-"}</p>
                    </div>
                    <div className="rounded-md bg-muted/60 px-2 py-2">
                      <p className="text-[11px] text-muted-foreground">Sau cân bằng</p>
                      <p className="text-lg font-semibold">{trainingPlan?.summary.after_balance ?? "-"}</p>
                    </div>
                    <div className="rounded-md bg-muted/60 px-2 py-2">
                      <p className="text-[11px] text-muted-foreground">Tổng train</p>
                      <p className="text-lg font-semibold">{trainingPlan?.summary.final_train ?? "-"}</p>
                    </div>
                  </div>
                  <details className="mt-3 text-xs text-muted-foreground">
                    <summary className="cursor-pointer select-none hover:text-foreground">Xem chi tiết snapshot</summary>
                    <div className="mt-2 grid gap-2 sm:grid-cols-2">
                      <span>Gold train: <b className="text-foreground">{trainingPlan?.summary.gold_train ?? "-"}</b></span>
                      <span>MLflow đủ điều kiện: <b className="text-foreground">{trainingPlan?.summary.eligible_mlflow ?? "-"}</b></span>
                      <span>Trùng loại: <b className="text-foreground">{trainingPlan?.summary.duplicates_skipped ?? "-"}</b></span>
                      <span>Validation/Test gold: <b className="text-foreground">{trainingPlan?.summary.gold_validation ?? "-"}/{trainingPlan?.summary.gold_test ?? "-"}</b></span>
                    </div>
                  </details>
                </div>
              </div>

              {selectedModelKind === "phobert" && selectedTrainingMode === "finetune" && (
                <div>
                  <label className="text-xs text-muted-foreground">Base model (required for finetune)</label>
                  <Input
                    value={finetuneBaseModel}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setFinetuneBaseModel(e.target.value)}
                    placeholder="Select an installed PhoBERT model artifact"
                    className="mt-1"
                    list="finetune-base-models"
                  />
                  <datalist id="finetune-base-models">
                    {availableModels
                      .filter((model) => !isDeprecatedModel(model))
                      .map((model) => (
                        <option key={`base-${model}`} value={model}>
                          {getModelLabel(model)}
                        </option>
                      ))}
                  </datalist>
                  <p className="mt-1 text-xs text-muted-foreground">Finetune bundles this exact PhoBERT checkpoint. An empty or incomplete base model is rejected; it never falls back to the original pretrained checkpoint.</p>
                </div>
              )}

              <div className="flex flex-wrap items-center justify-between gap-2 border-t pt-3">
                <p className="text-xs text-muted-foreground">Bundle path, checksum và bằng chứng dataset được lưu cùng run.</p>
                <Button
                  onClick={handleTriggerDO}
                  disabled={
                    kaggleTriggerPending || loading || doHasActiveRun || doPreflight?.ready === false || !trainingPlan
                  }
                >
                  {kaggleTriggerPending
                    ? "Đang tạo bundle & kích hoạt..."
                    : doHasActiveRun
                      ? "Kaggle đang chạy"
                      : "Tạo bundle & kích hoạt Kaggle"}
                </Button>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-4">
              <div className="rounded-xl border bg-background p-3 md:col-span-2">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Run hiện tại</p>
                  <div className="flex items-center gap-1">
                    {doStatus?.trigger_source && <Badge variant="outline">{doStatus.trigger_source === "automation" ? "Automation" : "Manual"}</Badge>}
                    <Badge variant={doBadgeVariant as "default" | "secondary" | "destructive" | "outline"}>{doStatusValue}</Badge>
                  </div>
                </div>
                <p className="mt-2 break-all text-sm font-medium">{doRunId}</p>
                <div className="mt-3 flex items-center gap-3">
                  <Progress value={doProgress} className="h-2 flex-1" />
                  <span className="text-sm font-semibold tabular-nums">{doProgress}%</span>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">{doCurrentStage ? doStageLabels[doCurrentStage] || doCurrentStage : "Chưa có run đang hoạt động"}</p>
              </div>
              <div className="rounded-xl border bg-background p-3">
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Cấu hình của run</p>
                <p className="mt-2 text-sm font-medium">{doModelLabel}</p>
                <p className="mt-1 text-xs text-muted-foreground">Training operation: {doTrainingMode}</p>
                {doAutomationMode && <p className="mt-1 text-xs text-muted-foreground">Automation mode: {doAutomationMode}</p>}
                <p className="mt-1 text-xs text-muted-foreground">Execution: {doStatusValue}</p>
              </div>
              <div className="rounded-xl border bg-background p-3">
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Artifact</p>
                <Badge className="mt-2" variant={doIsMockArtifact ? "destructive" : doHasRealArtifact ? "secondary" : "outline"}>
                  {doIsDryRun ? "NOT APPLICABLE" : doIsMockArtifact ? "PLACEHOLDER" : doHasRealArtifact ? "SẴN SÀNG" : "CHƯA CÓ"}
                </Badge>
                <p className="mt-2 text-xs text-muted-foreground">
                  {doIsDryRun
                    ? "Kaggle submission skipped; no model artifact is expected."
                    : doHasEvidenceDuration ? `${doEvidenceDurationSeconds.toFixed(2)} giây train` : hasDoTrainDuration ? `${doTrainDuration.toFixed(2)} phút train` : "Chưa có thời lượng"}
                </p>
              </div>
            </div>

            {doStatusValue === "completed" && (
              <Card className="p-4 border-primary/20 bg-primary/5 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="font-medium">Kaggle metrics</h3>
                  <Badge variant={doHasRealArtifact ? "secondary" : "outline"}>{doHasRealArtifact ? "exported model ready" : "no real artifact"}</Badge>
                </div>
                <div className="grid gap-3 md:grid-cols-5">
                  <div>
                    <p className="text-xs text-muted-foreground">f1_toxic</p>
                    <p className="text-xl font-semibold">{formatMetric(doMetrics?.f1_toxic)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">macro_f1</p>
                    <p className="text-xl font-semibold">{formatMetric(doMetrics?.macro_f1)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">accuracy</p>
                    <p className="text-xl font-semibold">{formatMetric(doMetrics?.accuracy)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">precision</p>
                    <p className="text-xl font-semibold">{formatMetric(doMetrics?.precision)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">recall</p>
                    <p className="text-xl font-semibold">{formatMetric(doMetrics?.recall)}</p>
                  </div>
                </div>
                {doPreviousMetrics && doPreviousRun ? (
                  <div className="space-y-3 rounded-md border bg-background p-3">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <p className="text-sm font-medium">So sánh với lần train hoàn tất trước</p>
                        <p className="text-xs text-muted-foreground">
                          Hiện tại <b>{doRunId}</b> so với <b>{doPreviousRun.run_id}</b> · {formatIsoTs(doPreviousRun.created_at)}
                        </p>
                      </div>
                      <Badge variant={doF1Deltas.length === 2 && doF1Deltas.every((value) => value >= 0) ? "secondary" : "outline"}>
                        {doComparisonSummary}
                      </Badge>
                    </div>

                    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
                      {doRunComparisonData.map((item) => {
                        const arrow = typeof item.delta !== "number" ? "" : item.delta > 0 ? "↑" : item.delta < 0 ? "↓" : "→";
                        return (
                          <div key={item.metric} className="rounded-md border p-3">
                            <p className="text-xs text-muted-foreground">{item.label}</p>
                            <div className="mt-1 flex items-end justify-between gap-2">
                              <p className="text-lg font-semibold">{formatMetric(item.current)}</p>
                              <p className={`text-sm font-semibold ${metricDeltaClass(item.delta)}`}>
                                {arrow} {formatMetricDelta(item.delta)}
                              </p>
                            </div>
                            <p className="mt-1 text-[11px] text-muted-foreground">Lần trước: {formatMetric(item.previous)}</p>
                          </div>
                        );
                      })}
                    </div>

                    <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_260px]">
                      <div className="h-64 rounded-md border p-3">
                        <p className="mb-2 text-sm font-medium">Metric test: hiện tại và lần trước</p>
                        <ResponsiveContainer width="100%" height="88%">
                          <BarChart data={doRunComparisonData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                            <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                            <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                            <RechartTooltip formatter={(value) => formatMetric(Number(value))} />
                            <Legend />
                            <Bar dataKey="previous" name="Lần trước" fill="#94a3b8" radius={[3, 3, 0, 0]} />
                            <Bar dataKey="current" name="Hiện tại" fill="#2563eb" radius={[3, 3, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="space-y-2 rounded-md border p-3">
                        <p className="text-sm font-medium">Thay đổi dữ liệu train</p>
                        <div className="rounded border p-2">
                          <p className="text-xs text-muted-foreground">Train thực dùng</p>
                          <p className="font-semibold">
                            {doPreviousEvidence?.used_train ?? "-"} → {doEvidence?.used_train ?? "-"}
                          </p>
                        </div>
                        <div className="rounded border p-2">
                          <p className="text-xs text-muted-foreground">Mẫu MLflow mới</p>
                          <p className="font-semibold">
                            {doPreviousEvidence?.included_mlflow_count ?? "-"} → {doEvidence?.included_mlflow_count ?? "-"}
                          </p>
                        </div>
                        <div className={`rounded border p-2 text-xs ${doComparableTestSet ? "bg-emerald-500/10" : "bg-amber-500/10"}`}>
                          Test set: {Number.isFinite(doPreviousTestSize) ? doPreviousTestSize : "-"} → {Number.isFinite(doCurrentTestSize) ? doCurrentTestSize : "-"}
                          <br />
                          {doComparableTestSet ? "Cùng kích thước test; có thể đối chiếu trực tiếp hơn." : "Test set khác kích thước; chỉ nên xem như tham khảo."}
                        </div>
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Delta chỉ thể hiện chênh lệch số học. Precision và recall có thể đánh đổi lẫn nhau; phần này không tự động đồng nghĩa với promotion/deployment.
                    </p>
                  </div>
                ) : (
                  doMetrics && (
                    <p className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
                      Chưa có lần train hoàn tất trước đó cùng loại model để so sánh.
                    </p>
                  )
                )}
                {doEvidence && (
                  <div className="space-y-3 rounded-md border bg-background p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <p className="text-sm font-medium">Bằng chứng dataset đã train</p>
                        <p className="text-xs text-muted-foreground">Đối chiếu bundle snapshot với dữ liệu LR Smoke thực sự sử dụng.</p>
                      </div>
                      <Badge variant={doEvidence.included_all_expected_mlflow && doBundleEvidenceVerified ? "secondary" : "destructive"}>
                        {doEvidence.included_all_expected_mlflow && doBundleEvidenceVerified ? "PROVENANCE VERIFIED" : "PROVENANCE MISMATCH"}
                      </Badge>
                    </div>
                    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                      <div className="rounded-md border p-2">
                        <p className="text-xs text-muted-foreground">Bundle train</p>
                        <p className="text-lg font-semibold">{doEvidence.raw_train ?? "-"}</p>
                      </div>
                      <div className="rounded-md border p-2">
                        <p className="text-xs text-muted-foreground">LR train thực dùng</p>
                        <p className="text-lg font-semibold">{doEvidence.used_train ?? "-"}</p>
                      </div>
                      <div className="rounded-md border p-2">
                        <p className="text-xs text-muted-foreground">Gold thực dùng</p>
                        <p className="text-lg font-semibold">{doEvidence.used_gold ?? "-"}</p>
                      </div>
                      <div className="rounded-md border p-2">
                        <p className="text-xs text-muted-foreground">MLflow mới đã dùng</p>
                        <p className="text-lg font-semibold">
                          {doEvidence.included_mlflow_count ?? "-"}/{doEvidence.expected_mlflow_count ?? "-"}
                        </p>
                      </div>
                    </div>
                    <p className="break-all text-xs text-muted-foreground">
                      MLflow IDs: {doEvidence.included_mlflow_ids?.join(", ") || "-"}
                    </p>
                    <p className="break-all text-xs text-muted-foreground">
                      IDs SHA-256: {doEvidence.included_mlflow_ids_sha256 || "-"}
                    </p>
                  </div>
                )}
                {doMetrics && (
                  <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_280px]">
                    <div className="h-64 rounded-md border bg-background p-3">
                      <p className="mb-2 text-sm font-medium">Validation và test metrics</p>
                      <ResponsiveContainer width="100%" height="88%">
                        <BarChart data={doMetricChartData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                          <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                          <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                          <RechartTooltip />
                          <Legend />
                          <Bar dataKey="validation" name="Validation" fill="#2563eb" radius={[3, 3, 0, 0]} />
                          <Bar dataKey="test" name="Test" fill="#16a34a" radius={[3, 3, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="rounded-md border bg-background p-3">
                      <p className="mb-3 text-sm font-medium">Test confusion matrix</p>
                      {doTestConfusion ? (
                        <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-center text-xs">
                          <span />
                          <span className="p-2 text-muted-foreground">Dự đoán Clean</span>
                          <span className="p-2 text-muted-foreground">Dự đoán Toxic</span>
                          <span className="flex items-center p-2 text-left text-muted-foreground">Thật Clean</span>
                          <span className="rounded bg-emerald-500/15 p-3 text-lg font-semibold">{doTestConfusion.tn ?? "-"}</span>
                          <span className="rounded bg-amber-500/15 p-3 text-lg font-semibold">{doTestConfusion.fp ?? "-"}</span>
                          <span className="flex items-center p-2 text-left text-muted-foreground">Thật Toxic</span>
                          <span className="rounded bg-amber-500/15 p-3 text-lg font-semibold">{doTestConfusion.fn ?? "-"}</span>
                          <span className="rounded bg-emerald-500/15 p-3 text-lg font-semibold">{doTestConfusion.tp ?? "-"}</span>
                        </div>
                      ) : (
                        <p className="text-xs text-muted-foreground">Artifact chưa có confusion matrix.</p>
                      )}
                    </div>
                  </div>
                )}
                <p className="text-xs break-all text-muted-foreground">run={doRunId} · checksum={doChecksum || "-"}</p>
                {!doMetrics && <p className="text-xs text-amber-700 dark:text-amber-300">Artifact hoàn tất nhưng chưa tìm thấy metrics.json trong ZIP.</p>}
                {doHasRealArtifact && (
                  <div className="rounded-md border bg-background p-3 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-violet-600" />
                        <div>
                          <p className="text-sm font-medium">Gemini Evaluate</p>
                          <p className="text-xs text-muted-foreground">Nhận định hỗ trợ admin; không thay thế production gate.</p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" onClick={() => void handleGeminiEvaluate(false)} disabled={geminiEvaluating}>
                          {geminiEvaluating ? "Đang đánh giá..." : doStatus?.gemini_evaluation ? "Xem lại" : "Đánh giá kết quả"}
                        </Button>
                        {doStatus?.gemini_evaluation && (
                          <Button variant="ghost" size="sm" onClick={() => void handleGeminiEvaluate(true)} disabled={geminiEvaluating}>
                            Đánh giá lại
                          </Button>
                        )}
                      </div>
                    </div>
                    {doStatus?.gemini_evaluation && (
                      <div className="space-y-2 text-sm">
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge variant={doStatus.gemini_evaluation.evaluation.verdict === "promote" ? "default" : doStatus.gemini_evaluation.evaluation.verdict === "hold" ? "destructive" : "secondary"}>
                            {doStatus.gemini_evaluation.evaluation.verdict === "promote" ? "ĐỀ XUẤT PROMOTE" : doStatus.gemini_evaluation.evaluation.verdict === "hold" ? "NÊN GIỮ LẠI" : "CẦN ADMIN XEM"}
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            Gemini: {doStatus.gemini_evaluation.model || "-"} · so với run {doStatus.gemini_evaluation.previous_run_id || "trước đó chưa có"}
                          </span>
                        </div>
                        <p className="whitespace-pre-wrap break-words leading-relaxed">{doStatus.gemini_evaluation.evaluation.summary}</p>
                        {doStatus.gemini_evaluation.evaluation.recommendation && (
                          <p className="rounded bg-muted/60 p-2 text-xs whitespace-pre-wrap break-words">
                            <span className="font-semibold">Khuyến nghị: </span>{doStatus.gemini_evaluation.evaluation.recommendation}
                          </p>
                        )}
                        <div className="grid gap-2 md:grid-cols-2">
                          {doStatus.gemini_evaluation.evaluation.strengths.length > 0 && (
                            <div className="rounded border border-emerald-500/30 bg-emerald-500/5 p-2 text-xs">
                              <p className="mb-1 font-semibold text-emerald-700 dark:text-emerald-300">Điểm tích cực</p>
                              <ul className="list-disc space-y-1 pl-4">{doStatus.gemini_evaluation.evaluation.strengths.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul>
                            </div>
                          )}
                          {doStatus.gemini_evaluation.evaluation.risks.length > 0 && (
                            <div className="rounded border border-amber-500/30 bg-amber-500/5 p-2 text-xs">
                              <p className="mb-1 font-semibold text-amber-700 dark:text-amber-300">Rủi ro / cần kiểm tra</p>
                              <ul className="list-disc space-y-1 pl-4">{doStatus.gemini_evaluation.evaluation.risks.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul>
                            </div>
                          )}
                        </div>
                        {doStatus.gemini_evaluation.evaluation.metric_observations.length > 0 && (
                          <p className="text-xs text-muted-foreground">
                            <span className="font-semibold text-foreground">Quan sát metric: </span>
                            {doStatus.gemini_evaluation.evaluation.metric_observations.join(" · ")}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </Card>
            )}

            <details className="rounded-xl border bg-muted/15 p-3" open={doPreflight?.ready === false || doStatusValue === "failed"}>
              <summary className="cursor-pointer select-none text-sm font-medium hover:text-primary">
                Thông tin kỹ thuật, preflight và nhật ký
              </summary>
              <div className="mt-3 space-y-3">
            <div className="rounded-md border p-3 bg-background/70 space-y-1">
              <p className="text-sm font-medium">Run provenance</p>
              <p className="text-xs text-muted-foreground">
                {doIsDryRun ? <>Đây là <b>automation simulation record</b>. Execution: <b>Dry run</b> · Kaggle submission: <b>Skipped</b>.</> : <>Đây là <b>{doIsMockRun ? "mock/test run" : "real run"}</b>.</>} Nguồn cập nhật status: <b>{doStatusSource}</b>.
              </p>
              <p className="text-xs text-muted-foreground">
                Created: {formatIsoTs((doStatus?.created_at as string | undefined) || null)} | Updated:{" "}
                {formatIsoTs((doStatus?.updated_at as string | undefined) || null)}
              </p>
            </div>

            {doPreflight && (
              <div className={`rounded-md border p-3 space-y-2 ${doPreflight.ready ? "bg-muted/20" : "border-destructive/40 bg-destructive/5"}`}>
                <div className="flex items-center gap-2">
                  <Badge variant={doPreflight.ready ? "secondary" : "destructive"}>{doPreflight.ready ? "READY" : "NOT READY"}</Badge>
                  <p className="text-xs text-muted-foreground">Preflight checked at: {doPreflight.checked_at || "-"}</p>
                </div>
                {doPreflight.missing.length > 0 && (
                  <p className="text-xs text-destructive">Missing env: {doPreflight.missing.join(", ")}</p>
                )}
                {doPreflight.warnings.length > 0 && (
                  <ul className="list-disc ml-5 text-xs text-muted-foreground space-y-1">
                    {doPreflight.warnings.map((w) => (
                      <li key={w}>{w}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {doIsRestricted && (
              <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 space-y-1">
                <p className="text-xs font-medium text-amber-700 dark:text-amber-300">GPU droplet bị restricted bởi account tier</p>
                <p className="text-xs text-muted-foreground">
                  Bạn có thể chuyển sang <b>CPU mode</b> để chạy tạm cho demo, hoặc mở ticket nâng tier để bật GPU droplet.
                </p>
              </div>
            )}

            {doIsPlaceholder && (
              <div className="rounded-md border border-destructive/40 bg-destructive/5 p-3 space-y-1">
                <p className="text-xs font-medium text-destructive">Kaggle pipeline đang ở placeholder mode</p>
                <p className="text-xs text-muted-foreground">
                  Backend hiện không thực thi tạo droplet thật. Hãy kiểm tra lại backend version đang chạy và restart server.
                </p>
              </div>
            )}

            {doApiCallEvidence && (
              <div className="rounded-md border p-3 bg-muted/20">
                <p className="text-xs text-muted-foreground">API call evidence</p>
                <p className="text-xs font-medium break-all">{doApiCallEvidence}</p>
              </div>
            )}

            <div className="grid gap-2">
              {doStages.map((stage, idx) => {
                const currentIdx = doStages.findIndex((s) => s === doCurrentStage);
                const isDone =
                  doStatusValue === "completed" || doStatusValue === "dry_run" || (currentIdx >= 0 && idx < currentIdx);
                const isRunning = doStatusValue === "running" && stage === doCurrentStage;
                const isFailed = doStatusValue === "failed" && stage === doCurrentStage;
                const variant = isFailed ? "destructive" : isRunning ? "default" : isDone ? "secondary" : "outline";
                const stateText = isFailed ? "FAILED" : isRunning ? "RUNNING" : isDone ? "DONE" : "PENDING";
                return (
                  <div key={stage} className="rounded-md border p-3 text-sm flex items-center justify-between gap-3">
                    <div>
                      <p>{doStageLabels[stage] || stage}</p>
                      <p className="text-xs text-muted-foreground">At: {formatIsoTs(doStageTimestamps[stage] || null)}</p>
                    </div>
                    <Badge variant={variant as "default" | "secondary" | "destructive" | "outline"}>{stateText}</Badge>
                  </div>
                );
              })}
            </div>

            <div className="rounded-md border p-3 text-sm space-y-2">
              <p className="font-medium">Artifact</p>
              {doIsDryRun ? (
                <p className="text-xs text-muted-foreground">Not applicable for dry run. Kaggle submission was skipped, so no trained model artifact or checksum is expected.</p>
              ) : doIsMockArtifact && (
                <p className="text-xs text-amber-700 dark:text-amber-300">
                  Artifact này là placeholder từ mock webhook, không phải model ZIP thật từ Kaggle.
                </p>
              )}
              {!doIsDryRun && <>
                <p className="text-xs break-all">URI: {doArtifactUri || "-"}</p>
                <p className="text-xs break-all">Checksum (sha256): {doChecksum || "-"}</p>
              </>}
              {doErrorMessage && <p className="text-xs text-destructive break-all">Error: {doErrorMessage}</p>}
            </div>

            <div className="rounded-md border p-3 text-sm space-y-2 bg-background/70">
              <p className="font-medium">Training log</p>
              <div className="max-h-56 overflow-auto space-y-1">
                {doLogEvents.length === 0 && doLogs.length === 0 ? (
                  <p className="text-xs text-muted-foreground">Chưa có log.</p>
                ) : doLogEvents.length > 0 ? (
                  doLogEvents.map((event, idx) => {
                    const msg = String(event.message || "");
                    const ts = typeof event.ts === "string" ? event.ts : "";
                    const src = typeof event.source === "string" ? event.source : "";
                    return (
                      <p key={`${idx}-${msg.slice(0, 16)}`} className="text-xs text-muted-foreground">
                        [{formatIsoTs(ts || null)}] [{src || "unknown"}] {msg}
                      </p>
                    );
                  })
                ) : (
                  doLogs.map((line, idx) => (
                    <p key={`${idx}-${line.slice(0, 16)}`} className="text-xs text-muted-foreground">
                      {line}
                    </p>
                  ))
                )}
              </div>
              <details>
                <summary className="cursor-pointer text-xs text-muted-foreground">View raw JSON</summary>
                <pre className="text-xs whitespace-pre-wrap mt-2">{JSON.stringify(doStatus || { status: "idle" }, null, 2)}</pre>
              </details>
            </div>
              </div>
            </details>
          </Card>
        </TabsContent>

        <TabsContent value="step5" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card className="p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-medium">So sánh với production cùng model family</h3>
                <Badge variant="outline">{comparePayload?.model_family || "chưa có family"}</Badge>
              </div>
              <div className="grid gap-2 text-sm sm:grid-cols-2">
                <div className="rounded-md border p-3">
                  <p className="text-xs text-muted-foreground">Production hiện tại</p>
                  <p className="break-all font-semibold">{comparePayload?.current?.model || "-"}</p>
                </div>
                <div className="rounded-md border p-3">
                  <p className="text-xs text-muted-foreground">Candidate</p>
                  <p className="break-all font-semibold">{comparePayload?.candidate?.model || "-"}</p>
                  <p className="break-all text-[11px] text-muted-foreground">run={comparePayload?.candidate?.source_run_id || "-"}</p>
                  <p className="break-all text-[11px] text-muted-foreground">feedback snapshot={comparePayload?.candidate?.feedback_snapshot_sha256 || "-"}</p>
                </div>
              </div>
              <div className="space-y-2">
                {productionComparisonData.map((item) => (
                  <div key={item.metric} className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-3 rounded-md border px-3 py-2 text-sm">
                    <span>{item.label}</span>
                    <span className="text-muted-foreground">{formatMetric(item.current)}</span>
                    <span className="font-medium">{formatMetric(item.candidate)}</span>
                    <span className={`font-semibold ${metricDeltaClass(item.delta)}`}>{formatMetricDelta(item.delta)}</span>
                  </div>
                ))}
              </div>
              <div className={`rounded-md border p-3 text-xs ${comparePayload?.test_comparability_verified ? "bg-emerald-500/10" : "bg-amber-500/10"}`}>
                {comparePayload?.test_comparability_verified
                  ? "Đã xác minh cùng semantic test-set fingerprint."
                  : "Chưa xác minh được test-set fingerprint; promotion bị khóa."}
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant="outline" onClick={() => refreshCompare(comparePayload?.candidate?.source_run_id || doRunId || undefined)}>
                  Refresh compare
                </Button>
                <Button
                  variant="secondary"
                  onClick={handleDownloadKaggleArtifact}
                  disabled={doStatusValue !== "completed" || !doHasRealArtifact || !doArtifactDownloadUrl}
                >
                  Download exported model
                </Button>
              </div>
            </Card>

            <Card className="p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-medium">Production gate</h3>
                <Badge variant={comparePayload?.promotion_enabled ? "default" : "secondary"}>
                  {comparePayload?.promotion_enabled ? "READY" : "BLOCKED"}
                </Badge>
              </div>
              <div className="space-y-2">
                {(comparePayload?.gate_checks || []).map((check) => (
                  <div key={check.name} className="flex items-start justify-between gap-3 rounded-md border p-2 text-sm">
                    <span>
                      {check.name}
                      {check.detail && <span className="mt-0.5 block text-[11px] text-muted-foreground">{check.detail}</span>}
                    </span>
                    <Badge variant={check.passed ? "default" : "secondary"}>{check.passed ? "PASS" : "WAIT"}</Badge>
                  </div>
                ))}
              </div>
              <div className="flex flex-wrap gap-2">
                <Dialog open={promotionDialogOpen} onOpenChange={setPromotionDialogOpen}>
                  <DialogTrigger asChild>
                    <Button disabled={!comparePayload?.promotion_enabled}>Promote to {comparePayload?.model_family || "family"} Production</Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Xác nhận promotion</DialogTitle>
                      <DialogDescription>
                        Production pointer của <b>{comparePayload?.model_family}</b> sẽ chuyển từ <b>{comparePayload?.current?.model || "-"}</b> sang <b>{comparePayload?.candidate?.model || "-"}</b>.
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-2 text-xs">
                      <p className="break-all">Run: {comparePayload?.candidate?.source_run_id || "-"}</p>
                      <p className="break-all">SHA-256: {comparePayload?.candidate?.artifact_checksum || "-"}</p>
                    </div>
                    <div className="flex justify-end gap-2">
                      <Button variant="outline" onClick={() => setPromotionDialogOpen(false)}>Hủy</Button>
                      <Button onClick={handlePromote}>Xác nhận promote</Button>
                    </div>
                  </DialogContent>
                </Dialog>
                <Button
                  variant="outline"
                  onClick={handleRollback}
                  disabled={!comparePayload?.current?.rollback_available}
                >
                  <RotateCcw className="h-4 w-4" />
                  Rollback
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Promotion cài artifact theo version bất biến và chỉ đổi production slot của đúng model family. Rollback không xóa artifact.
              </p>
            </Card>
          </div>

          <Card className="p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="font-medium">Model Registry</h3>
                <p className="text-xs text-muted-foreground">Completed Kaggle artifacts are retained as candidates until an explicit lifecycle action.</p>
              </div>
              <Button size="sm" variant="outline" onClick={() => void refreshModelRegistry()}>Refresh registry</Button>
            </div>
            <div className="space-y-2">
              {registryModels.map((model) => (
                <div key={model.model_id} className="flex flex-wrap items-center justify-between gap-3 rounded-md border p-3 text-sm">
                  <div className="min-w-56 flex-1">
                    <p className="break-all font-medium">{model.model_id}</p>
                    <p className="text-xs text-muted-foreground">
                      {model.model_family} · {model.training_mode || "retrain"} · {formatIsoTs(model.created_at || null)}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
                    <span>Macro-F1 {formatMetric(model.metrics?.macro_f1)}</span>
                    <span>F1 Toxic {formatMetric(model.metrics?.f1_toxic)}</span>
                    <span>Accuracy {formatMetric(model.metrics?.accuracy)}</span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant={model.status === "production" ? "default" : model.status === "archived" ? "secondary" : "outline"}>{model.status}</Badge>
                    {!model.artifact_available && <Badge variant="destructive">Artifact unavailable</Badge>}
                    <details className="text-xs text-muted-foreground">
                      <summary className="cursor-pointer">Details</summary>
                      <div className="mt-2 max-w-sm space-y-1 rounded-md bg-muted/35 p-2">
                        <p className="break-all">Run: {model.source_run_id}</p>
                        <p className="break-all">Base: {model.base_model || "-"}</p>
                        <p className="break-all">SHA-256: {model.artifact_checksum || "-"}</p>
                      </div>
                    </details>
                    <Button size="sm" variant="outline" onClick={() => void refreshCompare(model.source_run_id)}>Compare</Button>
                    <Button size="sm" variant="outline" disabled={!model.artifact_available} onClick={() => window.open(buildApiUrl(`/api/mlflow/registry/download?model_id=${encodeURIComponent(model.model_id)}`), "_blank", "noopener,noreferrer")}>Download</Button>
                    {model.status !== "production" && model.status !== "deleted" && (
                      <Button size="sm" variant="outline" onClick={() => void promote(model.source_run_id, model.artifact_checksum).then(() => Promise.all([refreshModelRegistry(), refreshCompare(model.source_run_id)]))}>Promote</Button>
                    )}
                    {model.status === "candidate" && <Button size="sm" variant="outline" onClick={() => void updateModelRegistryLifecycle(model.model_id, "archive")}>Archive</Button>}
                    {model.status !== "production" && model.status !== "deleted" && (
                      <Button size="sm" variant="destructive" onClick={() => { if (window.confirm(`Delete registry entry ${model.model_id}? The original training run and artifact are retained.`)) void updateModelRegistryLifecycle(model.model_id, "delete"); }}>Delete</Button>
                    )}
                  </div>
                </div>
              ))}
              {registryModels.length === 0 && <p className="rounded-md border border-dashed p-3 text-sm text-muted-foreground">No completed Kaggle model artifacts have been registered yet.</p>}
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
