import { useEffect, useMemo, useRef, useState, type ChangeEvent, type ComponentProps, type MouseEvent, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { BarChart3, Check, EyeOff, GripHorizontal, History, Lock, MessageCircle, Plus, RotateCcw, Sparkles, ThumbsUp, Unlock } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip as RechartTooltip, XAxis, YAxis } from "recharts";
import { toast } from "sonner";
import { Card } from "@/app/components/ui/card";
import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { Badge } from "@/app/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { Progress } from "@/app/components/ui/progress";
import { Checkbox } from "@/app/components/ui/checkbox";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/app/components/ui/tooltip";
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
  type MlflowGeminiReviewSuggestion,
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
    <Tooltip>
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
    lastBundlePath,
    doStatus,
    doPreflight,
    ingest,
    refreshOverview,
    refreshCandidates,
    refreshTrainingPreview,
    refreshTrainingPlan,
    reviewTrainingPreview,
    geminiReviewTrainingPreview,
    geminiReviewCandidates,
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
    geminiEvaluateKaggleRun,
    clearDOSession,
    refreshCompare,
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
  const [crawlHistoryOpen, setCrawlHistoryOpen] = useState(false);
  const [importModelName, setImportModelName] = useState("");
  const [importModelZipFile, setImportModelZipFile] = useState<File | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);
  const [includeUnusedInExport, setIncludeUnusedInExport] = useState(false);
  const [unusedScope, setUnusedScope] = useState<MlflowUnusedScope>("all");
  const [historyDecision, setHistoryDecision] = useState<"all" | "accepted" | "rejected" | "discarded">("all");
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
  const kaggleTriggerPendingRef = useRef(false);
  const trainingPreviewResizeRef = useRef<{ pointerId: number; startY: number; startHeight: number } | null>(null);

  useEffect(() => {
    void refreshOverview();
    void refreshCandidates(undefined, 1, "all_batches");
    void refreshThresholdStatus(activeBatchId);
    void refreshTrainingPreview(1, "all_batches");
    void refreshReviewHistory(undefined, historyDecision, 1, "all_batches");
    void refreshCompare();
    void refreshDOPreflight();
  }, []);

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
    void refreshReviewHistory(undefined, historyDecision, 1, "all_batches");
  }, [historyDecision]);

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

  const handleCandidateRowToggle = (event: MouseEvent<HTMLDivElement>, id: number) => {
    const target = event.target as HTMLElement;
    if (target.closest("button, input, textarea, select, option, a")) {
      return;
    }
    toggleCandidate(id);
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
    try {
      const result = await ingest(parsedUrls, selectedModel || undefined);
      const counts = result.counts || {};
      const summary = result.crawl_summary || null;
      const total = Number(counts.total || 0);
      const candidateCount = Number(counts.candidate || 0);
      setStatusText(`Đã ingest batch ${result.batch_id}`);
      setCrawlSummary(summary);
      setSelectedCandidateIds([]);
      void refreshTrainingPreview(1, "all_batches");

      if (total <= 0) {
        toast.warning("Crawl hoàn tất nhưng không tìm thấy comment.");
      } else {
        toast.success(`Ingest thành công: ${total} segments, ${candidateCount} candidates.`);
      }
    } catch {
      setStatusText("Ingest thất bại.");
      toast.error("Ingest thất bại.");
    }
  };

  const handleBulkReview = async (action: "include_toxic" | "include_clean" | "drop") => {
    if (selectedCandidateIds.length === 0) return;
    const selectedItems = candidates.filter((item) => selectedCandidateIds.includes(item.id));
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
      setSelectedCandidateIds([]);
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
    try {
      setCandidateGeminiSuggestions((prev) => {
        const next = { ...prev };
        selectedCandidateIds.forEach((id) => delete next[id]);
        return next;
      });
      const payload = await geminiReviewCandidates(selectedCandidateIds);
      const next = Object.fromEntries(payload.suggestions.map((item) => [item.id, item]));
      setCandidateGeminiSuggestions((prev) => ({ ...prev, ...next }));
      toast.success(`Gemini đã review ${payload.reviewed}/${payload.requested} dòng Manual Verify.`);
    } catch (error) {
      const detail = error instanceof Error && error.message ? error.message : "Gemini review thất bại.";
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

  const handleGeminiReviewPreview = async () => {
    if (selectedPreviewIds.length === 0) {
      toast.warning("Chọn ít nhất 1 dòng preview để Gemini review.");
      return;
    }
    setGeminiReviewing(true);
    try {
      setGeminiSuggestions((prev) => {
        const next = { ...prev };
        selectedPreviewIds.forEach((id) => delete next[id]);
        return next;
      });
      const payload = await geminiReviewTrainingPreview(selectedPreviewIds);
      const next = Object.fromEntries(payload.suggestions.map((item) => [item.id, item]));
      setGeminiSuggestions((prev) => ({ ...prev, ...next }));
      toast.success(`Gemini đã review ${payload.reviewed}/${payload.requested} dòng.`);
    } catch (error) {
      const detail = error instanceof Error && error.message ? error.message : "Gemini review thất bại.";
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
  });

  const clearAppliedGeminiSuggestions = (suggestions: MlflowGeminiReviewSuggestion[]) => {
    const appliedIds = new Set(suggestions.map((suggestion) => suggestion.id));
    setGeminiSuggestions((prev) => Object.fromEntries(Object.entries(prev).filter(([id]) => !appliedIds.has(Number(id)))));
  };

  const handleApplyGeminiSuggestions = async (suggestions: MlflowGeminiReviewSuggestion[]) => {
    if (suggestions.length === 0) return;
    setGeminiApplying(true);
    try {
      await reviewTrainingPreview(suggestions.map(buildGeminiReviewUpdate));
      clearAppliedGeminiSuggestions(suggestions);
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
        `Đã clear MLFlow: do_run=${rows.mlflow_do_run}, artifacts=${rows.mlflow_training_artifact}, items=${rows.mlflow_comment_item}, batches=${rows.mlflow_crawl_batch}.`,
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
      toast.success(`Đã trigger Kaggle run ${payload.run_id} (${trainingLabel}).`);
    } catch (error) {
      const detail = error instanceof Error && error.message ? error.message : "Không rõ nguyên nhân.";
      setStatusText(`Trigger Kaggle pipeline thất bại: ${detail}`);
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
    try {
      const payload = await geminiEvaluateKaggleRun(runId, force);
      setStatusText(payload.status === "cached" ? "Đã tải nhận định Gemini đã lưu." : "Gemini đã đánh giá kết quả train mới.");
      toast.success(payload.status === "cached" ? "Đã tải nhận định Gemini." : "Gemini Evaluate hoàn tất.");
    } catch (error) {
      const detail = error instanceof Error && error.message ? error.message : "Gemini Evaluate thất bại.";
      setStatusText(detail);
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
  const doTrainingMode = ((doStatus?.training_mode as string | undefined) || selectedTrainingMode || "retrain").toLowerCase();
  const doBaseModel = (doStatus?.base_model as string | undefined) || (selectedTrainingMode === "finetune" ? finetuneBaseModel : "");
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

  useEffect(() => {
    const prev = prevDoStatusRef.current;
    if (prev === doStatusValue) return;

    if (doStatusValue === "running") {
      toast.message("Kaggle pipeline đang chạy.");
    } else if (doStatusValue === "completed") {
      toast.success("Kaggle pipeline hoàn tất.");
      if (doRunId) void refreshCompare(doRunId);
    } else if (doStatusValue === "failed") {
      if (doIsRestricted) {
        toast.error("GPU bị restricted. Hãy chuyển CPU hoặc mở ticket tăng tier.");
      } else {
        toast.error("Kaggle pipeline thất bại.");
      }
    }

    prevDoStatusRef.current = doStatusValue;
  }, [doIsRestricted, doRunId, doStatusValue, refreshCompare]);

  const doCompletedIndex = doStages.findIndex((s) => s === doCurrentStage);
  const doHasStageProgress = ["running", "failed", "completed", "dry_run"].includes(doStatusValue);
  const doProgress =
    doStatusValue === "completed"
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
          <div className="flex items-center gap-2">
            <Badge variant={ingestStageMeta.variant}>{ingestStageMeta.label}</Badge>
          </div>
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

      {ingestStage !== "idle" && (
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

      {hasNoBatch && (
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
          <TabsTrigger value="step5">Results & Gate</TabsTrigger>
        </TabsList>

        <TabsContent value="step1" className="space-y-4">
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
              <div>
                <h3 className="font-medium">Training Preview</h3>
                <p className="text-xs text-muted-foreground">
                  Danh sách chỉ hiển thị comment accepted đã qua gate; candidate chưa xác minh chỉ hiển thị trong Manual Verify.
                  Mẫu accepted phải được chọn cho training và có nhãn Độc hại hoặc Sạch hợp lệ mới đủ điều kiện vào accepted export set;
                  balanced export có thể lấy ít hơn. Checkbox đầu hàng chỉ chọn tạm thời
                  cho thao tác trên màn hình.
                </p>
                <p className="text-xs text-muted-foreground">
                  Nhãn tính xây dựng hiển thị khi DB có giá trị 0/1; NULL nghĩa là ẩn hoặc chưa có nhãn, không mặc định là độ tin cậy thấp.
                  Màu Điểm độc hại dùng gate mặc định 0.20/0.80 vì Preview API chưa trả threshold theo từng batch.
                </p>
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
            <div className="grid gap-2 md:grid-cols-4">
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Đã chọn cho training (đủ điều kiện)</p>
                <p className="text-xl font-semibold">{trainingPreview?.counts.selected ?? 0}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Độc hại / Sạch</p>
                <p className="text-xl font-semibold">
                  {trainingPreview?.counts.selected_toxic ?? 0} / {trainingPreview?.counts.selected_clean ?? 0}
                </p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Xuất cân bằng</p>
                <p className="text-xl font-semibold">{trainingPreview?.balance.balanced_count ?? 0}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Tính xây dựng</p>
                <p className="text-xl font-semibold">
                  {trainingPreview?.constructiveness.included ?? 0} có nhãn
                </p>
                <p className="text-xs text-muted-foreground">{trainingPreview?.constructiveness.masked ?? 0} ẩn hoặc chưa có nhãn</p>
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
                  return (
                    <motion.div
                      key={`preview-${item.id}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.18, delay: Math.min(index * 0.015, 0.12) }}
                      className="cursor-pointer rounded-md border border-border/70 p-2.5 transition-colors hover:border-primary/35 hover:bg-muted/30"
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
                        <div className="flex flex-wrap items-center gap-1">
                          <Button
                            size="sm"
                            variant={item.pseudo_label === 1 ? "destructive" : "outline"}
                            onClick={() => void handlePreviewToxicity(item.id, 1)}
                          >
                            Độc hại
                          </Button>
                          <Button
                            size="sm"
                            variant={item.pseudo_label === 0 ? "secondary" : "outline"}
                            onClick={() => void handlePreviewToxicity(item.id, 0)}
                          >
                            Sạch
                          </Button>
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
                          <IconButtonWithTooltip
                            label={item.is_locked ? "Mở khóa mẫu" : "Khóa mẫu"}
                            tooltip={lockPresentation.tooltip}
                            size="icon"
                            variant="outline"
                            onClick={() => handlePreviewLock(item.id, !Boolean(item.is_locked))}
                          >
                            {item.is_locked ? <Unlock className="h-4 w-4" /> : <Lock className="h-4 w-4" />}
                          </IconButtonWithTooltip>
                          <div className="ml-1 flex items-center gap-1 rounded-md border bg-muted/20 p-1" aria-label="Tính xây dựng">
                            <IconButtonWithTooltip
                              label="Có tính xây dựng"
                              tooltip={getConstructivenessPresentation(1).tooltip}
                              size="icon"
                              variant={item.constructiveness_label === 1 ? "default" : "ghost"}
                              onClick={() => handlePreviewConstructiveness(item.id, 1)}
                            >
                              <ThumbsUp className="h-4 w-4" />
                            </IconButtonWithTooltip>
                            <IconButtonWithTooltip
                              label="Không rõ hoặc không đóng góp"
                              tooltip={getConstructivenessPresentation(0).tooltip}
                              size="icon"
                              variant={item.constructiveness_label === 0 ? "default" : "ghost"}
                              onClick={() => handlePreviewConstructiveness(item.id, 0)}
                            >
                              <MessageCircle className="h-4 w-4" />
                            </IconButtonWithTooltip>
                            <IconButtonWithTooltip
                              label="Ẩn hoặc xóa nhãn tính xây dựng"
                              tooltip={getConstructivenessPresentation(null).tooltip}
                              size="icon"
                              variant={item.constructiveness_label == null ? "default" : "ghost"}
                              onClick={() => handlePreviewConstructiveness(item.id, null)}
                            >
                              <EyeOff className="h-4 w-4" />
                            </IconButtonWithTooltip>
                          </div>
                        </div>
                      </div>
                      {suggestion && (
                        <div className="mt-2 rounded-md border border-primary/25 bg-primary/5 p-2 text-xs">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <Badge variant="outline">Gemini</Badge>
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

          <Card className="p-4 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <h3 className="font-medium">Manual Verify (DB persisted pool)</h3>
              <div className="text-sm text-muted-foreground">
                {candidateTotal} items · page {candidatePage} · size {candidatePageSize}
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Danh sách chỉ hiển thị candidate/unverified chưa qua gate; comment accepted trong Training Preview không xuất hiện lại ở đây.
              Checkbox chỉ chọn hàng tạm thời để thao tác;
              Toxic, Clean và Remove mới cập nhật trực tiếp trạng thái DB trước export/retrain.
            </p>

            <div className="flex flex-wrap gap-2">
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
                return (
                  <div
                    key={item.id}
                    className="flex cursor-pointer items-start gap-2 rounded-md border border-border/70 p-2.5 transition-colors hover:border-primary/35 hover:bg-muted/30"
                    onClick={(event) => handleCandidateRowToggle(event, item.id)}
                  >
                    <Checkbox
                      checked={selectedCandidateIds.includes(item.id)}
                      onCheckedChange={() => toggleCandidate(item.id)}
                      aria-label="Chọn tạm thời hàng này để thao tác Manual Verify"
                    />
                    <div className="min-w-0 flex-1 space-y-1">
                      <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">{item.text}</p>
                      <div className="flex flex-wrap gap-1.5 text-xs text-muted-foreground">
                        <Badge variant="outline">domain={resolveDomainTag(item)}</Badge>
                        <MlflowBadge presentation={getScorePresentation(item.score, DEFAULT_MLFLOW_GATE_THRESHOLDS)} />
                        <MlflowBadge presentation={getToxicityPresentation(item.pseudo_label)} />
                        <MlflowBadge presentation={getConstructivenessPresentation(item.constructiveness_label)} />
                        <MlflowBadge presentation={getLockPresentation(item.is_locked)} />
                        <MlflowBadge presentation={getVerificationStatusPresentation(item.verification_status)} />
                        <Badge variant="outline">source={item.label_source ?? "-"}</Badge>
                        <Badge variant="outline">conf={item.label_confidence ?? "-"}</Badge>
                      </div>
                      <p className="text-xs text-muted-foreground break-all">{item.url}</p>
                      {suggestion && (
                        <div className="mt-2 space-y-2 rounded-md border border-primary/25 bg-primary/5 p-2 text-xs">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <Badge variant="outline">Gemini</Badge>
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

            <div className="flex flex-wrap gap-2">
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
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Total: <b>{reviewHistoryTotal}</b> · page <b>{reviewHistoryPage}</b>
            </p>
            <div className="space-y-1.5 max-h-72 overflow-auto pr-1">
              {reviewHistory.map((item) => (
                <div key={`history-${item.id}`} className="rounded-md border border-border/70 p-2.5 transition-colors hover:border-border hover:bg-muted/20">
                  <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">{item.text}</p>
                  <div className="mt-1 flex flex-wrap gap-1.5 text-xs text-muted-foreground">
                    <MlflowBadge presentation={getVerificationStatusPresentation(item.verification_status)} />
                    <MlflowBadge presentation={getGateBucketPresentation(item.gate_bucket)} />
                    <Badge variant="outline">domain={resolveDomainTag(item)}</Badge>
                    <MlflowBadge presentation={getScorePresentation(item.score, DEFAULT_MLFLOW_GATE_THRESHOLDS)} />
                    <MlflowBadge presentation={getToxicityPresentation(item.pseudo_label)} />
                    <Badge variant="outline">source={item.label_source ?? "-"}</Badge>
                    <Badge variant="outline">conf={item.label_confidence ?? "-"}</Badge>
                  </div>
                </div>
              ))}
              {reviewHistory.length === 0 && (
                <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  Chưa có history cho filter hiện tại.
                </p>
              )}
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="step4" className="space-y-4">
          <Card className="p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-medium">Pipeline tự động Google Kaggle (API trực tiếp)</h3>
              <div className="flex gap-2">
                <Button variant="outline" onClick={handleRefreshDOStatus}>
                  Refresh status
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    clearDOSession();
                    setStatusText("Đã clear Kaggle session hiện tại. Sẵn sàng trigger run mới.");
                  }}
                >
                  Clear session
                </Button>
                <Button
                  variant="secondary"
                  onClick={handleDownloadKaggleArtifact}
                  disabled={doStatusValue !== "completed" || !doHasRealArtifact || !doArtifactDownloadUrl}
                >
                  Download exported model
                </Button>
              </div>
            </div>

            <div className="rounded-md border p-3 space-y-3 bg-muted/20">
              <p className="text-sm font-medium">Compute target</p>
              <p className="text-xs text-muted-foreground">Flow tự động hiện chạy qua Google Kaggle (GPU runtime).</p>

              <p className="text-sm font-medium">Model kind</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  variant={selectedModelKind === "phobert" ? "default" : "outline"}
                  onClick={() => setSelectedModelKind("phobert")}
                >
                  PhoBERT
                </Button>
                <Button
                  type="button"
                  variant={selectedModelKind === "lr_smoke" ? "default" : "outline"}
                  onClick={() => {
                    setSelectedModelKind("lr_smoke");
                    setSelectedTrainingMode("retrain");
                  }}
                >
                  TF-IDF + LR (fast)
                </Button>
              </div>

              <p className="text-sm font-medium">Training mode</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  variant={selectedTrainingMode === "retrain" ? "default" : "outline"}
                  onClick={() => setSelectedTrainingMode("retrain")}
                >
                  Retrain
                </Button>
                <Button
                  type="button"
                  variant={selectedTrainingMode === "finetune" ? "default" : "outline"}
                  onClick={() => setSelectedTrainingMode("finetune")}
                  disabled={selectedModelKind === "lr_smoke"}
                >
                  Finetune
                </Button>
              </div>

              <p className="text-sm font-medium">Data policy</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  variant={balanceStrategy === "balanced_50_50" ? "default" : "outline"}
                  onClick={() => setBalanceStrategy("balanced_50_50")}
                >
                  Balanced 50/50
                </Button>
                <Button
                  type="button"
                  variant={balanceStrategy === "all" ? "default" : "outline"}
                  onClick={() => setBalanceStrategy("all")}
                >
                  Use all approved
                </Button>
              </div>

              {selectedModelKind === "phobert" && selectedTrainingMode === "finetune" && (
                <div>
                  <label className="text-xs text-muted-foreground">Base model (optional)</label>
                  <Input
                    value={finetuneBaseModel}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setFinetuneBaseModel(e.target.value)}
                    placeholder="vinai/phobert-base-v2"
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
                  <p className="mt-1 text-xs text-muted-foreground">Để trống để dùng base model mặc định của script finetune.</p>
                </div>
              )}

              <div className="space-y-2 rounded-md border bg-background p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <p className="text-sm font-medium">Dataset snapshot sẽ dùng cho run</p>
                    <p className="text-xs text-muted-foreground">Scope: toàn bộ batch · policy: {balanceStrategy === "balanced_50_50" ? "Balanced 50/50" : "Use all approved"}</p>
                  </div>
                  <Badge variant="outline">Tạo bundle mới khi bấm chạy</Badge>
                </div>
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
                  <div className="rounded-md border p-2">
                    <p className="text-xs text-muted-foreground">Gold train</p>
                    <p className="text-lg font-semibold">{trainingPlan?.summary.gold_train ?? "-"}</p>
                  </div>
                  <div className="rounded-md border p-2">
                    <p className="text-xs text-muted-foreground">MLflow đủ điều kiện</p>
                    <p className="text-lg font-semibold">{trainingPlan?.summary.eligible_mlflow ?? "-"}</p>
                  </div>
                  <div className="rounded-md border p-2">
                    <p className="text-xs text-muted-foreground">Sau cân bằng</p>
                    <p className="text-lg font-semibold">{trainingPlan?.summary.after_balance ?? "-"}</p>
                  </div>
                  <div className="rounded-md border p-2">
                    <p className="text-xs text-muted-foreground">Trùng bị loại</p>
                    <p className="text-lg font-semibold">{trainingPlan?.summary.duplicates_skipped ?? "-"}</p>
                  </div>
                  <div className="rounded-md border p-2">
                    <p className="text-xs text-muted-foreground">Tổng train cuối</p>
                    <p className="text-lg font-semibold">{trainingPlan?.summary.final_train ?? "-"}</p>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">
                  Thêm mới thực tế: <b>{trainingPlan?.summary.mlflow_added ?? "-"}</b> · Validation/Test giữ nguyên gold: {trainingPlan?.summary.gold_validation ?? "-"}/{trainingPlan?.summary.gold_test ?? "-"}.
                </p>
              </div>

              <p className="text-xs text-muted-foreground">
                Retrain phù hợp khi refresh dataset lớn; Finetune phù hợp khi thêm ít data/pseudo mới để giảm tài nguyên.
              </p>
              <div className="flex flex-wrap items-center justify-between gap-2 border-t pt-3">
                <p className="text-xs text-muted-foreground">Run sẽ lưu bundle path, checksum và thống kê dataset để truy vết.</p>
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
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Run ID</p>
                <p className="text-sm font-medium break-all">{doRunId}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Status</p>
                <Badge variant={doBadgeVariant as "default" | "secondary" | "destructive" | "outline"}>{doStatusValue}</Badge>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Training mode</p>
                <p className="text-sm font-medium uppercase">{doTrainingMode}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Base model</p>
                <p className="text-sm font-medium break-all">{doBaseModel || "default"}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Artifact type</p>
                <Badge variant={doIsMockArtifact ? "destructive" : doHasRealArtifact ? "secondary" : "outline"}>
                  {doIsMockArtifact ? "PLACEHOLDER" : doHasRealArtifact ? "REAL" : "NONE"}
                </Badge>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Train duration (thực tế)</p>
                <p className="text-sm font-medium">
                  {doHasEvidenceDuration ? `${doEvidenceDurationSeconds.toFixed(2)} giây` : hasDoTrainDuration ? `${doTrainDuration.toFixed(2)} phút` : "-"}
                </p>
              </div>
              <div className="rounded-md border p-3 md:col-span-2">
                <p className="text-xs text-muted-foreground">Bundle snapshot</p>
                <p className="break-all text-xs font-medium">{doStatus?.bundle_path || "-"}</p>
              </div>
              <div className="rounded-md border p-3 md:col-span-2">
                <p className="text-xs text-muted-foreground">Bundle SHA-256</p>
                <p className="break-all text-xs font-medium">{doStatus?.bundle_checksum || "-"}</p>
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

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">Pipeline progress</span>
                <span className="text-muted-foreground">{doProgress}%</span>
              </div>
              <Progress value={doProgress} className="h-2" />
            </div>

            <div className="rounded-md border p-3 bg-muted/20 space-y-1">
              <p className="text-sm font-medium">Run provenance</p>
              <p className="text-xs text-muted-foreground">
                Đây là <b>{doIsMockRun ? "mock/test run" : "real run"}</b>. Nguồn cập nhật status: <b>{doStatusSource}</b>.
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
              {doIsMockArtifact && (
                <p className="text-xs text-amber-700 dark:text-amber-300">
                  Artifact này là placeholder từ mock webhook, không phải model ZIP thật từ Kaggle.
                </p>
              )}
              <p className="text-xs break-all">URI: {doArtifactUri || "-"}</p>
              <p className="text-xs break-all">Checksum (sha256): {doChecksum || "-"}</p>
              {doErrorMessage && <p className="text-xs text-destructive break-all">Error: {doErrorMessage}</p>}
            </div>

            <div className="rounded-md border p-3 text-sm space-y-2">
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
        </TabsContent>
      </Tabs>
    </div>
  );
}
