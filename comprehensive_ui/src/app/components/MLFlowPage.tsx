import { useEffect, useMemo, useRef, useState, type ChangeEvent, type MouseEvent } from "react";
import { RotateCcw } from "lucide-react";
import { toast } from "sonner";
import { Card } from "@/app/components/ui/card";
import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { Badge } from "@/app/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { Progress } from "@/app/components/ui/progress";
import { Checkbox } from "@/app/components/ui/checkbox";
import { useMlflowStore, type MlflowUnusedScope } from "../../hooks/useMlflowStore";


interface MLFlowPageProps {
  availableModels: string[];
  onModelsChanged?: () => Promise<void> | void;
}

const MLFLOW_URLS_DRAFT_KEY = "viettoxic:mlflow:urlsText";
const MLFLOW_MODEL_DRAFT_KEY = "viettoxic:mlflow:selectedModel";
const MLFLOW_CLEAR_ALL_CONFIRM_TOKEN = "DELETE_ALL_MLFLOW_DATA";

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

export function MLFlowPage({ availableModels, onModelsChanged }: MLFlowPageProps) {
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
    requiredZipContents,
    doStatus,
    doPreflight,
    ingest,
    refreshOverview,
    refreshCandidates,
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
    refreshCompare,
    promote,
  } = useMlflowStore();

  const [urlsText, setUrlsText] = useState(() => {
    if (typeof window === "undefined") return "";
    return safeReadLocalStorageString(MLFLOW_URLS_DRAFT_KEY, "");
  });
  const [selectedModel, setSelectedModel] = useState<string>(() => {
    if (typeof window === "undefined") return availableModels[0] || "";
    return safeReadLocalStorageString(MLFLOW_MODEL_DRAFT_KEY, availableModels[0] || "");
  });
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<number[]>([]);
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
  const [activeTab, setActiveTab] = useState("step1");
  const [manualTabUnlocked, setManualTabUnlocked] = useState(false);
  const [selectedComputeMode, setSelectedComputeMode] = useState<"gpu" | "cpu" | "local_m1">("gpu");
  const [selectedTrainingMode, setSelectedTrainingMode] = useState<"retrain" | "finetune">("retrain");
  const [finetuneBaseModel, setFinetuneBaseModel] = useState("");
  const [cpuProfileInput, setCpuProfileInput] = useState("");
  const prevDoStatusRef = useRef<string>("idle");

  useEffect(() => {
    void refreshOverview();
    void refreshCandidates(undefined, 1, "all_batches");
    void refreshThresholdStatus(activeBatchId);
    void refreshReviewHistory(undefined, historyDecision, 1, "all_batches");
    void refreshCrawlHistory(1);
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
    const firstSelectable = availableModels.find((model) => !isDeprecatedModel(model)) || availableModels[0] || "";
    if (!finetuneBaseModel && firstSelectable) {
      setFinetuneBaseModel(firstSelectable);
      return;
    }
    if (finetuneBaseModel && isDeprecatedModel(finetuneBaseModel) && firstSelectable && finetuneBaseModel !== firstSelectable) {
      setFinetuneBaseModel(firstSelectable);
    }
  }, [availableModels, finetuneBaseModel]);

  useEffect(() => {
    void refreshReviewHistory(undefined, historyDecision, 1, "all_batches");
  }, [historyDecision]);

  useEffect(() => {
    safeWriteLocalStorageString(MLFLOW_URLS_DRAFT_KEY, urlsText);
  }, [urlsText]);

  useEffect(() => {
    if (!selectedModel) return;
    safeWriteLocalStorageString(MLFLOW_MODEL_DRAFT_KEY, selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    const availableIds = new Set(candidates.map((item) => item.id));
    setSelectedCandidateIds((prev) => prev.filter((id) => availableIds.has(id)));
  }, [candidates]);

  useEffect(() => {
    if (!manualTabUnlocked && activeTab === "step3") {
      setActiveTab("step4");
    }
  }, [activeTab, manualTabUnlocked]);

  useEffect(() => {
    if (cpuProfileInput.trim()) return;
    const fromPreflight =
      doPreflight && typeof doPreflight.config?.do_default_cpu_size === "string"
        ? doPreflight.config.do_default_cpu_size
        : "";
    if (fromPreflight.trim()) {
      setCpuProfileInput(fromPreflight.trim());
    }
  }, [cpuProfileInput, doPreflight]);

  const parsedUrls = useMemo(
    () =>
      urlsText
        .split(/\r?\n/)
        .map((u) => u.trim())
        .filter(Boolean),
    [urlsText],
  );

  const thresholdProgress = useMemo(() => {
    if (!thresholdStatus) return 0;
    const max = Math.max(1, thresholdStatus.target_max_test_stage || 10);
    return Math.min(100, (thresholdStatus.accepted_count / max) * 100);
  }, [thresholdStatus]);

  const ingestStageMeta = useMemo(() => {
    if (ingestStage === "crawl") return { label: "Crawl", variant: "default" as const };
    if (ingestStage === "inference") return { label: "Inference", variant: "default" as const };
    if (ingestStage === "finalize") return { label: "Finalize", variant: "default" as const };
    if (ingestStage === "completed") return { label: "Completed", variant: "secondary" as const };
    if (ingestStage === "error") return { label: "Error", variant: "destructive" as const };
    return { label: loading ? "Running" : "Idle", variant: loading ? ("default" as const) : ("secondary" as const) };
  }, [ingestStage, loading]);

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

  const handleCandidateRowToggle = (event: MouseEvent<HTMLDivElement>, id: number) => {
    const target = event.target as HTMLElement;
    if (target.closest("button, input, textarea, select, option, a")) {
      return;
    }
    toggleCandidate(id);
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
    } catch {
      setStatusText("Lưu review vào DB thất bại.");
      toast.error("Lưu review vào DB thất bại.");
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
      const rows = payload.deleted_rows;
      toast.success(
        `Đã clear MLFlow: do_run=${rows.mlflow_do_run}, artifacts=${rows.mlflow_training_artifact}, items=${rows.mlflow_comment_item}, batches=${rows.mlflow_crawl_batch}.`,
      );
    } catch {
      toast.error("Clear all MLFlow thất bại.");
    }
  };

  const handleExportBundle = async () => {
    try {
      const payload = await exportBundle(activeBatchId, {
        scope: "all_batches",
        includeUnused: includeUnusedInExport,
        unusedScope,
      });

      const downloadHref = payload.download_url.startsWith("http")
        ? payload.download_url
        : `${window.location.origin}${payload.download_url}`;
      const anchor = document.createElement("a");
      anchor.href = downloadHref;
      anchor.rel = "noopener noreferrer";
      anchor.download = payload.bundle_path.split("/").pop() || "mlflow_bundle.zip";
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);

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
    try {
      const payload = await triggerDO({
        computeMode: selectedComputeMode,
        trainingMode: selectedTrainingMode,
        baseModel: selectedTrainingMode === "finetune" ? finetuneBaseModel : undefined,
        cpuProfile: cpuProfileInput,
      });
      const modeLabel =
        selectedComputeMode === "local_m1" ? "LOCAL_M1" : selectedComputeMode.toUpperCase();
      const trainingLabel = selectedTrainingMode === "finetune" ? "FINETUNE" : "RETRAIN";
      setStatusText(`Đã trigger DO run ${payload.run_id} (${payload.status}) - ${trainingLabel}.`);
      toast.success(`Đã trigger DO ${modeLabel} run ${payload.run_id} (${trainingLabel}).`);
    } catch {
      setStatusText("Trigger DO pipeline thất bại.");
      toast.error("Trigger DO pipeline thất bại.");
    }
  };

  const handlePromote = async () => {
    const candidateModel = comparePayload?.candidate?.model;
    if (!candidateModel) {
      setStatusText("Chưa có candidate model để promote.");
      return;
    }
    try {
      const payload = await promote(candidateModel);
      setStatusText(payload.message || payload.status);
    } catch {
      setStatusText("Promote thất bại.");
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
    train_local_m1: "Train local trên M1",
    finalize_local_run: "Finalize local run",
  };

  const doStages =
    Array.isArray(doStatus?.stages) && (doStatus.stages as unknown[]).every((stage) => typeof stage === "string")
      ? (doStatus.stages as string[])
      : defaultDoStages;
  const doStatusValue = (doStatus?.status as string | undefined) || "idle";
  const doCurrentStage = (doStatus?.current_stage as string | undefined) || "";
  const doLogs = Array.isArray(doStatus?.logs) ? (doStatus?.logs as string[]) : [];
  const doRunId = (doStatus?.run_id as string | undefined) || "-";
  const doProvider = (doStatus?.provider as string | undefined) || "-";
  const doBatchId = (doStatus?.batch_id as string | undefined) || "-";
  const doGpuProfile = (doStatus?.gpu_profile as string | undefined) || "-";
  const doComputeMode = ((doStatus?.compute_mode as string | undefined) || selectedComputeMode || "gpu").toLowerCase();
  const doTrainingMode = ((doStatus?.training_mode as string | undefined) || selectedTrainingMode || "retrain").toLowerCase();
  const doBaseModel = (doStatus?.base_model as string | undefined) || (selectedTrainingMode === "finetune" ? finetuneBaseModel : "");
  const doDropletProfile =
    (doStatus?.droplet_profile as string | undefined) ||
    (doComputeMode === "cpu" ? (cpuProfileInput.trim() || doGpuProfile) : doGpuProfile);
  const doEtaEstimate = Number(doStatus?.eta_estimate_minutes);
  const doTrainDuration = Number(doStatus?.train_duration_minutes);
  const doCpuPercent = Number(doStatus?.cpu_percent);
  const doMemoryPercent = Number(doStatus?.memory_percent);
  const doTelemetryLastSampleAt = (doStatus?.telemetry_last_sample_at as string | undefined) || "";
  const doDropletId = (doStatus?.droplet_id as string | undefined) || "-";
  const doArtifactUri = (doStatus?.artifact_uri as string | undefined) || "";
  const doChecksum = (doStatus?.artifact_checksum as string | undefined) || "";
  const doSignedUrl = (doStatus?.signed_download_url as string | undefined) || "";
  const doErrorMessage = (doStatus?.error_message as string | undefined) || "";
  const doApiCallEvidence = doLogs.find((line) => line.startsWith("DO API request:")) || "";
  const doIsPlaceholder =
    doStatusValue === "placeholder" || doLogs.some((line) => line.toLowerCase().includes("placeholder flow only"));
  const doIsRestricted = /restricted|account tier|increase your account tier/i.test(doErrorMessage);
  const hasDoEtaEstimate = Number.isFinite(doEtaEstimate) && doEtaEstimate > 0;
  const hasDoTrainDuration = Number.isFinite(doTrainDuration) && doTrainDuration >= 0;
  const hasDoCpuPercent = Number.isFinite(doCpuPercent) && doCpuPercent >= 0;
  const hasDoMemoryPercent = Number.isFinite(doMemoryPercent) && doMemoryPercent >= 0;
  const hasDoTelemetrySample = doTelemetryLastSampleAt.length > 0;

  useEffect(() => {
    const prev = prevDoStatusRef.current;
    if (prev === doStatusValue) return;

    if (doStatusValue === "running") {
      toast.message("DO pipeline đang chạy.");
    } else if (doStatusValue === "completed") {
      toast.success("DO pipeline hoàn tất.");
    } else if (doStatusValue === "failed") {
      if (doIsRestricted) {
        toast.error("GPU bị restricted. Hãy chuyển CPU hoặc mở ticket tăng tier.");
      } else {
        toast.error("DO pipeline thất bại.");
      }
    }

    prevDoStatusRef.current = doStatusValue;
  }, [doIsRestricted, doStatusValue]);

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
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <Card className="p-5 border-border/80 bg-gradient-to-br from-background to-muted/30">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">Admin / ML Flow</p>
            <h1 className="text-2xl font-semibold">VietToxic Self-Learning Pipeline</h1>
            <p className="text-sm text-muted-foreground">Ingest → Auto Gate Persisted DB → Verify → Retrain decision</p>
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

      {(ingestStage !== "idle" || loading) && (
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
        <TabsList className={`w-full grid ${manualTabUnlocked ? "grid-cols-5" : "grid-cols-4"} h-auto`}>
          <TabsTrigger value="step1">Thu thập & Gán nhãn</TabsTrigger>
          <TabsTrigger value="step2">Dataset</TabsTrigger>
          {manualTabUnlocked && <TabsTrigger value="step3">Thủ công: Train & Import</TabsTrigger>}
          <TabsTrigger value="step4">Tự động: DigitalOcean</TabsTrigger>
          <TabsTrigger value="step5">Đánh giá & Gate</TabsTrigger>
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
                        {model}
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
                  void refreshThresholdStatus(activeBatchId);
                  void refreshReviewHistory(undefined, historyDecision, 1, "all_batches");
                  void refreshCrawlHistory(1);
                  void refreshCompare();
                }}
              >
                Refresh
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
          </Card>

          <div className="grid gap-4 md:grid-cols-3">
            <Card className="p-4 space-y-1">
              <p className="text-sm text-muted-foreground">Crawl mới</p>
              <p className="text-3xl font-semibold">{overview?.pipeline_counts?.crawled ?? 0}</p>
              <p className="text-xs text-muted-foreground">segments crawled</p>
            </Card>
            <Card className="p-4 space-y-1">
              <p className="text-sm text-muted-foreground">Infer + Pseudo-label</p>
              <p className="text-3xl font-semibold">{overview?.pipeline_counts?.inferred ?? 0}</p>
              <p className="text-xs text-muted-foreground">model: {overview?.model_name || selectedModel || "-"}</p>
            </Card>
            <Card className="p-4 space-y-1">
              <p className="text-sm text-muted-foreground">Gate 0.8 / 0.2</p>
              <p className="text-sm">
                Accepted: <b>{overview?.pipeline_counts?.accepted ?? 0}</b>
              </p>
              <p className="text-sm">
                Candidates: <b>{overview?.pipeline_counts?.candidate ?? 0}</b>
              </p>
              <p className="text-sm">
                Discarded (auto + manual reject): <b>{overview?.pipeline_counts?.discarded ?? 0}</b>
              </p>
            </Card>
          </div>

          <Card className="p-4 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <h3 className="font-medium">Lịch sử URL đã crawl (DB persisted)</h3>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">
                  Total: <b>{crawlHistoryTotal}</b> · page <b>{crawlHistoryPage}</b>
                </span>
                <Button size="sm" variant="outline" onClick={() => refreshCrawlHistory(crawlHistoryPage)}>
                  Refresh history
                </Button>
              </div>
            </div>
            <div className="space-y-1.5 max-h-72 overflow-auto pr-1">
              {crawlHistory.map((item) => (
                <div key={`${item.batch_id}:${item.url_hash}`} className="rounded-md border p-2">
                  <p className="text-sm truncate">{item.url}</p>
                  <p className="text-xs text-muted-foreground">
                    batch={item.batch_id} · segments={item.segment_count} · accepted={item.accepted_count} · candidate={item.candidate_count} · discarded={item.discarded_count}
                  </p>
                </div>
              ))}
              {crawlHistory.length === 0 && (
                <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  Chưa có lịch sử crawl.
                </p>
              )}
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="step2" className="space-y-4">
          <Card className="p-6 text-center space-y-4">
            <h3 className="text-xl font-semibold">Dataset toàn bộ DB (all batches)</h3>
            <p className="text-4xl font-bold">
              {thresholdStatus?.accepted_count ?? 0} / {thresholdStatus?.target_max_test_stage ?? 10}
            </p>
            <Progress value={thresholdProgress} />
            <p className="text-sm text-muted-foreground">
              {thresholdStatus?.is_ready ? "Đủ điều kiện retrain" : "Chưa đủ điều kiện retrain"}
            </p>
            <div className="flex justify-center gap-2">
              <Button onClick={handleExportBundle}>Tải dataset xuống</Button>
              <Button
                variant="outline"
                onClick={() => {
                  void refreshThresholdStatus(activeBatchId);
                  void refreshCandidates(undefined, candidatePage, "all_batches");
                }}
              >
                Refresh
              </Button>
            </div>
          </Card>

          <Card className="p-4 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <h3 className="font-medium">Manual verify (DB persisted pool)</h3>
              <div className="text-sm text-muted-foreground">
                {candidateTotal} items · page {candidatePage} · size {candidatePageSize}
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Chọn Toxic/Clean/Remove để cập nhật trực tiếp trong DB trước khi export/retrain.
            </p>

            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="outline" onClick={handleSelectAllCandidates} disabled={candidates.length === 0}>
                Select all
              </Button>
              <Button size="sm" variant="outline" onClick={handleUnselectAllCandidates} disabled={selectedCandidateIds.length === 0}>
                Unselect all
              </Button>
            </div>

            <div className="space-y-1.5 max-h-[34rem] overflow-auto pr-1">
              {candidates.map((item) => (
                <div
                  key={item.id}
                  className="flex cursor-pointer items-start gap-2 rounded-md border p-2 hover:bg-muted/40"
                  onClick={(event) => handleCandidateRowToggle(event, item.id)}
                >
                  <Checkbox checked={selectedCandidateIds.includes(item.id)} onCheckedChange={() => toggleCandidate(item.id)} />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm line-clamp-2">{item.text}</p>
                    <p className="text-xs text-muted-foreground">
                      domain={resolveDomainTag(item)} · score={item.score?.toFixed(3) ?? "-"} · pseudo={item.pseudo_label ?? "-"} · source={item.label_source ?? "-"} · conf={item.label_confidence ?? "-"} · {item.url}
                    </p>
                  </div>
                </div>
              ))}
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
                <div key={`history-${item.id}`} className="rounded-md border p-2">
                  <p className="text-sm line-clamp-2">{item.text}</p>
                  <p className="text-xs text-muted-foreground">
                    {item.verification_status} · bucket={item.gate_bucket} · domain={resolveDomainTag(item)} · score={item.score?.toFixed(3) ?? "-"} · pseudo={item.pseudo_label ?? "-"} · source={item.label_source ?? "-"} · conf={item.label_confidence ?? "-"}
                  </p>
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

        {manualTabUnlocked && (
          <TabsContent value="step3" className="space-y-4">
            <Card className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-medium">Bước 1 — Tải dataset bundle</h3>
              <Button onClick={handleExportBundle}>Download</Button>
            </div>
            <p className="text-sm text-muted-foreground">Bundle mặc định gồm accepted + candidate. Bạn có thể bật thêm unused/discarded nếu cần phân tích sâu.</p>
            <p className="text-xs text-muted-foreground">
              Export bundle mặc định chạy theo scope all_batches, tự tạo bộ `dataset/victsd_gold/*` đã merge accepted vào train và vẫn giữ accepted/candidate/unused để tương thích ngược.
            </p>
            <div className="rounded-md border p-3 space-y-3 bg-muted/20">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium">Bao gồm unused/discarded vào bundle</p>
                  <p className="text-xs text-muted-foreground">OFF: chỉ accepted + candidate · ON: thêm discarded theo phạm vi chọn.</p>
                </div>
                <label className="inline-flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={includeUnusedInExport}
                    onChange={(e) => setIncludeUnusedInExport(e.target.checked)}
                  />
                  Include unused
                </label>
              </div>
              {includeUnusedInExport && (
                <div>
                  <label className="text-xs text-muted-foreground">Phạm vi discarded</label>
                  <select
                    className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
                    value={unusedScope}
                    onChange={(e) => setUnusedScope(e.target.value as MlflowUnusedScope)}
                  >
                    <option value="all">Tất cả discarded</option>
                    <option value="auto_discarded">Auto discarded (theo ngưỡng)</option>
                    <option value="manual_rejected">Manual rejected (do người review)</option>
                  </select>
                </div>
              )}
            </div>
            {lastBundlePath && <p className="text-xs break-all">Bundle: {lastBundlePath}</p>}
            <div className="rounded-md border p-3 text-sm">
              <p className="font-medium mb-2">Required zip contents</p>
              <ul className="list-disc ml-5 space-y-1">
                {(requiredZipContents.length > 0
                  ? requiredZipContents
                  : [
                      "dataset/accepted_pseudo.jsonl",
                      "dataset/candidates_unverified.jsonl",
                      "manifest.json",
                      "config/training_config.yaml",
                      "config/gate_policy.json",
                    ]
                ).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          </Card>

            <Card className="p-4 space-y-3">
              <h3 className="font-medium">Bước 2 — Import model</h3>
              <div className="grid md:grid-cols-2 gap-3">
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
              <div className="flex gap-2">
                <Button onClick={handleImportModelZip}>Import ZIP model</Button>
                <Button variant="outline" onClick={() => refreshCompare()}>
                  Refresh compare source
                </Button>
              </div>
            </Card>
          </TabsContent>
        )}

        <TabsContent value="step4" className="space-y-4">
          <Card className="p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-medium">Pipeline tự động DigitalOcean (API trực tiếp)</h3>
              <div className="flex gap-2">
                <Button variant="outline" onClick={() => refreshDOPreflight()}>
                  Check preflight
                </Button>
                <Button variant="outline" onClick={() => refreshDOStatus()}>
                  Refresh status
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    setManualTabUnlocked(true);
                    setActiveTab("step3");
                  }}
                >
                  Thủ công
                </Button>
                <Button
                  onClick={handleTriggerDO}
                  disabled={selectedComputeMode !== "local_m1" && doPreflight?.ready === false}
                >
                  Kích hoạt DO Pipeline
                </Button>
              </div>
            </div>

            <div className="rounded-md border p-3 space-y-3 bg-muted/20">
              <p className="text-sm font-medium">Compute mode</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  variant={selectedComputeMode === "gpu" ? "default" : "outline"}
                  onClick={() => setSelectedComputeMode("gpu")}
                >
                  GPU
                </Button>
                <Button
                  type="button"
                  variant={selectedComputeMode === "cpu" ? "default" : "outline"}
                  onClick={() => setSelectedComputeMode("cpu")}
                >
                  CPU
                </Button>
                <Button
                  type="button"
                  variant={selectedComputeMode === "local_m1" ? "default" : "outline"}
                  onClick={() => setSelectedComputeMode("local_m1")}
                >
                  M1 Mac (local)
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
                >
                  Finetune
                </Button>
              </div>

              {selectedTrainingMode === "finetune" && (
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
                        <option key={`base-${model}`} value={model} />
                      ))}
                  </datalist>
                  <p className="mt-1 text-xs text-muted-foreground">Để trống để dùng base model mặc định của script finetune.</p>
                </div>
              )}

              {selectedComputeMode === "cpu" && (
                <div>
                  <label className="text-xs text-muted-foreground">CPU profile (optional slug)</label>
                  <Input
                    value={cpuProfileInput}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setCpuProfileInput(e.target.value)}
                    placeholder="s-8vcpu-16gb"
                    className="mt-1"
                  />
                </div>
              )}
              <p className="text-xs text-muted-foreground">
                Retrain phù hợp khi refresh dataset lớn; Finetune phù hợp khi thêm ít data/pseudo mới để giảm tài nguyên.
              </p>
            </div>

            <div className="grid gap-3 md:grid-cols-3">
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Run ID</p>
                <p className="text-sm font-medium break-all">{doRunId}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Status</p>
                <Badge variant={doBadgeVariant as "default" | "secondary" | "destructive" | "outline"}>{doStatusValue}</Badge>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Droplet ID</p>
                <p className="text-sm font-medium break-all">{doDropletId}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Provider</p>
                <p className="text-sm font-medium break-all">{doProvider}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Batch ID</p>
                <p className="text-sm font-medium break-all">{doBatchId}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Compute profile</p>
                <p className="text-sm font-medium break-all">{doDropletProfile || "-"}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Compute mode</p>
                <p className="text-sm font-medium uppercase">{doComputeMode}</p>
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
                <p className="text-xs text-muted-foreground">CPU usage</p>
                <p className="text-sm font-medium">
                  {hasDoCpuPercent ? `${doCpuPercent.toFixed(1)}%` : doStatusValue === "running" ? "Đang chờ sample..." : "-"}
                </p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Memory usage</p>
                <p className="text-sm font-medium">
                  {hasDoMemoryPercent ? `${doMemoryPercent.toFixed(1)}%` : doStatusValue === "running" ? "Đang chờ sample..." : "-"}
                </p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Last telemetry sample</p>
                <p className="text-sm font-medium break-all">{hasDoTelemetrySample ? doTelemetryLastSampleAt : "-"}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">ETA train (ước tính)</p>
                <p className="text-sm font-medium">{hasDoEtaEstimate ? `~${Math.round(doEtaEstimate)} phút` : "-"}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Train duration (thực tế)</p>
                <p className="text-sm font-medium">{hasDoTrainDuration ? `${doTrainDuration.toFixed(2)} phút` : "-"}</p>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">Pipeline progress</span>
                <span className="text-muted-foreground">{doProgress}%</span>
              </div>
              <Progress value={doProgress} className="h-2" />
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
                <p className="text-xs font-medium text-destructive">DO pipeline đang ở placeholder mode</p>
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
                  <div key={stage} className="rounded-md border p-3 text-sm flex items-center justify-between">
                    <span>{doStageLabels[stage] || stage}</span>
                    <Badge variant={variant as "default" | "secondary" | "destructive" | "outline"}>{stateText}</Badge>
                  </div>
                );
              })}
            </div>

            <div className="rounded-md border p-3 text-sm space-y-2">
              <p className="font-medium">Artifact</p>
              <p className="text-xs break-all">URI: {doArtifactUri || "-"}</p>
              <p className="text-xs break-all">Checksum (sha256): {doChecksum || "-"}</p>
              {doSignedUrl && (
                <a className="text-xs underline text-primary break-all" href={doSignedUrl} target="_blank" rel="noreferrer">
                  Download artifact (Spaces)
                </a>
              )}
              {doErrorMessage && <p className="text-xs text-destructive break-all">Error: {doErrorMessage}</p>}
            </div>

            <div className="rounded-md border p-3 text-sm space-y-2">
              <p className="font-medium">Training log</p>
              <div className="max-h-56 overflow-auto space-y-1">
                {doLogs.length === 0 ? (
                  <p className="text-xs text-muted-foreground">Chưa có log.</p>
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
              <h3 className="font-medium">Metrics comparison</h3>
              <div className="text-sm space-y-1">
                <p>
                  <b>Current:</b> {comparePayload?.current?.model || "-"}
                </p>
                <p>
                  f1_toxic: {(comparePayload?.current?.metrics?.f1_toxic as number | null | undefined)?.toFixed?.(3) ?? "-"}
                </p>
                <p>
                  macro_f1: {(comparePayload?.current?.metrics?.macro_f1 as number | null | undefined)?.toFixed?.(3) ?? "-"}
                </p>
              </div>
              <div className="text-sm space-y-1">
                <p>
                  <b>Candidate:</b> {comparePayload?.candidate?.model || "-"}
                </p>
                <p>
                  f1_toxic: {(comparePayload?.candidate?.metrics?.f1_toxic as number | null | undefined)?.toFixed?.(3) ?? "-"}
                </p>
                <p>
                  macro_f1: {(comparePayload?.candidate?.metrics?.macro_f1 as number | null | undefined)?.toFixed?.(3) ?? "-"}
                </p>
              </div>
              <Button variant="outline" onClick={() => refreshCompare()}>
                Refresh compare
              </Button>
            </Card>

            <Card className="p-4 space-y-3">
              <h3 className="font-medium">Gate conditions</h3>
              <div className="space-y-2">
                {(comparePayload?.gate_checks || []).map((check) => (
                  <div key={check.name} className="flex justify-between rounded-md border p-2 text-sm">
                    <span>{check.name}</span>
                    <Badge variant={check.passed ? "default" : "secondary"}>{check.passed ? "PASS" : "WAIT"}</Badge>
                  </div>
                ))}
              </div>
              <Button onClick={handlePromote}>Promote to Production</Button>
              <p className="text-xs text-muted-foreground">Promote dùng gate check hiện có; pipeline DO trả thêm artifact URI/checksum để kiểm tra trước khi promote.</p>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
