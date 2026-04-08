import { useEffect, useMemo, useState, type ChangeEvent, type MouseEvent } from "react";
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
}

const MLFLOW_URLS_DRAFT_KEY = "viettoxic:mlflow:urlsText";
const MLFLOW_MODEL_DRAFT_KEY = "viettoxic:mlflow:selectedModel";
const MLFLOW_CANDIDATE_SELECTIONS_KEY = "viettoxic:mlflow:selectedCandidatesByBatch";
const MLFLOW_CLEAR_ALL_CONFIRM_TOKEN = "DELETE_ALL_MLFLOW_DATA";

const safeReadLocalStorage = <T,>(key: string, fallback: T): T => {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

const safeWriteLocalStorage = (key: string, value: unknown) => {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore quota / private mode errors
  }
};

export function MLFlowPage({ availableModels }: MLFlowPageProps) {
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
    batches,
    reviewHistory,
    reviewHistoryTotal,
    reviewHistoryPage,
    comparePayload,
    lastBundlePath,
    requiredZipContents,
    doStatus,
    ingest,
    refreshOverview,
    refreshBatches,
    refreshCandidates,
    refreshReviewHistory,
    reviewCandidates,
    clearMlflowBatch,
    clearMlflowAll,
    refreshThresholdStatus,
    exportBundle,
    importArtifact,
    triggerDO,
    refreshDOStatus,
    refreshCompare,
    promote,
  } = useMlflowStore();

  const [urlsText, setUrlsText] = useState(() => {
    if (typeof window === "undefined") return "";
    const raw = window.localStorage.getItem(MLFLOW_URLS_DRAFT_KEY);
    return raw ?? "";
  });
  const [selectedModel, setSelectedModel] = useState<string>(() => {
    if (typeof window === "undefined") return availableModels[0] || "";
    const raw = window.localStorage.getItem(MLFLOW_MODEL_DRAFT_KEY);
    return raw || availableModels[0] || "";
  });
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<number[]>([]);
  const [minCandidateScore, setMinCandidateScore] = useState(0.5);
  const [scoreSortMode, setScoreSortMode] = useState<"high_to_low" | "low_to_high">("high_to_low");
  const [artifactRunName, setArtifactRunName] = useState("");
  const [artifactPath, setArtifactPath] = useState("");
  const [statusText, setStatusText] = useState<string | null>(null);
  const [includeUnusedInExport, setIncludeUnusedInExport] = useState(false);
  const [unusedScope, setUnusedScope] = useState<MlflowUnusedScope>("all");
  const [historyDecision, setHistoryDecision] = useState<"all" | "accepted" | "rejected" | "discarded">("all");
  const [crawlSummary, setCrawlSummary] = useState<{
    status_counts?: Record<string, number>;
    timeout_count?: number;
    total_urls?: number;
  } | null>(null);

  useEffect(() => {
    void refreshBatches();
    if (!activeBatchId) {
      void refreshOverview();
      void refreshCompare();
      return;
    }
    void refreshOverview(activeBatchId);
    void refreshCandidates(activeBatchId, 1);
    void refreshThresholdStatus(activeBatchId);
    void refreshReviewHistory(activeBatchId, historyDecision, 1);
    void refreshCompare();
  }, []);
  useEffect(() => {
    if (!selectedModel && availableModels[0]) {
      setSelectedModel(availableModels[0]);
    }
  }, [availableModels, selectedModel]);

  useEffect(() => {
    if (!activeBatchId) return;
    void refreshReviewHistory(activeBatchId, historyDecision, 1);
  }, [activeBatchId, historyDecision]);
  useEffect(() => {
    safeWriteLocalStorage(MLFLOW_URLS_DRAFT_KEY, urlsText);
  }, [urlsText]);

  useEffect(() => {
    if (!selectedModel) return;
    safeWriteLocalStorage(MLFLOW_MODEL_DRAFT_KEY, selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    if (!activeBatchId) {
      setSelectedCandidateIds([]);
      return;
    }
    const map = safeReadLocalStorage<Record<string, number[]>>(MLFLOW_CANDIDATE_SELECTIONS_KEY, {});
    const persisted = map[activeBatchId] || [];
    const availableIds = new Set(candidates.map((item) => item.id));
    const sanitized = persisted.filter((id) => availableIds.has(id));
    setSelectedCandidateIds(sanitized);
    if (sanitized.length !== persisted.length) {
      map[activeBatchId] = sanitized;
      safeWriteLocalStorage(MLFLOW_CANDIDATE_SELECTIONS_KEY, map);
    }
  }, [activeBatchId, candidates]);
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

  const filteredCandidates = useMemo(() => {
    const withScore = candidates.filter((item) => {
      const score = Number(item.score ?? 0);
      return score >= minCandidateScore;
    });

    return withScore.sort((a, b) => {
      const scoreA = Number(a.score ?? 0);
      const scoreB = Number(b.score ?? 0);
      return scoreSortMode === "high_to_low" ? scoreB - scoreA : scoreA - scoreB;
    });
  }, [candidates, minCandidateScore, scoreSortMode]);

  const persistSelectionByBatch = (batchId: string | null, ids: number[]) => {
    if (!batchId) return;
    const map = safeReadLocalStorage<Record<string, number[]>>(MLFLOW_CANDIDATE_SELECTIONS_KEY, {});
    map[batchId] = ids;
    safeWriteLocalStorage(MLFLOW_CANDIDATE_SELECTIONS_KEY, map);
  };

  const toggleCandidate = (id: number) => {
    setSelectedCandidateIds((prev) => {
      const next = prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id];
      persistSelectionByBatch(activeBatchId, next);
      return next;
    });
  };

  const handleSelectAllCandidates = () => {
    const ids = filteredCandidates.map((item) => item.id);
    setSelectedCandidateIds(ids);
    persistSelectionByBatch(activeBatchId, ids);
  };

  const handleUnselectAllCandidates = () => {
    setSelectedCandidateIds([]);
    persistSelectionByBatch(activeBatchId, []);
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
      persistSelectionByBatch(result.batch_id, []);

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
      await refreshBatches();
      setSelectedCandidateIds([]);
      persistSelectionByBatch(activeBatchId, []);

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

  const handleBatchChange = async (nextBatchId: string) => {
    if (!nextBatchId || nextBatchId === activeBatchId) return;

    setSelectedCandidateIds([]);
    persistSelectionByBatch(activeBatchId, []);

    try {
      await refreshOverview(nextBatchId);
      await refreshCandidates(nextBatchId, 1);
      await refreshThresholdStatus(nextBatchId);
      await refreshReviewHistory(nextBatchId, historyDecision, 1);
      setStatusText(`Đã chuyển sang batch ${nextBatchId}.`);
    } catch {
      setStatusText("Chuyển batch thất bại.");
      toast.error("Chuyển batch thất bại.");
    }
  };

  const handleClearCurrentBatch = async () => {
    if (!activeBatchId) return;

    const confirmed = window.confirm(`Xóa toàn bộ dữ liệu MLFlow của batch ${activeBatchId}?`);
    if (!confirmed) return;

    try {
      const payload = await clearMlflowBatch(activeBatchId);
      setSelectedCandidateIds([]);
      persistSelectionByBatch(activeBatchId, []);
      const rows = payload.deleted_rows;
      setStatusText(
        `Đã clear batch ${payload.batch_id}: do_run=${rows.mlflow_do_run}, items=${rows.mlflow_comment_item}, batches=${rows.mlflow_crawl_batch}.`,
      );
      toast.success(`Đã clear batch ${payload.batch_id}.`);
    } catch {
      setStatusText("Clear current batch thất bại.");
      toast.error("Clear current batch thất bại.");
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
      setStatusText(
        `Đã clear toàn bộ MLFlow: do_run=${rows.mlflow_do_run}, artifacts=${rows.mlflow_training_artifact}, items=${rows.mlflow_comment_item}, batches=${rows.mlflow_crawl_batch}.`,
      );
      toast.success("Đã clear toàn bộ dữ liệu MLFlow.");
    } catch {
      setStatusText("Clear all MLFlow thất bại.");
      toast.error("Clear all MLFlow thất bại.");
    }
  };

  const handleExportBundle = async () => {
    try {
      const payload = await exportBundle(activeBatchId, {
        includeUnused: includeUnusedInExport,
        unusedScope,
      });
      setStatusText(
        `Đã tạo bundle train/import. accepted ${payload.count}, candidate ${payload.candidate_count}, unused ${payload.unused_count}.`,
      );
    } catch {
      setStatusText("Export bundle thất bại.");
    }
  };

  const handleImportArtifact = async () => {
    if (!artifactRunName.trim() || !artifactPath.trim()) {
      setStatusText("Nhập run name và artifact path trước khi import.");
      return;
    }
    try {
      await importArtifact(artifactRunName.trim(), artifactPath.trim());
      setStatusText("Đã lưu metadata artifact import.");
    } catch {
      setStatusText("Import artifact thất bại.");
    }
  };

  const handleTriggerDO = async () => {
    try {
      const payload = await triggerDO();
      setStatusText(`Đã trigger DO placeholder run ${payload.run_id}.`);
    } catch {
      setStatusText("Trigger DO pipeline thất bại.");
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

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <Card className="p-5 border-border/80 bg-gradient-to-br from-background to-muted/30">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-wider text-muted-foreground">Admin / ML Flow</p>
            <h1 className="text-2xl font-semibold">VietToxic Self-Learning Pipeline</h1>
            <p className="text-sm text-muted-foreground">Ingest → Infer → Gate → Review → Retrain decision</p>
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

      <Tabs defaultValue="step1" className="space-y-4">
        <TabsList className="w-full grid grid-cols-5 h-auto">
          <TabsTrigger value="step1">Thu thập & Gán nhãn</TabsTrigger>
          <TabsTrigger value="step2">Kiểm tra Threshold</TabsTrigger>
          <TabsTrigger value="step3">Thủ công: Train & Import</TabsTrigger>
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
                  {availableModels.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
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
                  void refreshOverview(activeBatchId);
                  void refreshCandidates(activeBatchId, 1);
                  void refreshThresholdStatus(activeBatchId);
                  void refreshReviewHistory(activeBatchId, historyDecision, 1);
                  void refreshBatches();
                }}
              >
                Refresh
              </Button>
              <select
                className="rounded-md border bg-background px-3 py-2 text-sm"
                value={activeBatchId || ""}
                onChange={(e) => {
                  void handleBatchChange(e.target.value);
                }}
              >
                <option value="" disabled>
                  Chọn batch
                </option>
                {batches.map((batch) => (
                  <option key={batch.batch_id} value={batch.batch_id}>
                    {batch.batch_id} · {batch.status || "-"}
                  </option>
                ))}
              </select>
              <Button variant="destructive" onClick={handleClearCurrentBatch} disabled={!activeBatchId}>
                Clear current batch
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
              <h3 className="font-medium">Candidate review</h3>
              <div className="text-sm text-muted-foreground">
                {candidateTotal} candidates · page {candidatePage} · size {candidatePageSize}
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Chọn Toxic/Clean/Remove để lưu trực tiếp vào DB cho các phase tiếp theo.
            </p>
            <div className="grid gap-2 md:grid-cols-3">
              <label className="text-sm">
                Min score
                <Input
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={minCandidateScore}
                  onChange={(e: ChangeEvent<HTMLInputElement>) => {
                    const raw = Number(e.target.value);
                    const clamped = Number.isFinite(raw) ? Math.max(0, Math.min(1, raw)) : 0;
                    setMinCandidateScore(clamped);
                  }}
                  className="mt-1"
                />
              </label>
              <label className="text-sm">
                Sort by score
                <select
                  className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
                  value={scoreSortMode}
                  onChange={(e) => setScoreSortMode(e.target.value as "high_to_low" | "low_to_high")}
                >
                  <option value="high_to_low">High → Low</option>
                  <option value="low_to_high">Low → High</option>
                </select>
              </label>
              <div className="text-sm text-muted-foreground md:self-end">
                Hiển thị: <b>{filteredCandidates.length}</b> / {candidates.length}
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="outline" onClick={handleSelectAllCandidates} disabled={filteredCandidates.length === 0}>
                Select all (filtered)
              </Button>
              <Button size="sm" variant="outline" onClick={handleUnselectAllCandidates} disabled={selectedCandidateIds.length === 0}>
                Unselect all
              </Button>
            </div>
            <div className="space-y-1.5 max-h-[34rem] overflow-auto pr-1">
              {filteredCandidates.map((item) => (
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
              {filteredCandidates.length === 0 && (
                <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  Không có candidate nào thỏa bộ lọc score hiện tại.
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
              <Button size="icon" variant="outline" onClick={() => refreshCandidates(activeBatchId, candidatePage)}>
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
                <Button size="sm" variant="outline" onClick={() => refreshReviewHistory(activeBatchId, historyDecision, reviewHistoryPage)}>
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

        <TabsContent value="step2" className="space-y-4">
          <Card className="p-6 text-center space-y-4">
            <h3 className="text-xl font-semibold">Kiểm tra threshold (test max=10)</h3>
            <p className="text-4xl font-bold">
              {thresholdStatus?.accepted_count ?? 0} / {thresholdStatus?.target_max_test_stage ?? 10}
            </p>
            <Progress value={thresholdProgress} />
            <p className="text-sm text-muted-foreground">
              {thresholdStatus?.is_ready ? "Đủ điều kiện retrain" : "Chưa đủ điều kiện retrain"}
            </p>
            <div className="flex justify-center gap-2">
              <Button onClick={handleExportBundle}>Tải dataset xuống</Button>
              <Button variant="outline" onClick={() => refreshThresholdStatus(activeBatchId)}>
                Refresh
              </Button>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="step3" className="space-y-4">
          <Card className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-medium">Bước 1 — Tải dataset bundle</h3>
              <Button onClick={handleExportBundle}>Download</Button>
            </div>
            <p className="text-sm text-muted-foreground">Bundle mặc định gồm accepted + candidate. Bạn có thể bật thêm unused/discarded nếu cần phân tích sâu.</p>
            <p className="text-xs text-muted-foreground">
              Export bundle lấy dữ liệu từ MLFlow DB theo batch hiện tại (accepted/candidate/unused). Endpoint này không tự merge với victsd_gold.
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
                placeholder="run_20260408_retrain"
                value={artifactRunName}
                onChange={(e: ChangeEvent<HTMLInputElement>) => setArtifactRunName(e.target.value)}
              />
              <Input
                placeholder="models/options/phobert/run_20260408_retrain"
                value={artifactPath}
                onChange={(e: ChangeEvent<HTMLInputElement>) => setArtifactPath(e.target.value)}
              />
            </div>
            <div className="flex gap-2">
              <Button onClick={handleImportArtifact}>Import metadata</Button>
              <Button variant="outline" onClick={() => refreshCompare()}>
                Refresh compare source
              </Button>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="step4" className="space-y-4">
          <Card className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-medium">Pipeline tự động DigitalOcean (placeholder)</h3>
              <Button onClick={handleTriggerDO}>Kích hoạt DO Pipeline</Button>
            </div>
            <div className="grid gap-2 md:grid-cols-2">
              {[
                "Trigger VM",
                "Upload data + train files",
                "Train trên VM",
                "Lưu artifact",
                "Tải về / upload destination",
                "Destroy VM",
              ].map((stage) => (
                <div key={stage} className="rounded-md border p-3 text-sm">
                  {stage}
                </div>
              ))}
            </div>
            <div className="rounded-md border p-3 text-sm space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-medium">Training log</span>
                <Button size="sm" variant="outline" onClick={() => refreshDOStatus()}>
                  Refresh log
                </Button>
              </div>
              <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(doStatus || { status: "idle" }, null, 2)}</pre>
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
              <p className="text-xs text-muted-foreground">Nút promote hiện đang chạy ở chế độ placeholder.</p>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
