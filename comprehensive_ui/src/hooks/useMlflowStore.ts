import { useCallback, useState } from "react";

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, message: string, detail: unknown) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

const parseJsonResponse = async <T,>(response: Response): Promise<T> => {
  const raw = await response.text();
  if (!response.ok) {
    let detail: unknown = null;
    let message = raw || "API request failed";

    try {
      const parsed = raw ? (JSON.parse(raw) as { detail?: unknown; message?: unknown }) : null;
      detail = parsed?.detail ?? null;
      if (typeof parsed?.detail === "string") {
        message = parsed.detail;
      } else if (typeof parsed?.message === "string") {
        message = parsed.message;
      }
    } catch {
      detail = null;
    }

    throw new ApiError(response.status, message, detail);
  }
  return JSON.parse(raw) as T;
};

export interface MlflowOverview {
  active_batch_id: string;
  model_name?: string;
  status?: string;
  source_job_id?: string;
  last_run_at?: string;
  pipeline_counts: {
    crawled: number;
    inferred: number;
    accepted: number;
    candidate: number;
    discarded: number;
  };
}

export interface MlflowCandidate {
  id: number;
  batch_id: string;
  url: string;
  url_hash: string;
  segment_id?: string | null;
  domain_category?: string | null;
  text: string;
  score?: number | null;
  pseudo_label?: number | null;
  label_source?: string | null;
  label_confidence?: string | null;
  gate_bucket: string;
  verification_status: string;
  reviewed_at?: string | null;
  created_at?: string | null;
}

export interface MlflowBatchSummary {
  batch_id: string;
  model_id?: string;
  status?: string;
  source_job_id?: string;
  created_at?: string;
  completed_at?: string | null;
  counts?: Record<string, number>;
}

export interface MlflowThresholdStatus {
  batch_id: string;
  scope?: "all_batches" | "batch";
  accepted_count: number;
  accepted_count_current_batch?: number;
  target_max_test_stage: number;
  remaining_to_target: number;
  is_ready: boolean;
}

export type MlflowUnusedScope = "all" | "auto_discarded" | "manual_rejected";

export interface MlflowComparePayload {
  current?: {
    model?: string | null;
    metrics?: Record<string, number | null>;
    created_at?: string | null;
  };
  candidate?: {
    model?: string | null;
    artifact_path?: string | null;
    notes?: string | null;
    metrics?: Record<string, number | null>;
    created_at?: string | null;
  };
  gate_checks?: Array<{ name: string; delta: number | null; passed: boolean }>;
  promotion_enabled?: boolean;
  promotion_mode?: string;
}

export interface MlflowDeletedRows {
  mlflow_do_run: number;
  mlflow_comment_item: number;
  mlflow_crawl_batch: number;
  mlflow_training_artifact: number;
}

export interface MlflowClearBatchResponse {
  scope: "batch";
  batch_id: string;
  deleted_rows: MlflowDeletedRows;
}

export interface MlflowClearAllResponse {
  scope: "all";
  deleted_rows: MlflowDeletedRows;
}

export type MlflowIngestStage = "idle" | "crawl" | "inference" | "finalize" | "completed" | "error";

export function useMlflowStore() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeBatchId, setActiveBatchId] = useState<string | null>(null);
  const [overview, setOverview] = useState<MlflowOverview | null>(null);
  const [candidates, setCandidates] = useState<MlflowCandidate[]>([]);
  const [candidateTotal, setCandidateTotal] = useState(0);
  const [candidatePage, setCandidatePage] = useState(1);
  const [candidatePageSize] = useState(25);
  const [thresholdStatus, setThresholdStatus] = useState<MlflowThresholdStatus | null>(null);
  const [batches, setBatches] = useState<MlflowBatchSummary[]>([]);
  const [reviewHistory, setReviewHistory] = useState<MlflowCandidate[]>([]);
  const [reviewHistoryTotal, setReviewHistoryTotal] = useState(0);
  const [reviewHistoryPage, setReviewHistoryPage] = useState(1);
  const [comparePayload, setComparePayload] = useState<MlflowComparePayload | null>(null);
  const [lastBundlePath, setLastBundlePath] = useState<string | null>(null);
  const [requiredZipContents, setRequiredZipContents] = useState<string[]>([]);
  const [doRunId, setDoRunId] = useState<string | null>(null);
  const [doStatus, setDoStatus] = useState<Record<string, unknown> | null>(null);
  const [hasNoBatch, setHasNoBatch] = useState(false);
  const [ingestStage, setIngestStage] = useState<MlflowIngestStage>("idle");
  const [ingestProgress, setIngestProgress] = useState(0);
  const [ingestStageMessage, setIngestStageMessage] = useState<string | null>(null);

  const run = useCallback(async <T,>(fn: () => Promise<T>) => {
    setLoading(true);
    setError(null);
    try {
      return await fn();
    } catch (err) {
      if (
        err instanceof ApiError &&
        err.status === 404 &&
        (err.message === "No mlflow batch found" || err.detail === "No mlflow batch found")
      ) {
        setHasNoBatch(true);
        setOverview(null);
        setCandidates([]);
        setCandidateTotal(0);
        setThresholdStatus(null);
        setComparePayload(null);
      } else {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
      }
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshBatches = useCallback(
    async (limit = 50) => {
      return run(async () => {
        const payload = await parseJsonResponse<{ items: MlflowBatchSummary[]; total: number }>(
          await fetch(buildApiUrl(`/api/mlflow/batches?limit=${limit}`)),
        );
        setBatches(payload.items || []);
        return payload;
      });
    },
    [run],
  );

  const refreshOverview = useCallback(
    async (batchId?: string | null) => {
      return run(async () => {
        const qs = batchId
          ? `?batch_id=${encodeURIComponent(batchId)}&strict_batch=true`
          : "";
        try {
          const payload = await parseJsonResponse<MlflowOverview>(await fetch(buildApiUrl(`/api/mlflow/overview${qs}`)));
          setHasNoBatch(false);
          setOverview(payload);
          setActiveBatchId(payload.active_batch_id);
          return payload;
        } catch (err) {
          if (
            err instanceof ApiError &&
            err.status === 404 &&
            (err.message === "No mlflow batch found" || err.detail === "No mlflow batch found")
          ) {
            setHasNoBatch(true);
            setOverview(null);
            setActiveBatchId(null);
            setCandidates([]);
            setCandidateTotal(0);
            setThresholdStatus(null);
            setComparePayload(null);
            return null;
          }
          throw err;
        }
      });
    },
    [run],
  );

  const refreshCandidates = useCallback(
    async (batchId?: string | null, page = candidatePage) => {
      return run(async () => {
        const useBatch = batchId || activeBatchId;
        if (!useBatch) return;
        const qs = `?batch_id=${encodeURIComponent(useBatch)}&page=${page}&page_size=${candidatePageSize}&strict_batch=true`;
        const payload = await parseJsonResponse<{ items: MlflowCandidate[]; total: number; page: number }>(
          await fetch(buildApiUrl(`/api/mlflow/candidates${qs}`)),
        );
        setCandidates(payload.items || []);
        setCandidateTotal(payload.total || 0);
        setCandidatePage(payload.page || page);
      });
    },
    [activeBatchId, candidatePage, candidatePageSize, run],
  );

  const refreshThresholdStatus = useCallback(
    async (batchId?: string | null) => {
      return run(async () => {
        const useBatch = batchId || activeBatchId;
        if (!useBatch) return;
        const qs = `?batch_id=${encodeURIComponent(useBatch)}&strict_batch=true`;
        const payload = await parseJsonResponse<MlflowThresholdStatus>(
          await fetch(buildApiUrl(`/api/mlflow/threshold-status${qs}`)),
        );
        setThresholdStatus(payload);
        return payload;
      });
    },
    [activeBatchId, run],
  );

  const refreshReviewHistory = useCallback(
    async (batchId?: string | null, decision = "all", page = reviewHistoryPage) => {
      return run(async () => {
        const useBatch = batchId || activeBatchId;
        if (!useBatch) return;
        const qs = `?batch_id=${encodeURIComponent(useBatch)}&decision=${encodeURIComponent(decision)}&page=${page}&page_size=${candidatePageSize}&strict_batch=true`;
        const payload = await parseJsonResponse<{ items: MlflowCandidate[]; total: number; page: number }>(
          await fetch(buildApiUrl(`/api/mlflow/review-history${qs}`)),
        );
        setReviewHistory(payload.items || []);
        setReviewHistoryTotal(payload.total || 0);
        setReviewHistoryPage(payload.page || page);
        return payload;
      });
    },
    [activeBatchId, reviewHistoryPage, candidatePageSize, run],
  );

  const exportBundle = useCallback(
    async (batchId?: string | null, options?: { includeUnused?: boolean; unusedScope?: MlflowUnusedScope }) => {
      return run(async () => {
        const useBatch = batchId || activeBatchId;
        if (!useBatch) throw new Error("No active batch to export");
        const payload = await parseJsonResponse<{
          bundle_path: string;
          required_zip_contents: string[];
          count: number;
          candidate_count: number;
          unused_count: number;
          include_unused: boolean;
          unused_scope: MlflowUnusedScope;
        }>(
          await fetch(buildApiUrl("/api/mlflow/manual/export-bundle"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              batch_id: useBatch,
              include_unused: Boolean(options?.includeUnused),
              unused_scope: options?.unusedScope || "all",
            }),
          }),
        );
        setLastBundlePath(payload.bundle_path);
        setRequiredZipContents(payload.required_zip_contents || []);
        return payload;
      });
    },
    [activeBatchId, run],
  );

  const importArtifact = useCallback(
    async (runName: string, artifactPath: string, notes?: string) => {
      return run(async () => {
        return parseJsonResponse<{ import_id: number; status: string }>(
          await fetch(buildApiUrl("/api/mlflow/manual/import-artifact"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ run_name: runName, artifact_path: artifactPath, notes }),
          }),
        );
      });
    },
    [run],
  );

  const triggerDO = useCallback(
    async () => {
      return run(async () => {
        const payload = await parseJsonResponse<{ run_id: string }>(
          await fetch(buildApiUrl("/api/mlflow/do/trigger"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ batch_id: activeBatchId, provider: "digitalocean", dry_run: true }),
          }),
        );
        setDoRunId(payload.run_id);
        await refreshDOStatus(payload.run_id);
        return payload;
      });
    },
    [activeBatchId, run],
  );

  const refreshDOStatus = useCallback(
    async (runId?: string | null) => {
      return run(async () => {
        const target = runId || doRunId;
        if (!target) return;
        const payload = await parseJsonResponse<Record<string, unknown>>(
          await fetch(buildApiUrl(`/api/mlflow/do/status?run_id=${encodeURIComponent(target)}`)),
        );
        setDoStatus(payload);
        return payload;
      });
    },
    [doRunId, run],
  );

  const refreshCompare = useCallback(async () => {
    return run(async () => {
      const payload = await parseJsonResponse<MlflowComparePayload>(
        await fetch(buildApiUrl("/api/mlflow/compare/latest")),
      );
      setComparePayload(payload);
      return payload;
    });
  }, [run]);

  const promote = useCallback(
    async (candidateModel: string) => {
      return run(async () => {
        return parseJsonResponse<{ status: string; message?: string }>(
          await fetch(buildApiUrl("/api/mlflow/promote"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ candidate_model: candidateModel }),
          }),
        );
      });
    },
    [run],
  );

  const reviewCandidates = useCallback(
    async (
      updates: Array<{
        id: number;
        action?: "include_toxic" | "include_clean" | "drop";
        decision?: "accept" | "reject";
        pseudo_label?: 0 | 1;
      }>,
    ) => {
      return run(async () => {
        const payload = await parseJsonResponse<{ updated: number }>(
          await fetch(buildApiUrl("/api/mlflow/candidates/review"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ updates }),
          }),
        );
        await refreshOverview(activeBatchId);
        await refreshCandidates(activeBatchId, candidatePage);
        await refreshThresholdStatus(activeBatchId);
        await refreshReviewHistory(activeBatchId, "all", 1);
        return payload;
      });
    },
    [activeBatchId, candidatePage, refreshCandidates, refreshOverview, refreshReviewHistory, refreshThresholdStatus, run],
  );

  const clearMlflowBatch = useCallback(
    async (batchId?: string | null) => {
      return run(async () => {
        const useBatch = batchId || activeBatchId;
        if (!useBatch) {
          throw new Error("No active batch to clear");
        }

        const payload = await parseJsonResponse<MlflowClearBatchResponse>(
          await fetch(buildApiUrl("/api/mlflow/clear-batch"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ batch_id: useBatch }),
          }),
        );

        await refreshBatches();
        let nextBatchId: string | null = null;
        try {
          const latestOverview = await refreshOverview();
          nextBatchId = latestOverview?.active_batch_id || null;
        } catch {
          nextBatchId = null;
        }

        if (nextBatchId) {
          await refreshCandidates(nextBatchId, 1);
          await refreshThresholdStatus(nextBatchId);
          await refreshReviewHistory(nextBatchId, "all", 1);
        } else {
          setCandidates([]);
          setCandidateTotal(0);
          setThresholdStatus(null);
          setReviewHistory([]);
          setReviewHistoryTotal(0);
        }

        return payload;
      });
    },
    [activeBatchId, refreshBatches, refreshCandidates, refreshOverview, refreshReviewHistory, refreshThresholdStatus, run],
  );

  const clearMlflowAll = useCallback(
    async (confirmToken: string) => {
      return run(async () => {
        const payload = await parseJsonResponse<MlflowClearAllResponse>(
          await fetch(buildApiUrl("/api/mlflow/clear-all"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ confirm_token: confirmToken }),
          }),
        );

        await refreshBatches();
        try {
          await refreshOverview();
        } catch {
          // expected when all batches are cleared
        }
        setCandidates([]);
        setCandidateTotal(0);
        setThresholdStatus(null);
        setReviewHistory([]);
        setReviewHistoryTotal(0);
        return payload;
      });
    },
    [refreshBatches, refreshOverview, run],
  );

  const ingest = useCallback(
    async (urls: string[], modelName?: string) => {
      setIngestStage("crawl");
      setIngestProgress(8);
      setIngestStageMessage("Đang crawl comment...");

      let stageTimer: ReturnType<typeof setInterval> | null = null;
      const startStageRamp = () => {
        stageTimer = setInterval(() => {
          setIngestProgress((prev) => {
            if (prev < 45) {
              setIngestStage("crawl");
              setIngestStageMessage("Đang crawl comment...");
              return Math.min(45, prev + 3);
            }
            if (prev < 82) {
              setIngestStage("inference");
              setIngestStageMessage("Đang infer và gate candidates...");
              return Math.min(82, prev + 2);
            }
            setIngestStage("finalize");
            setIngestStageMessage("Đang hoàn tất batch...");
            return Math.min(95, prev + 1);
          });
        }, 900);
      };

      const stopStageRamp = () => {
        if (stageTimer) {
          clearInterval(stageTimer);
          stageTimer = null;
        }
      };

      startStageRamp();
      try {
        const payload = await run(async () => {
          const ingestPayload = await parseJsonResponse<{
            batch_id: string;
            model_name?: string;
            counts?: Record<string, number>;
            crawl_summary?: {
              status_counts?: Record<string, number>;
              timeout_count?: number;
              total_urls?: number;
            };
          }>(
            await fetch(buildApiUrl("/api/mlflow/ingest"), {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                urls,
                options: {
                  model_name: modelName || undefined,
                },
              }),
            }),
          );
          const b = ingestPayload.batch_id;
          setIngestStage("finalize");
          setIngestStageMessage("Đang đồng bộ trạng thái từ backend...");
          setIngestProgress((prev) => Math.max(prev, 90));
          setActiveBatchId(b);
          await refreshOverview(b);
          await refreshCandidates(b, 1);
          await refreshThresholdStatus(b);
          await refreshReviewHistory(b, "all", 1);
          await refreshBatches();
          return ingestPayload;
        });

        stopStageRamp();
        setIngestStage("completed");
        setIngestProgress(100);
        setIngestStageMessage("Hoàn tất ingest + infer + gate.");
        return payload;
      } catch (err) {
        stopStageRamp();
        setIngestStage("error");
        setIngestStageMessage(err instanceof Error ? err.message : "Ingest thất bại");
        throw err;
      }
    },
    [refreshBatches, refreshCandidates, refreshOverview, refreshReviewHistory, refreshThresholdStatus, run],
  );

  return {
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
    doRunId,
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
    setError,
  };
}
