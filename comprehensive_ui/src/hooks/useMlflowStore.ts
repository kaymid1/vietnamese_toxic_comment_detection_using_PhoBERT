import { useCallback, useEffect, useRef, useState } from "react";

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");
const API_BASE_WITHOUT_API_SUFFIX = API_BASE.replace(/\/api$/i, "");
const API_FALLBACK_BASES = Array.from(
  new Set(
    ["", API_BASE, "http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:8001", "http://localhost:8001"]
      .map((value) => value.trim().replace(/\/+$/, ""))
      .filter(Boolean),
  ),
);
const API_FALLBACK_FAILURE_COOLDOWN_MS = 30000;
let lastSuccessfulApiBase: string | null = null;
const apiBaseFailureUntil = new Map<string, number>();

const buildApiUrlFromBase = (base: string, path: string) => {
  if (/^https?:\/\//i.test(path)) return path;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (!base) return normalizedPath;

  const baseEndsWithApi = /\/api$/i.test(base);
  const pathStartsWithApi = /^\/api(?:\/|$)/i.test(normalizedPath);
  if (baseEndsWithApi && pathStartsWithApi) {
    return `${base.replace(/\/api$/i, "")}${normalizedPath}`;
  }

  return `${base}${normalizedPath}`;
};

const isNetworkFetchError = (error: unknown) =>
  error instanceof TypeError && /failed to fetch|networkerror|load failed/i.test(error.message.toLowerCase());

export const buildApiUrl = (path: string) => {
  if (/^https?:\/\//i.test(path)) return path;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (!API_BASE) return normalizedPath;
  const baseEndsWithApi = /\/api$/i.test(API_BASE);
  const pathStartsWithApi = /^\/api(?:\/|$)/i.test(normalizedPath);
  if (baseEndsWithApi && pathStartsWithApi) {
    return `${API_BASE_WITHOUT_API_SUFFIX}${normalizedPath}`;
  }
  return `${API_BASE}${normalizedPath}`;
};

export const fetchApiWithFallback = async (urlOrPath: string, init?: RequestInit): Promise<Response> => {
  const normalizedPath = /^https?:\/\//i.test(urlOrPath)
    ? (() => {
        try {
          const parsed = new URL(urlOrPath);
          return `${parsed.pathname}${parsed.search}`;
        } catch {
          return urlOrPath;
        }
      })()
    : urlOrPath;

  const now = Date.now();
  const candidateBases = [lastSuccessfulApiBase || "", ...API_FALLBACK_BASES, API_BASE]
    .map((value) => value.trim())
    .filter((value, index, values) => values.indexOf(value) === index);

  let lastError: unknown = null;
  const tried: string[] = [];

  for (const base of candidateBases) {
    const blockedUntil = apiBaseFailureUntil.get(base) || 0;
    if (base !== (lastSuccessfulApiBase || "") && blockedUntil > now) {
      continue;
    }
    const url = /^https?:\/\//i.test(urlOrPath) ? buildApiUrlFromBase(base, normalizedPath) : buildApiUrlFromBase(base, urlOrPath);
    tried.push(url);
    try {
      const response = await fetch(url, init);
      lastSuccessfulApiBase = base;
      apiBaseFailureUntil.delete(base);
      return response;
    } catch (error) {
      if (!isNetworkFetchError(error)) throw error;
      lastError = error;
      apiBaseFailureUntil.set(base, Date.now() + API_FALLBACK_FAILURE_COOLDOWN_MS);
      if (lastSuccessfulApiBase === base) {
        lastSuccessfulApiBase = null;
      }
    }
  }

  throw new Error(
    `Cannot reach backend API for ${urlOrPath}. Tried: ${tried.join(" | ")}. ${(lastError as Error | null)?.message || ""}`.trim(),
  );
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

interface UseMlflowStoreOptions {
  adminToken?: string | null;
  onUnauthorized?: () => void;
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
  has_data?: boolean;
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
  constructiveness_score?: number | null;
  constructiveness_label?: number | null;
  constructiveness_confidence?: string | null;
  selected_for_training?: number | null;
  training_review_status?: string | null;
  is_locked?: number | null;
  label_source?: string | null;
  label_confidence?: string | null;
  gate_bucket: string;
  verification_status: string;
  reviewed_at?: string | null;
  created_at?: string | null;
}

export interface MlflowTrainingPreview {
  scope: "batch" | "all_batches";
  batch_id?: string | null;
  items: MlflowCandidate[];
  total: number;
  page: number;
  page_size: number;
  counts: {
    selected: number;
    selected_toxic: number;
    selected_clean: number;
    candidate: number;
    removed: number;
  };
  balance: {
    strategy: string;
    balanced_count: number;
    toxic_available: number;
    clean_available: number;
  };
  constructiveness: {
    included: number;
    masked: number;
  };
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
  has_data?: boolean;
}

export interface MlflowCrawlHistoryItem {
  batch_id: string;
  url: string;
  url_hash: string;
  domain_category?: string | null;
  segment_count: number;
  accepted_count: number;
  candidate_count: number;
  discarded_count: number;
  last_seen_at?: string | null;
}

export type MlflowUnusedScope = "all" | "auto_discarded" | "manual_rejected";
export type MlflowExportScope = "all_batches" | "batch";

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
    source_run_id?: string | null;
    raw_metrics?: MlflowKaggleMetrics | null;
    created_at?: string | null;
  };
  gate_checks?: Array<{ name: string; delta: number | null; passed: boolean }>;
  promotion_enabled?: boolean;
  promotion_mode?: string;
}

export interface MlflowKaggleMetrics {
  f1_toxic?: number | null;
  macro_f1?: number | null;
  accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  source_member?: string | null;
  run_name?: string | null;
  mode?: string | null;
  splits?: Record<string, Record<string, number>>;
}

export interface MlflowKaggleStatus {
  run_id?: string;
  batch_id?: string | null;
  provider?: string;
  gpu_profile?: string | null;
  compute_mode?: string;
  model_kind?: string;
  training_mode?: string;
  base_model?: string | null;
  status?: string;
  current_stage?: string | null;
  logs?: string[];
  log_events?: Array<Record<string, unknown>>;
  stages?: string[];
  artifact_uri?: string | null;
  artifact_kind?: string;
  artifact_download_url?: string | null;
  artifact_checksum?: string | null;
  metrics?: MlflowKaggleMetrics | null;
  error_message?: string | null;
  run_mode?: string;
  status_source?: string;
  stage_timestamps?: Record<string, string | null>;
  created_at?: string | null;
  updated_at?: string | null;
  job_id?: string | null;
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

export interface MlflowDOPreflight {
  ready: boolean;
  missing: string[];
  warnings: string[];
  checks: Record<string, boolean>;
  config: Record<string, unknown>;
  checked_at?: string;
}

export type MlflowIngestStage = "idle" | "crawl" | "inference" | "finalize" | "completed" | "error";

export interface MlflowDOTriggerOptions {
  computeMode?: "gpu" | "cpu" | "local_m1";
  modelKind?: "phobert" | "lr_smoke";
  trainingMode?: "retrain" | "finetune";
  baseModel?: string;
  cpuProfile?: string;
  gpuProfile?: string;
  dryRun?: boolean;
}

export interface MlflowImportModelZipResponse {
  status: string;
  model_id: string;
  model_name: string;
  model_type: string;
  model_path: string;
  validated: boolean;
}

const DO_TERMINAL_STATUSES = new Set(["completed", "failed", "dry_run", "placeholder"]);
const DO_POLL_INTERVAL_MS = 4000;
const DO_MAX_POLL_ATTEMPTS = 21600;

export function useMlflowStore(options: UseMlflowStoreOptions = {}) {
  const { adminToken, onUnauthorized } = options;
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
  const [trainingPreview, setTrainingPreview] = useState<MlflowTrainingPreview | null>(null);
  const [reviewHistoryTotal, setReviewHistoryTotal] = useState(0);
  const [reviewHistoryPage, setReviewHistoryPage] = useState(1);
  const [crawlHistory, setCrawlHistory] = useState<MlflowCrawlHistoryItem[]>([]);
  const [crawlHistoryTotal, setCrawlHistoryTotal] = useState(0);
  const [crawlHistoryPage, setCrawlHistoryPage] = useState(1);
  const [crawlHistoryUnavailable, setCrawlHistoryUnavailable] = useState(false);
  const [comparePayload, setComparePayload] = useState<MlflowComparePayload | null>(null);
  const [lastBundlePath, setLastBundlePath] = useState<string | null>(null);
  const [requiredZipContents, setRequiredZipContents] = useState<string[]>([]);
  const [doRunId, setDoRunId] = useState<string | null>(null);
  const [doStatus, setDoStatus] = useState<MlflowKaggleStatus | null>(null);
  const [doPreflight, setDoPreflight] = useState<MlflowDOPreflight | null>(null);
  const [hasNoBatch, setHasNoBatch] = useState(false);
  const [ingestStage, setIngestStage] = useState<MlflowIngestStage>("idle");
  const [ingestProgress, setIngestProgress] = useState(0);
  const [ingestStageMessage, setIngestStageMessage] = useState<string | null>(null);
  const doPollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const doPollAttemptRef = useRef(0);

  const authorizedFetch = useCallback(
    async (url: string, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      if (adminToken) {
        headers.set("Authorization", `Bearer ${adminToken}`);
      }
      const response = await fetchApiWithFallback(url, { ...init, headers });
      if (response.status === 401) {
        onUnauthorized?.();
      }
      return response;
    },
    [adminToken, onUnauthorized],
  );

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
        setReviewHistory([]);
        setReviewHistoryTotal(0);
        setCrawlHistory([]);
        setCrawlHistoryTotal(0);
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
          await authorizedFetch(buildApiUrl(`/api/mlflow/batches?limit=${limit}`)),
        );
        setBatches(payload.items || []);
        return payload;
      });
    },
    [authorizedFetch, run],
  );

  const refreshOverview = useCallback(
    async (batchId?: string | null) => {
      return run(async () => {
        const qs = batchId
          ? `?batch_id=${encodeURIComponent(batchId)}&strict_batch=true`
          : "";
        try {
          const payload = await parseJsonResponse<MlflowOverview>(await authorizedFetch(buildApiUrl(`/api/mlflow/overview${qs}`)));
          const isNoBatch = payload.has_data === false || !payload.active_batch_id;
          setHasNoBatch(isNoBatch);
          setOverview(payload);
          setActiveBatchId(payload.active_batch_id || null);
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
            setReviewHistory([]);
            setReviewHistoryTotal(0);
            setCrawlHistory([]);
            setCrawlHistoryTotal(0);
            return null;
          }
          throw err;
        }
      });
    },
    [authorizedFetch, run],
  );

  const refreshCandidates = useCallback(
    async (batchId?: string | null, page = candidatePage, scope: "batch" | "all_batches" = "all_batches") => {
      return run(async () => {
        const useBatch = batchId || activeBatchId;
        if (scope === "batch" && !useBatch) return;
        const query = new URLSearchParams({
          page: String(page),
          page_size: String(candidatePageSize),
          scope,
        });
        if (scope === "batch" && useBatch) {
          query.set("batch_id", useBatch);
          query.set("strict_batch", "true");
        }

        const payload = await parseJsonResponse<{ items: MlflowCandidate[]; total: number; page: number }>(
          await authorizedFetch(buildApiUrl(`/api/mlflow/candidates?${query.toString()}`)),
        );
        setCandidates(payload.items || []);
        setCandidateTotal(payload.total || 0);
        setCandidatePage(payload.page || page);
      });
    },
    [activeBatchId, authorizedFetch, candidatePage, candidatePageSize, run],
  );

  const refreshThresholdStatus = useCallback(
    async (batchId?: string | null) => {
      return run(async () => {
        const useBatch = batchId || activeBatchId;
        const qs = useBatch
          ? `?batch_id=${encodeURIComponent(useBatch)}&strict_batch=true`
          : "";
        const payload = await parseJsonResponse<MlflowThresholdStatus>(
          await authorizedFetch(buildApiUrl(`/api/mlflow/threshold-status${qs}`)),
        );
        if (payload.has_data === false) {
          setHasNoBatch(true);
        }
        setThresholdStatus(payload);
        return payload;
      });
    },
    [activeBatchId, authorizedFetch, run],
  );

  const refreshTrainingPreview = useCallback(
    async (page = 1, scope: "batch" | "all_batches" = "all_batches") => {
      return run(async () => {
        const query = new URLSearchParams({
          page: String(page),
          page_size: "50",
          scope,
        });
        if (scope === "batch" && activeBatchId) {
          query.set("batch_id", activeBatchId);
          query.set("strict_batch", "true");
        }
        const payload = await parseJsonResponse<MlflowTrainingPreview>(
          await authorizedFetch(buildApiUrl(`/api/mlflow/training-preview?${query.toString()}`)),
        );
        setTrainingPreview(payload);
        return payload;
      });
    },
    [activeBatchId, authorizedFetch, run],
  );

  const reviewTrainingPreview = useCallback(
    async (
      updates: Array<{
        id: number;
        selected_for_training?: boolean;
        pseudo_label?: 0 | 1;
        constructiveness_label?: 0 | 1;
        clear_constructiveness?: boolean;
        lock_state?: boolean;
      }>,
    ) => {
      return run(async () => {
        const payload = await parseJsonResponse<{ status: string; updated: number; skipped_locked?: number }>(
          await authorizedFetch(buildApiUrl("/api/mlflow/training-preview/review"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ updates }),
          }),
        );
        await refreshTrainingPreview(trainingPreview?.page || 1, "all_batches");
        return payload;
      });
    },
    [authorizedFetch, refreshTrainingPreview, run, trainingPreview?.page],
  );

  const refreshReviewHistory = useCallback(
    async (batchId?: string | null, decision = "all", page = reviewHistoryPage, scope: "batch" | "all_batches" = "all_batches") => {
      return run(async () => {
        const useBatch = batchId || activeBatchId;
        if (scope === "batch" && !useBatch) return;

        const query = new URLSearchParams({
          decision: String(decision),
          page: String(page),
          page_size: String(candidatePageSize),
          scope,
        });
        if (scope === "batch" && useBatch) {
          query.set("batch_id", useBatch);
          query.set("strict_batch", "true");
        }

        const payload = await parseJsonResponse<{ items: MlflowCandidate[]; total: number; page: number }>(
          await authorizedFetch(buildApiUrl(`/api/mlflow/review-history?${query.toString()}`)),
        );
        setReviewHistory(payload.items || []);
        setReviewHistoryTotal(payload.total || 0);
        setReviewHistoryPage(payload.page || page);
        return payload;
      });
    },
    [activeBatchId, authorizedFetch, reviewHistoryPage, candidatePageSize, run],
  );

  const refreshCrawlHistory = useCallback(
    async (page = crawlHistoryPage, options?: { allowUnavailableFallback?: boolean }) => {
      return run(async () => {
        if (crawlHistoryUnavailable && !options?.allowUnavailableFallback) {
          const fallback = { items: [] as MlflowCrawlHistoryItem[], total: 0, page };
          setCrawlHistory(fallback.items);
          setCrawlHistoryTotal(0);
          setCrawlHistoryPage(page);
          return fallback;
        }

        const query = new URLSearchParams({
          page: String(page),
          page_size: String(candidatePageSize),
        });

        try {
          const payload = await parseJsonResponse<{ items: MlflowCrawlHistoryItem[]; total: number; page: number }>(
            await authorizedFetch(buildApiUrl(`/api/mlflow/crawl-history?${query.toString()}`)),
          );
          setCrawlHistoryUnavailable(false);
          setCrawlHistory(payload.items || []);
          setCrawlHistoryTotal(payload.total || 0);
          setCrawlHistoryPage(payload.page || page);
          return payload;
        } catch (err) {
          if (err instanceof ApiError && err.status === 404) {
            setCrawlHistoryUnavailable(true);
            const fallback = { items: [] as MlflowCrawlHistoryItem[], total: 0, page };
            setCrawlHistory(fallback.items);
            setCrawlHistoryTotal(0);
            setCrawlHistoryPage(page);
            return fallback;
          }
          throw err;
        }
      });
    },
    [authorizedFetch, crawlHistoryPage, candidatePageSize, crawlHistoryUnavailable, run],
  );

  const exportBundle = useCallback(
    async (
      batchId?: string | null,
      options?: {
        includeUnused?: boolean;
        unusedScope?: MlflowUnusedScope;
        scope?: MlflowExportScope;
        modelKind?: "phobert" | "lr_smoke";
        trainingMode?: "retrain" | "finetune";
        balanceStrategy?: "balanced_50_50" | "all";
        includeBaseModel?: boolean;
        baseModel?: string;
      },
    ) => {
      return run(async () => {
        const useBatch = batchId || activeBatchId;
        const exportScope = options?.scope || "all_batches";
        if (exportScope === "batch" && !useBatch) throw new Error("No active batch to export in batch scope");
        const payload = await parseJsonResponse<{
          bundle_path: string;
          download_url: string;
          scope: MlflowExportScope;
          batch_id: string | null;
          required_zip_contents: string[];
          count: number;
          count_before_balance?: number;
          candidate_count: number;
          unused_count: number;
          model_kind?: "phobert" | "lr_smoke";
          training_mode?: "retrain" | "finetune";
          balance_strategy?: "balanced_50_50" | "all";
          balance_stats?: Record<string, unknown>;
          constructiveness?: { included: number; masked: number };
          base_model_bundled?: boolean;
          include_unused: boolean;
          unused_scope: MlflowUnusedScope;
          merge_stats?: {
            base_train_count: number;
            added_to_train: number;
            skipped_duplicate: number;
            skipped_empty: number;
            final_train_count: number;
          };
        }>(
          await authorizedFetch(buildApiUrl("/api/mlflow/manual/export-bundle"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              batch_id: useBatch,
              scope: exportScope,
              bundle_profile: options?.modelKind === "lr_smoke" ? "clean_victsd_gold" : "full_bundle",
              model_kind: options?.modelKind || "phobert",
              training_mode: options?.trainingMode || "finetune",
              balance_strategy: options?.balanceStrategy || "balanced_50_50",
              include_base_model: Boolean(options?.includeBaseModel),
              base_model: options?.baseModel || undefined,
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
    [activeBatchId, authorizedFetch, run],
  );

  const importArtifact = useCallback(
    async (runName: string, artifactPath: string, notes?: string) => {
      return run(async () => {
        return parseJsonResponse<{ import_id: number; status: string }>(
          await authorizedFetch(buildApiUrl("/api/mlflow/manual/import-artifact"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ run_name: runName, artifact_path: artifactPath, notes }),
          }),
        );
      });
    },
    [authorizedFetch, run],
  );

  const importModelZip = useCallback(
    async (modelName: string, modelZip: File) => {
      return run(async () => {
        const formData = new FormData();
        formData.append("model_name", modelName);
        formData.append("model_zip", modelZip);
        return parseJsonResponse<MlflowImportModelZipResponse>(
          await authorizedFetch(buildApiUrl("/api/models/import-zip"), {
            method: "POST",
            body: formData,
          }),
        );
      });
    },
    [authorizedFetch, run],
  );

  const refreshDOPreflight = useCallback(async () => {
    return run(async () => {
      try {
        const payload = await parseJsonResponse<MlflowDOPreflight>(
          await authorizedFetch(buildApiUrl("/api/mlflow/kaggle/preflight")),
        );
        setDoPreflight(payload);
        return payload;
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          const fallback: MlflowDOPreflight = {
            ready: false,
            missing: [],
            warnings: [
              "Backend đang chạy chưa có endpoint /api/mlflow/kaggle/preflight. Hãy restart backend với code mới.",
            ],
            checks: {},
            config: {},
            checked_at: new Date().toISOString(),
          };
          setDoPreflight(fallback);
          return fallback;
        }
        throw err;
      }
    });
  }, [authorizedFetch, run]);

  const refreshDOStatus = useCallback(
    async (runId?: string | null) => {
      return run(async () => {
        const target = runId || doRunId;
        if (!target) return;
        const payload = await parseJsonResponse<MlflowKaggleStatus>(
          await authorizedFetch(buildApiUrl(`/api/mlflow/kaggle/status?run_id=${encodeURIComponent(target)}`)),
        );
        setDoStatus(payload);
        return payload;
      });
    },
    [authorizedFetch, doRunId, run],
  );

  const stopDOPolling = useCallback(() => {
    if (doPollTimerRef.current) {
      clearInterval(doPollTimerRef.current);
      doPollTimerRef.current = null;
    }
    doPollAttemptRef.current = 0;
  }, []);

  const clearDOSession = useCallback(() => {
    stopDOPolling();
    setDoRunId(null);
    setDoStatus(null);
  }, [stopDOPolling]);

  const startDOPolling = useCallback(
    (runId: string) => {
      stopDOPolling();
      doPollAttemptRef.current = 0;

      doPollTimerRef.current = setInterval(() => {
        doPollAttemptRef.current += 1;
        const attempt = doPollAttemptRef.current;
        void refreshDOStatus(runId)
          .then((payload) => {
            const status = typeof payload?.status === "string" ? payload.status : "";
            if (DO_TERMINAL_STATUSES.has(status)) {
              stopDOPolling();
              return;
            }
            if (attempt >= DO_MAX_POLL_ATTEMPTS) {
              stopDOPolling();
            }
          })
          .catch(() => {
            if (attempt >= DO_MAX_POLL_ATTEMPTS) {
              stopDOPolling();
            }
          });
      }, DO_POLL_INTERVAL_MS);
    },
    [refreshDOStatus, stopDOPolling],
  );

  useEffect(() => {
    return () => {
      stopDOPolling();
    };
  }, [stopDOPolling]);

  const triggerDO = useCallback(
    async (options?: MlflowDOTriggerOptions) => {
      return run(async () => {
        const trainingMode = options?.trainingMode === "finetune" ? "finetune" : "retrain";
        const modelKind = options?.modelKind === "lr_smoke" ? "lr_smoke" : "phobert";
        const baseModel = (options?.baseModel && options.baseModel.trim()) || undefined;

        const payloadBody: Record<string, unknown> = {
          batch_id: activeBatchId,
          provider: "kaggle",
          dry_run: options?.dryRun ?? false,
          compute_mode: "kaggle",
          model_kind: modelKind,
          training_mode: trainingMode,
          training_scope: "light_only",
        };
        if (baseModel) {
          payloadBody.base_model = baseModel;
        }

        const payload = await parseJsonResponse<{
          run_id: string;
          status: string;
          stages: string[];
          dry_run: boolean;
          training_mode?: "retrain" | "finetune";
          base_model?: string | null;
        }>(
          await authorizedFetch(buildApiUrl("/api/mlflow/kaggle/trigger"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payloadBody),
          }),
        );

        if (payload.status === "placeholder") {
          throw new Error(
            "Backend đang trả status=placeholder (không chạy Colab thật). Hãy restart backend với phiên bản mới có Colab runtime flow.",
          );
        }

        setDoRunId(payload.run_id);
        const latest = await refreshDOStatus(payload.run_id);
        const latestStatus = typeof latest?.status === "string" ? latest.status : "";
        if (!DO_TERMINAL_STATUSES.has(latestStatus)) {
          startDOPolling(payload.run_id);
        }
        return payload;
      });
    },
    [activeBatchId, authorizedFetch, doPreflight, refreshDOStatus, run, startDOPolling],
  );

  const refreshCompare = useCallback(async () => {
    return run(async () => {
      const payload = await parseJsonResponse<MlflowComparePayload>(
        await authorizedFetch(buildApiUrl("/api/mlflow/compare/latest")),
      );
      setComparePayload(payload);
      return payload;
    });
  }, [authorizedFetch, run]);

  const promote = useCallback(
    async (candidateModel: string) => {
      return run(async () => {
        return parseJsonResponse<{ status: string; message?: string }>(
          await authorizedFetch(buildApiUrl("/api/mlflow/promote"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ candidate_model: candidateModel }),
          }),
        );
      });
    },
    [authorizedFetch, run],
  );

  const reviewCandidates = useCallback(
    async (
      updates: Array<{
        id: number;
        action?: "include_toxic" | "include_clean" | "drop";
        decision?: "accept" | "reject";
        pseudo_label?: 0 | 1;
        lock_state?: boolean;
      }>,
    ) => {
      return run(async () => {
        const payload = await parseJsonResponse<{ updated: number; skipped_locked?: number; locked_updated?: number }>(
          await authorizedFetch(buildApiUrl("/api/mlflow/candidates/review"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ updates }),
          }),
        );

        const latestOverview = await refreshOverview();
        const nextBatchId = latestOverview?.active_batch_id || null;

        await refreshCandidates(undefined, 1, "all_batches");
        if (nextBatchId) {
          await refreshThresholdStatus(nextBatchId);
        } else {
          setThresholdStatus(null);
        }
        await refreshReviewHistory(undefined, "all", 1, "all_batches");
        await refreshCrawlHistory(1);
        return payload;
      });
    },
    [authorizedFetch, refreshCandidates, refreshCrawlHistory, refreshOverview, refreshReviewHistory, refreshThresholdStatus, run],
  );

  const clearMlflowBatch = useCallback(
    async (batchId?: string | null) => {
      return run(async () => {
        const useBatch = batchId || activeBatchId;
        if (!useBatch) {
          throw new Error("No active batch to clear");
        }

        const payload = await parseJsonResponse<MlflowClearBatchResponse>(
          await authorizedFetch(buildApiUrl("/api/mlflow/clear-batch"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ batch_id: useBatch }),
          }),
        );

        await refreshBatches();
        const latestOverview = await refreshOverview();
        const nextBatchId = latestOverview?.active_batch_id || null;

        await refreshCandidates(undefined, 1, "all_batches");
        if (nextBatchId) {
          await refreshThresholdStatus(nextBatchId);
        } else {
          setThresholdStatus(null);
        }
        await refreshReviewHistory(undefined, "all", 1, "all_batches");
        await refreshCrawlHistory(1);

        return payload;
      });
    },
    [activeBatchId, authorizedFetch, refreshBatches, refreshCandidates, refreshCrawlHistory, refreshOverview, refreshReviewHistory, refreshThresholdStatus, run],
  );

  const clearMlflowAll = useCallback(
    async (confirmToken: string) => {
      return run(async () => {
        const payload = await parseJsonResponse<MlflowClearAllResponse>(
          await authorizedFetch(buildApiUrl("/api/mlflow/clear-all"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ confirm_token: confirmToken }),
          }),
        );

        await refreshBatches();
        setOverview(null);
        setActiveBatchId(null);
        setHasNoBatch(true);
        setCandidates([]);
        setCandidateTotal(0);
        setThresholdStatus(null);
        setReviewHistory([]);
        setReviewHistoryTotal(0);
        setReviewHistoryPage(1);
        setCrawlHistory([]);
        setCrawlHistoryTotal(0);
        setCrawlHistoryPage(1);
        return payload;
      });
    },
    [authorizedFetch, refreshBatches, run],
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
          return parseJsonResponse<{
            batch_id: string;
            model_name?: string;
            counts?: Record<string, number>;
            crawl_summary?: {
              status_counts?: Record<string, number>;
              timeout_count?: number;
              total_urls?: number;
            };
          }>(
            await authorizedFetch(buildApiUrl("/api/mlflow/ingest"), {
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
        });

        const b = payload.batch_id;
        setIngestStage("finalize");
        setIngestStageMessage("Đang đồng bộ trạng thái từ backend...");
        setIngestProgress((prev) => Math.max(prev, 90));
        setActiveBatchId(b);

        const refreshWarnings: string[] = [];
        const tryRefresh = async (label: string, fn: () => Promise<unknown>) => {
          try {
            await fn();
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            refreshWarnings.push(`${label}: ${message}`);
            console.warn(`[mlflow] post-ingest refresh failed: ${label}`, err);
          }
        };

        await tryRefresh("overview", () => refreshOverview(b));
        await tryRefresh("candidates", () => refreshCandidates(undefined, 1, "all_batches"));
        await tryRefresh("training-preview", () => refreshTrainingPreview(1, "all_batches"));
        await tryRefresh("threshold-status", () => refreshThresholdStatus(b));
        await tryRefresh("review-history", () => refreshReviewHistory(undefined, "all", 1, "all_batches"));
        await tryRefresh("crawl-history", () => refreshCrawlHistory(1, { allowUnavailableFallback: true }));
        await tryRefresh("batches", () => refreshBatches());

        stopStageRamp();
        setIngestStage("completed");
        setIngestProgress(100);
        if (refreshWarnings.length > 0) {
          setIngestStageMessage(`Hoàn tất ingest + infer + gate. Có ${refreshWarnings.length} panel chưa đồng bộ.`);
        } else {
          setIngestStageMessage("Hoàn tất ingest + infer + gate.");
        }
        setError(null);
        return payload;
      } catch (err) {
        stopStageRamp();
        setIngestStage("error");
        setIngestStageMessage(err instanceof Error ? err.message : "Ingest thất bại");
        throw err;
      }
    },
    [authorizedFetch, refreshBatches, refreshCandidates, refreshCrawlHistory, refreshOverview, refreshReviewHistory, refreshThresholdStatus, refreshTrainingPreview, run],
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
    trainingPreview,
    candidateTotal,
    candidatePage,
    candidatePageSize,
    thresholdStatus,
    batches,
    reviewHistory,
    reviewHistoryTotal,
    reviewHistoryPage,
    crawlHistory,
    crawlHistoryTotal,
    crawlHistoryPage,
    comparePayload,
    lastBundlePath,
    requiredZipContents,
    doRunId,
    doStatus,
    doPreflight,
    ingest,
    refreshOverview,
    refreshBatches,
    refreshCandidates,
    refreshTrainingPreview,
    reviewTrainingPreview,
    refreshReviewHistory,
    refreshCrawlHistory,
    reviewCandidates,
    clearMlflowBatch,
    clearMlflowAll,
    refreshThresholdStatus,
    exportBundle,
    importArtifact,
    importModelZip,
    triggerDO,
    refreshDOPreflight,
    refreshDOStatus,
    clearDOSession,
    refreshCompare,
    promote,
    setError,
  };
}
