import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import { Navigation } from "@/app/components/Navigation";
import { HomePage } from "@/app/components/HomePage";
import { I18nContext, createTranslator } from "@/app/i18n/context";
import type { Language } from "@/app/i18n/messages";
import { ResultsPage } from "@/app/components/ResultsPage";
import { DatasetPage } from "@/app/components/DatasetPage";
import { SyntheticGenerationPage } from "@/app/components/SyntheticGenerationPage";
import { ModelPage } from "@/app/components/ModelPage";
import { ContactPage } from "@/app/components/ContactPage";
import { MLFlowPage } from "@/app/components/MLFlowPage";
import { SystemSettingsPage } from "@/app/components/SystemSettingsPage";
import { Toaster } from "@/app/components/ui/sonner";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Input } from "@/app/components/ui/input";

interface ApiSegment {
  segment_id: string;
  score: number;
  text_preview: string;
  text?: string;
  html_tags?: string[] | null;
  og_types?: string[] | null;
  ai_learned?: boolean | null;
  ai_learned_label?: string | null;
  segment_hash?: string | null;
  toxic_label?: number | null;
  constructiveness_score?: number | null;
  constructiveness_label?: number | null;
  seg_threshold_used?: number | null;
}

interface ApiResult {
  url: string;
  url_hash?: string | null;
  status: "ok" | "error" | "skipped";
  crawl_status?: string | null;
  error?: string | null;
  warnings?: string[] | null;
  comment_cap_hit?: boolean | null;
  max_comments_per_url?: number | null;
  crawled_comment_count?: number | null;
  crawl_output_dir?: string | null;
  segments_path?: string | null;
  videos?: Record<string, unknown>[];
  html_tags?: string[] | null;
  og_types?: string[] | null;
  seg_threshold_used?: number | null;
  page_toxic?: number | null;
  toxicity?: {
    overall?: number | null;
    by_segment?: ApiSegment[];
  };
  constructiveness?: {
    overall?: number | null;
    by_segment?: ApiSegment[];
    meta?: {
      available?: boolean;
      threshold?: number;
      total_segments?: number;
      segments_with_score?: number;
      segments_without_score?: number;
      segments_with_label?: number;
      constructive_segments?: number;
      non_constructive_segments?: number;
      missing_reason?: string | null;
    };
  };
}


interface DomainThresholds {
  news?: number;
  social?: number;
  forum?: number;
  unknown?: number;
}

interface AnalyzeResponse {
  job_id: string;
  source_job_id?: string;
  flow_state?: "completed";
  model_name?: string;
  thresholds?: {
    seg_threshold?: number;
    page_threshold?: number;
  };
  thresholds_by_domain?: DomainThresholds;
  results?: ApiResult[];
}

interface AnalyzeOptions {
  batch_size: number;
  max_length: number;
  page_threshold: number;
  seg_threshold: number;
  crawl_timeout_sec: number;
  max_load_more_clicks: number;
  max_comments_per_url: number;
  enable_video: boolean;
  selenium_fallback_mode: "auto";
}

interface CompareModelResponse {
  model_name?: string;
  thresholds?: {
    seg_threshold?: number;
    page_threshold?: number;
  };
  thresholds_by_domain?: DomainThresholds;
  results: ApiResult[];
}

interface AnalyzeCompareResponse {
  job_id: string;
  flow_state?: "completed";
  models?: Record<string, CompareModelResponse>;
}

interface ModelsResponse {
  models?: string[];
  default?: string | null;
}


interface ScanHistoryItem {
  id: string;
  savedAt: string;
  jobId: string | null;
  modelId: string | null;
  thresholds: AnalyzeResponse["thresholds"] | null;
  thresholdsByDomain: DomainThresholds | null;
  result: ApiResult;
}

interface AdminSession {
  token: string;
  expires_at: string;
  username?: string;
}

interface AdminSessionResponse {
  authenticated?: boolean;
  username?: string;
  expires_at?: string;
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const SCAN_HISTORY_KEY = "viettoxic:scan-history";
const ADMIN_SESSION_KEY = "viettoxic:admin-session";
const THEME_KEY = "viettoxic:theme";
const LANGUAGE_KEY = "viettoxic:language";
const MAX_SCAN_HISTORY = 120;

const API_BASE = RAW_API_BASE.replace(/\/+$/, "");
const DEFAULT_ANALYZE_OPTIONS: AnalyzeOptions = {
  batch_size: 8,
  max_length: 256,
  page_threshold: 0.25,
  seg_threshold: 0.4,
  crawl_timeout_sec: 90,
  max_load_more_clicks: 4,
  max_comments_per_url: 50,
  enable_video: false,
  selenium_fallback_mode: "auto",
};
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

const getInitialTheme = (): "light" | "dark" => {
  if (typeof window === "undefined") return "light";
  const storedTheme = window.localStorage.getItem(THEME_KEY);
  if (storedTheme === "light" || storedTheme === "dark") return storedTheme;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
};

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

const buildApiUrlFromBase = (base: string, path: string) => {
  if (/^https?:\/\//i.test(path)) return path;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (!base) return normalizedPath;
  return `${base}${normalizedPath}`;
};

const isNetworkFetchError = (error: unknown) =>
  error instanceof TypeError && /failed to fetch|networkerror|load failed/i.test(error.message.toLowerCase());

const fetchApiWithFallback = async (path: string, init?: RequestInit): Promise<Response> => {
  const now = Date.now();
  const candidates = [lastSuccessfulApiBase || "", ...API_FALLBACK_BASES, API_BASE]
    .map((item) => item.trim())
    .filter((value, index, values) => values.indexOf(value) === index);
  let lastError: unknown = null;
  const tried: string[] = [];

  for (const candidate of candidates) {
    const blockedUntil = apiBaseFailureUntil.get(candidate) || 0;
    if (candidate !== (lastSuccessfulApiBase || "") && blockedUntil > now) {
      continue;
    }
    const url = buildApiUrlFromBase(candidate, path);
    tried.push(url);
    try {
      const response = await fetch(url, init);
      lastSuccessfulApiBase = candidate;
      apiBaseFailureUntil.delete(candidate);
      return response;
    } catch (error) {
      if (!isNetworkFetchError(error)) throw error;
      lastError = error;
      apiBaseFailureUntil.set(candidate, Date.now() + API_FALLBACK_FAILURE_COOLDOWN_MS);
      if (lastSuccessfulApiBase === candidate) {
        lastSuccessfulApiBase = null;
      }
    }
  }

  throw new Error(
    `Cannot reach backend API for ${path}. Tried: ${tried.join(" | ")}. ${(lastError as Error | null)?.message || ""}`.trim(),
  );
};

const readAdminSession = (): AdminSession | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(ADMIN_SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<AdminSession>;
    if (typeof parsed.token === "string" && parsed.token.trim()) {
      return {
        token: parsed.token,
        expires_at: typeof parsed.expires_at === "string" ? parsed.expires_at : "",
        username: typeof parsed.username === "string" ? parsed.username : undefined,
      };
    }
  } catch {
    window.localStorage.removeItem(ADMIN_SESSION_KEY);
  }
  return null;
};

const writeAdminSession = (session: AdminSession | null) => {
  if (typeof window === "undefined") return;
  if (!session) {
    window.localStorage.removeItem(ADMIN_SESSION_KEY);
    return;
  }
  window.localStorage.setItem(ADMIN_SESSION_KEY, JSON.stringify(session));
};

const normalizeModelId = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, "");
const isDeprecatedModel = (model: string) => model.toLowerCase().includes("deprecated");

const sortModelsForSelection = (models: string[]) =>
  [...models].sort((a, b) => {
    const aDeprecated = isDeprecatedModel(a);
    const bDeprecated = isDeprecatedModel(b);
    if (aDeprecated !== bDeprecated) {
      return aDeprecated ? 1 : -1;
    }
    return a.localeCompare(b);
  });

const pickPreferredModel = (models: string[]): string | null => {
  if (models.length === 0) return null;
  const target = normalizeModelId("phobert_lora_v2");
  const exact = models.find((model) => normalizeModelId(model) === target);
  if (exact) return exact;
  const partial = models.find((model) => normalizeModelId(model).includes(target));
  if (partial) return partial;
  const loraFallback = models.find((model) => normalizeModelId(model).includes(normalizeModelId("phobert_lora")));
  if (loraFallback) return loraFallback;
  return null;
};

const parseJsonResponse = async <T,>(response: Response): Promise<T> => {
  const contentType = response.headers.get("content-type") || "";
  const raw = await response.text();

  if (!response.ok) {
    let message = raw || "API request failed";
    try {
      const parsed = raw ? (JSON.parse(raw) as { detail?: unknown; message?: unknown }) : null;
      if (typeof parsed?.detail === "string") {
        message = parsed.detail;
      } else if (typeof parsed?.message === "string") {
        message = parsed.message;
      }
    } catch {
      // Keep raw response text when the error body is not JSON.
    }
    throw new Error(message);
  }

  if (!contentType.includes("application/json")) {
    const preview = raw.slice(0, 120).replace(/\s+/g, " ");
    throw new Error(
      `API did not return JSON (content-type: ${contentType || "unknown"}). Response starts with: ${preview}`,
    );
  }

  return JSON.parse(raw) as T;
};

const normalizeInputUrl = (raw: string): string | null => {
  const cleaned = raw.trim().replace(/^["'`]+|["'`]+$/g, "").replace(/[),.;]+$/g, "");
  if (!cleaned) return null;
  const withScheme = /^https?:\/\//i.test(cleaned) ? cleaned : `https://${cleaned}`;
  try {
    const parsed = new URL(withScheme);
    const protocol = parsed.protocol.toLowerCase();
    if (protocol !== "http:" && protocol !== "https:") return null;
    const host = parsed.hostname.trim().toLowerCase();
    if (!host || !host.includes(".")) return null;
    const path = parsed.pathname || "";
    const query = parsed.search || "";
    return `${protocol}//${host}${path}${query}`;
  } catch {
    return null;
  }
};

const normalizeInputUrls = (rawUrls: string[]): string[] => {
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const raw of rawUrls) {
    const url = normalizeInputUrl(raw);
    if (!url || seen.has(url)) continue;
    seen.add(url);
    normalized.push(url);
  }
  return normalized;
};

const mergeUniqueWarnings = (left?: string[] | null, right?: string[] | null) => {
  const merged = [...(left || []), ...(right || [])]
    .map((item) => String(item || "").trim())
    .filter((item) => item.length > 0);
  return Array.from(new Set(merged));
};

const mergeApiResult = (
  previous: ApiResult | null,
  latest: ApiResult,
  pageThreshold: number,
): ApiResult => {
  if (!previous || latest.status !== "ok") return latest;
  const prevSegments = previous.toxicity?.by_segment || [];
  const nextSegments = latest.toxicity?.by_segment || [];
  if (prevSegments.length === 0) return latest;

  const combinedByKey = new Map<string, ApiSegment>();
  const pushSegment = (segment: ApiSegment) => {
    const textKey = (segment.text || segment.text_preview || "").trim();
    const key = textKey || segment.segment_id || `${segment.score}:${segment.text_preview}`;
    if (!combinedByKey.has(key)) {
      combinedByKey.set(key, segment);
    } else {
      const existing = combinedByKey.get(key)!;
      if (segment.score > existing.score) {
        combinedByKey.set(key, segment);
      }
    }
  };
  prevSegments.forEach(pushSegment);
  nextSegments.forEach(pushSegment);

  const mergedSegments = Array.from(combinedByKey.values());
  const toxicityOverall =
    mergedSegments.length > 0
      ? mergedSegments.reduce((sum, segment) => sum + (segment.score || 0), 0) / mergedSegments.length
      : latest.toxicity?.overall ?? previous.toxicity?.overall ?? null;
  const constructiveSegments = mergedSegments.filter((segment) => typeof segment.constructiveness_score === "number");
  const constructivenessOverall =
    constructiveSegments.length > 0
      ? constructiveSegments.reduce((sum, segment) => sum + (segment.constructiveness_score || 0), 0) /
        constructiveSegments.length
      : latest.constructiveness?.overall ?? previous.constructiveness?.overall ?? null;

  const segThresholdUsed =
    typeof latest.seg_threshold_used === "number"
      ? latest.seg_threshold_used
      : typeof previous.seg_threshold_used === "number"
        ? previous.seg_threshold_used
        : 0.5;
  const resolvedPageThreshold = Number.isFinite(pageThreshold) ? pageThreshold : 0.5;

  return {
    ...latest,
    warnings: mergeUniqueWarnings(previous.warnings, latest.warnings),
    comment_cap_hit: Boolean(latest.comment_cap_hit),
    crawled_comment_count: mergedSegments.length,
    toxicity: {
      overall: toxicityOverall,
      by_segment: mergedSegments,
    },
    page_toxic:
      toxicityOverall !== null
        ? toxicityOverall >= resolvedPageThreshold
          ? 1
          : 0
        : latest.page_toxic ?? previous.page_toxic ?? null,
    constructiveness: {
      overall: constructivenessOverall,
      by_segment: constructiveSegments,
      meta: {
        ...(latest.constructiveness?.meta || previous.constructiveness?.meta || {}),
        total_segments: mergedSegments.length,
        segments_with_score: constructiveSegments.length,
        segments_without_score: Math.max(0, mergedSegments.length - constructiveSegments.length),
      },
    },
  };
};

const readScanHistory = (): ScanHistoryItem[] => {
  try {
    const raw = window.localStorage.getItem(SCAN_HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const filtered = parsed.filter((item): item is ScanHistoryItem => {
      if (!item || typeof item !== "object") return false;
      const candidate = item as Partial<ScanHistoryItem>;
      return (
        typeof candidate.id === "string" &&
        !!candidate.result &&
        typeof candidate.result.url === "string" &&
        !!normalizeInputUrl(candidate.result.url)
      );
    });
    if (filtered.length !== parsed.length) {
      window.localStorage.setItem(SCAN_HISTORY_KEY, JSON.stringify(filtered.slice(0, MAX_SCAN_HISTORY)));
    }
    return filtered;
  } catch {
    return [];
  }
};

const writeScanHistory = (items: ScanHistoryItem[]) => {
  window.localStorage.setItem(SCAN_HISTORY_KEY, JSON.stringify(items.slice(0, MAX_SCAN_HISTORY)));
};

const createHistoryEntries = (params: {
  results: ApiResult[];
  jobId: string | null;
  modelId: string | null;
  thresholds: AnalyzeResponse["thresholds"] | null;
  thresholdsByDomain: DomainThresholds | null;
}): ScanHistoryItem[] => {
  const { results, jobId, modelId, thresholds, thresholdsByDomain } = params;
  const savedAt = new Date().toISOString();
  return results
    .filter((result) => !!normalizeInputUrl(result.url))
    .map((result, index) => ({
    id: `${result.url_hash || result.url}-${modelId || "unknown"}-${Date.now()}-${index}`,
    savedAt,
    jobId,
    modelId,
    thresholds,
    thresholdsByDomain,
    result,
    }));
};

function AdminLoginPage({
  onLogin,
  loading,
  error,
}: {
  onLogin: (username: string, password: string) => Promise<void>;
  loading: boolean;
  error: string | null;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await onLogin(username, password);
  };

  return (
    <div className="dashboard-page mx-auto flex min-h-[60vh] max-w-md items-center">
      <Card className="w-full border bg-card p-6 shadow-sm">
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div>
            <p className="text-sm font-medium text-muted-foreground">Admin access</p>
            <h2 className="mt-1 text-2xl font-semibold text-foreground">ML Flow login</h2>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground" htmlFor="admin-username">
              Username
            </label>
            <Input
              id="admin-username"
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              disabled={loading}
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground" htmlFor="admin-password">
              Password
            </label>
            <Input
              id="admin-password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              disabled={loading}
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <Button type="submit" className="w-full" disabled={loading || !username.trim() || !password}>
            {loading ? "Signing in..." : "Sign in"}
          </Button>
        </form>
      </Card>
    </div>
  );
}

export default function App() {
  const [currentPage, setCurrentPage] = useState("home");
  const [analysisResults, setAnalysisResults] = useState<ApiResult[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [analysisModelId, setAnalysisModelId] = useState<string | null>(null);
  const [scanHistory, setScanHistory] = useState<ScanHistoryItem[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [thresholds, setThresholds] = useState<AnalyzeResponse["thresholds"] | null>(null);
  const [thresholdsByDomain, setThresholdsByDomain] = useState<DomainThresholds | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [compareModels, setCompareModels] = useState<Record<string, CompareModelResponse> | null>(null);
  const [activeResultModel, setActiveResultModel] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<number | null>(null);
  const [scanMoreLoadingByUrl, setScanMoreLoadingByUrl] = useState<Record<string, boolean>>({});
  const [theme, setTheme] = useState<"light" | "dark">(getInitialTheme);
  const [language, setLanguage] = useState<Language>("vi");
  const [mlflowMounted, setMlflowMounted] = useState(false);
  const [adminSession, setAdminSession] = useState<AdminSession | null>(() => readAdminSession());
  const [adminLoginLoading, setAdminLoginLoading] = useState(false);
  const [adminLoginError, setAdminLoginError] = useState<string | null>(null);

  useEffect(() => {
    setScanHistory(readScanHistory());
  }, []);

  useEffect(() => {
    const stored = readAdminSession();
    if (!stored?.token) return;

    let cancelled = false;
    void (async () => {
      try {
        const payload = await parseJsonResponse<AdminSessionResponse>(
          await fetchApiWithFallback("/api/admin/session", {
            headers: { Authorization: `Bearer ${stored.token}` },
          }),
        );
        if (cancelled) return;
        const verified = {
          ...stored,
          username: payload.username || stored.username,
          expires_at: payload.expires_at || stored.expires_at,
        };
        writeAdminSession(verified);
        setAdminSession(verified);
      } catch {
        if (cancelled) return;
        writeAdminSession(null);
        setAdminSession(null);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const storedLanguage = window.localStorage.getItem(LANGUAGE_KEY);
    if (storedLanguage === "vi" || storedLanguage === "en") {
      setLanguage(storedLanguage);
    }
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem(LANGUAGE_KEY, language);
  }, [language]);

  const t = useMemo(() => createTranslator(language), [language]);

  const loadModels = useCallback(async () => {
    setModelsLoading(true);
    setModelsError(null);
    try {
      const response = await fetchApiWithFallback("/api/models");
      const data = await parseJsonResponse<ModelsResponse>(response);
      const models = Array.isArray(data.models)
        ? data.models.filter((name): name is string => typeof name === "string")
        : [];

      const sortedModels = sortModelsForSelection(models);
      const nonDeprecatedModels = sortedModels.filter((model) => !isDeprecatedModel(model));
      const apiDefault = data.default && sortedModels.includes(data.default) ? data.default : null;
      const preferred = pickPreferredModel(nonDeprecatedModels);
      const resolvedDefault =
        preferred ||
        (apiDefault && !isDeprecatedModel(apiDefault) ? apiDefault : null) ||
        nonDeprecatedModels[0] ||
        sortedModels[0] ||
        null;

      setAvailableModels(sortedModels);
      const stored = window.localStorage.getItem("viettoxic:models");
      const legacyStored = window.localStorage.getItem("viettoxic:model");
      let parsedStored: unknown = null;
      try {
        parsedStored = stored ? JSON.parse(stored) : null;
      } catch {
        parsedStored = null;
      }

      const fromArray = Array.isArray(parsedStored)
        ? parsedStored.filter(
            (name): name is string =>
              typeof name === "string" && sortedModels.includes(name) && !isDeprecatedModel(name),
          )
        : [];
      const fromLegacy =
        legacyStored && sortedModels.includes(legacyStored) && !isDeprecatedModel(legacyStored)
          ? [legacyStored]
          : [];
      const selected = (fromArray.length > 0 ? fromArray : fromLegacy).slice(0, 2);
      const fallback = resolvedDefault ? [resolvedDefault] : [];
      setSelectedModels(selected.length > 0 ? selected : fallback);
    } catch (error) {
      const message = error instanceof Error ? error.message : t("app.cannotLoadModels");
      setModelsError(message);
      setAvailableModels([]);
      setSelectedModels([]);
    } finally {
      setModelsLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const clearAdminSession = useCallback(() => {
    writeAdminSession(null);
    setAdminSession(null);
  }, []);

  const handleAdminUnauthorized = useCallback(() => {
    clearAdminSession();
    setMlflowMounted(false);
    setCurrentPage((page) =>
      page === "admin_mlflow" || page === "mlflow" || page === "admin_system_settings" ? "admin_login" : page,
    );
  }, [clearAdminSession]);

  const handleAdminLogout = useCallback(() => {
    clearAdminSession();
    setMlflowMounted(false);
    setCurrentPage("home");
  }, [clearAdminSession]);

  const handleAdminLogin = useCallback(async (username: string, password: string) => {
    setAdminLoginLoading(true);
    setAdminLoginError(null);
    try {
      const payload = await parseJsonResponse<AdminSession>(
        await fetchApiWithFallback("/api/admin/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username: username.trim(), password }),
        }),
      );
      writeAdminSession(payload);
      setAdminSession(payload);
      setMlflowMounted(true);
      setCurrentPage("admin_mlflow");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Admin login failed";
      setAdminLoginError(message);
    } finally {
      setAdminLoginLoading(false);
    }
  }, []);

  const handleNavigate = (page: string) => {
    if ((page === "admin_mlflow" || page === "mlflow" || page === "admin_system_settings") && !adminSession?.token) {
      setCurrentPage("admin_login");
      return;
    }
    setCurrentPage(page);
  };

  useEffect(() => {
    if ((currentPage === "mlflow" || currentPage === "admin_mlflow") && adminSession?.token) {
      setMlflowMounted(true);
    }
  }, [adminSession?.token, currentPage]);

  const handleToggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const handleSetLanguage = (nextLanguage: Language) => {
    setLanguage(nextLanguage);
  };

  const appendHistory = (entries: ScanHistoryItem[]) => {
    if (entries.length === 0) return;
    setScanHistory((prev) => {
      const merged = [...entries, ...prev].slice(0, MAX_SCAN_HISTORY);
      writeScanHistory(merged);
      return merged;
    });
  };

  const handleAnalyze = async (
    urls: string[],
    modelNames: string[],
    optionOverrides?: Partial<AnalyzeOptions>,
  ) => {
    try {
      const normalizedUrls = normalizeInputUrls(urls);
      if (normalizedUrls.length === 0) {
        throw new Error("No valid URLs provided.");
      }

      setErrorMessage(null);
      setCompareModels(null);
      setActiveResultModel(null);
      setAnalysisProgress(0);

      const baseOptions: AnalyzeOptions = {
        ...DEFAULT_ANALYZE_OPTIONS,
        ...(optionOverrides || {}),
      };

      if (modelNames.length >= 2) {
        const requestBody: Record<string, unknown> = {
          urls: normalizedUrls,
          options: {
            ...baseOptions,
            model_names: modelNames,
          },
        };

        const data = await parseJsonResponse<AnalyzeCompareResponse>(
          await fetchApiWithFallback("/api/analyze_compare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
          }),
        );

        const comparePayloads = data.models || {};
        const firstModel = modelNames.find((name) => comparePayloads[name]) ?? Object.keys(comparePayloads)[0] ?? null;
        const selectedPayload = firstModel ? comparePayloads[firstModel] : null;

        setAnalysisProgress(100);
        setJobId(data.job_id);
        setCompareModels(Object.keys(comparePayloads).length > 0 ? comparePayloads : null);
        setActiveResultModel(firstModel);
        setAnalysisModelId(firstModel);
        setThresholds(selectedPayload?.thresholds || null);
        setThresholdsByDomain(selectedPayload?.thresholds_by_domain || null);
        setAnalysisResults(selectedPayload?.results || []);

        const compareHistoryEntries = Object.entries(comparePayloads).flatMap(([modelKey, payload]) =>
          createHistoryEntries({
            results: payload?.results || [],
            jobId: data.job_id,
            modelId: modelKey,
            thresholds: payload?.thresholds || null,
            thresholdsByDomain: payload?.thresholds_by_domain || null,
          }),
        );
        appendHistory(compareHistoryEntries);
        setCurrentPage("results");
        return;
      }

      const selected = modelNames[0] || null;
      const requestBody: Record<string, unknown> = {
        urls: normalizedUrls,
        options: selected ? { ...baseOptions, model_name: selected } : baseOptions,
      };

      const data = await parseJsonResponse<AnalyzeResponse>(
        await fetchApiWithFallback("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        }),
      );

      const resolvedModel = data.model_name || selected || null;
      setAnalysisProgress(100);
      setJobId(data.job_id);
      setAnalysisModelId(resolvedModel);
      setThresholds(data.thresholds || null);
      setThresholdsByDomain(data.thresholds_by_domain || null);
      setAnalysisResults(data.results || []);
      appendHistory(
        createHistoryEntries({
          results: data.results || [],
          jobId: data.job_id,
          modelId: resolvedModel,
          thresholds: data.thresholds || null,
          thresholdsByDomain: data.thresholds_by_domain || null,
        }),
      );
      setCurrentPage("results");
    } catch (error) {
      const message = error instanceof Error ? error.message : t("app.unknownError");
      setErrorMessage(message);
    } finally {
      setAnalysisProgress(null);
    }
  };

  const handleScanMoreForUrl = async (result: ApiResult) => {
    const modelForRescan = analysisModelId || selectedModels[0] || null;
    if (!modelForRescan) {
      setErrorMessage(t("app.cannotLoadModels"));
      return;
    }
    const normalizedTargetUrl = normalizeInputUrl(result.url);
    if (!normalizedTargetUrl) {
      setErrorMessage("Cannot scan more: current result does not contain a valid URL.");
      return;
    }

    const currentCapRaw = Number(result.max_comments_per_url ?? DEFAULT_ANALYZE_OPTIONS.max_comments_per_url);
    const currentCap = Number.isFinite(currentCapRaw) ? Math.max(0, Math.floor(currentCapRaw)) : DEFAULT_ANALYZE_OPTIONS.max_comments_per_url;
    const nextCap = Math.min(5000, Math.max(50, currentCap + 50));

    try {
      setErrorMessage(null);
      setScanMoreLoadingByUrl((prev) => ({ ...prev, [normalizedTargetUrl]: true }));
      const requestBody: Record<string, unknown> = {
        urls: [normalizedTargetUrl],
        options: {
          ...DEFAULT_ANALYZE_OPTIONS,
          model_name: modelForRescan,
          max_comments_per_url: nextCap,
          max_load_more_clicks: 8,
          crawl_timeout_sec: 120,
        },
      };

      const data = await parseJsonResponse<AnalyzeResponse>(
        await fetchApiWithFallback("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        }),
      );

      const nextResult = (data.results || [])[0];
      if (!nextResult) {
        throw new Error(t("results.cannotAnalyzeUrl"));
      }
      const existingResult =
        analysisResults.find((item) => item.url === normalizedTargetUrl || item.url === result.url) || null;
      const mergedResult = mergeApiResult(
        existingResult,
        nextResult,
        Number(data.thresholds?.page_threshold ?? thresholds?.page_threshold ?? 0.5),
      );

      const resolvedModel = data.model_name || modelForRescan;
      setJobId(data.job_id);
      setAnalysisModelId(resolvedModel);
      setThresholds(data.thresholds || null);
      setThresholdsByDomain(data.thresholds_by_domain || null);
      setAnalysisResults((prev) => {
        const found = prev.some((item) => item.url === normalizedTargetUrl || item.url === result.url);
        if (!found) return [mergedResult, ...prev];
        return prev.map((item) =>
          item.url === normalizedTargetUrl || item.url === result.url ? mergedResult : item,
        );
      });

      if (compareModels && activeResultModel && compareModels[activeResultModel]) {
        setCompareModels((prev) => {
          if (!prev || !activeResultModel || !prev[activeResultModel]) return prev;
          const payload = prev[activeResultModel];
          const currentResults = payload.results || [];
          const hasUrl = currentResults.some((item) => item.url === normalizedTargetUrl || item.url === result.url);
          const nextResults = hasUrl
            ? currentResults.map((item) =>
                item.url === normalizedTargetUrl || item.url === result.url ? mergedResult : item,
              )
            : [mergedResult, ...currentResults];
          return {
            ...prev,
            [activeResultModel]: {
              ...payload,
              model_name: resolvedModel,
              thresholds: data.thresholds || payload.thresholds,
              thresholds_by_domain: data.thresholds_by_domain || payload.thresholds_by_domain,
              results: nextResults,
            },
          };
        });
      }

      appendHistory(
        createHistoryEntries({
          results: [mergedResult],
          jobId: data.job_id,
          modelId: resolvedModel,
          thresholds: data.thresholds || null,
          thresholdsByDomain: data.thresholds_by_domain || null,
        }),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : t("app.unknownError");
      setErrorMessage(message);
    } finally {
      setScanMoreLoadingByUrl((prev) => {
        const next = { ...prev };
        delete next[normalizedTargetUrl];
        return next;
      });
    }
  };

  const handleScanAgain = () => {
    setCurrentPage("home");
    setAnalysisResults([]);
    setJobId(null);
    setAnalysisModelId(null);
    setScanMoreLoadingByUrl({});
    setThresholds(null);
    setThresholdsByDomain(null);
    setCompareModels(null);
    setActiveResultModel(null);
  };

  const handleSelectResultModel = (modelName: string) => {
    if (!compareModels || !compareModels[modelName]) return;
    const payload = compareModels[modelName];
    setActiveResultModel(modelName);
    setAnalysisModelId(modelName);
    setThresholds(payload.thresholds || null);
    setThresholdsByDomain(payload.thresholds_by_domain || null);
    setAnalysisResults(payload.results || []);
  };

  const handleLoadFromHistory = (item: ScanHistoryItem) => {
    setCurrentPage("results");
    setCompareModels(null);
    setActiveResultModel(null);
    setJobId(item.jobId);
    setAnalysisModelId(item.modelId);
    setThresholds(item.thresholds);
    setThresholdsByDomain(item.thresholdsByDomain || null);
    setAnalysisResults([item.result]);
  };

  const handleTryNow = () => {
    setCurrentPage("home");
  };

  return (
    <div className="dashboard-app min-h-screen">
      <I18nContext.Provider value={{ language, setLanguage: handleSetLanguage, t }}>
        <Navigation
          currentPage={currentPage}
          onNavigate={handleNavigate}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          language={language}
          onSetLanguage={handleSetLanguage}
          adminAuthenticated={Boolean(adminSession?.token)}
          adminUsername={adminSession?.username}
          onAdminLogout={handleAdminLogout}
        />

        <main className="dashboard-main">
          <div className="dashboard-content">
            {currentPage === "home" && (
              <HomePage
                onAnalyze={handleAnalyze}
                availableModels={availableModels}
                selectedModels={selectedModels}
                onSelectModels={(modelNames: string[]) => {
                  const sanitized = Array.from(new Set(modelNames))
                    .filter((name) => !isDeprecatedModel(name))
                    .slice(0, 2);
                  setSelectedModels(sanitized);
                  window.localStorage.setItem("viettoxic:models", JSON.stringify(sanitized));
                  if (sanitized[0]) {
                    window.localStorage.setItem("viettoxic:model", sanitized[0]);
                  }
                }}
                modelsLoading={modelsLoading}
                modelsError={modelsError}
                errorMessage={errorMessage}
                onClearError={() => setErrorMessage(null)}
                analysisProgress={analysisProgress}
              />
            )}

            {currentPage === "results" && (
              <ResultsPage
                results={analysisResults}
                errorMessage={errorMessage}
                jobId={jobId}
                thresholds={thresholds}
                thresholdsByDomain={thresholdsByDomain}
                modelId={analysisModelId}
                compareModelNames={compareModels ? Object.keys(compareModels) : []}
                activeResultModel={activeResultModel}
                onSelectResultModel={handleSelectResultModel}
                scanHistory={scanHistory}
                onLoadHistoryItem={handleLoadFromHistory}
                onScanMore={handleScanMoreForUrl}
                scanMoreLoadingByUrl={scanMoreLoadingByUrl}
                onScanAgain={handleScanAgain}
              />
            )}

            {currentPage === "dataset" && <DatasetPage />}

            {currentPage === "dataset_synthetic" && (
              <SyntheticGenerationPage onBack={() => setCurrentPage("dataset")} />
            )}

            {currentPage === "admin_login" && (
              <AdminLoginPage
                onLogin={handleAdminLogin}
                loading={adminLoginLoading}
                error={adminLoginError}
              />
            )}

            {mlflowMounted && adminSession?.token && (
              <div className={currentPage === "admin_mlflow" || currentPage === "mlflow" ? "" : "hidden"}>
                <MLFlowPage
                  availableModels={availableModels}
                  onModelsChanged={loadModels}
                  adminToken={adminSession.token}
                  onAdminUnauthorized={handleAdminUnauthorized}
                />
              </div>
            )}

            {currentPage === "admin_system_settings" && adminSession?.token && (
              <SystemSettingsPage
                adminToken={adminSession.token}
                onAdminUnauthorized={handleAdminUnauthorized}
              />
            )}

            {currentPage === "model" && <ModelPage onTryNow={handleTryNow} />}

            {currentPage === "contact" && <ContactPage />}
          </div>
        </main>
        <Toaster position="top-right" />
      </I18nContext.Provider>
    </div>
  );
}
