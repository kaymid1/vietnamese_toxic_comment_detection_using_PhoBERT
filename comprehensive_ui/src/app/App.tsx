import { useEffect, useRef, useState } from "react";
import { Navigation } from "@/app/components/Navigation";
import { HomePage } from "@/app/components/HomePage";
import { ResultsPage } from "@/app/components/ResultsPage";
import { DatasetPage } from "@/app/components/DatasetPage";
import { SyntheticGenerationPage } from "@/app/components/SyntheticGenerationPage";
import { ModelPage } from "@/app/components/ModelPage";
import { ProtocolPage } from "@/app/components/ProtocolPage";
import { ContactPage } from "@/app/components/ContactPage";

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
  seg_threshold_used?: number | null;
}

interface ApiResult {
  url: string;
  url_hash?: string | null;
  status: "ok" | "error" | "skipped";
  error?: string | null;
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
}

interface PendingFallbackUrl {
  url: string;
  url_hash: string;
  reason?: string;
  trafilatura_text_len?: number;
}

interface AnalyzeResponse {
  job_id: string;
  source_job_id?: string;
  flow_state?: "awaiting_user_choice" | "completed";
  pending_fallback_urls?: PendingFallbackUrl[];
  model_name?: string;
  thresholds?: {
    seg_threshold?: number;
    page_threshold?: number;
  };
  results?: ApiResult[];
}

interface CompareModelResponse {
  model_name?: string;
  thresholds?: {
    seg_threshold?: number;
    page_threshold?: number;
  };
  results: ApiResult[];
}

interface AnalyzeCompareResponse {
  job_id: string;
  flow_state?: "awaiting_user_choice" | "completed";
  pending_fallback_urls?: PendingFallbackUrl[];
  models?: Record<string, CompareModelResponse>;
}

interface ModelsResponse {
  models?: string[];
  default?: string | null;
}

interface FallbackDecisionPayload {
  url: string;
  url_hash: string;
  action: "use_selenium" | "skip";
}

interface ScanHistoryItem {
  id: string;
  savedAt: string;
  jobId: string | null;
  modelId: string | null;
  thresholds: AnalyzeResponse["thresholds"] | null;
  result: ApiResult;
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const SCAN_HISTORY_KEY = "viettoxic:scan-history";
const THEME_KEY = "viettoxic:theme";
const MAX_SCAN_HISTORY = 120;
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

const normalizeModelId = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, "");

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
    throw new Error(raw || "API request failed");
  }

  if (!contentType.includes("application/json")) {
    const preview = raw.slice(0, 120).replace(/\s+/g, " ");
    throw new Error(
      `API did not return JSON (content-type: ${contentType || "unknown"}). Response starts with: ${preview}`,
    );
  }

  return JSON.parse(raw) as T;
};

const readScanHistory = (): ScanHistoryItem[] => {
  try {
    const raw = window.localStorage.getItem(SCAN_HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is ScanHistoryItem => {
      if (!item || typeof item !== "object") return false;
      const candidate = item as Partial<ScanHistoryItem>;
      return typeof candidate.id === "string" && !!candidate.result && typeof candidate.result.url === "string";
    });
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
}): ScanHistoryItem[] => {
  const { results, jobId, modelId, thresholds } = params;
  const savedAt = new Date().toISOString();
  return results.map((result, index) => ({
    id: `${result.url_hash || result.url}-${modelId || "unknown"}-${Date.now()}-${index}`,
    savedAt,
    jobId,
    modelId,
    thresholds,
    result,
  }));
};

export default function App() {
  const [currentPage, setCurrentPage] = useState("home");
  const [analysisResults, setAnalysisResults] = useState<ApiResult[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [analysisModelId, setAnalysisModelId] = useState<string | null>(null);
  const [scanHistory, setScanHistory] = useState<ScanHistoryItem[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [thresholds, setThresholds] = useState<AnalyzeResponse["thresholds"] | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [compareModels, setCompareModels] = useState<Record<string, CompareModelResponse> | null>(null);
  const [activeResultModel, setActiveResultModel] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<number | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [fallbackPrompt, setFallbackPrompt] = useState<{
    items: PendingFallbackUrl[];
    decisions: Record<string, "use_selenium" | "skip">;
  } | null>(null);
  const fallbackResolverRef = useRef<((value: FallbackDecisionPayload[] | null) => void) | null>(null);

  useEffect(() => {
    setScanHistory(readScanHistory());
  }, []);

  useEffect(() => {
    const storedTheme = window.localStorage.getItem(THEME_KEY);
    if (storedTheme === "light" || storedTheme === "dark") {
      setTheme(storedTheme);
      return;
    }

    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(prefersDark ? "dark" : "light");
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    let isMounted = true;

    const loadModels = async () => {
      setModelsLoading(true);
      setModelsError(null);
      try {
        const response = await fetch(buildApiUrl("/api/models"));
        const data = await parseJsonResponse<ModelsResponse>(response);
        const models = Array.isArray(data.models)
          ? data.models.filter((name): name is string => typeof name === "string")
          : [];

        const sortedModels = [...models].sort((a, b) => a.localeCompare(b));
        const apiDefault = data.default && sortedModels.includes(data.default) ? data.default : null;
        const preferred = pickPreferredModel(sortedModels);
        const resolvedDefault = preferred || apiDefault || sortedModels[0] || null;

        if (!isMounted) return;

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
          ? parsedStored.filter((name): name is string => typeof name === "string" && sortedModels.includes(name))
          : [];
        const fromLegacy = legacyStored && sortedModels.includes(legacyStored) ? [legacyStored] : [];
        const selected = (fromArray.length > 0 ? fromArray : fromLegacy).slice(0, 2);
        const fallback = resolvedDefault ? [resolvedDefault] : [];
        setSelectedModels(selected.length > 0 ? selected : fallback);
      } catch (error) {
        if (!isMounted) return;
        const message = error instanceof Error ? error.message : "Không thể tải danh sách model";
        setModelsError(message);
        setAvailableModels([]);
        setSelectedModels([]);
      } finally {
        if (isMounted) {
          setModelsLoading(false);
        }
      }
    };

    void loadModels();

    return () => {
      isMounted = false;
    };
  }, []);

  const handleNavigate = (page: string) => {
    setCurrentPage(page);
  };

  const handleToggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const appendHistory = (entries: ScanHistoryItem[]) => {
    if (entries.length === 0) return;
    setScanHistory((prev) => {
      const merged = [...entries, ...prev].slice(0, MAX_SCAN_HISTORY);
      writeScanHistory(merged);
      return merged;
    });
  };

  const askFallbackDecisions = (pending: PendingFallbackUrl[]): Promise<FallbackDecisionPayload[] | null> => {
    return new Promise((resolve) => {
      const initial: Record<string, "use_selenium" | "skip"> = {};
      pending.forEach((item) => {
        initial[item.url_hash] = "use_selenium";
      });
      setFallbackPrompt({ items: pending, decisions: initial });
      fallbackResolverRef.current = resolve;
    });
  };

  const handleFallbackDecisionChange = (urlHash: string, action: "use_selenium" | "skip") => {
    setFallbackPrompt((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        decisions: {
          ...prev.decisions,
          [urlHash]: action,
        },
      };
    });
  };

  const handleFallbackApplyAll = (action: "use_selenium" | "skip") => {
    setFallbackPrompt((prev) => {
      if (!prev) return prev;
      const next: Record<string, "use_selenium" | "skip"> = {};
      prev.items.forEach((item) => {
        next[item.url_hash] = action;
      });
      return { ...prev, decisions: next };
    });
  };

  const closeFallbackPrompt = (payload: FallbackDecisionPayload[] | null) => {
    const resolver = fallbackResolverRef.current;
    fallbackResolverRef.current = null;
    setFallbackPrompt(null);
    resolver?.(payload);
  };

  const handleFallbackConfirm = () => {
    if (!fallbackPrompt) return;
    const payload: FallbackDecisionPayload[] = fallbackPrompt.items.map((item) => ({
      url: item.url,
      url_hash: item.url_hash,
      action: fallbackPrompt.decisions[item.url_hash] || "use_selenium",
    }));
    closeFallbackPrompt(payload);
  };

  const handleFallbackCancel = () => {
    closeFallbackPrompt(null);
  };

  const handleAnalyze = async (urls: string[], modelNames: string[]) => {
    try {
      setErrorMessage(null);
      setCompareModels(null);
      setActiveResultModel(null);
      setAnalysisProgress(0);

      const baseOptions: Record<string, unknown> = {
        batch_size: 8,
        max_length: 256,
        page_threshold: 0.25,
        seg_threshold: 0.4,
        enable_video: true,
        selenium_fallback_mode: "ask",
      };

      if (modelNames.length >= 2) {
        const requestBody: Record<string, unknown> = {
          urls,
          options: {
            ...baseOptions,
            model_names: modelNames,
          },
        };

        let data = await parseJsonResponse<AnalyzeCompareResponse>(
          await fetch(buildApiUrl("/api/analyze_compare"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
          }),
        );

        if (data.flow_state === "awaiting_user_choice" && data.pending_fallback_urls && data.pending_fallback_urls.length > 0) {
          const fallbackDecisions = await askFallbackDecisions(data.pending_fallback_urls);
          if (!fallbackDecisions) {
            throw new Error("Bạn đã hủy thao tác chuyển qua Selenium.");
          }
          data = await parseJsonResponse<AnalyzeCompareResponse>(
            await fetch(buildApiUrl("/api/analyze_compare"), {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                ...requestBody,
                pending_job_id: data.job_id,
                fallback_decisions: fallbackDecisions,
              }),
            }),
          );
        }

        const comparePayloads = data.models || {};
        const firstModel = modelNames.find((name) => comparePayloads[name]) ?? Object.keys(comparePayloads)[0] ?? null;
        const selectedPayload = firstModel ? comparePayloads[firstModel] : null;

        setAnalysisProgress(100);
        setJobId(data.job_id);
        setCompareModels(Object.keys(comparePayloads).length > 0 ? comparePayloads : null);
        setActiveResultModel(firstModel);
        setAnalysisModelId(firstModel);
        setThresholds(selectedPayload?.thresholds || null);
        setAnalysisResults(selectedPayload?.results || []);

        const compareHistoryEntries = Object.entries(comparePayloads).flatMap(([modelKey, payload]) =>
          createHistoryEntries({
            results: payload?.results || [],
            jobId: data.job_id,
            modelId: modelKey,
            thresholds: payload?.thresholds || null,
          }),
        );
        appendHistory(compareHistoryEntries);
        setCurrentPage("results");
        return;
      }

      const selected = modelNames[0] || null;
      const requestBody: Record<string, unknown> = {
        urls,
        options: selected ? { ...baseOptions, model_name: selected } : baseOptions,
      };

      let data = await parseJsonResponse<AnalyzeResponse>(
        await fetch(buildApiUrl("/api/analyze"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        }),
      );

      if (data.flow_state === "awaiting_user_choice" && data.pending_fallback_urls && data.pending_fallback_urls.length > 0) {
        const fallbackDecisions = await askFallbackDecisions(data.pending_fallback_urls);
          if (!fallbackDecisions) {
            throw new Error("Bạn đã hủy thao tác chuyển qua Selenium.");
          }
        data = await parseJsonResponse<AnalyzeResponse>(
          await fetch(buildApiUrl("/api/analyze"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...requestBody,
              pending_job_id: data.job_id,
              fallback_decisions: fallbackDecisions,
            }),
          }),
        );
      }

      const resolvedModel = data.model_name || selected || null;
      setAnalysisProgress(100);
      setJobId(data.job_id);
      setAnalysisModelId(resolvedModel);
      setThresholds(data.thresholds || null);
      setAnalysisResults(data.results || []);
      appendHistory(
        createHistoryEntries({
          results: data.results || [],
          jobId: data.job_id,
          modelId: resolvedModel,
          thresholds: data.thresholds || null,
        }),
      );
      setCurrentPage("results");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setErrorMessage(message);
    } finally {
      setAnalysisProgress(null);
    }
  };

  const handleScanAgain = () => {
    setCurrentPage("home");
    setAnalysisResults([]);
    setJobId(null);
    setAnalysisModelId(null);
    setThresholds(null);
    setCompareModels(null);
    setActiveResultModel(null);
  };

  const handleSelectResultModel = (modelName: string) => {
    if (!compareModels || !compareModels[modelName]) return;
    const payload = compareModels[modelName];
    setActiveResultModel(modelName);
    setAnalysisModelId(modelName);
    setThresholds(payload.thresholds || null);
    setAnalysisResults(payload.results || []);
  };

  const handleLoadFromHistory = (item: ScanHistoryItem) => {
    setCurrentPage("results");
    setCompareModels(null);
    setActiveResultModel(null);
    setJobId(item.jobId);
    setAnalysisModelId(item.modelId);
    setThresholds(item.thresholds);
    setAnalysisResults([item.result]);
  };

  const handleTryNow = () => {
    setCurrentPage("home");
  };

  return (
    <div className="min-h-screen">
      <Navigation
        currentPage={currentPage}
        onNavigate={handleNavigate}
        theme={theme}
        onToggleTheme={handleToggleTheme}
      />

      {currentPage === "home" && (
        <HomePage
          onAnalyze={handleAnalyze}
          availableModels={availableModels}
          selectedModels={selectedModels}
          onSelectModels={(modelNames: string[]) => {
            const sanitized = Array.from(new Set(modelNames)).slice(0, 2);
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

      {fallbackPrompt && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4">
          <div className="w-full max-w-2xl rounded-2xl border border-gray-200 bg-white p-5 shadow-xl">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Một số URL cần chuyển qua Selenium</h3>
              <p className="mt-1 text-sm text-gray-600">Chọn theo từng URL: chuyển qua Selenium hoặc bỏ qua.</p>
            </div>

            <div className="mb-3 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => handleFallbackApplyAll("use_selenium")}
                className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs text-blue-700"
              >
                Dùng Selenium cho tất cả
              </button>
              <button
                type="button"
                onClick={() => handleFallbackApplyAll("skip")}
                className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-800"
              >
                Skip tất cả
              </button>
            </div>

            <div className="max-h-80 space-y-3 overflow-auto pr-1">
              {fallbackPrompt.items.map((item) => {
                const value = fallbackPrompt.decisions[item.url_hash] || "use_selenium";
                return (
                  <div key={item.url_hash} className="rounded-lg border border-gray-200 p-3">
                    <p className="break-all text-sm text-gray-800">{item.url}</p>
                    <p className="mt-1 text-xs text-gray-500">Trafilatura text length: {item.trafilatura_text_len ?? 0}</p>
                    <div className="mt-2 flex gap-2">
                      <button
                        type="button"
                        onClick={() => handleFallbackDecisionChange(item.url_hash, "use_selenium")}
                        className={`rounded-full border px-3 py-1 text-xs ${
                          value === "use_selenium"
                            ? "border-blue-500 bg-blue-600 text-white"
                            : "border-gray-300 bg-white text-gray-700"
                        }`}
                      >
                        Dùng Selenium
                      </button>
                      <button
                        type="button"
                        onClick={() => handleFallbackDecisionChange(item.url_hash, "skip")}
                        className={`rounded-full border px-3 py-1 text-xs ${
                          value === "skip"
                            ? "border-amber-500 bg-amber-500 text-white"
                            : "border-gray-300 bg-white text-gray-700"
                        }`}
                      >
                        Skip URL này
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={handleFallbackCancel}
                className="rounded-md border border-gray-300 px-4 py-2 text-sm text-gray-700"
              >
                Hủy
              </button>
              <button
                type="button"
                onClick={handleFallbackConfirm}
                className="rounded-md bg-[var(--viet-primary)] px-4 py-2 text-sm text-white"
              >
                Tiếp tục
              </button>
            </div>
          </div>
        </div>
      )}

      {currentPage === "results" && (
        <ResultsPage
          results={analysisResults}
          jobId={jobId}
          thresholds={thresholds}
          modelId={analysisModelId}
          compareModelNames={compareModels ? Object.keys(compareModels) : []}
          activeResultModel={activeResultModel}
          onSelectResultModel={handleSelectResultModel}
          scanHistory={scanHistory}
          onLoadHistoryItem={handleLoadFromHistory}
          onScanAgain={handleScanAgain}
        />
      )}

      {currentPage === "dataset" && <DatasetPage />}

      {currentPage === "dataset_synthetic" && (
        <SyntheticGenerationPage onBack={() => setCurrentPage("dataset")} />
      )}

      {currentPage === "protocol" && <ProtocolPage />}

      {currentPage === "model" && <ModelPage onTryNow={handleTryNow} />}

      {currentPage === "contact" && <ContactPage />}
    </div>
  );
}
