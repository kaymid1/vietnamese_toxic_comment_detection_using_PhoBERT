import { useEffect, useMemo, useState } from "react";
import { Navigation } from "@/app/components/Navigation";
import { HomePage } from "@/app/components/HomePage";
import { I18nContext, createTranslator } from "@/app/i18n/context";
import type { Language } from "@/app/i18n/messages";
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

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const SCAN_HISTORY_KEY = "viettoxic:scan-history";
const THEME_KEY = "viettoxic:theme";
const LANGUAGE_KEY = "viettoxic:language";
const DATASET_VERSION_KEY = "viettoxic:dataset-version";
const MAX_SCAN_HISTORY = 120;

type DatasetVersion = "v1" | "latest";
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
  thresholdsByDomain: DomainThresholds | null;
}): ScanHistoryItem[] => {
  const { results, jobId, modelId, thresholds, thresholdsByDomain } = params;
  const savedAt = new Date().toISOString();
  return results.map((result, index) => ({
    id: `${result.url_hash || result.url}-${modelId || "unknown"}-${Date.now()}-${index}`,
    savedAt,
    jobId,
    modelId,
    thresholds,
    thresholdsByDomain,
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
  const [thresholdsByDomain, setThresholdsByDomain] = useState<DomainThresholds | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [compareModels, setCompareModels] = useState<Record<string, CompareModelResponse> | null>(null);
  const [activeResultModel, setActiveResultModel] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<number | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [language, setLanguage] = useState<Language>("vi");
  const [datasetVersion, setDatasetVersion] = useState<DatasetVersion>("v1");

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
    const storedLanguage = window.localStorage.getItem(LANGUAGE_KEY);
    if (storedLanguage === "vi" || storedLanguage === "en") {
      setLanguage(storedLanguage);
    }
  }, []);

  useEffect(() => {
    const storedDatasetVersion = window.localStorage.getItem(DATASET_VERSION_KEY);
    if (storedDatasetVersion === "v1" || storedDatasetVersion === "latest") {
      setDatasetVersion(storedDatasetVersion);
    }
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem(LANGUAGE_KEY, language);
  }, [language]);

  useEffect(() => {
    window.localStorage.setItem(DATASET_VERSION_KEY, datasetVersion);
  }, [datasetVersion]);

  const t = useMemo(() => createTranslator(language), [language]);

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
        const message = error instanceof Error ? error.message : t("app.cannotLoadModels");
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
  }, [t]);

  const handleNavigate = (page: string) => {
    setCurrentPage(page);
  };

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
        enable_video: false,
        selenium_fallback_mode: "auto",
      };

      if (modelNames.length >= 2) {
        const requestBody: Record<string, unknown> = {
          urls,
          options: {
            ...baseOptions,
            model_names: modelNames,
          },
        };

        const data = await parseJsonResponse<AnalyzeCompareResponse>(
          await fetch(buildApiUrl("/api/analyze_compare"), {
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
        urls,
        options: selected ? { ...baseOptions, model_name: selected } : baseOptions,
      };

      const data = await parseJsonResponse<AnalyzeResponse>(
        await fetch(buildApiUrl("/api/analyze"), {
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

  const handleScanAgain = () => {
    setCurrentPage("home");
    setAnalysisResults([]);
    setJobId(null);
    setAnalysisModelId(null);
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
    <div className="min-h-screen">
      <I18nContext.Provider value={{ language, setLanguage: handleSetLanguage, t }}>
        <Navigation
          currentPage={currentPage}
          onNavigate={handleNavigate}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          language={language}
          onSetLanguage={handleSetLanguage}
          datasetVersion={datasetVersion}
          onSetDatasetVersion={setDatasetVersion}
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

      {currentPage === "results" && (
        <ResultsPage
          results={analysisResults}
          jobId={jobId}
          thresholds={thresholds}
          thresholdsByDomain={thresholdsByDomain}
          modelId={analysisModelId}
          compareModelNames={compareModels ? Object.keys(compareModels) : []}
          activeResultModel={activeResultModel}
          onSelectResultModel={handleSelectResultModel}
          scanHistory={scanHistory}
          onLoadHistoryItem={handleLoadFromHistory}
          onScanAgain={handleScanAgain}
        />
      )}

      {currentPage === "dataset" && (
        <DatasetPage
          datasetVersion={datasetVersion}
          onNavigateToProtocol={() => setCurrentPage("protocol")}
        />
      )}

      {currentPage === "dataset_synthetic" && (
        <SyntheticGenerationPage onBack={() => setCurrentPage("dataset")} />
      )}

      {currentPage === "protocol" && <ProtocolPage />}

      {currentPage === "model" && <ModelPage onTryNow={handleTryNow} />}

      {currentPage === "contact" && <ContactPage />}
      </I18nContext.Provider>
    </div>
  );
}
