import { useEffect, useState } from "react";
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
  status: "ok" | "error";
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

interface AnalyzeResponse {
  job_id: string;
  source_job_id?: string;
  model_name?: string;
  thresholds?: {
    seg_threshold?: number;
    page_threshold?: number;
  };
  results: ApiResult[];
}

interface CompareResponse {
  job_id: string;
  models: Record<string, AnalyzeResponse>;
}

interface ModelsResponse {
  models?: string[];
  default?: string | null;
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
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

export default function App() {
  const [currentPage, setCurrentPage] = useState("home");
  const [analysisResults, setAnalysisResults] = useState<ApiResult[]>([]);
  const [compareResults, setCompareResults] = useState<Record<string, AnalyzeResponse> | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [analysisModelId, setAnalysisModelId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [thresholds, setThresholds] = useState<AnalyzeResponse["thresholds"] | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

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
        const resolvedDefault =
          apiDefault || (sortedModels.includes("phobert/v2") ? "phobert/v2" : sortedModels[0] || null);

        if (!isMounted) return;

        setAvailableModels(sortedModels);
        const stored = window.localStorage.getItem("viettoxic:model");
        const storedModel = stored && sortedModels.includes(stored) ? stored : null;
        setSelectedModel((prev) => {
          const candidate = prev && sortedModels.includes(prev) ? prev : storedModel || resolvedDefault;
          return candidate;
        });
      } catch (error) {
        if (!isMounted) return;
        const message = error instanceof Error ? error.message : "Không thể tải danh sách model";
        setModelsError(message);
        setAvailableModels([]);
        setSelectedModel(null);
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

  const handleAnalyze = async (urls: string[], modelName?: string | null) => {
    try {
      setErrorMessage(null);
      const options: Record<string, unknown> = {
        batch_size: 8,
        max_length: 256,
        page_threshold: 0.25,
        seg_threshold: 0.4,
        enable_video: true,
      };
      if (modelName) {
        options.model_name = modelName;
      }

      const response = await fetch(buildApiUrl("/api/analyze"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          urls,
          options,
        }),
      });
      const data = await parseJsonResponse<AnalyzeResponse>(response);
      setJobId(data.job_id);
      setAnalysisModelId(data.model_name || modelName || null);
      setThresholds(data.thresholds || null);
      setAnalysisResults(data.results || []);
      setCompareResults(null);
      setCurrentPage("results");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setErrorMessage(message);
    }
  };

  const handleCompare = async (urls: string[], modelNames: string[]) => {
    try {
      setErrorMessage(null);
      const response = await fetch(buildApiUrl("/api/analyze_compare"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          urls,
          options: {
            batch_size: 8,
            max_length: 256,
            page_threshold: 0.25,
            seg_threshold: 0.4,
            enable_video: true,
            model_names: modelNames,
          },
        }),
      });
      const data = await parseJsonResponse<CompareResponse>(response);
      setJobId(data.job_id);
      setCompareResults(data.models || {});
      const firstModel = modelNames[0] || null;
      setAnalysisModelId(firstModel);
      const firstPayload = data.models?.[firstModel];
      setThresholds(firstPayload?.thresholds || null);
      setAnalysisResults(firstPayload?.results || []);
      setCurrentPage("results");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setErrorMessage(message);
    }
  };

  const handleScanAgain = () => {
    setCurrentPage("home");
    setAnalysisResults([]);
    setCompareResults(null);
    setJobId(null);
    setAnalysisModelId(null);
    setThresholds(null);
  };

  const handleRerun = async (sourceJobId: string, modelName?: string | null) => {
    try {
      setErrorMessage(null);
      const options: Record<string, unknown> = {
        batch_size: 8,
        max_length: 256,
        page_threshold: 0.25,
        seg_threshold: 0.4,
        enable_video: true,
      };

      const payload: Record<string, unknown> = {
        job_id: sourceJobId,
        options,
        prefer_merged: true,
      };
      if (modelName) {
        payload.model_name = modelName;
      }

      const response = await fetch(buildApiUrl("/api/analyze/rerun"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await parseJsonResponse<AnalyzeResponse>(response);
      setJobId(data.job_id);
      setAnalysisModelId(data.model_name || modelName || null);
      setThresholds(data.thresholds || null);
      setAnalysisResults(data.results || []);
      setCompareResults(null);
      setCurrentPage("results");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setErrorMessage(message);
    }
  };

  const handleTryNow = () => {
    setCurrentPage("home");
  };

  return (
    <div className="min-h-screen">
      <Navigation currentPage={currentPage} onNavigate={handleNavigate} />

      {currentPage === "home" && (
        <HomePage
          onAnalyze={handleAnalyze}
          onCompare={handleCompare}
          onRerun={handleRerun}
          compareMode={compareMode}
          onToggleCompare={setCompareMode}
          availableModels={availableModels}
          selectedModel={selectedModel}
          onSelectModel={(modelName) => {
            window.localStorage.setItem("viettoxic:model", modelName);
            setSelectedModel(modelName);
          }}
          modelsLoading={modelsLoading}
          modelsError={modelsError}
          errorMessage={errorMessage}
          onClearError={() => setErrorMessage(null)}
        />
      )}

      {currentPage === "results" && (
        <ResultsPage
          results={analysisResults}
          compareResults={compareResults}
          jobId={jobId}
          thresholds={thresholds}
          modelId={analysisModelId}
          onSelectModel={(modelId) => {
            setAnalysisModelId(modelId);
            const payload = compareResults?.[modelId];
            setThresholds(payload?.thresholds || null);
            setAnalysisResults(payload?.results || []);
          }}
          onScanAgain={handleScanAgain}
        />
      )}

      {currentPage === "dataset" && <DatasetPage onOpenSyntheticPage={() => setCurrentPage("dataset_synthetic")} />}

      {currentPage === "dataset_synthetic" && (
        <SyntheticGenerationPage onBack={() => setCurrentPage("dataset")} />
      )}

      {currentPage === "protocol" && <ProtocolPage />}

      {currentPage === "model" && <ModelPage onTryNow={handleTryNow} />}

      {currentPage === "contact" && <ContactPage />}
    </div>
  );
}
