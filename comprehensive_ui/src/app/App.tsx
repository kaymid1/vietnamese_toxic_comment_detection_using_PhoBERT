import { useState } from "react";
import { Navigation } from "@/app/components/Navigation";
import { HomePage } from "@/app/components/HomePage";
import { ResultsPage } from "@/app/components/ResultsPage";
import { ModelPage } from "@/app/components/ModelPage";
import { ContactPage } from "@/app/components/ContactPage";

interface ApiSegment {
  segment_id: string;
  score: number;
  text_preview: string;
}

interface ApiResult {
  url: string;
  status: "ok" | "error";
  error?: string | null;
  crawl_output_dir?: string | null;
  segments_path?: string | null;
  toxicity?: {
    overall?: number | null;
    by_segment?: ApiSegment[];
  };
}

interface AnalyzeResponse {
  job_id: string;
  results: ApiResult[];
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function App() {
  const [currentPage, setCurrentPage] = useState("home");
  const [analysisResults, setAnalysisResults] = useState<ApiResult[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleNavigate = (page: string) => {
    setCurrentPage(page);
  };

  const handleAnalyze = async (urls: string[]) => {
    try {
      setErrorMessage(null);
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          urls,
          options: {
            batch_size: 8,
            max_length: 256,
            page_threshold: 0.25,
            seg_threshold: 0.4,
          },
        }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "API request failed");
      }
      const data = (await response.json()) as AnalyzeResponse;
      setJobId(data.job_id);
      setAnalysisResults(data.results || []);
      setCurrentPage("results");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setErrorMessage(message);
    }
  };

  const handleScanAgain = () => {
    setCurrentPage("home");
    setAnalysisResults([]);
    setJobId(null);
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
          errorMessage={errorMessage}
          onClearError={() => setErrorMessage(null)}
        />
      )}
      
      {currentPage === "results" && (
        <ResultsPage
          results={analysisResults}
          jobId={jobId}
          onScanAgain={handleScanAgain}
        />
      )}
      
      {currentPage === "model" && <ModelPage onTryNow={handleTryNow} />}
      
      {currentPage === "contact" && <ContactPage />}
    </div>
  );
}
