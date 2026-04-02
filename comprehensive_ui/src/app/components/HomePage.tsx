import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import { Button } from "@/app/components/ui/button";
import { Textarea } from "@/app/components/ui/textarea";

interface HomePageProps {
  onAnalyze: (urls: string[], modelNames: string[]) => Promise<void>;
  availableModels: string[];
  selectedModels: string[];
  onSelectModels: (modelNames: string[]) => void;
  modelsLoading: boolean;
  modelsError?: string | null;
  errorMessage?: string | null;
  onClearError?: () => void;
  analysisProgress?: number | null;
}

export function HomePage({
  onAnalyze,
  availableModels,
  selectedModels,
  onSelectModels,
  modelsLoading,
  modelsError,
  errorMessage,
  onClearError,
  analysisProgress,
}: HomePageProps) {
  const [urlInput, setUrlInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [uiProgress, setUiProgress] = useState(0);

  const primaryModel = selectedModels[0] ?? "";
  const compareModel = selectedModels[1] ?? "";
  const compareEnabled = selectedModels.length > 1;
  const compareCandidates = useMemo(
    () => availableModels.filter((model) => model !== primaryModel),
    [availableModels, primaryModel],
  );
  const targetProgress = Math.max(0, Math.min(100, Math.round(analysisProgress ?? 0)));
  const progressPercent = isProcessing
    ? Math.max(1, Math.min(100, Math.round(uiProgress)))
    : Math.max(0, Math.min(100, Math.round(uiProgress)));
  const progressCircumference = 2 * Math.PI * 7;
  const progressOffset = progressCircumference * (1 - progressPercent / 100);

  useEffect(() => {
    if (!isProcessing) {
      setUiProgress(0);
      return;
    }

    const tick = window.setInterval(() => {
      setUiProgress((prev) => {
        const synced = targetProgress > 0 ? Math.max(prev, targetProgress) : prev;
        const step = synced < 70 ? 2.4 : synced < 90 ? 0.9 : synced < 97 ? 0.25 : 0;
        return Math.min(97, synced + step);
      });
    }, 120);

    return () => window.clearInterval(tick);
  }, [isProcessing, targetProgress]);

  const handleAnalyze = async () => {
    if (!urlInput.trim() || selectedModels.length === 0) return;

    const urls = urlInput
      .split(/[\n,]/)
      .map((url) => url.trim())
      .filter((url) => url.length > 0);

    if (urls.length === 0) return;

    onClearError?.();
    setUiProgress(1);
    setIsProcessing(true);

    try {
      await onAnalyze(urls, selectedModels);
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePrimaryModelChange = (nextModel: string) => {
    if (!nextModel) return;
    if (!compareEnabled) {
      onSelectModels([nextModel]);
      return;
    }
    const nextCompare = compareModel && compareModel !== nextModel ? compareModel : "";
    onSelectModels(nextCompare ? [nextModel, nextCompare] : [nextModel]);
  };

  const handleToggleCompare = () => {
    if (!compareEnabled) {
      const fallback = compareCandidates[0];
      onSelectModels(fallback ? [primaryModel, fallback] : [primaryModel]);
      return;
    }
    onSelectModels(primaryModel ? [primaryModel] : []);
  };

  const handleCompareModelChange = (nextModel: string) => {
    if (!nextModel || !primaryModel || nextModel === primaryModel) {
      onSelectModels(primaryModel ? [primaryModel] : []);
      return;
    }
    onSelectModels([primaryModel, nextModel]);
  };

  const handleUrlInputChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setUrlInput(event.target.value);
  };

  const handlePrimarySelectChange = (event: ChangeEvent<HTMLSelectElement>) => {
    handlePrimaryModelChange(event.target.value);
  };

  const handleCompareSelectChange = (event: ChangeEvent<HTMLSelectElement>) => {
    handleCompareModelChange(event.target.value);
  };

  return (
    <div className="min-h-screen bg-background px-4 py-16 sm:px-6 lg:px-8">
      <div className="mx-auto w-full max-w-5xl">
        <div className="mb-8 text-center">
          <h1 className="text-4xl tracking-tight text-primary">VietToxic Detector</h1>
          <p className="mt-3 text-muted-foreground">Phân tích URL với bố cục gọn, tập trung vào thao tác chính.</p>
        </div>

        <div className="rounded-[28px] border border-border bg-card p-6 shadow-sm sm:p-8">
          <Textarea
            placeholder="Dán URL vào đây (mỗi dòng một URL)&#10;https://example.com/article-1"
            className="min-h-[220px] resize-none border-0 bg-transparent px-0 text-lg shadow-none focus-visible:ring-0"
            value={urlInput}
            onChange={handleUrlInputChange}
            disabled={isProcessing}
          />

          <div className="mt-4 flex flex-col gap-3 border-t border-border pt-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-muted-foreground">Model</span>
              <select
                className="h-9 rounded-full border border-border bg-card px-3 text-sm text-foreground"
                value={primaryModel}
                onChange={handlePrimarySelectChange}
                disabled={isProcessing || modelsLoading || availableModels.length === 0}
              >
                {modelsLoading && <option value="">Đang tải...</option>}
                {!modelsLoading && availableModels.length === 0 && <option value="">Không có model</option>}
                {availableModels.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>

              <button
                type="button"
                onClick={handleToggleCompare}
                disabled={isProcessing || modelsLoading || !primaryModel || compareCandidates.length === 0}
                className={`h-9 rounded-full border px-3 text-sm transition ${
                  compareEnabled
                    ? "border-primary bg-accent text-primary"
                    : "border-border bg-card text-muted-foreground"
                } disabled:opacity-50`}
              >
                Compare
              </button>

              {compareEnabled && (
                <select
                  className="h-9 rounded-full border border-border bg-card px-3 text-sm text-foreground"
                  value={compareModel}
                  onChange={handleCompareSelectChange}
                  disabled={isProcessing || modelsLoading || compareCandidates.length === 0}
                >
                  {compareCandidates.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              )}
            </div>

            <Button
              onClick={handleAnalyze}
              disabled={isProcessing || !urlInput.trim() || modelsLoading || selectedModels.length === 0}
              className="h-10 rounded-full px-5"
            >
              {isProcessing ? (
                <span className="flex items-center gap-2">
                  <svg className="h-4 w-4" viewBox="0 0 16 16" aria-hidden="true">
                    <circle cx="8" cy="8" r="7" fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth="2" />
                    <circle
                      cx="8"
                      cy="8"
                      r="7"
                      fill="none"
                      stroke="white"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeDasharray={progressCircumference}
                      strokeDashoffset={progressOffset}
                      transform="rotate(-90 8 8)"
                    />
                  </svg>
                  <span>Đang xử lý {progressPercent}%</span>
                </span>
              ) : (
                "Quét nội dung"
              )}
            </Button>
          </div>

          {modelsError && <p className="mt-4 text-sm text-destructive">Không thể tải model: {modelsError}</p>}
          {errorMessage && <p className="mt-2 text-sm text-destructive">{errorMessage}</p>}
        </div>
      </div>
    </div>
  );
}
