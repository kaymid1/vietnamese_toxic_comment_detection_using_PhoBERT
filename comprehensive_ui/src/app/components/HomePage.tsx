import { useEffect, useState } from "react";
import { Button } from "@/app/components/ui/button";
import { Textarea } from "@/app/components/ui/textarea";
import { AlertCircle, Brain, FileSearch, TrendingUp } from "lucide-react";
import { Progress } from "@/app/components/ui/progress";

interface HomePageProps {
  onAnalyze: (urls: string[], modelName?: string | null) => Promise<void>;
  onCompare: (urls: string[], modelNames: string[]) => Promise<void>;
  onRerun: (jobId: string, modelName?: string | null) => Promise<void>;
  compareMode: boolean;
  onToggleCompare: (value: boolean) => void;
  availableModels: string[];
  selectedModel: string | null;
  onSelectModel: (modelName: string) => void;
  modelsLoading: boolean;
  modelsError?: string | null;
  errorMessage?: string | null;
  onClearError?: () => void;
}

export function HomePage({
  onAnalyze,
  onCompare,
  onRerun,
  compareMode,
  onToggleCompare,
  availableModels,
  selectedModel,
  onSelectModel,
  modelsLoading,
  modelsError,
  errorMessage,
  onClearError,
}: HomePageProps) {
  const [urlInput, setUrlInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [compareInitialized, setCompareInitialized] = useState(false);
  const [rerunJobId, setRerunJobId] = useState("");
  const [rerunModelId, setRerunModelId] = useState<string | null>(null);
  const [rerunStatus, setRerunStatus] = useState<string | null>(null);
  const [rerunLoading, setRerunLoading] = useState(false);

  useEffect(() => {
    if (compareMode) {
      if (!compareInitialized) {
        setSelectedModels(selectedModel ? [selectedModel] : []);
        setCompareInitialized(true);
      }
    } else {
      setCompareInitialized(false);
      if (selectedModel) {
        setSelectedModels([selectedModel]);
      }
    }
  }, [compareMode, compareInitialized, selectedModel]);

  const handleAnalyze = async () => {
    if (!urlInput.trim()) return;

    // Parse URLs (separated by newlines or commas)
    const urls = urlInput
      .split(/[\n,]/)
      .map((url) => url.trim())
      .filter((url) => url.length > 0);

    if (urls.length === 0) return;

    const modelsForCompare = selectedModels.filter((model) => availableModels.includes(model));
    if (compareMode) {
      if (modelsForCompare.length < 2) return;
    } else if (!selectedModel) {
      return;
    }

    onClearError?.();
    setIsProcessing(true);
    setProgress(10);

    const interval = setInterval(() => {
      setProgress((prev) => (prev >= 90 ? 90 : prev + 10));
    }, 500);

    try {
      if (compareMode) {
        await onCompare(urls, modelsForCompare);
      } else {
        await onAnalyze(urls, selectedModel);
      }
      setProgress(100);
      setTimeout(() => setIsProcessing(false), 300);
    } finally {
      clearInterval(interval);
    }
  };

  const handleRerun = async () => {
    if (!rerunJobId.trim()) {
      setRerunStatus("Vui lòng nhập job_id cần re-run.");
      return;
    }
    if (compareMode && selectedModels.length > 0 && !rerunModelId) {
      setRerunStatus("Chọn model để re-run từ job compare.");
      return;
    }
    try {
      setRerunStatus(null);
      setRerunLoading(true);
      await onRerun(rerunJobId.trim(), rerunModelId ?? null);
      setRerunStatus("Đã re-run thành công.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Re-run thất bại";
      setRerunStatus(message);
    } finally {
      setRerunLoading(false);
    }
  };

  useEffect(() => {
    if (!compareMode) {
      setRerunModelId(null);
    } else if (selectedModels.length > 0 && !rerunModelId) {
      setRerunModelId(selectedModels[0]);
    }
  }, [compareMode, selectedModels, rerunModelId]);

  return (
    <div style={{ backgroundColor: "var(--viet-bg)" }} className="min-h-screen">
      {/* Hero Section */}
      <section className="pt-20 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h1
            className="text-5xl mb-6 tracking-tight"
            style={{ color: "var(--viet-primary)" }}
          >
            VietToxic Detector – Phát Hiện Nội Dung Độc Hại Tiếng Việt
          </h1>
          <p className="text-xl text-gray-600 mb-12 max-w-3xl mx-auto leading-relaxed">
            Phân tích nội dung từ URL bằng AI PhoBERT để phát hiện ngôn ngữ độc hại
            và hỗ trợ người đọc.
          </p>

          {/* Feature Icons */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
            <div className="bg-white p-6 rounded-xl shadow-sm">
              <div
                className="w-12 h-12 rounded-lg mx-auto mb-4 flex items-center justify-center"
                style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
              >
                <Brain className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
              </div>
              <h3 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                AI PhoBERT
              </h3>
              <p className="text-gray-600">
                Mô hình ngôn ngữ tiên tiến được tinh chỉnh cho tiếng Việt
              </p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-sm">
              <div
                className="w-12 h-12 rounded-lg mx-auto mb-4 flex items-center justify-center"
                style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
              >
                <FileSearch className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
              </div>
              <h3 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                Phân Tích Chi Tiết
              </h3>
              <p className="text-gray-600">
                Tự động cào và phân tích nội dung từ bất kỳ URL nào
              </p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-sm">
              <div
                className="w-12 h-12 rounded-lg mx-auto mb-4 flex items-center justify-center"
                style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
              >
                <TrendingUp className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
              </div>
              <h3 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                Kết Quả Trực Quan
              </h3>
              <p className="text-gray-600">
                Xem điểm số độc hại, biểu đồ và khuyến nghị rõ ràng
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* URL Input Section */}
      <section className="pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mx-auto">
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <h2 className="text-2xl mb-4" style={{ color: "var(--viet-primary)" }}>
              Nhập URL để Phân Tích
            </h2>
            <p className="text-gray-600 mb-6">
              Nhập một hoặc nhiều URL của bài viết hoặc trang web bạn muốn kiểm tra.
              Mỗi URL trên một dòng hoặc phân cách bằng dấu phẩy.
            </p>

            <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
              <div>
                <Textarea
                  placeholder="https://example.com/article1&#10;https://example.com/article2"
                  className="min-h-[160px] mb-4 text-base border-gray-300 focus:border-[var(--viet-primary)] focus:ring-[var(--viet-primary)]"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  disabled={isProcessing}
                />
              </div>
              <div className="rounded-xl border border-gray-200 bg-gray-50 p-4">
                <h3 className="text-sm font-semibold text-gray-700 mb-2">Re-run theo Job ID</h3>
                <p className="text-xs text-gray-600 mb-3">
                  Dùng lại dữ liệu crawl đã có (bao gồm segments.jsonl đã chỉnh sửa).
                </p>
                <div className="space-y-2 mb-3">
                  <label className="block text-xs font-medium text-gray-600" htmlFor="rerun-job-id">
                    Job ID
                  </label>
                  <input
                    id="rerun-job-id"
                    className="w-full h-9 rounded-md border border-gray-300 px-3 text-sm bg-white"
                    value={rerunJobId}
                    onChange={(event) => setRerunJobId(event.target.value)}
                    placeholder="Nhập job_id"
                  />
                </div>
                {compareMode && selectedModels.length > 0 && (
                  <div className="space-y-2 mb-3">
                    <label className="block text-xs font-medium text-gray-600">Model</label>
                    <select
                      className="w-full h-9 rounded-md border border-gray-300 px-3 text-sm bg-white"
                      value={rerunModelId ?? ""}
                      onChange={(event) => setRerunModelId(event.target.value)}
                      disabled={selectedModels.length === 0}
                    >
                      <option value="">Chọn model</option>
                      {selectedModels.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                <Button
                  onClick={handleRerun}
                  disabled={rerunLoading}
                  className="h-9 w-full text-sm"
                  style={{ backgroundColor: "var(--viet-primary)" }}
                >
                  {rerunLoading ? "Đang re-run..." : "Re-run"}
                </Button>
                {rerunStatus && <p className="mt-2 text-xs text-gray-600">{rerunStatus}</p>}
              </div>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Model</label>
              <div className="flex items-center gap-3 mb-3">
                <input
                  id="compare-mode"
                  type="checkbox"
                  checked={compareMode}
                  onChange={(e) => onToggleCompare(e.target.checked)}
                  disabled={isProcessing}
                />
                <label htmlFor="compare-mode" className="text-sm text-gray-700">
                  So sánh nhiều model (chạy song song)
                </label>
              </div>

              {!compareMode && (
                <select
                  className="w-full h-11 rounded-md border border-gray-300 px-3 text-sm bg-white disabled:bg-gray-100 disabled:text-gray-500"
                  value={selectedModel ?? ""}
                  onChange={(e) => onSelectModel(e.target.value)}
                  disabled={isProcessing || modelsLoading || availableModels.length === 0}
                >
                  {modelsLoading && <option value="">Đang tải danh sách model...</option>}
                  {!modelsLoading && availableModels.length === 0 && (
                    <option value="">Không có model khả dụng</option>
                  )}
                  {availableModels.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              )}

              {compareMode && (
                <div className="grid gap-2 rounded-md border border-gray-200 p-3">
                  {availableModels.map((model) => {
                    const checked = selectedModels.includes(model);
                    return (
                      <label key={model} className="flex items-center gap-2 text-sm text-gray-700">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(e) => {
                            setSelectedModels((prev) => {
                              if (e.target.checked) {
                                return Array.from(new Set([...prev, model]));
                              }
                              return prev.filter((m) => m !== model);
                            });
                          }}
                          disabled={isProcessing}
                        />
                        {model}
                      </label>
                    );
                  })}
                  {selectedModels.length > 0 && selectedModels.length < 2 && (
                    <p className="text-xs text-amber-700">Chọn ít nhất 2 model để so sánh.</p>
                  )}
                </div>
              )}

              {modelsError && (
                <p className="mt-2 text-sm text-red-700">
                  Không thể tải danh sách model: {modelsError}
                </p>
              )}
              {!modelsLoading && !modelsError && availableModels.length === 0 && (
                <p className="mt-2 text-sm text-amber-700">
                  Không tìm thấy model trong thư mục `models/options/`. Vui lòng kiểm tra backend.
                </p>
              )}
            </div>

            <div className="flex items-start gap-3 mb-6 p-4 bg-blue-50 rounded-lg">
              <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-gray-700">
                Hệ thống sẽ tự động cào dữ liệu từ các URL và chạy mô hình đã chọn
                để phát hiện nội dung độc hại. Quá trình này có thể mất vài giây.
              </p>
            </div>

            {isProcessing && (
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600">
                    Đang cào dữ liệu và chạy mô hình {selectedModel ?? "N/A"}…
                  </span>
                  <span className="text-sm" style={{ color: "var(--viet-primary)" }}>
                    {progress}%
                  </span>
                </div>
                <Progress value={progress} className="h-2" />
              </div>
            )}

            {errorMessage && (
              <div className="mb-6 p-4 rounded-lg border border-red-200 bg-red-50 text-sm text-red-700">
                {errorMessage}
              </div>
            )}

            <Button
              onClick={handleAnalyze}
              disabled={
                isProcessing ||
                !urlInput.trim() ||
                modelsLoading ||
                (!compareMode && !selectedModel) ||
                (compareMode && selectedModels.length < 2)
              }
              className="w-full h-12 text-base"
              style={{
                backgroundColor:
                  isProcessing ||
                  !urlInput.trim() ||
                  modelsLoading ||
                  (!compareMode && !selectedModel) ||
                  (compareMode && selectedModels.length < 2)
                    ? "#94a3b8"
                    : "var(--viet-primary)",
              }}
            >
              {isProcessing ? "Đang xử lý..." : compareMode ? "So Sánh Model" : "Quét Nội Dung"}
            </Button>

            <p className="text-sm text-gray-500 text-center mt-4">
              Kết quả phân tích sẽ hiển thị ngay sau khi quét xong
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h4 className="mb-3" style={{ color: "var(--viet-primary)" }}>
                Về VietToxic Detector
              </h4>
              <p className="text-sm text-gray-600">
                Hệ thống AI phát hiện nội dung độc hại bằng mô hình PhoBERT được
                tinh chỉnh cho tiếng Việt.
              </p>
            </div>

            <div>
              <h4 className="mb-3" style={{ color: "var(--viet-primary)" }}>
                Lưu Ý Quan Trọng
              </h4>
              <p className="text-sm text-gray-600">
                Kết quả dự đoán mang tính hỗ trợ và không thay thế được đánh giá
                của con người. Vui lòng sử dụng cẩn trọng.
              </p>
            </div>

            <div>
              <h4 className="mb-3" style={{ color: "var(--viet-primary)" }}>
                Quyền Riêng Tư
              </h4>
              <p className="text-sm text-gray-600">
                Chúng tôi không lưu trữ URL hoặc nội dung bạn phân tích. Mọi dữ
                liệu chỉ được xử lý tạm thời.
              </p>
            </div>
          </div>

          <div className="mt-8 pt-8 border-t border-gray-200 text-center text-sm text-gray-500">
            © 2026 VietToxic Detector. Được phát triển với PhoBERT.
          </div>
        </div>
      </footer>
    </div>
  );
}
