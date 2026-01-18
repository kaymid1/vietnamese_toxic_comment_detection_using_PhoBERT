import { useState } from "react";
import { Button } from "@/app/components/ui/button";
import { Textarea } from "@/app/components/ui/textarea";
import { AlertCircle, Brain, FileSearch, TrendingUp } from "lucide-react";
import { Progress } from "@/app/components/ui/progress";

interface HomePageProps {
  onAnalyze: (urls: string[]) => Promise<void>;
  errorMessage?: string | null;
  onClearError?: () => void;
}

export function HomePage({ onAnalyze, errorMessage, onClearError }: HomePageProps) {
  const [urlInput, setUrlInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleAnalyze = async () => {
    if (!urlInput.trim()) return;

    // Parse URLs (separated by newlines or commas)
    const urls = urlInput
      .split(/[\n,]/)
      .map((url) => url.trim())
      .filter((url) => url.length > 0);

    if (urls.length === 0) return;

    onClearError?.();
    setIsProcessing(true);
    setProgress(10);

    const interval = setInterval(() => {
      setProgress((prev) => (prev >= 90 ? 90 : prev + 10));
    }, 500);

    try {
      await onAnalyze(urls);
      setProgress(100);
      setTimeout(() => setIsProcessing(false), 300);
    } finally {
      clearInterval(interval);
    }
  };

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

            <Textarea
              placeholder="https://example.com/article1&#10;https://example.com/article2"
              className="min-h-[160px] mb-4 text-base border-gray-300 focus:border-[var(--viet-primary)] focus:ring-[var(--viet-primary)]"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              disabled={isProcessing}
            />

            <div className="flex items-start gap-3 mb-6 p-4 bg-blue-50 rounded-lg">
              <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-gray-700">
                Hệ thống sẽ tự động cào dữ liệu từ các URL và chạy mô hình PhoBERT
                để phát hiện nội dung độc hại. Quá trình này có thể mất vài giây.
              </p>
            </div>

            {isProcessing && (
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600">
                    Đang cào dữ liệu và chạy mô hình PhoBERT…
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
              disabled={isProcessing || !urlInput.trim()}
              className="w-full h-12 text-base"
              style={{
                backgroundColor: isProcessing || !urlInput.trim() ? "#94a3b8" : "var(--viet-primary)",
              }}
            >
              {isProcessing ? "Đang xử lý..." : "Quét Nội Dung"}
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
