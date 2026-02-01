import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Progress } from "@/app/components/ui/progress";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/app/components/ui/tooltip";
import { AlertTriangle, CheckCircle, Download, ExternalLink, Info, RotateCcw } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip as RechartTooltip } from "recharts";

interface SegmentData {
  segment_id: string;
  score: number;
  text_preview: string;
  text?: string;
}

interface ResultData {
  url: string;
  status: "ok" | "error";
  error?: string | null;
  crawl_output_dir?: string | null;
  segments_path?: string | null;
  toxicity?: {
    overall?: number | null;
    by_segment?: SegmentData[];
  };
}

interface ResultsPageProps {
  results: ResultData[];
  jobId?: string | null;
  thresholds?: {
    seg_threshold?: number;
    page_threshold?: number;
  } | null;
  onScanAgain: () => void;
}

const parseDomain = (url: string) => {
  try {
    const urlObj = new URL(url.startsWith("http") ? url : `https://${url}`);
    return urlObj.hostname;
  } catch {
    return "unknown";
  }
};

export function ResultsPage({ results, jobId, thresholds, onScanAgain }: ResultsPageProps) {
  return (
    <div style={{ backgroundColor: "var(--viet-bg)" }} className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl mb-3" style={{ color: "var(--viet-primary)" }}>
            Kết Quả Phân Tích
          </h1>
          <p className="text-gray-600">
            Phân tích hoàn tất cho {results.length} URL
          </p>
          {jobId && (
            <p className="text-sm text-gray-500 mt-2">
              Job ID: {jobId}
            </p>
          )}
        </div>

        {/* Results */}
        {results.map((result, index) => {
          const domain = parseDomain(result.url);
          const segments = result.toxicity?.by_segment ?? [];
          const overallScore = result.toxicity?.overall;
          const overallPercent = typeof overallScore === "number" ? Math.round(overallScore * 100) : null;
          const segThreshold = thresholds?.seg_threshold ?? 0.5;
          const toxicCount = segments.filter((s) => s.score >= segThreshold).length;
          const safeCount = segments.length - toxicCount;
          const isToxic = overallPercent !== null ? overallPercent > 50 : false;
          const topSegments = [...segments]
            .sort((a, b) => b.score - a.score)
            .slice(0, 3);

          return (
            <Card key={index} className="bg-white p-8 mb-6 shadow-lg">
              {/* URL Header */}
              <div className="mb-6 pb-6 border-b border-gray-200">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <h2 className="text-2xl mb-2" style={{ color: "var(--viet-primary)" }}>
                      Phân tích URL
                    </h2>
                    <div className="flex items-center gap-2 text-gray-600">
                      <span className="text-sm font-medium">{domain}</span>
                      <a href={result.url} target="_blank" rel="noreferrer">
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    </div>
                    <p className="text-sm text-gray-500 mt-1 break-all">{result.url}</p>
                    <p className="text-xs mt-2" style={{ color: result.status === "ok" ? "var(--viet-safe)" : "var(--viet-toxic)" }}>
                      Trạng thái: {result.status === "ok" ? "Thành công" : "Lỗi"}
                    </p>
                  </div>
                </div>
              </div>

              {result.status !== "ok" && (
                <div className="mb-8 p-4 rounded-lg border border-red-200 bg-red-50 text-sm text-red-700">
                  {result.error || "Không thể phân tích URL này."}
                </div>
              )}

              {result.status === "ok" && (
                <>
                  {/* Toxicity Score */}
                  <div className="mb-8">
                    <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl flex items-center gap-2" style={{ color: "var(--viet-primary)" }}>
                  Điểm Độc Hại Tổng Thể
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-blue-50 text-blue-700">
                        <Info className="w-4 h-4" />
                      </span>
                    </TooltipTrigger>
                    <TooltipContent side="top">
                      Điểm này là ước lượng rủi ro ở mức toàn trang (tính từ xác suất mô hình trên các đoạn), không đồng nghĩa chắc chắn có đoạn bị gắn nhãn độc hại.
                    </TooltipContent>
                  </Tooltip>
                </h3>
                      <span className="text-3xl" style={{ 
                        color: isToxic ? "var(--viet-toxic)" : "var(--viet-safe)" 
                      }}>
                        {overallPercent !== null ? `${overallPercent}%` : "--"}
                      </span>
                    </div>
                    <Progress 
                      value={overallPercent ?? 0} 
                      className="h-4"
                      style={{
                        backgroundColor: "#e5e7eb",
                      }}
                    />
                    <div className="flex justify-between mt-2 text-sm">
                      <span style={{ color: "var(--viet-safe)" }}>An Toàn</span>
                      <span style={{ color: "var(--viet-toxic)" }}>Độc Hại</span>
                    </div>
                  </div>

                  {/* Visual Analytics */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                    {/* Pie Chart */}
                    <div>
                      <h3 className="mb-4" style={{ color: "var(--viet-primary)" }}>
                        Phân Bố Nội Dung
                      </h3>
                      <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                          <Pie
                            data={[
                              { name: "Độc hại", value: overallPercent ?? 0 },
                              { name: "An toàn", value: 100 - (overallPercent ?? 0) },
                            ]}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            <Cell fill="var(--viet-toxic)" />
                            <Cell fill="var(--viet-safe)" />
                          </Pie>
                    <RechartTooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>

                    {/* Stats */}
                    <div className="space-y-4">
                      <h3 className="mb-4" style={{ color: "var(--viet-primary)" }}>
                        Thống Kê Chi Tiết
                      </h3>
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-gray-600">Tổng đoạn văn phân tích:</span>
                          <span className="text-xl">{segments.length}</span>
                        </div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-gray-600">Đoạn độc hại phát hiện:</span>
                          <span className="text-xl" style={{ color: "var(--viet-toxic)" }}>
                            {toxicCount}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-600">Đoạn an toàn:</span>
                          <span className="text-xl" style={{ color: "var(--viet-safe)" }}>
                            {safeCount}
                          </span>
                </div>
              </div>
            </div>

            {segments.length > 0 && (
              <div className="mb-8">
                <h3 className="text-xl mb-4" style={{ color: "var(--viet-primary)" }}>
                  Đoạn có rủi ro cao nhất (Top 3)
                </h3>
                <div className="space-y-3">
                  {topSegments.map((segment, idx) => {
                    const segmentIsToxic = segment.score >= segThreshold;
                    const percent = (segment.score * 100).toFixed(1);
                    const fullText = segment.text || segment.text_preview;
                    return (
                      <div
                        key={segment.segment_id || idx}
                        className="p-4 rounded-lg border border-gray-200 bg-white"
                      >
                        <div className="flex items-start justify-between gap-4 mb-2">
                          <div className="flex items-center gap-3">
                            <span className="text-sm font-medium text-gray-600">#{idx + 1}</span>
                            <span className="text-sm font-semibold" style={{ color: "var(--viet-primary)" }}>
                              {percent}%
                            </span>
                          </div>
                          <span
                            className={`text-xs px-2 py-1 rounded-full ${
                              segmentIsToxic ? "bg-red-100 text-red-700" : "bg-yellow-100 text-yellow-800"
                            }`}
                          >
                            {segmentIsToxic ? "Độc hại (>= seg_threshold)" : "Rủi ro (chưa vượt ngưỡng)"}
                          </span>
                        </div>
                        <details className="text-sm text-gray-700">
                          <summary className="cursor-pointer">
                            {segment.text_preview}
                          </summary>
                          <div className="mt-2 flex items-center justify-between gap-3">
                            <p className="whitespace-pre-wrap flex-1">{fullText}</p>
                            <button
                              type="button"
                              className="text-xs text-blue-700 hover:underline"
                              onClick={() => navigator.clipboard?.writeText(fullText)}
                            >
                              Copy
                            </button>
                          </div>
                        </details>
                      </div>
                    );
                  })}
                </div>
                <p className="text-xs text-gray-500 mt-3">
                  Các đoạn dưới đây có xác suất độc hại cao nhất theo mô hình. Chúng có thể chưa đủ để được gắn nhãn ‘Độc hại’ nếu chưa vượt ngưỡng seg_threshold.
                </p>
              </div>
            )}
                  </div>

                  {/* Content Explainability */}
                  <div className="mb-8">
                    <h3 className="text-xl mb-4" style={{ color: "var(--viet-primary)" }}>
                      Chi Tiết Phát Hiện (XAI - Explainable AI)
                    </h3>
                    {segments.length === 0 && (
                      <p className="text-sm text-gray-500">
                        Chưa có dữ liệu chi tiết theo từng đoạn.
                      </p>
                    )}
                    <div className="space-y-3">
                      {segments.map((segment, idx) => {
                        const segmentIsToxic = segment.score >= segThreshold;
                        return (
                          <div
                            key={segment.segment_id || idx}
                            className="p-4 rounded-lg border-l-4"
                            style={{
                              backgroundColor: segmentIsToxic ? "rgba(255, 51, 51, 0.05)" : "rgba(0, 204, 102, 0.05)",
                              borderLeftColor: segmentIsToxic ? "var(--viet-toxic)" : "var(--viet-safe)",
                            }}
                          >
                            <div className="flex items-start justify-between gap-4 mb-2">
                              <p className="flex-1">{segment.text_preview}</p>
                              <div className="flex items-center gap-2 flex-shrink-0">
                                <span className="text-sm font-medium" style={{ 
                                  color: segmentIsToxic ? "var(--viet-toxic)" : "var(--viet-safe)" 
                                }}>
                                  {(segment.score * 100).toFixed(1)}%
                                </span>
                                {segmentIsToxic ? (
                                  <AlertTriangle className="w-5 h-5" style={{ color: "var(--viet-toxic)" }} />
                                ) : (
                                  <CheckCircle className="w-5 h-5" style={{ color: "var(--viet-safe)" }} />
                                )}
                              </div>
                            </div>
                            <p className="text-xs text-gray-500">
                              {segmentIsToxic 
                                ? "Đoạn này được gắn nhãn độc hại do mô hình PhoBERT dự đoán"
                                : "Đoạn này được đánh giá là an toàn"
                              }
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Recommendations */}
                  <div className={`p-6 rounded-lg border-l-4 ${
                    isToxic 
                      ? "bg-red-50 border-red-500" 
                      : "bg-green-50 border-green-500"
                  }`}>
                    <div className="flex items-start gap-3">
                      {isToxic ? (
                        <AlertTriangle className="w-6 h-6 mt-1 flex-shrink-0" style={{ color: "var(--viet-toxic)" }} />
                      ) : (
                        <CheckCircle className="w-6 h-6 mt-1 flex-shrink-0" style={{ color: "var(--viet-safe)" }} />
                      )}
                      <div>
                        <h4 className="mb-2" style={{ 
                          color: isToxic ? "var(--viet-toxic)" : "var(--viet-safe)" 
                        }}>
                          {isToxic ? "⚠️ Cảnh Báo Nội Dung" : "✅ Nội Dung An Toàn"}
                        </h4>
                        <p className="text-gray-700">
                          {isToxic 
                            ? "Nội dung có thể chứa yếu tố độc hại. Khuyến nghị đọc cẩn trọng và tránh lan truyền. Hãy kiểm chứng thông tin từ nhiều nguồn đáng tin cậy."
                            : "Nội dung tương đối an toàn cho người đọc. Tuy nhiên, vẫn nên duy trì suy nghĩ phản biện khi tiếp nhận thông tin."
                          }
                        </p>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </Card>
          );
        })}

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center mt-8">
          <Button
            onClick={onScanAgain}
            className="h-12 px-8"
            style={{ backgroundColor: "var(--viet-primary)" }}
          >
            <RotateCcw className="w-5 h-5 mr-2" />
            Quét URL Khác
          </Button>
          <Button
            variant="outline"
            className="h-12 px-8 border-2"
            style={{ 
              borderColor: "var(--viet-primary)", 
              color: "var(--viet-primary)" 
            }}
          >
            <Download className="w-5 h-5 mr-2" />
            Xuất Báo Cáo (PDF)
          </Button>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 p-6 bg-white rounded-lg shadow-sm">
          <p className="text-sm text-gray-600 text-center">
            <strong>Lưu ý:</strong> Kết quả phân tích được tạo bởi AI và mang tính chất tham khảo. 
            Độ chính xác của mô hình PhoBERT hiện tại là ~71% (Macro-F1). 
            Vui lòng không hoàn toàn dựa vào kết quả này để đưa ra các quyết định quan trọng.
          </p>
        </div>
      </div>
    </div>
  );
}
