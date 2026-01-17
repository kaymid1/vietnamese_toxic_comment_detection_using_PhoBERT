import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Progress } from "@/app/components/ui/progress";
import { AlertTriangle, CheckCircle, Download, ExternalLink, RotateCcw } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";

interface ResultData {
  url: string;
  title: string;
  domain: string;
  toxicityScore: number;
  toxicSegments: {
    text: string;
    confidence: number;
    isToxic: boolean;
  }[];
}

interface ResultsPageProps {
  results: ResultData[];
  onScanAgain: () => void;
}

export function ResultsPage({ results, onScanAgain }: ResultsPageProps) {
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
        </div>

        {/* Results */}
        {results.map((result, index) => (
          <Card key={index} className="bg-white p-8 mb-6 shadow-lg">
            {/* URL Header */}
            <div className="mb-6 pb-6 border-b border-gray-200">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <h2 className="text-2xl mb-2" style={{ color: "var(--viet-primary)" }}>
                    {result.title}
                  </h2>
                  <div className="flex items-center gap-2 text-gray-600">
                    <span className="text-sm font-medium">{result.domain}</span>
                    <ExternalLink className="w-4 h-4" />
                  </div>
                  <p className="text-sm text-gray-500 mt-1 break-all">{result.url}</p>
                </div>
              </div>
            </div>

            {/* Toxicity Score */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl" style={{ color: "var(--viet-primary)" }}>
                  Điểm Độc Hại Tổng Thể
                </h3>
                <span className="text-3xl" style={{ 
                  color: result.toxicityScore > 50 ? "var(--viet-toxic)" : "var(--viet-safe)" 
                }}>
                  {result.toxicityScore}%
                </span>
              </div>
              <Progress 
                value={result.toxicityScore} 
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
                        { name: "Độc hại", value: result.toxicityScore },
                        { name: "An toàn", value: 100 - result.toxicityScore },
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
                    <Tooltip />
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
                    <span className="text-xl">{result.toxicSegments.length}</span>
                  </div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-gray-600">Đoạn độc hại phát hiện:</span>
                    <span className="text-xl" style={{ color: "var(--viet-toxic)" }}>
                      {result.toxicSegments.filter((s) => s.isToxic).length}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Đoạn an toàn:</span>
                    <span className="text-xl" style={{ color: "var(--viet-safe)" }}>
                      {result.toxicSegments.filter((s) => !s.isToxic).length}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Content Explainability */}
            <div className="mb-8">
              <h3 className="text-xl mb-4" style={{ color: "var(--viet-primary)" }}>
                Chi Tiết Phát Hiện (XAI - Explainable AI)
              </h3>
              <div className="space-y-3">
                {result.toxicSegments.map((segment, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-lg border-l-4"
                    style={{
                      backgroundColor: segment.isToxic ? "rgba(255, 51, 51, 0.05)" : "rgba(0, 204, 102, 0.05)",
                      borderLeftColor: segment.isToxic ? "var(--viet-toxic)" : "var(--viet-safe)",
                    }}
                  >
                    <div className="flex items-start justify-between gap-4 mb-2">
                      <p className="flex-1">{segment.text}</p>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className="text-sm font-medium" style={{ 
                          color: segment.isToxic ? "var(--viet-toxic)" : "var(--viet-safe)" 
                        }}>
                          {(segment.confidence * 100).toFixed(1)}%
                        </span>
                        {segment.isToxic ? (
                          <AlertTriangle className="w-5 h-5" style={{ color: "var(--viet-toxic)" }} />
                        ) : (
                          <CheckCircle className="w-5 h-5" style={{ color: "var(--viet-safe)" }} />
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-gray-500">
                      {segment.isToxic 
                        ? "Đoạn này được gắn nhãn độc hại do mô hình PhoBERT dự đoán"
                        : "Đoạn này được đánh giá là an toàn"
                      }
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            <div className={`p-6 rounded-lg border-l-4 ${
              result.toxicityScore > 50 
                ? "bg-red-50 border-red-500" 
                : "bg-green-50 border-green-500"
            }`}>
              <div className="flex items-start gap-3">
                {result.toxicityScore > 50 ? (
                  <AlertTriangle className="w-6 h-6 mt-1 flex-shrink-0" style={{ color: "var(--viet-toxic)" }} />
                ) : (
                  <CheckCircle className="w-6 h-6 mt-1 flex-shrink-0" style={{ color: "var(--viet-safe)" }} />
                )}
                <div>
                  <h4 className="mb-2" style={{ 
                    color: result.toxicityScore > 50 ? "var(--viet-toxic)" : "var(--viet-safe)" 
                  }}>
                    {result.toxicityScore > 50 ? "⚠️ Cảnh Báo Nội Dung" : "✅ Nội Dung An Toàn"}
                  </h4>
                  <p className="text-gray-700">
                    {result.toxicityScore > 50 
                      ? "Nội dung có thể chứa yếu tố độc hại. Khuyến nghị đọc cẩn trọng và tránh lan truyền. Hãy kiểm chứng thông tin từ nhiều nguồn đáng tin cậy."
                      : "Nội dung tương đối an toàn cho người đọc. Tuy nhiên, vẫn nên duy trì suy nghĩ phản biện khi tiếp nhận thông tin."
                    }
                  </p>
                </div>
              </div>
            </div>
          </Card>
        ))}

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
