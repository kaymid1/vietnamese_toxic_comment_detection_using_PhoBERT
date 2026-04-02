import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Progress } from "@/app/components/ui/progress";
import { AlertTriangle, CheckCircle, Download, ExternalLink, RotateCcw } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip as RechartTooltip } from "recharts";

interface SegmentData {
  segment_id: string;
  score: number;
  text_preview: string;
  text?: string;
  toxic_label?: number | null;
}

interface ResultData {
  url: string;
  url_hash?: string | null;
  status: "ok" | "error" | "skipped";
  error?: string | null;
  seg_threshold_used?: number | null;
  page_toxic?: number | null;
  html_tags?: string[] | null;
  og_types?: string[] | null;
  videos?: Array<{
    video_id?: string | null;
    platform?: string | null;
    video_url?: string | null;
    title?: string | null;
    channel?: string | null;
    upload_date?: string | null;
    duration?: number | null;
    transcript?: Array<{ text: string }> | null;
    error?: string | null;
  }>;
  toxicity?: {
    overall?: number | null;
    by_segment?: SegmentData[];
  };
}

interface HistoryItem {
  id: string;
  savedAt: string;
  jobId: string | null;
  modelId: string | null;
  thresholds?: {
    seg_threshold?: number;
    page_threshold?: number;
  } | null;
  result: ResultData;
}

interface ResultsPageProps {
  results: ResultData[];
  jobId?: string | null;
  thresholds?: {
    seg_threshold?: number;
    page_threshold?: number;
  } | null;
  modelId?: string | null;
  compareModelNames?: string[];
  activeResultModel?: string | null;
  onSelectResultModel?: (modelName: string) => void;
  scanHistory?: HistoryItem[];
  onLoadHistoryItem?: (item: HistoryItem) => void;
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

const formatThreshold = (value?: number | null) => {
  if (typeof value !== "number") return "--";
  return value.toFixed(2);
};

const getSeverityClasses = (score: number | null) => {
  if (score === null) return "bg-gray-100 text-gray-700 border-gray-300";
  if (score < 35) return "bg-green-100 text-green-700 border-green-300";
  if (score < 70) return "bg-yellow-100 text-yellow-800 border-yellow-300";
  return "bg-red-100 text-red-700 border-red-300";
};

export function ResultsPage({
  results,
  jobId,
  thresholds,
  modelId,
  compareModelNames,
  activeResultModel,
  onSelectResultModel,
  scanHistory,
  onLoadHistoryItem,
  onScanAgain,
}: ResultsPageProps) {
  return (
    <div style={{ backgroundColor: "var(--viet-bg)" }} className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl mb-3" style={{ color: "var(--viet-primary)" }}>
            Kết Quả Phân Tích
          </h1>
          <p className="text-gray-600">Phân tích hoàn tất cho {results.length} URL</p>
          {jobId && <p className="text-sm text-gray-500 mt-2">Job ID: {jobId}</p>}
          {modelId && (
            <p className="text-sm text-gray-500">
              Model đang xem: <span className="font-medium text-gray-700">{modelId}</span>
            </p>
          )}
          {compareModelNames && compareModelNames.length > 1 && onSelectResultModel && (
            <div className="mt-3 max-w-sm">
              <label className="block text-xs text-gray-500 mb-1">Chuyển model kết quả</label>
              <select
                className="w-full h-10 rounded-md border border-gray-300 px-3 text-sm bg-white"
                value={activeResultModel ?? ""}
                onChange={(e) => onSelectResultModel(e.target.value)}
              >
                {compareModelNames.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {results.map((result, index) => {
          const domain = parseDomain(result.url);
          const segments = result.toxicity?.by_segment ?? [];
          const overallScore = result.toxicity?.overall;
          const overallPercent = typeof overallScore === "number" ? Math.round(overallScore * 100) : null;
          const effectiveSegThreshold =
            typeof result.seg_threshold_used === "number"
              ? result.seg_threshold_used
              : thresholds?.seg_threshold ?? 0.5;
          const toxicCount = segments.filter((s) => s.score >= effectiveSegThreshold).length;
          const safeCount = segments.length - toxicCount;
          const pageToxicFlag = typeof result.page_toxic === "number" ? result.page_toxic : null;
          const pageThreshold = thresholds?.page_threshold ?? 0.5;
          const isToxic =
            pageToxicFlag !== null
              ? pageToxicFlag === 1
              : typeof overallScore === "number"
                ? overallScore >= pageThreshold
                : false;
          const videos = result.videos ?? [];
          const topSegments = [...segments].sort((a, b) => b.score - a.score).slice(0, 3);

          return (
            <Card key={result.url_hash ?? result.url ?? index} className="bg-white p-8 mb-6 shadow-lg">
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
                    {result.status === "ok" && (
                      <div className="mt-2 text-xs text-gray-500 space-y-1">
                        <p>HTML tag: {(result.html_tags && result.html_tags[0]) || "unknown"}</p>
                        <p>OG: {(result.og_types && result.og_types.length > 0) ? result.og_types.join(", ") : "--"}</p>
                        <p>
                          Ngưỡng dùng: <span className="font-medium text-gray-700">{formatThreshold(result.seg_threshold_used)}</span>
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {result.status === "error" && (
                <div className="mb-8 p-4 rounded-lg border border-red-200 bg-red-50 text-sm text-red-700">
                  {result.error || "Không thể phân tích URL này."}
                </div>
              )}

              {result.status === "skipped" && (
                <div className="mb-8 p-4 rounded-lg border border-amber-200 bg-amber-50 text-sm text-amber-800">
                  URL này đã được bỏ qua theo lựa chọn của bạn (không chuyển qua Selenium).
                </div>
              )}

              {result.status === "ok" && (
                <>
                  <div className="mb-8">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-xl" style={{ color: "var(--viet-primary)" }}>
                        Điểm Độc Hại Tổng Thể
                      </h3>
                      <span
                        className="text-3xl"
                        style={{ color: isToxic ? "var(--viet-toxic)" : "var(--viet-safe)" }}
                      >
                        {overallPercent !== null ? `${overallPercent}%` : "--"}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-4 text-xs text-gray-500">
                      <span>Ngưỡng page: {formatThreshold(thresholds?.page_threshold)}</span>
                      <span>Ngưỡng segment hiệu lực: {formatThreshold(effectiveSegThreshold)}</span>
                    </div>
                    <Progress value={overallPercent ?? 0} className="h-4" style={{ backgroundColor: "#e5e7eb" }} />
                    <div className="flex justify-between mt-2 text-sm">
                      <span style={{ color: "var(--viet-safe)" }}>An Toàn</span>
                      <span style={{ color: "var(--viet-toxic)" }}>Độc Hại</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
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
                  </div>

                  {topSegments.length > 0 && (
                    <div className="mb-8">
                      <h3 className="text-xl mb-4" style={{ color: "var(--viet-primary)" }}>
                        Đoạn có rủi ro cao nhất (Top 3)
                      </h3>
                      <div className="space-y-3">
                        {topSegments.map((segment, idx) => {
                          const segmentIsToxic =
                            segment.toxic_label !== undefined && segment.toxic_label !== null
                              ? segment.toxic_label === 1
                              : segment.score >= effectiveSegThreshold;
                          const percent = (segment.score * 100).toFixed(1);
                          const fullText = segment.text || segment.text_preview;

                          return (
                            <div key={segment.segment_id || idx} className="p-4 rounded-lg border border-gray-200 bg-white">
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
                                  {segmentIsToxic ? "Độc hại (>= ngưỡng)" : "Rủi ro (chưa vượt ngưỡng)"}
                                </span>
                              </div>
                              <details className="text-sm text-gray-700">
                                <summary className="cursor-pointer">{segment.text_preview}</summary>
                                <p className="mt-2 whitespace-pre-wrap">{fullText}</p>
                              </details>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  <div
                    className={`p-6 rounded-lg border-l-4 ${
                      isToxic ? "bg-red-50 border-red-500" : "bg-green-50 border-green-500"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {isToxic ? (
                        <AlertTriangle className="w-6 h-6 mt-1 flex-shrink-0" style={{ color: "var(--viet-toxic)" }} />
                      ) : (
                        <CheckCircle className="w-6 h-6 mt-1 flex-shrink-0" style={{ color: "var(--viet-safe)" }} />
                      )}
                      <div>
                        <h4
                          className="mb-2"
                          style={{ color: isToxic ? "var(--viet-toxic)" : "var(--viet-safe)" }}
                        >
                          {isToxic ? "⚠️ Cảnh Báo Nội Dung" : "✅ Nội Dung An Toàn"}
                        </h4>
                        <p className="text-gray-700">
                          {isToxic
                            ? "Nội dung có thể chứa yếu tố độc hại. Khuyến nghị đọc cẩn trọng và tránh lan truyền."
                            : "Nội dung tương đối an toàn cho người đọc. Tuy nhiên, vẫn nên duy trì suy nghĩ phản biện."}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-8">
                    <h3 className="text-xl mb-4" style={{ color: "var(--viet-primary)" }}>
                      Video Phát Hiện
                    </h3>
                    {videos.length === 0 && <p className="text-sm text-gray-500">Không phát hiện video.</p>}
                    <div className="space-y-4">
                      {videos.map((video, vIdx) => (
                        <div key={`${video.video_id || vIdx}`} className="p-4 rounded-lg border border-gray-200 bg-white">
                          <p className="text-sm text-gray-600">
                            {video.platform || "video"} {video.video_id ? `• ${video.video_id}` : ""}
                          </p>
                          <p className="text-base font-semibold mt-1" style={{ color: "var(--viet-primary)" }}>
                            {video.title || "Untitled"}
                          </p>
                          {video.video_url && <p className="text-xs text-gray-500 break-all mt-1">{video.video_url}</p>}
                          {video.error && <p className="text-xs text-red-600 mt-2">Lỗi video: {video.error}</p>}
                          <div className="mt-2 text-xs text-gray-600">
                            {video.channel && <span>Kênh: {video.channel} </span>}
                            {video.upload_date && <span>• Ngày: {video.upload_date} </span>}
                            {typeof video.duration === "number" && <span>• {Math.round(video.duration)}s </span>}
                          </div>
                          {video.transcript && video.transcript.length > 0 && (
                            <details className="mt-3 text-sm text-gray-700">
                              <summary className="cursor-pointer">Xem transcript ({video.transcript.length} dòng)</summary>
                              <div className="mt-2 space-y-1">
                                {video.transcript.slice(0, 5).map((seg, sIdx) => (
                                  <p key={sIdx} className="text-xs text-gray-700">
                                    {seg.text}
                                  </p>
                                ))}
                              </div>
                            </details>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </Card>
          );
        })}

        {scanHistory && scanHistory.length > 0 && onLoadHistoryItem && (
          <div className="mt-10 rounded-xl border border-gray-200 bg-white p-5">
            <h3 className="mb-3 text-lg" style={{ color: "var(--viet-primary)" }}>
              URL đã quét gần đây
            </h3>
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {scanHistory.map((item) => {
                const score =
                  typeof item.result.toxicity?.overall === "number"
                    ? Math.round(item.result.toxicity.overall * 100)
                    : null;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => onLoadHistoryItem(item)}
                    className="w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-left hover:bg-gray-100"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-gray-800">{item.result.url}</p>
                        <p className="text-xs text-gray-500">
                          {item.modelId || "unknown model"} • {new Date(item.savedAt).toLocaleString()}
                        </p>
                      </div>
                      <span
                        className={`shrink-0 rounded-full border px-2 py-1 text-xs font-semibold ${getSeverityClasses(score)}`}
                      >
                        {score !== null ? `${score}% toxic` : "--"}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div className="flex flex-col sm:flex-row gap-4 justify-center mt-8">
          <Button onClick={onScanAgain} className="h-12 px-8" style={{ backgroundColor: "var(--viet-primary)" }}>
            <RotateCcw className="w-5 h-5 mr-2" />
            Quét URL Khác
          </Button>
          <Button
            variant="outline"
            className="h-12 px-8 border-2"
            style={{ borderColor: "var(--viet-primary)", color: "var(--viet-primary)" }}
          >
            <Download className="w-5 h-5 mr-2" />
            Xuất Báo Cáo (PDF)
          </Button>
        </div>
      </div>
    </div>
  );
}
