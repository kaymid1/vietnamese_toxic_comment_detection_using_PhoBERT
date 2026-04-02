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
  if (score === null) return "bg-muted text-muted-foreground border-border";
  if (score < 35) return "bg-background-success text-text-success border-border-success";
  if (score < 70) return "bg-background-warning text-text-warning border-border-warning";
  return "bg-background-danger text-text-danger border-border-danger";
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
    <div className="min-h-screen bg-background py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl mb-3 text-primary">Kết Quả Phân Tích</h1>
          <p className="text-muted-foreground">Phân tích hoàn tất cho {results.length} URL</p>
          {jobId && <p className="text-sm text-muted-foreground mt-2">Job ID: {jobId}</p>}
          {modelId && (
            <p className="text-sm text-muted-foreground">
              Model đang xem: <span className="font-medium text-foreground">{modelId}</span>
            </p>
          )}
          {compareModelNames && compareModelNames.length > 1 && onSelectResultModel && (
            <div className="mt-3 max-w-sm">
              <label className="block text-xs text-muted-foreground mb-1">Chuyển model kết quả</label>
              <select
                className="w-full h-10 rounded-md border border-border px-3 text-sm bg-card text-foreground"
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
            <Card key={result.url_hash ?? result.url ?? index} className="bg-card p-8 mb-6 shadow-lg">
              <div className="mb-6 pb-6 border-b border-border">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <h2 className="text-2xl mb-2 text-primary">Phân tích URL</h2>
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <span className="text-sm font-medium">{domain}</span>
                      <a href={result.url} target="_blank" rel="noreferrer" className="text-muted-foreground hover:text-primary">
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1 break-all">{result.url}</p>
                    {result.status === "ok" && (
                      <div className="mt-2 text-xs text-muted-foreground space-y-1">
                        <p>HTML tag: {(result.html_tags && result.html_tags[0]) || "unknown"}</p>
                        <p>OG: {(result.og_types && result.og_types.length > 0) ? result.og_types.join(", ") : "--"}</p>
                        <p>
                          Ngưỡng dùng: <span className="font-medium text-foreground">{formatThreshold(result.seg_threshold_used)}</span>
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {result.status === "error" && (
                <div className="mb-8 p-4 rounded-lg border border-border-danger bg-background-danger text-sm text-text-danger">
                  {result.error || "Không thể phân tích URL này."}
                </div>
              )}

              {result.status === "skipped" && (
                <div className="mb-8 p-4 rounded-lg border border-border-warning bg-background-warning text-sm text-text-warning">
                  URL này đã được bỏ qua theo lựa chọn của bạn (không chuyển qua Selenium).
                </div>
              )}

              {result.status === "ok" && (
                <>
                  <div className="mb-8">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-xl text-primary">Điểm Độc Hại Tổng Thể</h3>
                      <span className={`text-3xl ${isToxic ? "text-text-danger" : "text-text-success"}`}>
                        {overallPercent !== null ? `${overallPercent}%` : "--"}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
                      <span>Ngưỡng page: {formatThreshold(thresholds?.page_threshold)}</span>
                      <span>Ngưỡng segment hiệu lực: {formatThreshold(effectiveSegThreshold)}</span>
                    </div>
                    <Progress value={overallPercent ?? 0} className="h-4" />
                    <div className="flex justify-between mt-2 text-sm">
                      <span className="text-text-success">An Toàn</span>
                      <span className="text-text-danger">Độc Hại</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                    <div>
                      <h3 className="mb-4 text-primary">Phân Bố Nội Dung</h3>
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
                            <Cell fill="var(--color-text-danger)" />
                            <Cell fill="var(--color-text-success)" />
                          </Pie>
                          <RechartTooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="space-y-4">
                      <h3 className="mb-4 text-primary">Thống Kê Chi Tiết</h3>
                      <div className="bg-background-secondary p-4 rounded-lg border border-border">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-muted-foreground">Tổng đoạn văn phân tích:</span>
                          <span className="text-xl">{segments.length}</span>
                        </div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-muted-foreground">Đoạn độc hại phát hiện:</span>
                          <span className="text-xl text-text-danger">{toxicCount}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Đoạn an toàn:</span>
                          <span className="text-xl text-text-success">{safeCount}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {topSegments.length > 0 && (
                    <div className="mb-8">
                      <h3 className="text-xl mb-4 text-primary">Đoạn có rủi ro cao nhất (Top 3)</h3>
                      <div className="space-y-3">
                        {topSegments.map((segment, idx) => {
                          const segmentIsToxic =
                            segment.toxic_label !== undefined && segment.toxic_label !== null
                              ? segment.toxic_label === 1
                              : segment.score >= effectiveSegThreshold;
                          const percent = (segment.score * 100).toFixed(1);
                          const fullText = segment.text || segment.text_preview;

                          return (
                            <div key={segment.segment_id || idx} className="p-4 rounded-lg border border-border bg-card">
                              <div className="flex items-start justify-between gap-4 mb-2">
                                <div className="flex items-center gap-3">
                                  <span className="text-sm font-medium text-muted-foreground">#{idx + 1}</span>
                                  <span className="text-sm font-semibold text-primary">{percent}%</span>
                                </div>
                                <span
                                  className={`text-xs px-2 py-1 rounded-full border ${
                                    segmentIsToxic
                                      ? "bg-background-danger text-text-danger border-border-danger"
                                      : "bg-background-warning text-text-warning border-border-warning"
                                  }`}
                                >
                                  {segmentIsToxic ? "Độc hại (>= ngưỡng)" : "Rủi ro (chưa vượt ngưỡng)"}
                                </span>
                              </div>
                              <details className="text-sm text-foreground">
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
                      isToxic ? "bg-background-danger border-border-danger" : "bg-background-success border-border-success"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {isToxic ? (
                        <AlertTriangle className="w-6 h-6 mt-1 flex-shrink-0 text-text-danger" />
                      ) : (
                        <CheckCircle className="w-6 h-6 mt-1 flex-shrink-0 text-text-success" />
                      )}
                      <div>
                        <h4 className={`mb-2 ${isToxic ? "text-text-danger" : "text-text-success"}`}>
                          {isToxic ? "⚠️ Cảnh Báo Nội Dung" : "✅ Nội Dung An Toàn"}
                        </h4>
                        <p className="text-foreground">
                          {isToxic
                            ? "Nội dung có thể chứa yếu tố độc hại. Khuyến nghị đọc cẩn trọng và tránh lan truyền."
                            : "Nội dung tương đối an toàn cho người đọc. Tuy nhiên, vẫn nên duy trì suy nghĩ phản biện."}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-8">
                    <h3 className="text-xl mb-4 text-primary">Video Phát Hiện</h3>
                    {videos.length === 0 && <p className="text-sm text-muted-foreground">Không phát hiện video.</p>}
                    <div className="space-y-4">
                      {videos.map((video, vIdx) => (
                        <div key={`${video.video_id || vIdx}`} className="p-4 rounded-lg border border-border bg-card">
                          <p className="text-sm text-muted-foreground">
                            {video.platform || "video"} {video.video_id ? `• ${video.video_id}` : ""}
                          </p>
                          <p className="text-base font-semibold mt-1 text-primary">{video.title || "Untitled"}</p>
                          {video.video_url && <p className="text-xs text-muted-foreground break-all mt-1">{video.video_url}</p>}
                          {video.error && <p className="text-xs text-destructive mt-2">Lỗi video: {video.error}</p>}
                          <div className="mt-2 text-xs text-muted-foreground">
                            {video.channel && <span>Kênh: {video.channel} </span>}
                            {video.upload_date && <span>• Ngày: {video.upload_date} </span>}
                            {typeof video.duration === "number" && <span>• {Math.round(video.duration)}s </span>}
                          </div>
                          {video.transcript && video.transcript.length > 0 && (
                            <details className="mt-3 text-sm text-foreground">
                              <summary className="cursor-pointer">Xem transcript ({video.transcript.length} dòng)</summary>
                              <div className="mt-2 space-y-1">
                                {video.transcript.slice(0, 5).map((seg, sIdx) => (
                                  <p key={sIdx} className="text-xs text-foreground">
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
          <div className="mt-10 rounded-xl border border-border bg-card p-5">
            <h3 className="mb-3 text-lg text-primary">URL đã quét gần đây</h3>
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
                    className="w-full rounded-lg border border-border bg-background-secondary px-3 py-2 text-left hover:bg-muted"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-foreground">{item.result.url}</p>
                        <p className="text-xs text-muted-foreground">
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
          <Button onClick={onScanAgain} className="h-12 px-8">
            <RotateCcw className="w-5 h-5 mr-2" />
            Quét URL Khác
          </Button>
          <Button variant="outline" className="h-12 px-8 border-2 border-primary text-primary hover:bg-accent">
            <Download className="w-5 h-5 mr-2" />
            Xuất Báo Cáo (PDF)
          </Button>
        </div>
      </div>
    </div>
  );
}
