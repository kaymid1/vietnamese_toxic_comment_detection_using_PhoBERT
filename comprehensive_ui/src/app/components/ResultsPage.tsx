import { useEffect, useMemo, useState } from "react";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Label } from "@/app/components/ui/label";
import { Progress } from "@/app/components/ui/progress";
import { RadioGroup, RadioGroupItem } from "@/app/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/app/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/app/components/ui/tooltip";
import { AlertTriangle, CheckCircle, Download, ExternalLink, Info, RotateCcw } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip as RechartTooltip } from "recharts";

interface SegmentData {
  segment_id: string;
  score: number;
  text_preview: string;
  text?: string;
  html_tags?: string[] | null;
  og_types?: string[] | null;
  ai_learned?: boolean | null;
  ai_learned_label?: string | null;
  segment_hash?: string | null;
  context_segment_hash?: string | null;
  toxic_label?: number | null;
  toxic_prob_adjusted?: number | null;
  ai_learned_mode?: string | null;
  learned_support?: number | null;
  learned_agreement?: number | null;
  seg_threshold_used?: number | null;
}

interface SegmentFeedbackItem {
  url: string;
  url_hash: string;
  model_id: string;
  html_tag: string;
  html_tag_override?: string | null;
  segment_id: string;
  text: string;
  score?: number | null;
  seg_threshold_used?: number | null;
  label: string;
  segment_hash?: string | null;
  context_segment_hash?: string | null;
}

interface ResultData {
  url: string;
  status: "ok" | "error";
  error?: string | null;
  crawl_output_dir?: string | null;
  segments_path?: string | null;
  html_tags?: string[] | null;
  og_types?: string[] | null;
  seg_threshold_used?: number | null;
  page_toxic?: number | null;
  videos?: Array<{
    video_id?: string | null;
    platform?: string | null;
    video_url?: string | null;
    page_url?: string | null;
    title?: string | null;
    channel?: string | null;
    upload_date?: string | null;
    duration?: number | null;
    view_count?: number | null;
    transcript?: Array<{ text: string; start?: number; duration?: number }> | null;
    language?: string | null;
    has_auto_generated?: boolean | null;
    error?: string | null;
  }>;
  toxicity?: {
    overall?: number | null;
    by_segment?: SegmentData[];
  };
}

type PageLabel = "toxic" | "clean" | "unsure";


interface ResultsPageProps {
  results: ResultData[];
  compareResults?: Record<string, {
    results: ResultData[];
    thresholds?: {
      seg_threshold?: number;
      page_threshold?: number;
    } | null;
  }> | null;
  jobId?: string | null;
  thresholds?: {
    seg_threshold?: number;
    page_threshold?: number;
  } | null;
  modelId?: string | null;
  onSelectModel?: (modelId: string) => void;
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

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const slugifyId = (value: string) => value.replace(/[^a-zA-Z0-9_-]/g, "-");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

const formatThreshold = (value?: number | null) => {
  if (typeof value !== "number") return "--";
  return value.toFixed(2);
};

const badgeStyles: Record<string, { label: string; text: string; bg: string }> = {
  schema: { label: "SCHEMA", text: "#0f766e", bg: "#ccfbf1" },
  og: { label: "OG", text: "#1d4ed8", bg: "#dbeafe" },
  header: { label: "HEADER", text: "#7c3aed", bg: "#ede9fe" },
  unknown: { label: "UNKNOWN", text: "#6b7280", bg: "#f3f4f6" },
  attention: { label: "ATTN", text: "#b45309", bg: "#fef3c7" },
};

const htmlTagOptions = ["unknown", "attention"] as const;

export function ResultsPage({
  results,
  compareResults,
  jobId,
  thresholds,
  modelId,
  onSelectModel,
  onScanAgain,
}: ResultsPageProps) {
  const [pageLabels, setPageLabels] = useState<Record<string, PageLabel>>({});
  const [pageHtmlTags, setPageHtmlTags] = useState<Record<string, string>>({});
  const [segmentHtmlTags, setSegmentHtmlTags] = useState<Record<string, string>>({});
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [askAiStatus, setAskAiStatus] = useState<string | null>(null);
  const [askAiResponse, setAskAiResponse] = useState<string | null>(null);
  const [askAiQuestion, setAskAiQuestion] = useState<string>("");
  const [askAiTarget, setAskAiTarget] = useState<ResultData | null>(null);
  const [segmentFeedbackStatus, setSegmentFeedbackStatus] = useState<string | null>(null);

  const jobModelId = modelId ?? "";
  const compareModelIds = compareResults ? Object.keys(compareResults) : [];

  const labeledCount = useMemo(
    () => Object.values(pageLabels).filter((value) => value && value !== "unsure").length,
    [pageLabels],
  );

  const hasLabels = labeledCount > 0;

  const handleLabelChange = (url: string, value: PageLabel) => {
    setPageLabels((prev) => ({ ...prev, [url]: value }));
  };

  const handlePageHtmlTagChange = (url: string, value: string) => {
    setPageHtmlTags((prev) => ({ ...prev, [url]: value }));
  };

  const handleSegmentHtmlTagChange = (segmentId: string, value: string) => {
    setSegmentHtmlTags((prev) => ({ ...prev, [segmentId]: value }));
  };

  const handleSubmitFeedback = async () => {
    if (!jobId || !jobModelId) {
      setFeedbackStatus("Thiếu job_id hoặc model_id để gửi đánh giá.");
      return;
    }

    const items = results
      .map((result) => {
        const label = pageLabels[result.url];
        if (!label || label === "unsure") return null;
        const defaultTag = (result.html_tags && result.html_tags[0]) || "unknown";
        const selectedTag = pageHtmlTags[result.url] ?? defaultTag;
        return {
          url: result.url,
          url_hash: result.url_hash ?? result.url,
          html_tag: defaultTag,
          html_tag_override: selectedTag !== defaultTag ? selectedTag : null,
          seg_threshold_used: result.seg_threshold_used ?? null,
          score_overall: result.toxicity?.overall ?? null,
          label,
        };
      })
      .filter(Boolean);

    if (!items.length) {
      setFeedbackStatus("Chưa có đánh giá nào để gửi.");
      return;
    }

    try {
      setFeedbackStatus(null);
      const response = await fetch(buildApiUrl("/api/feedback"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: jobId,
          model_id: jobModelId,
          items,
        }),
      });

      const text = await response.text();
      if (!response.ok) {
        throw new Error(text || "Gửi đánh giá thất bại");
      }

      setFeedbackStatus("Đã lưu đánh giá thành công.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Gửi đánh giá thất bại";
      setFeedbackStatus(message);
    }
  };


  const handleAskAi = async () => {
    if (!askAiTarget) return;
    try {
      setAskAiStatus(null);
      setAskAiResponse(null);
      const segments = (askAiTarget.toxicity?.by_segment ?? [])
        .slice()
        .sort((a, b) => b.score - a.score)
        .slice(0, 5)
        .map((seg) => ({
          text: seg.text || seg.text_preview,
          text_preview: seg.text_preview,
          score: seg.score,
        }));

      const response = await fetch(buildApiUrl("/api/ask-ai"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: askAiTarget.url,
          html_tag: askAiTarget.html_tags?.[0] ?? undefined,
          overall: askAiTarget.toxicity?.overall ?? undefined,
          thresholds: {
            page_threshold: thresholds?.page_threshold,
            seg_threshold: askAiTarget.seg_threshold_used ?? thresholds?.seg_threshold,
          },
          segments,
          question: askAiQuestion || undefined,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || "Ask AI thất bại");
      }
      setAskAiResponse(data?.answer || "(No response)");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Ask AI thất bại";
      setAskAiStatus(message);
    }
  };

  const handleSegmentFeedback = async (items: SegmentFeedbackItem[]) => {
    if (!jobId || !jobModelId || items.length === 0) {
      setSegmentFeedbackStatus("Thiếu job_id hoặc dữ liệu feedback.");
      return;
    }
    try {
      setSegmentFeedbackStatus(null);
      const response = await fetch(buildApiUrl("/api/feedback/segment"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: jobId,
          items,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || "Gửi feedback segment thất bại");
      }
      setSegmentFeedbackStatus(`Đã lưu ${data?.inserted ?? 0} segment.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Gửi feedback segment thất bại";
      setSegmentFeedbackStatus(message);
    }
  };

  useEffect(() => {
    setPageLabels({});
    setPageHtmlTags({});
    setSegmentHtmlTags({});
    setFeedbackStatus(null);
  }, [jobModelId]);

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
          {jobModelId && (
            <p className="text-sm text-gray-500">
              Model đang xem: <span className="font-medium text-gray-700">{jobModelId}</span>
            </p>
          )}
          {compareModelIds.length > 1 && onSelectModel && (
            <div className="mt-4 flex flex-wrap gap-2">
              {compareModelIds.map((model) => (
                <button
                  key={model}
                  type="button"
                  onClick={() => onSelectModel(model)}
                  className={`px-3 py-1 rounded-full text-sm border ${
                    model === jobModelId
                      ? "border-blue-600 text-blue-700 bg-blue-50"
                      : "border-gray-300 text-gray-600 bg-white"
                  }`}
                >
                  {model}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Feedback Controls */}
        <Card className="bg-white p-6 mb-8 shadow-sm">
          <h2 className="text-xl mb-3" style={{ color: "var(--viet-primary)" }}>
            Gán nhãn theo HTML tags
          </h2>
          <p className="text-sm text-gray-600 mb-3">
            Quy trình: (1) Gán nhãn thủ công → (2) Gửi đánh giá. Nhãn sẽ được lưu và dùng lại (AI_LEARNED)
            cho các segment trùng tag ở lần quét sau.
          </p>
          <div className="flex flex-wrap gap-3">
            <Button
              onClick={handleSubmitFeedback}
              disabled={!hasLabels}
              className="h-10 px-5"
              style={{ backgroundColor: "var(--viet-primary)" }}
            >
              Gửi đánh giá ({labeledCount})
            </Button>
          </div>
          {feedbackStatus && (
            <p className="mt-3 text-sm text-gray-600">{feedbackStatus}</p>
          )}
        </Card>

        {/* Results */}
        {results.map((result, index) => {
          const domain = parseDomain(result.url);
          const segments = result.toxicity?.by_segment ?? [];
          const overallScore = result.toxicity?.overall;
          const overallPercent = typeof overallScore === "number" ? Math.round(overallScore * 100) : null;
          const pageHtmlTag = pageHtmlTags[result.url] ?? result.html_tags?.[0] ?? "unknown";
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
          const topSegments = [...segments]
            .sort((a, b) => b.score - a.score)
            .slice(0, 3);

          return (
            <Card key={result.url_hash ?? result.url ?? index} className="bg-white p-8 mb-6 shadow-lg">
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
                    {result.status === "ok" && (
                      <div className="mt-2 text-xs text-gray-500 space-y-1">
                        {(() => {
                          const og = result.og_types ?? [];
                          const tagLabel = pageHtmlTag;
                          return (
                            <div className="space-y-1">
                              <p className="flex items-center gap-2">
                                <span
                                  className="px-2 py-0.5 rounded-full text-[11px] font-semibold tracking-wide"
                                  style={{ backgroundColor: badgeStyles.unknown.bg, color: badgeStyles.unknown.text }}
                                >
                                  TAG
                                </span>
                                <span className="text-gray-600">HTML tag: {tagLabel}</span>
                              </p>
                              <p className="text-gray-500">OG: {og.length ? og.join(", ") : "--"}</p>
                            </div>
                          );
                        })()}
                        <p>
                          Ngưỡng dùng: <span className="font-medium text-gray-700">{formatThreshold(result.seg_threshold_used)}</span>
                        </p>
                      </div>
                    )}
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
                  <div className="mb-8">
                    <h3 className="text-xl mb-4" style={{ color: "var(--viet-primary)" }}>
                      Đánh giá thủ công (theo trang)
                    </h3>
                    <div className="flex flex-col gap-3">
                      <RadioGroup
                        value={pageLabels[result.url] ?? "unsure"}
                        onValueChange={(value) => handleLabelChange(result.url, value as PageLabel)}
                        className="flex flex-wrap gap-6"
                      >
                        {(() => {
                          const baseId = slugifyId(result.url);
                          return (
                            <>
                              <div className="flex items-center gap-2">
                                <RadioGroupItem value="toxic" id={`${baseId}-toxic`} />
                                <Label htmlFor={`${baseId}-toxic`} className="text-sm">Độc hại</Label>
                              </div>
                              <div className="flex items-center gap-2">
                                <RadioGroupItem value="clean" id={`${baseId}-clean`} />
                                <Label htmlFor={`${baseId}-clean`} className="text-sm">An toàn</Label>
                              </div>
                              <div className="flex items-center gap-2">
                                <RadioGroupItem value="unsure" id={`${baseId}-unsure`} />
                                <Label htmlFor={`${baseId}-unsure`} className="text-sm">Chưa chắc</Label>
                              </div>
                            </>
                          );
                        })()}
                      </RadioGroup>
                      <div className="flex flex-wrap items-center gap-3 text-sm">
                        <span className="text-gray-600">HTML tag:</span>
                        <Select
                          value={pageHtmlTags[result.url] ?? result.html_tags?.[0] ?? "unknown"}
                          onValueChange={(value) => handlePageHtmlTagChange(result.url, value)}
                        >
                          <SelectTrigger className="h-8 w-[220px]">
                            <SelectValue placeholder="Chọn tag" />
                          </SelectTrigger>
                          <SelectContent>
                            {(result.html_tags && result.html_tags.length > 0 ? result.html_tags : htmlTagOptions).map(
                              (option) => (
                                <SelectItem key={option} value={option}>
                                  {option}
                                </SelectItem>
                              ),
                            )}
                          </SelectContent>
                        </Select>
                      </div>
                      <p className="text-xs text-gray-500">
                        Tag dùng để lưu feedback và kích hoạt AI_LEARNED khi gặp lại segment cùng tag.
                      </p>
                    </div>
                  </div>

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
                      <span
                        className="text-3xl"
                        style={{
                          color: isToxic ? "var(--viet-toxic)" : "var(--viet-safe)",
                        }}
                      >
                        {overallPercent !== null ? `${overallPercent}%` : "--"}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-4 text-xs text-gray-500">
                      <span>Ngưỡng page: {formatThreshold(thresholds?.page_threshold)}</span>
                      <span>Ngưỡng segment hiệu lực: {formatThreshold(effectiveSegThreshold)}</span>
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
                    const segmentIsToxic =
                      segment.toxic_label !== undefined && segment.toxic_label !== null
                        ? segment.toxic_label === 1
                        : segment.score >= effectiveSegThreshold;
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
                            {segmentIsToxic ? "Độc hại (>= ngưỡng)" : "Rủi ro (chưa vượt ngưỡng)"}
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
                  Các đoạn dưới đây có xác suất độc hại cao nhất theo mô hình. Chúng có thể chưa đủ để được gắn nhãn ‘Độc hại’ nếu chưa vượt ngưỡng hiện hành.
                </p>
              </div>
            )}
                  </div>

                  {/* Content Explainability */}
                  <div className="mb-8">
                    <h3 className="text-xl mb-4" style={{ color: "var(--viet-primary)" }}>
                      Chi Tiết Phát Hiện (XAI - Explainable AI)
                    </h3>
                    {segmentFeedbackStatus && (
                      <p className="text-sm text-gray-600 mb-3">{segmentFeedbackStatus}</p>
                    )}
                    {segments.length === 0 && (
                      <p className="text-sm text-gray-500">
                        Chưa có dữ liệu chi tiết theo từng đoạn.
                      </p>
                    )}
                    <div className="space-y-3">
                      {segments.map((segment, idx) => {
                        const segmentIsToxic =
                          segment.toxic_label !== undefined && segment.toxic_label !== null
                            ? segment.toxic_label === 1
                            : segment.score >= effectiveSegThreshold;
                        const nearThreshold = Math.abs(segment.score - effectiveSegThreshold) <= 0.05;
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
                                <span
                                  className="text-sm font-medium"
                                  style={{
                                    color: segmentIsToxic ? "var(--viet-toxic)" : "var(--viet-safe)",
                                  }}
                                >
                                  {(segment.score * 100).toFixed(1)}%
                                </span>
                                {segmentIsToxic ? (
                                  <AlertTriangle className="w-5 h-5" style={{ color: "var(--viet-toxic)" }} />
                                ) : (
                                  <CheckCircle className="w-5 h-5" style={{ color: "var(--viet-safe)" }} />
                                )}
                              </div>
                            </div>
                            <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
                              <span>
                                {segment.ai_learned
                                  ? `AI_PRIOR (${segment.ai_learned_label}) · support=${segment.learned_support ?? 0} · agreement=${((segment.learned_agreement ?? 0) * 100).toFixed(0)}%`
                                  : segment.ai_learned_mode === "conflict"
                                    ? "NEEDS_REVIEW (memory conflict)"
                                    : segmentIsToxic
                                      ? "Đoạn này được gắn nhãn độc hại do mô hình dự đoán"
                                      : "Đoạn này được đánh giá là an toàn"}
                              </span>
                              {nearThreshold && (
                                <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-700">
                                  Gần ngưỡng
                                </span>
                              )}
                            </div>
                            <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-gray-600">
                              <span>HTML tag:</span>
                              <Select
                                value={segmentHtmlTags[segment.segment_id] ?? pageHtmlTags[result.url] ?? result.html_tags?.[0] ?? "unknown"}
                                onValueChange={(value) => handleSegmentHtmlTagChange(segment.segment_id, value)}
                              >
                                <SelectTrigger className="h-7 w-[200px] text-xs">
                                  <SelectValue placeholder="Chọn tag" />
                                </SelectTrigger>
                                <SelectContent>
                                  {(result.html_tags && result.html_tags.length > 0 ? result.html_tags : htmlTagOptions).map(
                                    (option) => (
                                      <SelectItem key={option} value={option}>
                                        {option}
                                      </SelectItem>
                                    ),
                                  )}
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="mt-2 flex flex-wrap gap-2">
                              {segmentIsToxic ? (
                                <Button
                                  type="button"
                                  variant="outline"
                                  className="h-8 px-3 text-xs"
                                  onClick={() => {
                                    const defaultTag = result.html_tags?.[0] ?? "unknown";
                                    const selectedTag = segmentHtmlTags[segment.segment_id] ?? defaultTag;
                                    const payload: SegmentFeedbackItem = {
                                      url: result.url,
                                      url_hash: result.url_hash ?? result.url,
                                      model_id: jobModelId,
                                      html_tag: defaultTag,
                                      html_tag_override: selectedTag !== defaultTag ? selectedTag : null,
                                      segment_id: segment.segment_id,
                                      text: segment.text || segment.text_preview,
                                      score: segment.score,
                                      seg_threshold_used: effectiveSegThreshold,
                                      label: "clean",
                                      segment_hash: segment.segment_hash ?? null,
                                      context_segment_hash: segment.context_segment_hash ?? null,
                                    };
                                    void handleSegmentFeedback([payload]);
                                  }}
                                >
                                  Mark an toàn (segment)
                                </Button>
                              ) : (
                                <Button
                                  type="button"
                                  variant="outline"
                                  className="h-8 px-3 text-xs"
                                  onClick={() => {
                                    const defaultTag = result.html_tags?.[0] ?? "unknown";
                                    const selectedTag = segmentHtmlTags[segment.segment_id] ?? defaultTag;
                                    const payload: SegmentFeedbackItem = {
                                      url: result.url,
                                      url_hash: result.url_hash ?? result.url,
                                      model_id: jobModelId,
                                      html_tag: defaultTag,
                                      html_tag_override: selectedTag !== defaultTag ? selectedTag : null,
                                      segment_id: segment.segment_id,
                                      text: segment.text || segment.text_preview,
                                      score: segment.score,
                                      seg_threshold_used: effectiveSegThreshold,
                                      label: "toxic",
                                      segment_hash: segment.segment_hash ?? null,
                                      context_segment_hash: segment.context_segment_hash ?? null,
                                    };
                                    void handleSegmentFeedback([payload]);
                                  }}
                                >
                                  Mark độc hại (segment)
                                </Button>
                              )}
                              {!jobModelId && (
                                <span className="text-xs text-amber-700">Thiếu model_id để lưu feedback.</span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Recommendations */}
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
                          style={{
                            color: isToxic ? "var(--viet-toxic)" : "var(--viet-safe)",
                          }}
                        >
                          {isToxic ? "⚠️ Cảnh Báo Nội Dung" : "✅ Nội Dung An Toàn"}
                        </h4>
                        <p className="text-gray-700">
                          {isToxic
                            ? "Nội dung có thể chứa yếu tố độc hại. Khuyến nghị đọc cẩn trọng và tránh lan truyền. Hãy kiểm chứng thông tin từ nhiều nguồn đáng tin cậy."
                            : "Nội dung tương đối an toàn cho người đọc. Tuy nhiên, vẫn nên duy trì suy nghĩ phản biện khi tiếp nhận thông tin."}
                        </p>
                        <div className="mt-4 flex flex-wrap gap-3">
                          <Button
                            type="button"
                            variant="outline"
                            className="h-10 px-4"
                            onClick={() => {
                              setAskAiTarget(result);
                              setAskAiResponse(null);
                              setAskAiStatus(null);
                              setAskAiQuestion("");
                            }}
                          >
                            Ask AI (Gemini)
                          </Button>
                        </div>
                        {askAiTarget?.url === result.url && (
                          <div className="mt-4 rounded-lg border border-gray-200 p-4 bg-white">
                            <label className="text-sm text-gray-600 block mb-2">
                              Câu hỏi thêm (tuỳ chọn)
                            </label>
                            <input
                              type="text"
                              value={askAiQuestion}
                              onChange={(e) => setAskAiQuestion(e.target.value)}
                              className="w-full h-10 rounded-md border border-gray-300 px-3 text-sm"
                              placeholder="Ví dụ: Có nên đọc không? Lý do?"
                            />
                            <div className="mt-3 flex items-center gap-3">
                              <Button
                                type="button"
                                onClick={handleAskAi}
                                className="h-9 px-4"
                                style={{ backgroundColor: "var(--viet-primary)" }}
                              >
                                Gửi cho Gemini
                              </Button>
                              {askAiStatus && <span className="text-sm text-red-600">{askAiStatus}</span>}
                            </div>
                            {askAiResponse && (
                              <div className="mt-3 text-sm text-gray-700 whitespace-pre-wrap max-h-64 overflow-auto pr-1">
                                {askAiResponse}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Videos */}
                  <div className="mt-8">
                    <h3 className="text-xl mb-4" style={{ color: "var(--viet-primary)" }}>
                      Video Phát Hiện
                    </h3>
                    {videos.length === 0 && (
                      <p className="text-sm text-gray-500">Không phát hiện video.</p>
                    )}
                    <div className="space-y-4">
                      {videos.map((video, vIdx) => (
                        <div key={`${video.video_id || vIdx}`} className="p-4 rounded-lg border border-gray-200 bg-white">
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex-1">
                              <p className="text-sm text-gray-600">
                                {video.platform || "video"} {video.video_id ? `• ${video.video_id}` : ""}
                              </p>
                              <p className="text-base font-semibold mt-1" style={{ color: "var(--viet-primary)" }}>
                                {video.title || "Untitled"}
                              </p>
                              {video.video_url && (
                                <p className="text-xs text-gray-500 break-all mt-1">{video.video_url}</p>
                              )}
                              {video.error && (
                                <p className="text-xs text-red-600 mt-2">Lỗi video: {video.error}</p>
                              )}
                              <div className="mt-2 text-xs text-gray-600">
                                {video.channel && <span>Kênh: {video.channel} </span>}
                                {video.upload_date && <span>• Ngày: {video.upload_date} </span>}
                                {typeof video.duration === "number" && <span>• {Math.round(video.duration)}s </span>}
                              </div>
                            </div>
                          </div>
                          {video.transcript && video.transcript.length > 0 && (
                            <details className="mt-3 text-sm text-gray-700">
                              <summary className="cursor-pointer">
                                Xem transcript ({video.transcript.length} dòng)
                              </summary>
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
