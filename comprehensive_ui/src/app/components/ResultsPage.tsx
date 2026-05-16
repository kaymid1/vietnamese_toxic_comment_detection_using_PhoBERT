import { useState } from "react";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Progress } from "@/app/components/ui/progress";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/app/components/ui/tooltip";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/app/components/ui/dialog";
import { AlertTriangle, CheckCircle, CircleHelp, Download, ExternalLink, RotateCcw } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip as RechartTooltip } from "recharts";
import { useI18n } from "@/app/i18n/context";

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

interface DomainThresholds {
  news?: number;
  social?: number;
  forum?: number;
  unknown?: number;
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
  thresholdsByDomain?: DomainThresholds | null;
  result: ResultData;
}

interface ResultsPageProps {
  results: ResultData[];
  jobId?: string | null;
  thresholds?: {
    seg_threshold?: number;
    page_threshold?: number;
  } | null;
  thresholdsByDomain?: DomainThresholds | null;
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

const DOMAIN_KEYS: Array<keyof DomainThresholds> = ["news", "social", "forum", "unknown"];

export function ResultsPage({
  results,
  jobId,
  thresholds,
  thresholdsByDomain,
  modelId,
  compareModelNames,
  activeResultModel,
  onSelectResultModel,
  scanHistory,
  onLoadHistoryItem,
  onScanAgain,
}: ResultsPageProps) {
  const { language, t } = useI18n();
  const dateLocale = language === "vi" ? "vi-VN" : "en-US";
  const [thresholdDetailsResult, setThresholdDetailsResult] = useState<ResultData | null>(null);

  const getDomainThresholdLabel = (key: keyof DomainThresholds) => {
    if (key === "news") return t("results.thresholdDomainNews");
    if (key === "social") return t("results.thresholdDomainSocial");
    if (key === "forum") return t("results.thresholdDomainForum");
    return t("results.thresholdDomainUnknown");
  };

  return (
    <div className="min-h-screen bg-background py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl mb-3 text-primary">{t("results.title")}</h1>
          <p className="text-muted-foreground">{t("results.completedForUrls", { count: results.length })}</p>
          {jobId && <p className="text-sm text-muted-foreground mt-2">{t("results.jobId", { id: jobId })}</p>}
          {modelId && (
            <p className="text-sm text-muted-foreground">
              {t("results.viewingModel")} <span className="font-medium text-foreground">{modelId}</span>
            </p>
          )}
          {compareModelNames && compareModelNames.length > 1 && onSelectResultModel && (
            <div className="mt-3 max-w-sm">
              <label className="block text-xs text-muted-foreground mb-1">{t("results.switchResultModel")}</label>
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
                    <h2 className="text-2xl mb-2 text-primary">{t("results.analyzeUrl")}</h2>
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <span className="text-sm font-medium">{domain}</span>
                      <a href={result.url} target="_blank" rel="noreferrer" className="text-muted-foreground hover:text-primary">
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1 break-all">{result.url}</p>
                    {result.status === "ok" && (
                      <div className="mt-2 text-xs text-muted-foreground space-y-1">
                        <p>{t("results.htmlTag", { value: (result.html_tags && result.html_tags[0]) || "unknown" })}</p>
                        <p>{t("results.og", { value: (result.og_types && result.og_types.length > 0) ? result.og_types.join(", ") : "--" })}</p>
                        <p className="flex items-center gap-1.5">
                          <span>{t("results.usedThreshold")} <span className="font-medium text-foreground">{formatThreshold(result.seg_threshold_used)}</span></span>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <button
                                type="button"
                                className="inline-flex h-4 w-4 items-center justify-center rounded-full text-muted-foreground hover:text-foreground"
                                aria-label={t("results.thresholdInfoAria")}
                              >
                                <CircleHelp className="h-3.5 w-3.5" />
                              </button>
                            </TooltipTrigger>
                            <TooltipContent side="top" sideOffset={8} className="max-w-72 p-3">
                              <div className="space-y-2 text-xs">
                                <p className="font-semibold">{t("results.thresholdInfoTitle")}</p>
                                <div className="space-y-1">
                                  {DOMAIN_KEYS.map((domainKey) => (
                                    <div key={domainKey} className="flex items-center justify-between gap-3">
                                      <span>{getDomainThresholdLabel(domainKey)}</span>
                                      <span className="font-semibold">{formatThreshold(thresholdsByDomain?.[domainKey])}</span>
                                    </div>
                                  ))}
                                </div>
                                <button
                                  type="button"
                                  className="underline underline-offset-2 hover:opacity-80"
                                  onClick={() => setThresholdDetailsResult(result)}
                                >
                                  {t("results.thresholdReadMore")}
                                </button>
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {result.status === "error" && (
                <div className="mb-8 p-4 rounded-lg border border-border-danger bg-background-danger text-sm text-text-danger">
                  {result.error || t("results.cannotAnalyzeUrl")}
                </div>
              )}

              {result.status === "skipped" && (
                <div className="mb-8 p-4 rounded-lg border border-border-warning bg-background-warning text-sm text-text-warning">
                  {t("results.skippedByChoice")}
                </div>
              )}

              {result.status === "ok" && (
                <>
                  <div className="mb-8">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-xl text-primary">{t("results.overallToxicityScore")}</h3>
                      <span className={`text-3xl ${isToxic ? "text-text-danger" : "text-text-success"}`}>
                        {overallPercent !== null ? `${overallPercent}%` : "--"}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
                      <span>{t("results.pageThreshold", { value: formatThreshold(thresholds?.page_threshold) })}</span>
                      <span>{t("results.effectiveSegmentThreshold", { value: formatThreshold(effectiveSegThreshold) })}</span>
                    </div>
                    <Progress value={overallPercent ?? 0} className="h-4" />
                    <div className="flex justify-between mt-2 text-sm">
                      <span className="text-text-success">{t("results.safe")}</span>
                      <span className="text-text-danger">{t("results.toxic")}</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                    <div>
                      <h3 className="mb-4 text-primary">{t("results.contentDistribution")}</h3>
                      <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                          <Pie
                            data={[
                              { name: t("results.toxicContent"), value: overallPercent ?? 0 },
                              { name: t("results.safeContent"), value: 100 - (overallPercent ?? 0) },
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
                      <h3 className="mb-4 text-primary">{t("results.detailedStats")}</h3>
                      <div className="bg-background-secondary p-4 rounded-lg border border-border">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-muted-foreground">{t("results.totalSegments")}</span>
                          <span className="text-xl">{segments.length}</span>
                        </div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-muted-foreground">{t("results.detectedToxicSegments")}</span>
                          <span className="text-xl text-text-danger">{toxicCount}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">{t("results.safeSegments")}</span>
                          <span className="text-xl text-text-success">{safeCount}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {topSegments.length > 0 && (
                    <div className="mb-8">
                      <h3 className="text-xl mb-4 text-primary">{t("results.topRiskSegments")}</h3>
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
                                  {segmentIsToxic ? t("results.toxicAtOrAboveThreshold") : t("results.riskBelowThreshold")}
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
                          {isToxic ? t("results.warningContent") : t("results.safeContentTitle")}
                        </h4>
                        <p className="text-foreground">
                          {isToxic ? t("results.warningDescription") : t("results.safeDescription")}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-8">
                    <h3 className="text-xl mb-4 text-primary">{t("results.detectedVideos")}</h3>
                    {videos.length === 0 && <p className="text-sm text-muted-foreground">{t("results.noVideos")}</p>}
                    <div className="space-y-4">
                      {videos.map((video, vIdx) => (
                        <div key={`${video.video_id || vIdx}`} className="p-4 rounded-lg border border-border bg-card">
                          <p className="text-sm text-muted-foreground">
                            {video.platform || "video"} {video.video_id ? `• ${video.video_id}` : ""}
                          </p>
                          <p className="text-base font-semibold mt-1 text-primary">{video.title || t("results.untitled")}</p>
                          {video.video_url && <p className="text-xs text-muted-foreground break-all mt-1">{video.video_url}</p>}
                          {video.error && <p className="text-xs text-destructive mt-2">{t("results.videoError", { error: video.error })}</p>}
                          <div className="mt-2 text-xs text-muted-foreground">
                            {video.channel && <span>{t("results.channel", { value: video.channel })} </span>}
                            {video.upload_date && <span>• {t("results.date", { value: video.upload_date })} </span>}
                            {typeof video.duration === "number" && <span>• {Math.round(video.duration)}s </span>}
                          </div>
                          {video.transcript && video.transcript.length > 0 && (
                            <details className="mt-3 text-sm text-foreground">
                              <summary className="cursor-pointer">{t("results.viewTranscript", { count: video.transcript.length })}</summary>
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
            <h3 className="mb-3 text-lg text-primary">{t("results.recentScans")}</h3>
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
                          {item.modelId || t("results.unknownModel")} • {new Date(item.savedAt).toLocaleString(dateLocale)}
                        </p>
                      </div>
                      <span
                        className={`shrink-0 rounded-full border px-2 py-1 text-xs font-semibold ${getSeverityClasses(score)}`}
                      >
                        {score !== null ? t("results.toxicPercent", { score }) : "--"}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <Dialog open={!!thresholdDetailsResult} onOpenChange={(open: boolean) => !open && setThresholdDetailsResult(null)}>
          <DialogContent className="max-w-xl">
            <DialogHeader>
              <DialogTitle>{t("results.thresholdDetailsTitle")}</DialogTitle>
              <DialogDescription>{t("results.thresholdDetailsIntro")}</DialogDescription>
            </DialogHeader>
            <div className="space-y-3 text-sm">
              <div className="rounded-md border border-border bg-background-secondary p-3">
                <p className="break-all text-xs text-muted-foreground">{thresholdDetailsResult?.url}</p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <p>
                  <span className="text-muted-foreground">{t("results.thresholdDetailHtmlTag")}</span>{" "}
                  <span className="font-medium">{(thresholdDetailsResult?.html_tags && thresholdDetailsResult.html_tags[0]) || "--"}</span>
                </p>
                <p>
                  <span className="text-muted-foreground">{t("results.thresholdDetailOg")}</span>{" "}
                  <span className="font-medium">{(thresholdDetailsResult?.og_types && thresholdDetailsResult.og_types[0]) || "--"}</span>
                </p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <p>
                  <span className="text-muted-foreground">{t("results.thresholdDetailUsed")}</span>{" "}
                  <span className="font-medium">{formatThreshold(thresholdDetailsResult?.seg_threshold_used)}</span>
                </p>
                <p>
                  <span className="text-muted-foreground">{t("results.thresholdDetailClamp")}</span>{" "}
                  <span className="font-medium">0.40 - 0.85</span>
                </p>
              </div>
              <div className="rounded-md border border-border p-3">
                <p className="font-medium mb-2">{t("results.thresholdInfoTitle")}</p>
                <div className="space-y-1 text-xs">
                  {DOMAIN_KEYS.map((domainKey) => (
                    <div key={domainKey} className="flex items-center justify-between gap-3">
                      <span>{getDomainThresholdLabel(domainKey)}</span>
                      <span className="font-semibold">{formatThreshold(thresholdsByDomain?.[domainKey])}</span>
                    </div>
                  ))}
                </div>
              </div>
              <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                <li>{t("results.thresholdDetailRule1")}</li>
                <li>{t("results.thresholdDetailRule2")}</li>
                <li>{t("results.thresholdDetailRule3")}</li>
              </ul>
            </div>
          </DialogContent>
        </Dialog>

        <div className="flex flex-col sm:flex-row gap-4 justify-center mt-8">
          <Button onClick={onScanAgain} className="h-12 px-8">
            <RotateCcw className="w-5 h-5 mr-2" />
            {t("results.scanAnother")}
          </Button>
          <Button variant="outline" className="h-12 px-8 border-2 border-primary text-primary hover:bg-accent">
            <Download className="w-5 h-5 mr-2" />
            {t("results.exportPdf")}
          </Button>
        </div>
      </div>
    </div>
  );
}
