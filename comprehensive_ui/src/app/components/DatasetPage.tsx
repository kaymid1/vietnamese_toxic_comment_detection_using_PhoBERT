import { useEffect, useMemo, useState } from "react";
import { Button } from "@/app/components/ui/button";
import { Badge } from "@/app/components/ui/badge";
import { Card } from "@/app/components/ui/card";
import { Label } from "@/app/components/ui/label";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/app/components/ui/pagination";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { useI18n } from "@/app/i18n/context";

interface DatasetRow {
  text: string;
  label: number;
  meta?: {
    source?: string;
    split?: string;
    is_augmented?: boolean;
    created_at?: string;
    feedback_id?: number;
  };
}

interface DatasetStats {
  total: number;
  by_source: Record<string, { total: number; clean: number; toxic: number }>;
}

type DatasetVersion = "v1" | "latest";

interface DatasetPreviewResponse {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  items: DatasetRow[];
  stats?: DatasetStats;
  dataset_version?: string;
}

interface DatasetExportResponse {
  path: string;
  artifact_path?: string;
  manifest_path?: string;
  count: number;
  stats: DatasetStats;
  artifact_versions?: {
    dataset_version?: string;
    model_version?: string;
    policy_version?: string;
  };
}

interface DatasetPageProps {
  datasetVersion: DatasetVersion;
  onNavigateToProtocol?: () => void;
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

const labelText = (label: number, t: (key: string) => string) => (label === 1 ? t("dataset.filters.toxic") : t("dataset.filters.clean"));

const sourceLabel = (source: string, t: (key: string) => string) => {
  const normalized = source.trim().toLowerCase();
  const map: Record<string, string> = {
    all: t("dataset.filters.allSources"),
    victsd: "ViCTSD",
    victsd_augmented: "ViCTSD",
    vihsd: "UIT-ViHSD",
    vihsd_augmented: "UIT-ViHSD",
    "uit-vihsd_augmented": "UIT-ViHSD",
    unknown: t("dataset.common.unknown"),
  };
  return map[normalized] || source.replaceAll("_", " ");
};

const SOURCE_ORDER: string[] = ["victsd", "victsd_augmented", "vihsd", "uit-vihsd_augmented", "new_collected"];

const LATEST_HIDDEN_SOURCES = new Set(["vihsd", "vihsd_augmented", "uit-vihsd_augmented"]);

const isVisibleSourceOption = (source: string, isLegacyDataset: boolean) => {
  const normalized = source.trim().toLowerCase();
  if (normalized === "all") return false;
  if (!isLegacyDataset && LATEST_HIDDEN_SOURCES.has(normalized)) return false;
  return true;
};

const sortSourcesByPreferredOrder = (sources: string[]) => {
  const order = new Map(SOURCE_ORDER.map((value, index) => [value, index]));
  return [...sources].sort((a, b) => {
    const aNorm = a.trim().toLowerCase();
    const bNorm = b.trim().toLowerCase();
    const aIdx = order.get(aNorm);
    const bIdx = order.get(bNorm);
    if (aIdx !== undefined && bIdx !== undefined) return aIdx - bIdx;
    if (aIdx !== undefined) return -1;
    if (bIdx !== undefined) return 1;
    return aNorm.localeCompare(bNorm);
  });
};

const formatPercent = (value: number, total: number) => {
  if (!total) return "0.0%";
  return `${((value / total) * 100).toFixed(1)}%`;
};

const resolveDatasetVersionParam = (datasetVersion: DatasetVersion) =>
  datasetVersion === "latest" ? "latest" : "v1";

export function DatasetPage({ datasetVersion, onNavigateToProtocol }: DatasetPageProps) {
  const { t } = useI18n();
  const isLegacyDataset = datasetVersion === "v1";
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [totalPages, setTotalPages] = useState(1);
  const [totalRows, setTotalRows] = useState(0);
  const [sourceFilter, setSourceFilter] = useState("all");
  const [labelFilter, setLabelFilter] = useState("all");
  const [splitFilter, setSplitFilter] = useState("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exportStatus, setExportStatus] = useState<string | null>(null);
  const [selectedFeedback, setSelectedFeedback] = useState<number[]>([]);
  const [deleteStatus, setDeleteStatus] = useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [sourceOptions, setSourceOptions] = useState<string[]>([
    "victsd_augmented",
    "uit-vihsd_augmented",
    "new_collected",
  ]);

  useEffect(() => {
    const keys = Object.keys(stats?.by_source || {});
    if (!keys.length) return;
    setSourceOptions((prev) => {
      const merged = new Set([...prev, ...keys]);
      const filtered = Array.from(merged).filter((source) => isVisibleSourceOption(source, isLegacyDataset));
      return sortSourcesByPreferredOrder(filtered);
    });
  }, [isLegacyDataset, stats]);

  const availableSources = useMemo(() => {
    const merged = new Set([...sourceOptions, sourceFilter]);
    const filtered = Array.from(merged).filter((source) => isVisibleSourceOption(source, isLegacyDataset));
    return ["all", ...sortSourcesByPreferredOrder(filtered)];
  }, [isLegacyDataset, sourceFilter, sourceOptions]);

  const aggregatedStats = useMemo(() => {
    const bySource = stats?.by_source || {};
    const sources = Object.keys(bySource);
    let clean = 0;
    let toxic = 0;
    sources.forEach((source) => {
      clean += bySource[source]?.clean ?? 0;
      toxic += bySource[source]?.toxic ?? 0;
    });
    const total = (stats?.total ?? 0) || clean + toxic;
    return { total, clean, toxic, sources };
  }, [stats]);

  const sourceSummary = useMemo(() => {
    const bySource = stats?.by_source || {};
    const victsdBase = bySource.victsd || { total: 0, clean: 0, toxic: 0 };
    const victsdAug = bySource.victsd_augmented || { total: 0, clean: 0, toxic: 0 };
    const vihsdBase = bySource.vihsd || { total: 0, clean: 0, toxic: 0 };
    const vihsdAug = bySource.vihsd_augmented || bySource["uit-vihsd_augmented"] || { total: 0, clean: 0, toxic: 0 };
    return {
      victsd: {
        total: victsdBase.total + victsdAug.total,
        clean: victsdBase.clean + victsdAug.clean,
        toxic: victsdBase.toxic + victsdAug.toxic,
      },
      vihsd: {
        total: vihsdBase.total + vihsdAug.total,
        clean: vihsdBase.clean + vihsdAug.clean,
        toxic: vihsdBase.toxic + vihsdAug.toxic,
      },
    };
  }, [stats]);

  const latestSourceSummary = useMemo(() => {
    const bySource = stats?.by_source || {};
    const hiddenLatestSources = new Set(["vihsd", "vihsd_augmented", "uit-vihsd_augmented"]);

    return Object.entries(bySource)
      .filter(([source]) => !hiddenLatestSources.has(source.trim().toLowerCase()))
      .filter(([, counts]) => (counts?.total ?? 0) > 0)
      .sort((a, b) => (b[1]?.total ?? 0) - (a[1]?.total ?? 0));
  }, [stats]);

  const imbalanceRatioText = useMemo(() => {
    if (!aggregatedStats.toxic) return "N/A";
    return `${(aggregatedStats.clean / aggregatedStats.toxic).toFixed(1)}:1`;
  }, [aggregatedStats.clean, aggregatedStats.toxic]);

  const imbalanceStatus = useMemo(() => {
    if (!aggregatedStats.total) return null;
    const cleanRatio = aggregatedStats.clean / aggregatedStats.total;
    const toxicRatio = aggregatedStats.toxic / aggregatedStats.total;
    const dominant = cleanRatio >= toxicRatio ? "clean" : "toxic";
    const dominantRatio = Math.max(cleanRatio, toxicRatio);
    return {
      dominant,
      dominantRatio,
      isImbalanced: dominantRatio > 0.7,
    };
  }, [aggregatedStats]);

  const fetchPreview = async (targetPage: number, targetPageSize: number) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        page: String(targetPage),
        page_size: String(targetPageSize),
        include_stats: "true",
      });
      if (sourceFilter !== "all") params.set("source", sourceFilter);
      if (labelFilter !== "all") params.set("label", labelFilter === "toxic" ? "1" : "0");
      if (splitFilter !== "all") params.set("split", splitFilter);
      params.set("dataset_version", resolveDatasetVersionParam(datasetVersion));

      const response = await fetch(buildApiUrl(`/api/dataset/preview?${params.toString()}`));
      const data = (await response.json()) as DatasetPreviewResponse;
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }
      setRows(data.items || []);
      setStats(data.stats || null);
      setTotalPages(data.total_pages || 1);
      setTotalRows(data.total || 0);
      setSelectedFeedback([]);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("dataset.status.cannotLoadDataset");
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setPage(1);
    setSelectedFeedback([]);
  }, [sourceFilter, labelFilter, splitFilter, pageSize, datasetVersion]);

  useEffect(() => {
    void fetchPreview(page, pageSize);
  }, [page, pageSize, sourceFilter, labelFilter, splitFilter, datasetVersion]);

  const handleExport = async () => {
    setExportStatus(null);
    try {
      const body: Record<string, unknown> = {};
      if (sourceFilter !== "all") body.source = [sourceFilter];
      if (labelFilter !== "all") body.label = [labelFilter === "toxic" ? 1 : 0];
      if (splitFilter !== "all") body.split = [splitFilter];

      body.dataset_version = resolveDatasetVersionParam(datasetVersion);
      body.model_version = "phobert/baseline";
      body.policy_version = "policy-v1";

      const response = await fetch(buildApiUrl("/api/dataset/export"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = (await response.json()) as DatasetExportResponse;
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }
      const artifactPath = data.artifact_path || data.path;
      const manifestPath = data.manifest_path || t("dataset.common.na");
      setExportStatus(
        t("dataset.status.exportedRowsWithLineage", {
          count: data.count,
          path: artifactPath,
          manifest: manifestPath,
          datasetVersion: data.artifact_versions?.dataset_version || t("dataset.common.na"),
          modelVersion: data.artifact_versions?.model_version || t("dataset.common.na"),
          policyVersion: data.artifact_versions?.policy_version || t("dataset.common.na"),
        }),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : t("dataset.status.exportFailed");
      setExportStatus(message);
    }
  };

  const toggleFeedbackSelection = (feedbackId: number) => {
    setSelectedFeedback((prev) =>
      prev.includes(feedbackId) ? prev.filter((id) => id !== feedbackId) : [...prev, feedbackId],
    );
  };

  const handleDeleteFeedback = async () => {
    if (!selectedFeedback.length) return;
    const confirmed = window.confirm(t("dataset.status.confirmDeleteFeedback", { count: selectedFeedback.length }));
    if (!confirmed) return;
    setDeleteLoading(true);
    setDeleteStatus(null);
    try {
      const response = await fetch(buildApiUrl("/api/feedback/segment/delete"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: selectedFeedback }),
      });
      const data = (await response.json()) as { deleted: number };
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }
      setDeleteStatus(t("dataset.status.deletedFeedbackCount", { count: data.deleted }));
      setSelectedFeedback([]);
      await fetchPreview(page, pageSize);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("dataset.status.deleteFeedbackFailed");
      setDeleteStatus(message);
    } finally {
      setDeleteLoading(false);
    }
  };

  const renderPageLinks = () => {
    if (totalPages <= 1) return null;
    const pages: (number | "ellipsis")[] = [];
    const showPages = new Set([1, totalPages, page, page - 1, page + 1]);

    for (let i = 1; i <= totalPages; i += 1) {
      if (showPages.has(i) && i >= 1 && i <= totalPages) {
        pages.push(i);
      } else if (i === 2 && page > 3) {
        pages.push("ellipsis");
      } else if (i === totalPages - 1 && page < totalPages - 2) {
        pages.push("ellipsis");
      }
    }

    const deduped: (number | "ellipsis")[] = [];
    pages.forEach((item) => {
      if (item === "ellipsis" && deduped[deduped.length - 1] === "ellipsis") return;
      if (typeof item === "number" && deduped.includes(item)) return;
      deduped.push(item);
    });

    return deduped.map((item, idx) => {
      if (item === "ellipsis") {
        return (
          <PaginationItem key={`ellipsis-${idx}`}>
            <PaginationEllipsis />
          </PaginationItem>
        );
      }
      return (
        <PaginationItem key={item}>
          <PaginationLink
            href="#"
            isActive={item === page}
            onClick={(event) => {
              event.preventDefault();
              setPage(item);
            }}
          >
            {item}
          </PaginationLink>
        </PaginationItem>
      );
    });
  };

  return (
    <div className="min-h-screen bg-background py-12 px-4 sm:px-6 lg:px-8" >
      <div className="max-w-6xl mx-auto">
        <div className="mb-10 text-center">
          <h1 className="text-4xl mb-3 text-primary">
            {t("dataset.hero.title")}
          </h1>
          <p className="text-lg text-muted-foreground">
            {t("dataset.hero.subtitle")}
          </p>
        </div>

        <Card className="bg-card p-6 mb-8 shadow-lg">
          <div className="flex flex-wrap items-start justify-between gap-4 mb-6">
            <div>
              <h2 className="text-2xl text-primary">
                {t("dataset.analysis.title")}
              </h2>
              <p className="text-sm text-muted-foreground">{t("dataset.analysis.subtitle")}</p>
            </div>
            {isLegacyDataset && onNavigateToProtocol && (
              <Button
                type="button"
                variant="outline"
                onClick={onNavigateToProtocol}
                className="border-2 border-foreground text-text-danger hover:text-text-danger"
              >
                {t("dataset.analysis.protocolPageCta")}
              </Button>
            )}
          </div>
          <Tabs defaultValue="overview" className="mt-2">
            <TabsList className="w-full flex flex-wrap justify-start gap-2">
              <TabsTrigger value="overview">{t("dataset.tabs.overview")}</TabsTrigger>
              {isLegacyDataset && <TabsTrigger value="compare">{t("dataset.tabs.compare")}</TabsTrigger>}
              {isLegacyDataset && <TabsTrigger value="annotation">{t("dataset.tabs.annotation")}</TabsTrigger>}
              {isLegacyDataset && <TabsTrigger value="limitation">{t("dataset.tabs.limitation")}</TabsTrigger>}
              {isLegacyDataset && <TabsTrigger value="definition">{t("dataset.tabs.definition")}</TabsTrigger>}
            </TabsList>

            <TabsContent value="overview" className="mt-4 space-y-6">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.overview.distributionAfterMerge")}</p>
                <div className="mt-3 grid grid-cols-1 md:grid-cols-4 gap-3">
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <p className="text-xs text-muted-foreground">{t("dataset.overview.totalSamples")}</p>
                    <p className="text-2xl font-semibold">{aggregatedStats.total.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">{t("dataset.overview.sourcesSummary")}</p>
                  </div>
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <p className="text-xs text-muted-foreground">{t("dataset.overview.cleanNonToxic")}</p>
                    <p className="text-2xl font-semibold text-text-info">{aggregatedStats.clean.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">{t("dataset.overview.percentOfDataset", { percent: formatPercent(aggregatedStats.clean, aggregatedStats.total) })}</p>
                  </div>
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <p className="text-xs text-muted-foreground">{t("dataset.filters.toxic")}</p>
                    <p className="text-2xl font-semibold text-text-warning">{aggregatedStats.toxic.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">{t("dataset.overview.percentOfDataset", { percent: formatPercent(aggregatedStats.toxic, aggregatedStats.total) })}</p>
                  </div>
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <p className="text-xs text-muted-foreground">{t("dataset.overview.imbalanceRatio")}</p>
                    <p className="text-2xl font-semibold">{imbalanceRatioText}</p>
                    <p className="text-xs text-muted-foreground">{t("dataset.overview.cleanToxicCurrent")}</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="lg:col-span-2">
                  <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.overview.contributionBySource")}</p>
                  {isLegacyDataset ? (
                    <>
                      <div className="mt-3 flex flex-wrap gap-4 text-xs text-muted-foreground">
                        <span className="inline-flex items-center gap-2">
                          <span className="h-3 w-3 rounded-sm bg-text-info" />{t("dataset.compare.victsdBadge")}
                        </span>
                        <span className="inline-flex items-center gap-2">
                          <span className="h-3 w-3 rounded-sm bg-text-warning" />{t("dataset.compare.vihsdBadge")}
                        </span>
                      </div>

                      <div className="mt-4 space-y-3 text-sm">
                        <div className="flex items-center gap-3">
                          <div className="w-44 text-muted-foreground">{t("dataset.compare.victsdBadge")}</div>
                          <div className="flex-1 h-3 rounded bg-muted overflow-hidden">
                            <div className="flex h-full">
                              <div className="bg-text-info" style={{ width: formatPercent(sourceSummary.victsd.clean, sourceSummary.victsd.total) }} />
                              <div className="bg-text-warning" style={{ width: formatPercent(sourceSummary.victsd.toxic, sourceSummary.victsd.total) }} />
                            </div>
                          </div>
                          <div className="w-20 text-right text-xs text-muted-foreground">{sourceSummary.victsd.total.toLocaleString()}</div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="w-44 text-muted-foreground">{t("dataset.compare.vihsdBadge")}</div>
                          <div className="flex-1 h-3 rounded bg-muted overflow-hidden">
                            <div className="flex h-full">
                              <div className="bg-text-warning" style={{ width: formatPercent(sourceSummary.vihsd.toxic, sourceSummary.vihsd.total) }} />
                            </div>
                          </div>
                          <div className="w-20 text-right text-xs text-muted-foreground">{sourceSummary.vihsd.total.toLocaleString()}</div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="mt-4 space-y-3 text-sm">
                      {latestSourceSummary.map(([source, counts]) => (
                        <div key={source} className="flex items-center gap-3">
                          <div className="w-44 text-muted-foreground">{sourceLabel(source, t)}</div>
                          <div className="flex-1 h-3 rounded bg-muted overflow-hidden">
                            <div className="flex h-full">
                              <div className="bg-text-info" style={{ width: formatPercent(counts.clean, counts.total) }} />
                              <div className="bg-text-warning" style={{ width: formatPercent(counts.toxic, counts.total) }} />
                            </div>
                          </div>
                          <div className="w-20 text-right text-xs text-muted-foreground">{counts.total.toLocaleString()}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="mt-4 rounded-lg border-l-4 border-l-border-success bg-background-success p-4 text-sm text-text-success">
                    <strong>{t("dataset.overview.resultLabel")}</strong> {t("dataset.overview.dynamicRenderNote")}
                  </div>
                </div>

                {!isLegacyDataset && (
                  <Card className="border p-4 shadow-none">
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Giới hạn (latest)</p>
                    <ul className="mt-3 space-y-2 text-sm text-muted-foreground list-disc pl-4">
                      <li>Phạm vi nguồn vẫn thiên về dữ liệu ViCTSD + dữ liệu thu thập trong đồ án, chưa đại diện đầy đủ mọi ngữ cảnh tiếng Việt.</li>
                      <li>Nhãn mới đi theo luồng pseudo-label + tiêu chí duyệt, nên chất lượng dữ liệu phụ thuộc vào độ tốt của bộ tiêu chí và mức bao phủ của rule.</li>
                      <li>Các trường hợp nằm sát ngưỡng hoặc mơ hồ ngữ nghĩa vẫn có thể bị pseudo-label sai trước bước duyệt.</li>
                      <li>Mất cân bằng lớp clean/toxic vẫn còn đáng kể, có thể làm mô hình thiên lệch về lớp chiếm ưu thế.</li>
                    </ul>
                  </Card>
                )}
              </div>
            </TabsContent>

            {isLegacyDataset && (
              <>
                <TabsContent value="compare" className="mt-4 space-y-6">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.compare.datasetCharacteristics")}</p>
                    <div className="mt-3 grid grid-cols-1 lg:grid-cols-2 gap-4">
                      <Card className="border p-4 shadow-none">
                        <span className="inline-flex rounded-md bg-background-info px-2 py-1 text-xs font-medium text-text-info">{t("dataset.compare.victsdBadge")}</span>
                        <h3 className="mt-3 text-sm font-semibold">{t("dataset.compare.victsdTitle")}</h3>
                        <div className="mt-3 space-y-2 text-sm">
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.source")}</span><span className="text-right">{t("dataset.compare.victsd.source")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.originalSize")}</span><span className="text-right">{t("dataset.compare.victsd.originalSize")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.labelSchema")}</span><span className="text-right">{t("dataset.compare.victsd.labelSchema")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.binarizedTo")}</span><span className="text-right">{t("dataset.compare.victsd.binarizedTo")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.originalToxic")}</span><span className="text-right">{t("dataset.compare.victsd.originalToxic")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.textLength")}</span><span className="text-right">{t("dataset.compare.victsd.textLength")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.style")}</span><span className="text-right">{t("dataset.compare.victsd.style")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.paper")}</span><a className="text-text-info hover:underline" href="https://arxiv.org/abs/2103.10069" target="_blank" rel="noreferrer">arXiv 2103.10069</a></div>
                        </div>
                        <div className="mt-4 rounded-lg border-l-4 border-l-border-info bg-background-info p-3 text-sm text-text-info">
                          {t("dataset.compare.victsd.note")}
                        </div>
                      </Card>

                      <Card className="border p-4 shadow-none">
                        <span className="inline-flex rounded-md bg-background-success px-2 py-1 text-xs font-medium text-text-success">{t("dataset.compare.vihsdBadge")}</span>
                        <h3 className="mt-3 text-sm font-semibold">{t("dataset.compare.vihsdTitle")}</h3>
                        <div className="mt-3 space-y-2 text-sm">
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.source")}</span><span className="text-right">{t("dataset.compare.vihsd.source")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.originalSize")}</span><span className="text-right">{t("dataset.compare.vihsd.originalSize")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.labelSchema")}</span><span className="text-right">{t("dataset.compare.vihsd.labelSchema")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.take")}</span><span className="text-right">{t("dataset.compare.vihsd.take")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.samplesUsed")}</span><span className="text-right">{t("dataset.compare.vihsd.samplesUsed")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.textLength")}</span><span className="text-right">{t("dataset.compare.vihsd.textLength")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.style")}</span><span className="text-right">{t("dataset.compare.vihsd.style")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.fields.paper")}</span><a className="text-text-success hover:underline" href="https://arxiv.org/abs/2103.11528" target="_blank" rel="noreferrer">arXiv 2103.11528</a></div>
                        </div>
                        <div className="mt-4 rounded-lg border-l-4 border-l-border-info bg-background-info p-3 text-sm text-text-info">
                          {t("dataset.compare.vihsd.note")}
                        </div>
                      </Card>
                    </div>
                  </div>

                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.compare.similaritiesReason")}</p>
                    <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div className="rounded-lg border-l-4 border-l-border-success bg-background-success p-4 text-sm text-text-success">
                        <strong>{t("dataset.compare.similarity1Title")}</strong><br />{t("dataset.compare.similarity1Desc")}
                      </div>
                      <div className="rounded-lg border-l-4 border-l-border-success bg-background-success p-4 text-sm text-text-success">
                        <strong>{t("dataset.compare.similarity2Title")}</strong><br />{t("dataset.compare.similarity2Desc")}
                      </div>
                      <div className="rounded-lg border-l-4 border-l-border-success bg-background-success p-4 text-sm text-text-success">
                        <strong>{t("dataset.compare.similarity3Title")}</strong><br />{t("dataset.compare.similarity3Desc")}
                      </div>
                    </div>
                  </div>

                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.compare.empiricalEvidence")}</p>
                    <div className="mt-3 grid grid-cols-1 lg:grid-cols-2 gap-4">
                      <Card className="border p-4 shadow-none">
                        <p className="text-sm font-semibold">{t("dataset.compare.evidenceScoreTitle")}</p>
                        <div className="mt-3 space-y-2 text-sm">
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.evidenceOffensive")}</span><span className="font-medium">0.8726</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.evidenceHate")}</span><span className="font-medium">{t("dataset.compare.evidenceHateValue")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.compare.evidenceClean")}</span><span className="font-medium">0.1285</span></div>
                        </div>
                        <div className="mt-4 rounded-lg border-l-4 border-l-border-success bg-background-success p-3 text-sm text-text-success">
                          {t("dataset.compare.evidenceConclusion")}
                        </div>
                      </Card>

                      <Card className="border p-4 shadow-none">
                        <p className="text-sm font-semibold">{t("dataset.compare.distributionTitle")}</p>
                        <img
                          src="/src/assets/images/distribution_vihsd_toxic.png"
                          alt={t("dataset.compare.distributionAlt")}
                          className="mt-3 w-full rounded-lg border"
                        />
                        <p className="mt-3 text-sm text-muted-foreground">
                          {t("dataset.compare.distributionDesc")}
                        </p>
                      </Card>
                    </div>
                  </div>

                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.compare.differencesAcknowledged")}</p>
                    <div className="mt-3 rounded-lg border-l-4 border-l-border-warning bg-background-warning p-4 text-sm text-text-warning">
                      <strong>{t("dataset.compare.toxicShareTitle")}</strong> {t("dataset.compare.toxicShareDesc")}
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="annotation" className="mt-4 space-y-6">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.annotation.consistencyMock")}</p>
                    <div className="mt-3 rounded-lg border bg-muted/30 p-4 text-sm">
                      <p className="leading-7">
                        <strong>{t("dataset.annotation.consistencyTitle")}</strong> {t("dataset.annotation.consistencyDesc")}
                      </p>
                      <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-muted-foreground">
                        <div>{t("dataset.annotation.kappa1")}</div>
                        <div>{t("dataset.annotation.kappa2")}</div>
                        <div>{t("dataset.annotation.kappa3")}</div>
                        <div>{t("dataset.annotation.kappa4")}</div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.annotation.whyOffensiveToToxic")}</p>
                    <div className="mt-3 grid grid-cols-1 lg:grid-cols-2 gap-4">
                      <Card className="border p-4 shadow-none">
                        <span className="inline-flex rounded-md bg-background-success px-2 py-1 text-xs font-medium text-text-success">{t("dataset.annotation.offensiveBadge")}</span>
                        <div className="mt-3 space-y-2 text-sm">
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.annotation.fields.definition")}</span><span className="text-right">{t("dataset.annotation.offensive.definition")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.annotation.fields.overlap")}</span><span className="text-right">{t("dataset.annotation.offensive.overlap")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.annotation.fields.annotationBoundary")}</span><span className="text-right">{t("dataset.annotation.offensive.annotationBoundary")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.annotation.fields.labelNoise")}</span><span className="text-right">{t("dataset.annotation.offensive.labelNoise")}</span></div>
                        </div>
                        <div className="mt-4 rounded-lg border-l-4 border-l-border-success bg-background-success p-3 text-sm text-text-success">
                          {t("dataset.annotation.offensive.note")}
                        </div>
                      </Card>

                      <Card className="border p-4 shadow-none">
                        <span className="inline-flex rounded-md bg-background-danger px-2 py-1 text-xs font-medium text-text-danger">{t("dataset.annotation.hateBadge")}</span>
                        <div className="mt-3 space-y-2 text-sm">
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.annotation.fields.definition")}</span><span className="text-right">{t("dataset.annotation.hate.definition")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.annotation.fields.characteristics")}</span><span className="text-right">{t("dataset.annotation.hate.characteristics")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.annotation.fields.annotationBoundary")}</span><span className="text-right">{t("dataset.annotation.hate.annotationBoundary")}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-muted-foreground">{t("dataset.annotation.fields.labelNoise")}</span><span className="text-right">{t("dataset.annotation.hate.labelNoise")}</span></div>
                        </div>
                        <div className="mt-4 rounded-lg border-l-4 border-l-border-danger bg-background-danger p-3 text-sm text-text-danger">
                          {t("dataset.annotation.hate.note")}
                        </div>
                      </Card>
                    </div>
                  </div>

                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.annotation.faq")}</p>
                    <div className="mt-3 space-y-4 text-sm">
                      <div>
                        <div className="flex items-start gap-2 font-medium">
                          <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-background-info text-xs text-text-info">Q</span>
                          {t("dataset.annotation.q1")}
                        </div>
                        <p className="mt-2 pl-7 text-muted-foreground">
                          {t("dataset.annotation.a1")}
                        </p>
                      </div>
                      <div>
                        <div className="flex items-start gap-2 font-medium">
                          <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-background-info text-xs text-text-info">Q</span>
                          {t("dataset.annotation.q2")}
                        </div>
                        <p className="mt-2 pl-7 text-muted-foreground">
                          {t("dataset.annotation.a2")}
                        </p>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="limitation" className="mt-4 space-y-6">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.limitation.title")}</p>
                    <div className="mt-3 space-y-4 text-sm">
                      <div className="flex gap-3 border-b pb-4">
                        <div className="h-8 w-8 rounded-full bg-background-warning text-text-warning flex items-center justify-center text-xs font-semibold">L1</div>
                        <div>
                          <p className="font-medium">{t("dataset.limitation.l1Title")}</p>
                          <p className="text-muted-foreground">{t("dataset.limitation.l1Desc")}</p>
                        </div>
                      </div>
                      <div className="flex gap-3 border-b pb-4">
                        <div className="h-8 w-8 rounded-full bg-background-warning text-text-warning flex items-center justify-center text-xs font-semibold">L2</div>
                        <div>
                          <p className="font-medium">{t("dataset.limitation.l2Title")}</p>
                          <p className="text-muted-foreground">{t("dataset.limitation.l2Desc")}</p>
                        </div>
                      </div>
                      <div className="flex gap-3 border-b pb-4">
                        <div className="h-8 w-8 rounded-full bg-background-danger text-text-danger flex items-center justify-center text-xs font-semibold">L3</div>
                        <div>
                          <p className="font-medium">{t("dataset.limitation.l3Title")}</p>
                          <p className="text-muted-foreground">{t("dataset.limitation.l3Desc")}</p>
                        </div>
                      </div>
                      <div className="flex gap-3 border-b pb-4">
                        <div className="h-8 w-8 rounded-full bg-background-danger text-text-danger flex items-center justify-center text-xs font-semibold">L4</div>
                        <div>
                          <p className="font-medium">{t("dataset.limitation.l4Title")}</p>
                          <p className="text-muted-foreground">{t("dataset.limitation.l4Desc")}</p>
                        </div>
                      </div>
                      <div className="flex gap-3">
                        <div className="h-8 w-8 rounded-full bg-background-info text-text-info flex items-center justify-center text-xs font-semibold">L5</div>
                        <div>
                          <p className="font-medium">{t("dataset.limitation.l5Title")}</p>
                          <p className="text-muted-foreground">{t("dataset.limitation.l5Desc")}</p>
                        </div>
                      </div>
                    </div>
                    <div className="mt-4 rounded-lg border-l-4 border-l-border-info bg-background-info p-4 text-sm text-text-info">
                      <strong>{t("dataset.limitation.tipTitle")}</strong> {t("dataset.limitation.tipDesc")}
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="definition" className="mt-4 space-y-6">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{t("dataset.definition.title")}</p>
                    <div className="mt-3 text-sm text-muted-foreground space-y-3">
                      <p>
                        {t("dataset.definition.p1Prefix")}<strong>{t("dataset.definition.p1Label")}</strong>{t("dataset.definition.p1Mid")}
                        <strong>{t("dataset.definition.p1Clean")}</strong>, <strong>{t("dataset.definition.p1Toxic")}</strong>.
                      </p>
                      <p>
                        {t("dataset.definition.p2Prefix")}<strong>{t("dataset.definition.p2Constructiveness")}</strong>{t("dataset.definition.p2Suffix")}
                      </p>
                    </div>
                  </div>
                </TabsContent>
              </>
            )}
          </Tabs>
        </Card>

        <Card className="bg-card p-6 mb-8 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
            <div>
              <Label className="text-sm text-muted-foreground">{t("dataset.filters.source")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={sourceFilter}
                onChange={(event) => setSourceFilter(event.target.value)}
              >
                {availableSources.map((source) => (
                  <option key={source} value={source}>
                    {sourceLabel(source, t)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">{t("dataset.filters.label")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={labelFilter}
                onChange={(event) => setLabelFilter(event.target.value)}
              >
                <option value="all">{t("dataset.filters.all")}</option>
                <option value="clean">{t("dataset.filters.clean")}</option>
                <option value="toxic">{t("dataset.filters.toxic")}</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">{t("dataset.filters.split")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={splitFilter}
                onChange={(event) => setSplitFilter(event.target.value)}
              >
                <option value="all">{t("dataset.filters.all")}</option>
                <option value="train">{t("dataset.filters.train")}</option>
                <option value="validation">{t("dataset.filters.validation")}</option>
                <option value="test">{t("dataset.filters.test")}</option>
                <option value="feedback">{t("dataset.filters.feedback")}</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">{t("dataset.filters.pageSize")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={pageSize}
                onChange={(event) => setPageSize(Number(event.target.value))}
              >
                {[10, 25, 50, 100].map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-3 items-center">
            <Button onClick={() => fetchPreview(1, pageSize)} disabled={loading}>
              {loading ? t("dataset.status.loading") : t("dataset.actions.refresh")}
            </Button>
            <Button variant="outline" onClick={handleExport}>
              {t("dataset.actions.exportJsonl")}
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteFeedback}
              disabled={!selectedFeedback.length || deleteLoading}
            >
              {deleteLoading
                ? t("dataset.status.deleting")
                : t("dataset.actions.deleteFeedbackCount", { count: selectedFeedback.length })}
            </Button>
            {exportStatus && <span className="text-sm text-muted-foreground">{exportStatus}</span>}
            {deleteStatus && <span className="text-sm text-muted-foreground">{deleteStatus}</span>}
          </div>

          {error && <p className="mt-3 text-sm text-destructive">{error}</p>}
        </Card>

        <Card className="bg-card p-6 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <h2 className="text-2xl text-primary">
                {t("dataset.overview.title")}
              </h2>
              <p className="text-sm text-muted-foreground">
                {t("dataset.overview.currentFilterStats")}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge className="bg-background-info text-text-info">{t("dataset.overview.mergedDatasets", { count: aggregatedStats.sources.length })}</Badge>
              <Badge className={imbalanceStatus?.isImbalanced ? "bg-background-danger text-text-danger" : "bg-background-success text-text-success"}>
                {imbalanceStatus?.isImbalanced ? t("dataset.overview.imbalanced") : t("dataset.overview.balanced")}
              </Badge>
            </div>
          </div>

          {imbalanceStatus?.isImbalanced && (
            <div className="mb-6 rounded-lg border border-border-danger bg-background-danger px-4 py-3 text-sm text-text-danger">
              {t("dataset.overview.dominantClassWarning", {
                label: imbalanceStatus.dominant === "clean" ? t("dataset.filters.clean") : t("dataset.filters.toxic"),
                percent: formatPercent(
                  imbalanceStatus.dominant === "clean" ? aggregatedStats.clean : aggregatedStats.toxic,
                  aggregatedStats.total,
                ),
              })}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card className="bg-card p-4 border">
              <p className="text-sm text-muted-foreground">{t("dataset.overview.totalSamples")}</p>
              <p className="text-2xl text-primary">{aggregatedStats.total}</p>
            </Card>
            <Card className="bg-card p-4 border">
              <p className="text-sm text-muted-foreground">{t("dataset.filters.clean")}</p>
              <p className="text-xl text-foreground">
                {aggregatedStats.clean} ({formatPercent(aggregatedStats.clean, aggregatedStats.total)})
              </p>
            </Card>
            <Card className="bg-card p-4 border">
              <p className="text-sm text-muted-foreground">{t("dataset.filters.toxic")}</p>
              <p className="text-xl text-foreground">
                {aggregatedStats.toxic} ({formatPercent(aggregatedStats.toxic, aggregatedStats.total)})
              </p>
            </Card>
            <Card className="bg-card p-4 border">
              <p className="text-sm text-muted-foreground">{t("dataset.overview.page")}</p>
              <p className="text-sm text-foreground mt-2">
                {page} / {totalPages}
              </p>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="border rounded-lg overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("dataset.filters.source")}</TableHead>
                    <TableHead className="text-right">{t("dataset.filters.clean")}</TableHead>
                    <TableHead className="text-right">{t("dataset.filters.toxic")}</TableHead>
                    <TableHead className="text-right">{t("dataset.common.total")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {aggregatedStats.sources.map((source) => {
                    const counts = stats?.by_source?.[source];
                    if (!counts) return null;
                    return (
                      <TableRow key={source}>
                        <TableCell>{source}</TableCell>
                        <TableCell className="text-right">{counts.clean}</TableCell>
                        <TableCell className="text-right">{counts.toxic}</TableCell>
                        <TableCell className="text-right">{counts.total}</TableCell>
                      </TableRow>
                    );
                  })}
                  {!aggregatedStats.sources.length && (
                    <TableRow>
                      <TableCell colSpan={4} className="text-center text-sm text-muted-foreground">
                        --
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>

            <div className="border rounded-lg overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("dataset.overview.crossTab")}</TableHead>
                    <TableHead className="text-right">{t("dataset.filters.clean")}</TableHead>
                    <TableHead className="text-right">{t("dataset.filters.toxic")}</TableHead>
                    <TableHead className="text-right">{t("dataset.common.total")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {aggregatedStats.sources.map((source) => {
                    const counts = stats?.by_source?.[source];
                    if (!counts) return null;
                    return (
                      <TableRow key={`cross-${source}`}>
                        <TableCell>{source}</TableCell>
                        <TableCell className="text-right">
                          {counts.clean} ({formatPercent(counts.clean, counts.total)})
                        </TableCell>
                        <TableCell className="text-right">
                          {counts.toxic} ({formatPercent(counts.toxic, counts.total)})
                        </TableCell>
                        <TableCell className="text-right">{counts.total}</TableCell>
                      </TableRow>
                    );
                  })}
                  <TableRow>
                    <TableCell className="font-medium">{t("dataset.common.total")}</TableCell>
                    <TableCell className="text-right font-medium">
                      {aggregatedStats.clean} ({formatPercent(aggregatedStats.clean, aggregatedStats.total)})
                    </TableCell>
                    <TableCell className="text-right font-medium">
                      {aggregatedStats.toxic} ({formatPercent(aggregatedStats.toxic, aggregatedStats.total)})
                    </TableCell>
                    <TableCell className="text-right font-medium">{aggregatedStats.total}</TableCell>
                  </TableRow>
                  {!aggregatedStats.sources.length && (
                    <TableRow>
                      <TableCell colSpan={4} className="text-center text-sm text-muted-foreground">
                        --
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
        </Card>

        <Card className="bg-card p-6 shadow-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("dataset.table.select")}</TableHead>
                <TableHead>{t("dataset.table.text")}</TableHead>
                <TableHead>{t("dataset.filters.label")}</TableHead>
                <TableHead>{t("dataset.filters.source")}</TableHead>
                <TableHead>{t("dataset.filters.split")}</TableHead>
                <TableHead>{t("dataset.table.augmented")}</TableHead>
                <TableHead>{t("dataset.table.created")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row, idx) => {
                const feedbackId = row.meta?.feedback_id;
                const selectable = row.meta?.split === "feedback" && typeof feedbackId === "number";
                return (
                  <TableRow key={`${row.text.slice(0, 20)}-${idx}`}>
                    <TableCell>
                      {selectable ? (
                        <input
                          type="checkbox"
                          checked={selectedFeedback.includes(feedbackId)}
                          onChange={() => toggleFeedbackSelection(feedbackId)}
                        />
                      ) : (
                        t("dataset.common.na")
                      )}
                    </TableCell>
                    <TableCell className="max-w-[360px] truncate" title={row.text}>
                      {row.text}
                    </TableCell>
                    <TableCell>{labelText(row.label, t)}</TableCell>
                    <TableCell>{row.meta?.source ?? t("dataset.common.na")}</TableCell>
                    <TableCell>{row.meta?.split ?? t("dataset.common.na")}</TableCell>
                    <TableCell>{row.meta?.is_augmented ? t("dataset.common.yes") : t("dataset.common.no")}</TableCell>
                    <TableCell>{row.meta?.created_at ?? t("dataset.common.na")}</TableCell>
                  </TableRow>
                );
              })}
              {!rows.length && !loading && (
                <TableRow>
                  <TableCell colSpan={7} className="text-center text-sm text-muted-foreground">
                    {t("dataset.common.noData")}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>

          <div className="mt-6">
            <Pagination>
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    href="#"
                    onClick={(event) => {
                      event.preventDefault();
                      setPage((prev) => Math.max(1, prev - 1));
                    }}
                  />
                </PaginationItem>
                {renderPageLinks()}
                <PaginationItem>
                  <PaginationNext
                    href="#"
                    onClick={(event) => {
                      event.preventDefault();
                      setPage((prev) => Math.min(totalPages, prev + 1));
                    }}
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          </div>
        </Card>
      </div>
    </div>
  );
}
