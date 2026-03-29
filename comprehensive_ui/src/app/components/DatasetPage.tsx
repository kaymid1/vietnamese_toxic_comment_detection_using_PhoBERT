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

interface DatasetPreviewResponse {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  items: DatasetRow[];
  stats?: DatasetStats;
}

interface DatasetExportResponse {
  path: string;
  count: number;
  stats: DatasetStats;
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

const labelText = (label: number) => (label === 1 ? "toxic" : "clean");

const sourceLabel = (source: string) => {
  const normalized = source.trim().toLowerCase();
  const map: Record<string, string> = {
    all: "Tất cả nguồn",
    victsd_augmented: "ViCTSD",
    "uit-vihsd_augmented": "UIT-ViHSD",
    new_collected: "New collected",
    unknown: "Không xác định",
  };
  return map[normalized] || source.replaceAll("_", " ");
};

const SOURCE_ORDER: string[] = ["victsd_augmented", "uit-vihsd_augmented", "new_collected"];

const isVisibleSourceOption = (source: string) => {
  const normalized = source.trim().toLowerCase();
  return (
    normalized !== "all" &&
    normalized !== "victsd" &&
    normalized !== "vihsd" &&
    normalized !== "vihsd_augmented"
  );
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

interface DatasetPageProps {
  onOpenSyntheticPage?: () => void;
}

export function DatasetPage({ onOpenSyntheticPage }: DatasetPageProps) {
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
      const filtered = Array.from(merged).filter(isVisibleSourceOption);
      return sortSourcesByPreferredOrder(filtered);
    });
  }, [stats]);

  const availableSources = useMemo(() => {
    const merged = new Set([...sourceOptions, sourceFilter]);
    const filtered = Array.from(merged).filter(isVisibleSourceOption);
    return ["all", ...sortSourcesByPreferredOrder(filtered)];
  }, [sourceFilter, sourceOptions]);

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
    return {
      victsd: bySource.victsd_augmented || { total: 0, clean: 0, toxic: 0 },
      vihsd: bySource["uit-vihsd_augmented"] || bySource.vihsd_augmented || { total: 0, clean: 0, toxic: 0 },
      newCollected: bySource.new_collected || { total: 0, clean: 0, toxic: 0 },
    };
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
      const message = err instanceof Error ? err.message : "Không thể tải dataset";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setPage(1);
    setSelectedFeedback([]);
  }, [sourceFilter, labelFilter, splitFilter, pageSize]);

  useEffect(() => {
    void fetchPreview(page, pageSize);
  }, [page, pageSize, sourceFilter, labelFilter, splitFilter]);

  const handleExport = async () => {
    setExportStatus(null);
    try {
      const body: Record<string, unknown> = {};
      if (sourceFilter !== "all") body.source = [sourceFilter];
      if (labelFilter !== "all") body.label = [labelFilter === "toxic" ? 1 : 0];
      if (splitFilter !== "all") body.split = [splitFilter];

      const response = await fetch(buildApiUrl("/api/dataset/export"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = (await response.json()) as DatasetExportResponse;
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }
      setExportStatus(`Đã xuất ${data.count} dòng vào ${data.path}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Xuất dữ liệu thất bại";
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
    const confirmed = window.confirm(`Xoá ${selectedFeedback.length} feedback đã chọn?`);
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
      setDeleteStatus(`Đã xoá ${data.deleted} feedback.`);
      setSelectedFeedback([]);
      await fetchPreview(page, pageSize);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Xoá feedback thất bại";
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
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8" style={{ backgroundColor: "var(--viet-bg)" }}>
      <div className="max-w-6xl mx-auto">
        <div className="mb-10 text-center">
          <h1 className="text-4xl mb-3" style={{ color: "var(--viet-primary)" }}>
            Dataset Preview
          </h1>
          <p className="text-lg text-gray-600">
            Preview dữ liệu huấn luyện (augmented + non-augmented) và feedback mới thu thập.
          </p>
        </div>

        <Card className="bg-white p-6 mb-8 shadow-lg">
          <div className="flex flex-wrap items-start justify-between gap-4 mb-6">
            <div>
              <h2 className="text-2xl" style={{ color: "var(--viet-primary)" }}>
                Dataset analysis (defense)
              </h2>
              <p className="text-sm text-gray-600">Tổng hợp nhanh để bảo vệ: overview, so sánh, annotation và limitations.</p>
            </div>
          </div>
          <Tabs defaultValue="overview" className="mt-2">
            <TabsList className="w-full flex flex-wrap justify-start gap-2">
              <TabsTrigger value="overview">Tổng quan dataset</TabsTrigger>
              <TabsTrigger value="compare">So sánh chi tiết</TabsTrigger>
              <TabsTrigger value="annotation">Annotation &amp; merge</TabsTrigger>
              <TabsTrigger value="limitation">Limitations</TabsTrigger>
              <TabsTrigger value="definition">Toxicity definition</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="mt-4 space-y-6">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Phân phối sau khi merge</p>
                <div className="mt-3 grid grid-cols-1 md:grid-cols-4 gap-3">
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <p className="text-xs text-muted-foreground">Tổng samples</p>
                    <p className="text-2xl font-semibold">{aggregatedStats.total.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">ViCTSD + UIT-ViHSD + collected</p>
                  </div>
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <p className="text-xs text-muted-foreground">Clean (non-toxic)</p>
                    <p className="text-2xl font-semibold text-blue-700">{aggregatedStats.clean.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">{formatPercent(aggregatedStats.clean, aggregatedStats.total)} tổng dataset</p>
                  </div>
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <p className="text-xs text-muted-foreground">Toxic</p>
                    <p className="text-2xl font-semibold text-orange-600">{aggregatedStats.toxic.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">{formatPercent(aggregatedStats.toxic, aggregatedStats.total)} tổng dataset</p>
                  </div>
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <p className="text-xs text-muted-foreground">Imbalance ratio</p>
                    <p className="text-2xl font-semibold">{imbalanceRatioText}</p>
                    <p className="text-xs text-muted-foreground">Clean:Toxic theo dữ liệu hiện tại</p>
                  </div>
                </div>
              </div>

              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Đóng góp theo nguồn</p>
                <div className="mt-3 flex flex-wrap gap-4 text-xs text-muted-foreground">
                  <span className="inline-flex items-center gap-2">
                    <span className="h-3 w-3 rounded-sm bg-blue-600" />ViCTSD
                  </span>
                  <span className="inline-flex items-center gap-2">
                    <span className="h-3 w-3 rounded-sm bg-orange-500" />UIT-ViHSD
                  </span>
                  <span className="inline-flex items-center gap-2">
                    <span className="h-3 w-3 rounded-sm bg-zinc-400" />New collected
                  </span>
                </div>

                <div className="mt-4 space-y-3 text-sm">
                  <div className="flex items-center gap-3">
                    <div className="w-44 text-muted-foreground">ViCTSD</div>
                    <div className="flex-1 h-3 rounded bg-muted overflow-hidden">
                      <div className="flex h-full">
                        <div className="bg-blue-600" style={{ width: formatPercent(sourceSummary.victsd.clean, sourceSummary.victsd.total) }} />
                        <div className="bg-orange-500" style={{ width: formatPercent(sourceSummary.victsd.toxic, sourceSummary.victsd.total) }} />
                      </div>
                    </div>
                    <div className="w-20 text-right text-xs text-muted-foreground">{sourceSummary.victsd.total.toLocaleString()}</div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-44 text-muted-foreground">UIT-ViHSD</div>
                    <div className="flex-1 h-3 rounded bg-muted overflow-hidden">
                      <div className="flex h-full">
                        <div className="bg-orange-500" style={{ width: formatPercent(sourceSummary.vihsd.toxic, sourceSummary.vihsd.total) }} />
                      </div>
                    </div>
                    <div className="w-20 text-right text-xs text-muted-foreground">{sourceSummary.vihsd.total.toLocaleString()}</div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-44 text-muted-foreground">New collected</div>
                    <div className="flex-1 h-3 rounded bg-muted overflow-hidden">
                      <div className="flex h-full">
                        <div className="bg-blue-600" style={{ width: formatPercent(sourceSummary.newCollected.clean, sourceSummary.newCollected.total) }} />
                        <div className="bg-orange-500" style={{ width: formatPercent(sourceSummary.newCollected.toxic, sourceSummary.newCollected.total) }} />
                      </div>
                    </div>
                    <div className="w-20 text-right text-xs text-muted-foreground">{sourceSummary.newCollected.total.toLocaleString()}</div>
                  </div>
                </div>

                <div className="mt-4 rounded-lg border-l-4 border-l-green-600 bg-green-50 p-4 text-sm text-green-800">
                  <strong>Kết quả:</strong> Dữ liệu được render động từ API dataset preview theo bộ lọc hiện tại, không còn hardcode số lượng.
                </div>
              </div>
            </TabsContent>

            <TabsContent value="compare" className="mt-4 space-y-6">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Đặc điểm từng dataset</p>
                <div className="mt-3 grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card className="border p-4 shadow-none">
                    <span className="inline-flex rounded-md bg-blue-100 px-2 py-1 text-xs font-medium text-blue-700">ViCTSD</span>
                    <h3 className="mt-3 text-sm font-semibold">UIT-ViCTSD (2021)</h3>
                    <div className="mt-3 space-y-2 text-sm">
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Nguồn</span><span className="text-right">YouTube comments trên video tin tức VN</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Kích thước gốc</span><span className="text-right">10,000 comments</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Label schema</span><span className="text-right">Toxic / Constructive / Neutral</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Binarize thành</span><span className="text-right">Toxic vs Non-toxic</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Toxic gốc</span><span className="text-right">~10.8% (heavily imbalanced)</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Độ dài text</span><span className="text-right">Dài hơn, có ngữ cảnh tranh luận</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Style</span><span className="text-right">Informal, comment phản hồi video</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Paper</span><a className="text-blue-700 hover:underline" href="https://arxiv.org/abs/2103.10069" target="_blank" rel="noreferrer">arXiv 2103.10069</a></div>
                    </div>
                    <div className="mt-4 rounded-lg border-l-4 border-l-blue-600 bg-blue-50 p-3 text-sm text-blue-900">
                      10 domains: chính trị, thể thao, giải trí, kinh tế, sức khỏe... — đa dạng chủ đề.
                    </div>
                  </Card>

                  <Card className="border p-4 shadow-none">
                    <span className="inline-flex rounded-md bg-emerald-100 px-2 py-1 text-xs font-medium text-emerald-700">ViHSD</span>
                    <h3 className="mt-3 text-sm font-semibold">UIT-ViHSD (2021)</h3>
                    <div className="mt-3 space-y-2 text-sm">
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Nguồn</span><span className="text-right">Facebook posts + comments</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Kích thước gốc</span><span className="text-right">33,400 comments</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Label schema</span><span className="text-right">CLEAN / OFFENSIVE / HATE</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Bạn lấy</span><span className="text-right">Chỉ OFFENSIVE → map sang Toxic</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Samples dùng</span><span className="text-right">2,260 OFFENSIVE samples</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Độ dài text</span><span className="text-right">Ngắn hơn, nhiều viết tắt, emoji</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Style</span><span className="text-right">Informal, status/comment Facebook</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Paper</span><a className="text-emerald-700 hover:underline" href="https://arxiv.org/abs/2103.11528" target="_blank" rel="noreferrer">arXiv 2103.11528</a></div>
                    </div>
                    <div className="mt-4 rounded-lg border-l-4 border-l-blue-600 bg-blue-50 p-3 text-sm text-blue-900">
                      Comments chứa nhiều teencode, viết tắt (M.n, mik, Dm), slang — cần xử lý tiền xử lý riêng.
                    </div>
                  </Card>
                </div>
              </div>

              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Điểm tương đồng — lý do merge được</p>
                <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div className="rounded-lg border-l-4 border-l-green-600 bg-green-50 p-4 text-sm text-green-900">
                    <strong>Cùng domain</strong><br />Cả hai đều là social media informal Vietnamese text — domain shift nhỏ.
                  </div>
                  <div className="rounded-lg border-l-4 border-l-green-600 bg-green-50 p-4 text-sm text-green-900">
                    <strong>Cùng nguồn gốc</strong><br />Cả hai do UIT NLP Group xây dựng với quy trình annotation chuẩn.
                  </div>
                  <div className="rounded-lg border-l-4 border-l-green-600 bg-green-50 p-4 text-sm text-green-900">
                    <strong>Ngữ nghĩa gần nhau</strong><br />OFFENSIVE ≈ Toxic về mức độ xúc phạm, phù hợp mapping 1-1.
                  </div>
                </div>
              </div>

              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Empirical evidence từ inter-dataset scoring</p>
                <div className="mt-3 grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card className="border p-4 shadow-none">
                    <p className="text-sm font-semibold">Mean toxic_score (ViCTSD-trained model)</p>
                    <div className="mt-3 space-y-2 text-sm">
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Offensive (label=1)</span><span className="font-medium">0.8726</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Hate (label=2)</span><span className="font-medium">N/A (không có trong merged binary set)</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Clean (label=0)</span><span className="font-medium">0.1285</span></div>
                    </div>
                    <div className="mt-4 rounded-lg border-l-4 border-l-green-600 bg-green-50 p-3 text-sm text-green-900">
                      Khoảng cách điểm rất rõ giữa Offensive và Clean (0.8726 vs 0.1285), cho thấy mapping OFFENSIVE → Toxic có tính nhất quán thực nghiệm. Theo rule mean &gt; 0.6, quyết định merge là hợp lý.
                    </div>
                  </Card>

                  <Card className="border p-4 shadow-none">
                    <p className="text-sm font-semibold">Distribution toxic_score</p>
                    <img
                      src="/src/assets/images/distribution_vihsd_toxic.png"
                      alt="Distribution of toxic scores for clean, offensive, and hate groups"
                      className="mt-3 w-full rounded-lg border"
                    />
                    <p className="mt-3 text-sm text-muted-foreground">
                      Phân phối cho thấy cụm Offensive tập trung ở vùng điểm cao, còn Clean tập trung ở vùng thấp. Điều này củng cố thêm cơ sở empirical để merge ViHSD Offensive vào lớp Toxic.
                    </p>
                  </Card>
                </div>
              </div>

              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Điểm khác biệt — thừa nhận</p>
                <div className="mt-3 rounded-lg border-l-4 border-l-yellow-500 bg-yellow-50 p-4 text-sm text-yellow-900">
                  <strong>74.9% toxic samples đến từ ViHSD OFFENSIVE</strong> (2,260/3,019). Model đang học toxic từ nguồn nào nhiều hơn? → ViCTSD cung cấp label schema và negative examples; ViHSD cung cấp positive examples để cân bằng class — cả hai cùng định hình decision boundary.
                </div>
              </div>
            </TabsContent>

            <TabsContent value="annotation" className="mt-4 space-y-6">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Annotation consistency là gì? (mock)</p>
                <div className="mt-3 rounded-lg border bg-muted/30 p-4 text-sm">
                  <p className="leading-7">
                    <strong>Annotation consistency</strong> = mức độ đồng thuận giữa các annotator khi label cùng một câu text. Đo bằng <strong>Cohen&apos;s Kappa (κ)</strong> hoặc <strong>Fleiss&apos; Kappa</strong>.
                  </p>
                  <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-muted-foreground">
                    <div>κ = 0.0–0.2 → Gần như ngẫu nhiên</div>
                    <div>κ = 0.2–0.4 → Yếu</div>
                    <div>κ = 0.4–0.6 → Trung bình (acceptable)</div>
                    <div>κ = 0.6–0.8 → Tốt</div>
                  </div>
                </div>
              </div>

              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Tại sao OFFENSIVE → Toxic (không dùng HATE)?</p>
                <div className="mt-3 grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card className="border p-4 shadow-none">
                    <span className="inline-flex rounded-md bg-emerald-100 px-2 py-1 text-xs font-medium text-emerald-700">OFFENSIVE → Toxic ✓</span>
                    <div className="mt-3 space-y-2 text-sm">
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Định nghĩa</span><span className="text-right">Ngôn ngữ xúc phạm cá nhân, thô tục</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Overlap</span><span className="text-right">Cao với Toxic trong ViCTSD</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Annotation boundary</span><span className="text-right">Tương đối rõ ràng, consistent</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Label noise khi merge</span><span className="text-right">Thấp</span></div>
                    </div>
                    <div className="mt-4 rounded-lg border-l-4 border-l-green-600 bg-green-50 p-3 text-sm text-green-900">
                      Phù hợp để bổ sung positive examples cho bài toán toxic detection.
                    </div>
                  </Card>

                  <Card className="border p-4 shadow-none">
                    <span className="inline-flex rounded-md bg-red-100 px-2 py-1 text-xs font-medium text-red-700">HATE → Loại ✗</span>
                    <div className="mt-3 space-y-2 text-sm">
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Định nghĩa</span><span className="text-right">Kích động thù địch nhắm nhóm người</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Đặc điểm</span><span className="text-right">Nhắm ethnicity, religion, gender...</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Annotation boundary</span><span className="text-right">Khác guideline so với ViCTSD</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Label noise khi merge</span><span className="text-right">Cao — concept khác nhau</span></div>
                    </div>
                    <div className="mt-4 rounded-lg border-l-4 border-l-red-600 bg-red-50 p-3 text-sm text-red-900">
                      Thêm HATE gây label noise: annotator ViCTSD không train theo tiêu chí này.
                    </div>
                  </Card>
                </div>
              </div>

              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Câu hỏi thường gặp</p>
                <div className="mt-3 space-y-4 text-sm">
                  <div>
                    <div className="flex items-start gap-2 font-medium">
                      <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-indigo-100 text-xs text-indigo-700">Q</span>
                      Annotation boundary giữa OFFENSIVE và Toxic có consistent không?
                    </div>
                    <p className="mt-2 pl-7 text-muted-foreground">
                      Không hoàn toàn identical — đây là limitation được thừa nhận. Tuy nhiên cả hai đều do UIT NLP Group xây dựng, cùng hướng đến toxic/offensive content trên social media VN. Semantic overlap đủ cao để merge có ý nghĩa. Rủi ro label noise nhỏ được chấp nhận đổi lại việc cải thiện class balance đáng kể từ 8.2:1 xuống 2.1:1.
                    </p>
                  </div>
                  <div>
                    <div className="flex items-start gap-2 font-medium">
                      <span className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-indigo-100 text-xs text-indigo-700">Q</span>
                      Vì sao không merge CLEAN của ViHSD vào Non-toxic?
                    </div>
                    <p className="mt-2 pl-7 text-muted-foreground">
                      ViCTSD đã có đủ non-toxic samples (6,241 samples — 89.2%). Thêm CLEAN từ ViHSD sẽ khiến imbalance tệ hơn theo hướng ngược lại. Mục tiêu là cân bằng class, không phải tối đa hóa data.
                    </p>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="limitation" className="mt-4 space-y-6">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Limitations của dataset — chủ động nêu trước hội đồng</p>
                <div className="mt-3 space-y-4 text-sm">
                  <div className="flex gap-3 border-b pb-4">
                    <div className="h-8 w-8 rounded-full bg-yellow-100 text-yellow-900 flex items-center justify-center text-xs font-semibold">L1</div>
                    <div>
                      <p className="font-medium">Annotation guideline không hoàn toàn identical</p>
                      <p className="text-muted-foreground">ViCTSD và ViHSD được xây dựng bởi hai nhóm khác nhau với guideline riêng. OFFENSIVE trong ViHSD và Toxic trong ViCTSD overlap về ngữ nghĩa nhưng ranh giới không được định nghĩa thống nhất. Dẫn đến một lượng nhỏ label noise trong merged dataset.</p>
                    </div>
                  </div>
                  <div className="flex gap-3 border-b pb-4">
                    <div className="h-8 w-8 rounded-full bg-yellow-100 text-yellow-900 flex items-center justify-center text-xs font-semibold">L2</div>
                    <div>
                      <p className="font-medium">74.9% toxic samples đến từ ViHSD</p>
                      <p className="text-muted-foreground">Model học toxic chủ yếu từ Facebook data (ViHSD OFFENSIVE: 2,260/3,019 toxic samples). Điều này có thể khiến model nhạy hơn với style toxic của Facebook so với YouTube.</p>
                    </div>
                  </div>
                  <div className="flex gap-3 border-b pb-4">
                    <div className="h-8 w-8 rounded-full bg-red-100 text-red-700 flex items-center justify-center text-xs font-semibold">L3</div>
                    <div>
                      <p className="font-medium">Toàn bộ dataset là social media text — bias với formal domain</p>
                      <p className="text-muted-foreground">Cả ViCTSD (YouTube) và ViHSD (Facebook) đều là informal text. Model không có negative examples từ formal domain (báo chí, văn bản hành chính) → false positive cao khi inference trên news website. Đây là vấn đề cốt lõi dẫn đến việc phải implement domain-aware thresholding.</p>
                    </div>
                  </div>
                  <div className="flex gap-3 border-b pb-4">
                    <div className="h-8 w-8 rounded-full bg-red-100 text-red-700 flex items-center justify-center text-xs font-semibold">L4</div>
                    <div>
                      <p className="font-medium">Imbalance vẫn còn (2.1:1)</p>
                      <p className="text-muted-foreground">Sau khi merge, ratio Clean:Toxic là 2.1:1 — cải thiện đáng kể từ 8.2:1 nhưng vẫn chưa balanced hoàn toàn. Model vẫn có xu hướng predict Non-toxic nhiều hơn, đặc biệt với các trường hợp borderline.</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <div className="h-8 w-8 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center text-xs font-semibold">L5</div>
                    <div>
                      <p className="font-medium">Temporal &amp; domain coverage hạn chế</p>
                      <p className="text-muted-foreground">Cả hai dataset được thu thập trước 2021 — slang, cách viết tắt, và hình thức toxic mới (teencodes mới, emoji-based toxicity) có thể chưa được cover. New_collected (9 samples) quá nhỏ để bù đắp.</p>
                    </div>
                  </div>
                </div>
                <div className="mt-4 rounded-lg border-l-4 border-l-blue-600 bg-blue-50 p-4 text-sm text-blue-900">
                  <strong>Mẹo bảo vệ:</strong> Chủ động nêu L3 và L4 trước — đây là những limitation bạn đã nhận ra và đã có giải pháp (domain-aware thresholding). Hội đồng sẽ đánh giá cao việc bạn không né tránh mà đối mặt trực tiếp với limitation của mình.
                </div>
              </div>
            </TabsContent>

            <TabsContent value="definition" className="mt-4 space-y-6">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Cách định nghĩa nhãn toxic trong nghiên cứu</p>
                <div className="mt-3 text-sm text-muted-foreground space-y-3">
                  <p>
                    Trong nghiên cứu này, “toxic” được định nghĩa theo nhãn <strong>Toxicity</strong> gốc của ViCTSD. Label nhị phân được giữ nguyên:
                    <strong> 0 = non-toxic/clean</strong>, <strong>1 = toxic</strong>.
                  </p>
                  <p>
                    Không gộp thêm các mức độ khác hay tiêu chí <strong>Constructiveness</strong> để đảm bảo tính nhất quán với annotation guideline
                    gốc của dataset và tránh thay đổi semantics nhãn trong quá trình tiền xử lý.
                  </p>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </Card>

        <Card className="bg-white p-6 mb-8 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
            <div>
              <Label className="text-sm text-gray-600">Source</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={sourceFilter}
                onChange={(event) => setSourceFilter(event.target.value)}
              >
                {availableSources.map((source) => (
                  <option key={source} value={source}>
                    {sourceLabel(source)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <Label className="text-sm text-gray-600">Label</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={labelFilter}
                onChange={(event) => setLabelFilter(event.target.value)}
              >
                <option value="all">all</option>
                <option value="clean">clean</option>
                <option value="toxic">toxic</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-gray-600">Split</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={splitFilter}
                onChange={(event) => setSplitFilter(event.target.value)}
              >
                <option value="all">all</option>
                <option value="train">train</option>
                <option value="validation">validation</option>
                <option value="test">test</option>
                <option value="feedback">feedback</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-gray-600">Page size</Label>
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
              {loading ? "Đang tải..." : "Refresh"}
            </Button>
            <Button variant="outline" onClick={handleExport}>
              Export JSONL
            </Button>
            {onOpenSyntheticPage && (
              <Button className="bg-violet-600 hover:bg-violet-700 text-white font-semibold" onClick={onOpenSyntheticPage}>
                Synthetic Generation
              </Button>
            )}
            <Button
              variant="destructive"
              onClick={handleDeleteFeedback}
              disabled={!selectedFeedback.length || deleteLoading}
            >
              {deleteLoading ? "Đang xoá..." : `Xoá feedback (${selectedFeedback.length})`}
            </Button>
            {exportStatus && <span className="text-sm text-gray-600">{exportStatus}</span>}
            {deleteStatus && <span className="text-sm text-gray-600">{deleteStatus}</span>}
          </div>

          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
        </Card>

        <Card className="bg-white p-6 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <h2 className="text-2xl" style={{ color: "var(--viet-primary)" }}>
                Dataset Overview
              </h2>
              <p className="text-sm text-gray-600">
                Thống kê theo bộ lọc hiện tại.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge className="bg-blue-100 text-blue-700">Merged datasets: {aggregatedStats.sources.length}</Badge>
              <Badge className={imbalanceStatus?.isImbalanced ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"}>
                {imbalanceStatus?.isImbalanced ? "Imbalanced" : "Balanced"}
              </Badge>
            </div>
          </div>

          {imbalanceStatus?.isImbalanced && (
            <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              Lớp {imbalanceStatus.dominant} chiếm {formatPercent(
                imbalanceStatus.dominant === "clean" ? aggregatedStats.clean : aggregatedStats.toxic,
                aggregatedStats.total,
              )} tổng mẫu. Cân nhắc tăng dữ liệu cho lớp còn lại.
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card className="bg-white p-4 border">
              <p className="text-sm text-gray-500">Tổng mẫu</p>
              <p className="text-2xl" style={{ color: "var(--viet-primary)" }}>{aggregatedStats.total}</p>
            </Card>
            <Card className="bg-white p-4 border">
              <p className="text-sm text-gray-500">Clean</p>
              <p className="text-xl text-gray-800">
                {aggregatedStats.clean} ({formatPercent(aggregatedStats.clean, aggregatedStats.total)})
              </p>
            </Card>
            <Card className="bg-white p-4 border">
              <p className="text-sm text-gray-500">Toxic</p>
              <p className="text-xl text-gray-800">
                {aggregatedStats.toxic} ({formatPercent(aggregatedStats.toxic, aggregatedStats.total)})
              </p>
            </Card>
            <Card className="bg-white p-4 border">
              <p className="text-sm text-gray-500">Trang</p>
              <p className="text-sm text-gray-700 mt-2">
                {page} / {totalPages}
              </p>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="border rounded-lg overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Source</TableHead>
                    <TableHead className="text-right">Clean</TableHead>
                    <TableHead className="text-right">Toxic</TableHead>
                    <TableHead className="text-right">Total</TableHead>
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
                      <TableCell colSpan={4} className="text-center text-sm text-gray-500">
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
                    <TableHead>Cross-tab</TableHead>
                    <TableHead className="text-right">Clean</TableHead>
                    <TableHead className="text-right">Toxic</TableHead>
                    <TableHead className="text-right">Total</TableHead>
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
                    <TableCell className="font-medium">Total</TableCell>
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
                      <TableCell colSpan={4} className="text-center text-sm text-gray-500">
                        --
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
        </Card>

        <Card className="bg-white p-6 shadow-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Chọn</TableHead>
                <TableHead>Text</TableHead>
                <TableHead>Label</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Split</TableHead>
                <TableHead>Augmented</TableHead>
                <TableHead>Created</TableHead>
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
                        "--"
                      )}
                    </TableCell>
                    <TableCell className="max-w-[360px] truncate" title={row.text}>
                      {row.text}
                    </TableCell>
                    <TableCell>{labelText(row.label)}</TableCell>
                    <TableCell>{row.meta?.source ?? "--"}</TableCell>
                    <TableCell>{row.meta?.split ?? "--"}</TableCell>
                    <TableCell>{row.meta?.is_augmented ? "yes" : "no"}</TableCell>
                    <TableCell>{row.meta?.created_at ?? "--"}</TableCell>
                  </TableRow>
                );
              })}
              {!rows.length && !loading && (
                <TableRow>
                  <TableCell colSpan={7} className="text-center text-sm text-gray-500">
                    Không có dữ liệu
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
