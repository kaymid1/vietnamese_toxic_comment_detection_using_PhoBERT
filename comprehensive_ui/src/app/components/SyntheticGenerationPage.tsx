import { useEffect, useMemo, useState, type MouseEvent } from "react";
import { Button } from "@/app/components/ui/button";
import { Badge } from "@/app/components/ui/badge";
import { Card } from "@/app/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/app/components/ui/dialog";
import { Label } from "@/app/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/components/ui/table";
import { useI18n } from "@/app/i18n/context";

interface SyntheticRow {
  id: number;
  batch_id: string;
  text: string;
  label: number;
  toxicity?: number;
  constructiveness?: number | null;
  domain: "education" | "news" | "politic";
  style: "formal" | "informal";
  is_accepted: boolean;
  meta?: Record<string, unknown>;
  created_at?: string;
}

interface SyntheticStats {
  total_generated: number;
  accepted: number;
  rejected: number;
  acceptance_rate: number;
  by_domain: Record<string, { total: number; accepted: number; rejected: number }>;
  by_style: Record<string, { total: number; accepted: number; rejected: number }>;
  by_label: Record<string, { total: number; accepted: number; rejected: number }>;
  by_constructiveness?: Record<string, { total: number; accepted: number; rejected: number }>;
}

interface SyntheticPreviewResponse {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  items: SyntheticRow[];
  stats?: SyntheticStats;
}

interface SyntheticGenerateResponse {
  batch_id: string;
  generated_count: number;
  requested_count: number;
  validation_summary?: {
    length_bucket_target?: Record<string, number>;
    length_bucket_generated?: Record<string, number>;
  };
}

interface SyntheticGenerationPageProps {
  onBack: () => void;
  adminToken: string;
  onAdminUnauthorized: () => void;
}

interface SyntheticTransferSummary {
  batch_id?: string | null;
  eligible: number;
  toxic: number;
  clean: number;
  constructive: number;
  non_constructive: number;
  constructiveness_masked: number;
  already_transferred: number;
  ids: number[];
}

interface SyntheticGeminiSuggestion {
  id: number;
  toxicity_label: 0 | 1;
  constructiveness_label: 0 | 1 | null;
  confidence: "low" | "medium" | "high";
  reason: string;
  action: "apply" | "review_more";
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");
const API_FALLBACK_BASES = Array.from(
  new Set(
    ["", API_BASE, "http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:8001", "http://localhost:8001"]
      .map((value) => value.trim().replace(/\/+$/, ""))
      .filter(Boolean),
  ),
);
const API_FALLBACK_FAILURE_COOLDOWN_MS = 30000;
let lastSuccessfulApiBase: string | null = null;
const apiBaseFailureUntil = new Map<string, number>();

const buildApiUrlFromBase = (base: string, path: string) => {
  if (/^https?:\/\//i.test(path)) return path;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (!base) return normalizedPath;
  return `${base}${normalizedPath}`;
};

const isNetworkFetchError = (error: unknown) =>
  error instanceof TypeError && /failed to fetch|networkerror|load failed/i.test(error.message.toLowerCase());

const fetchApiWithFallback = async (path: string, init?: RequestInit): Promise<Response> => {
  const now = Date.now();
  const candidates = [lastSuccessfulApiBase || "", ...API_FALLBACK_BASES, API_BASE]
    .map((item) => item.trim())
    .filter((value, index, values) => values.indexOf(value) === index);

  let lastError: unknown = null;
  for (const candidate of candidates) {
    const blockedUntil = apiBaseFailureUntil.get(candidate) || 0;
    if (candidate !== (lastSuccessfulApiBase || "") && blockedUntil > now) continue;

    try {
      const response = await fetch(buildApiUrlFromBase(candidate, path), init);
      if (!candidate && response.status === 404) {
        lastError = new Error(`Relative API path returned 404 for ${path}`);
        continue;
      }
      lastSuccessfulApiBase = candidate;
      apiBaseFailureUntil.delete(candidate);
      return response;
    } catch (error) {
      if (!isNetworkFetchError(error)) throw error;
      lastError = error;
      apiBaseFailureUntil.set(candidate, Date.now() + API_FALLBACK_FAILURE_COOLDOWN_MS);
      if (lastSuccessfulApiBase === candidate) lastSuccessfulApiBase = null;
    }
  }

  throw new Error(`Cannot reach backend API for ${path}. ${(lastError as Error | null)?.message || ""}`.trim());
};

export function SyntheticGenerationPage({ onBack, adminToken, onAdminUnauthorized }: SyntheticGenerationPageProps) {
  const { t } = useI18n();
  const [domain, setDomain] = useState<"education" | "news" | "politic">("education");
  const [style, setStyle] = useState<"formal" | "informal">("formal");
  const [label, setLabel] = useState<0 | 1>(1);
  const [constructiveness, setConstructiveness] = useState<0 | 1 | null>(null);
  const [count, setCount] = useState(10);

  const [rows, setRows] = useState<SyntheticRow[]>([]);
  const [checkedMap, setCheckedMap] = useState<Record<number, boolean>>({});
  const [editedTextMap, setEditedTextMap] = useState<Record<number, string>>({});
  const [editedLabelMap, setEditedLabelMap] = useState<Record<number, 0 | 1>>({});
  const [editedConstructivenessMap, setEditedConstructivenessMap] = useState<Record<number, 0 | 1 | null>>({});
  const [stats, setStats] = useState<SyntheticStats | null>(null);
  const [batchIdFilter, setBatchIdFilter] = useState<string>("");

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [totalPages, setTotalPages] = useState(1);

  const [loading, setLoading] = useState(false);
  const [generateLoading, setGenerateLoading] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [transferLoading, setTransferLoading] = useState(false);
  const [transferDialogOpen, setTransferDialogOpen] = useState(false);
  const [transferSummary, setTransferSummary] = useState<SyntheticTransferSummary | null>(null);
  const [geminiSuggestions, setGeminiSuggestions] = useState<Record<number, SyntheticGeminiSuggestion>>({});
  const [geminiReviewing, setGeminiReviewing] = useState(false);
  const [geminiApplying, setGeminiApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const authorizedFetch = async (path: string, init?: RequestInit): Promise<Response> => {
    const headers = new Headers(init?.headers);
    headers.set("Authorization", `Bearer ${adminToken}`);
    const response = await fetchApiWithFallback(path, { ...init, headers });
    if (response.status === 401) {
      onAdminUnauthorized();
      throw new Error(t("synthetic.adminSessionExpired"));
    }
    return response;
  };

  const acceptedCountCurrentPage = useMemo(() => {
    return rows.filter((row) => checkedMap[row.id] ?? row.is_accepted).length;
  }, [rows, checkedMap]);

  const visibleGeminiSuggestions = useMemo(
    () => rows.map((row) => geminiSuggestions[row.id]).filter(Boolean) as SyntheticGeminiSuggestion[],
    [geminiSuggestions, rows],
  );

  const constructivenessStats = useMemo(() => {
    const byConstructiveness = stats?.by_constructiveness || {};
    const included = (byConstructiveness["0"]?.total || 0) + (byConstructiveness["1"]?.total || 0);
    const masked = byConstructiveness.masked?.total || 0;
    return { included, masked };
  }, [stats]);

  const fetchPreview = async (targetPage: number, targetPageSize: number) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        page: String(targetPage),
        page_size: String(targetPageSize),
        include_stats: "true",
      });
      if (batchIdFilter.trim()) params.set("batch_id", batchIdFilter.trim());
      params.set("reviewed", "false");

      const response = await authorizedFetch(`/api/dataset/synthetic/preview?${params.toString()}`);
      const data = (await response.json()) as SyntheticPreviewResponse;
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }

      setRows(data.items || []);
      setStats(data.stats || null);
      setTotalPages(data.total_pages || 1);

      const nextChecked: Record<number, boolean> = {};
      const nextEditedText: Record<number, string> = {};
      const nextEditedLabel: Record<number, 0 | 1> = {};
      const nextEditedConstructiveness: Record<number, 0 | 1 | null> = {};
      (data.items || []).forEach((item) => {
        nextChecked[item.id] = item.is_accepted;
        nextEditedText[item.id] = item.text;
        nextEditedLabel[item.id] = item.label === 1 ? 1 : 0;
        nextEditedConstructiveness[item.id] =
          item.constructiveness === 1 ? 1 : item.constructiveness === 0 ? 0 : null;
      });
      setCheckedMap(nextChecked);
      setEditedTextMap(nextEditedText);
      setEditedLabelMap(nextEditedLabel);
      setEditedConstructivenessMap(nextEditedConstructiveness);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.loadingDataset");
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchPreview(page, pageSize);
  }, [page, pageSize, batchIdFilter]);

  const handleGenerate = async () => {
    setGenerateLoading(true);
    setStatus(null);
    setError(null);
    try {
      const response = await authorizedFetch("/api/dataset/synthetic/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ domain, style, label, constructiveness, count }),
      });
      const data = (await response.json()) as SyntheticGenerateResponse;
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }

      const generated = data.validation_summary?.length_bucket_generated || {};
      const target = data.validation_summary?.length_bucket_target || {};
      const bucketText = ["very_short", "short_medium", "medium_long", "long"]
        .map((key) => `${key}:${generated[key] ?? 0}/${target[key] ?? 0}`)
        .join(" | ");

      setBatchIdFilter(data.batch_id);
      setPage(1);
      setStatus(t("synthetic.generatedStatus", { generated: data.generated_count, requested: data.requested_count, batch: data.batch_id, bucket: bucketText }));
      await fetchPreview(1, pageSize);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.generateFailed");
      setError(message);
    } finally {
      setGenerateLoading(false);
    }
  };

  const toggleChecked = (id: number) => {
    setCheckedMap((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const hasAnySelected = rows.some((row) => checkedMap[row.id] ?? row.is_accepted);

  const handleCopyRowText = async (rowId: number) => {
    const text = editedTextMap[rowId] ?? rows.find((row) => row.id === rowId)?.text ?? "";
    try {
      if (!navigator?.clipboard?.writeText) {
        throw new Error(t("synthetic.clipboardUnavailable"));
      }
      await navigator.clipboard.writeText(text);
      setStatus(t("synthetic.copiedRow", { id: rowId }));
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.copyFailed");
      setError(message);
    }
  };

  const handlePasteRowText = async (rowId: number) => {
    try {
      if (!navigator?.clipboard?.readText) {
        throw new Error(t("synthetic.clipboardUnavailable"));
      }
      const pasted = await navigator.clipboard.readText();
      setEditedTextMap((prev) => ({
        ...prev,
        [rowId]: pasted,
      }));
      setStatus(t("synthetic.pastedRow", { id: rowId }));
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.pasteFailed");
      setError(message);
    }
  };

  const handleAcceptAll = () => {
    const next = { ...checkedMap };
    rows.forEach((row) => {
      next[row.id] = true;
    });
    setCheckedMap(next);
  };

  const handleUnselectAll = () => {
    const next = { ...checkedMap };
    rows.forEach((row) => {
      next[row.id] = false;
    });
    setCheckedMap(next);
  };

  const handleRowToggle = (event: MouseEvent<HTMLTableRowElement>, rowId: number) => {
    const target = event.target as HTMLElement;
    if (target.closest("button, input, textarea, select, option, a")) {
      return;
    }
    toggleChecked(rowId);
  };

  const handleGeminiReview = async () => {
    const ids = rows
      .filter((row) => checkedMap[row.id] ?? row.is_accepted)
      .map((row) => row.id);
    if (!ids.length) {
      setStatus(t("synthetic.selectRowsForGemini"));
      return;
    }

    setGeminiReviewing(true);
    setStatus(null);
    setError(null);
    try {
      const response = await authorizedFetch("/api/dataset/synthetic/gemini-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids }),
      });
      const data = (await response.json()) as { suggestions?: SyntheticGeminiSuggestion[]; reviewed?: number };
      if (!response.ok) throw new Error(JSON.stringify(data));
      const next = { ...geminiSuggestions };
      (data.suggestions || []).forEach((suggestion) => {
        next[suggestion.id] = suggestion;
      });
      setGeminiSuggestions(next);
      setStatus(t("synthetic.geminiReviewed", { count: data.reviewed || 0 }));
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.geminiReviewFailed");
      setError(message);
    } finally {
      setGeminiReviewing(false);
    }
  };

  const handleApplyGeminiSuggestions = async (suggestions: SyntheticGeminiSuggestion[]) => {
    const updates = suggestions.flatMap((suggestion) => {
      const row = rows.find((item) => item.id === suggestion.id);
      if (!row) return [];
      return [{
        id: row.id,
        is_accepted: checkedMap[row.id] ?? row.is_accepted,
        text: (editedTextMap[row.id] ?? row.text).trim(),
        label: suggestion.toxicity_label,
        constructiveness: suggestion.constructiveness_label,
        review_method: "gemini_assisted" as const,
        label_confidence: suggestion.confidence,
      }];
    });
    if (!updates.length) return;

    setGeminiApplying(true);
    setStatus(null);
    setError(null);
    try {
      const response = await authorizedFetch("/api/dataset/synthetic/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates }),
      });
      const data = (await response.json()) as { updated: number };
      if (!response.ok) throw new Error(JSON.stringify(data));
      const appliedIds = new Set(suggestions.map((item) => item.id));
      setGeminiSuggestions((current) =>
        Object.fromEntries(Object.entries(current).filter(([id]) => !appliedIds.has(Number(id)))),
      );
      setStatus(t("synthetic.geminiApplied", { count: data.updated }));
      await fetchPreview(page, pageSize);
      await handleOpenTransferDialog();
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.geminiApplyFailed");
      setError(message);
    } finally {
      setGeminiApplying(false);
    }
  };

  const handleSaveReview = async () => {
    const updates = rows
      .map((row) => {
        const nextAccepted = checkedMap[row.id] ?? row.is_accepted;
        const nextText = (editedTextMap[row.id] ?? row.text).trim();
        const nextLabel = editedLabelMap[row.id] ?? (row.label === 1 ? 1 : 0);
        const nextConstructiveness =
          editedConstructivenessMap[row.id] === 1 ? 1 : editedConstructivenessMap[row.id] === 0 ? 0 : null;
        return {
          id: row.id,
          is_accepted: nextAccepted,
          text: nextText,
          label: nextLabel,
          constructiveness: nextConstructiveness,
        };
      });

    if (!updates.length) {
      setStatus(t("synthetic.noChanges"));
      return;
    }

    setSaveLoading(true);
    setStatus(null);
    setError(null);
    try {
      const response = await authorizedFetch("/api/dataset/synthetic/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates }),
      });
      const data = (await response.json()) as { updated: number };
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }
      setStatus(t("synthetic.savedReview", { count: data.updated }));
      await fetchPreview(page, pageSize);
      await handleOpenTransferDialog();
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.saveFailed");
      setError(message);
    } finally {
      setSaveLoading(false);
    }
  };

  const handleDeleteSelected = async () => {
    const selectedIds = rows
      .filter((row) => checkedMap[row.id] ?? row.is_accepted)
      .map((row) => row.id);

    if (!selectedIds.length) {
      setStatus(t("synthetic.noRowsToDelete"));
      return;
    }

    const confirmed = window.confirm(t("synthetic.confirmDelete", { count: selectedIds.length }));
    if (!confirmed) {
      return;
    }

    setDeleteLoading(true);
    setStatus(null);
    setError(null);
    try {
      const response = await authorizedFetch("/api/dataset/synthetic/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: selectedIds }),
      });
      const data = (await response.json()) as { deleted: number };
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }
      setStatus(t("synthetic.deletedRows", { count: data.deleted }));
      await fetchPreview(page, pageSize);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.deleteFailed");
      setError(message);
    } finally {
      setDeleteLoading(false);
    }
  };

  const handleOpenTransferDialog = async () => {
    setTransferLoading(true);
    setStatus(null);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (batchIdFilter.trim()) params.set("batch_id", batchIdFilter.trim());
      const query = params.toString();
      const response = await authorizedFetch(
        `/api/dataset/synthetic/training-preview-summary${query ? `?${query}` : ""}`,
      );
      const data = (await response.json()) as SyntheticTransferSummary;
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }
      setTransferSummary(data);
      setTransferDialogOpen(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.transferSummaryFailed");
      setError(message);
    } finally {
      setTransferLoading(false);
    }
  };

  const handleConfirmTransfer = async () => {
    if (!transferSummary?.ids.length) return;
    setTransferLoading(true);
    setError(null);
    try {
      const response = await authorizedFetch("/api/dataset/synthetic/transfer-to-training-preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: transferSummary.ids }),
      });
      const data = (await response.json()) as { transferred: number; toxic: number; clean: number; skipped: number };
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }
      setTransferDialogOpen(false);
      setTransferSummary(null);
      setStatus(
        t("synthetic.transferDone", {
          count: data.transferred,
          toxic: data.toxic,
          clean: data.clean,
          skipped: data.skipped,
        }),
      );
      await fetchPreview(page, pageSize);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.transferFailed");
      setError(message);
    } finally {
      setTransferLoading(false);
    }
  };

  return (
    <div className="dashboard-page">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-3xl text-primary">
              {t("synthetic.title")}
            </h1>
            <p className="text-sm text-muted-foreground">{t("synthetic.subtitle")}</p>
          </div>
          <Button variant="outline" onClick={onBack}>
            {t("synthetic.back")}
          </Button>
        </div>

        <Card className="bg-card p-6 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-6 gap-4 items-end">
            <div>
              <Label className="text-sm text-muted-foreground">{t("synthetic.domain")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={domain}
                onChange={(event) => setDomain(event.target.value as "education" | "news" | "politic")}
              >
                <option value="education">{t("synthetic.domainEducation")}</option>
                <option value="news">{t("synthetic.domainNews")}</option>
                <option value="politic">{t("synthetic.domainPolitic")}</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">{t("synthetic.style")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={style}
                onChange={(event) => setStyle(event.target.value as "formal" | "informal")}
              >
                <option value="formal">{t("synthetic.styleFormal")}</option>
                <option value="informal">{t("synthetic.styleInformal")}</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">{t("synthetic.label")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={label}
                onChange={(event) => setLabel(Number(event.target.value) as 0 | 1)}
              >
                <option value={1}>{t("synthetic.labelToxic")}</option>
                <option value={0}>{t("synthetic.labelClean")}</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">{t("synthetic.constructiveness")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={constructiveness === null ? "mask" : String(constructiveness)}
                onChange={(event) => {
                  const value = event.target.value;
                  setConstructiveness(value === "mask" ? null : (Number(value) as 0 | 1));
                }}
              >
                <option value="mask">{t("synthetic.constructivenessMask")}</option>
                <option value={1}>{t("synthetic.constructive")}</option>
                <option value={0}>{t("synthetic.nonConstructive")}</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">{t("synthetic.count")}</Label>
              <input
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                type="number"
                min={1}
                max={200}
                value={count}
                onChange={(event) => setCount(Math.max(1, Math.min(200, Number(event.target.value) || 1)))}
              />
            </div>
            <Button onClick={handleGenerate} disabled={generateLoading}>
              {generateLoading ? t("synthetic.generating") : t("synthetic.generate")}
            </Button>
          </div>

          <div className="mt-4 flex flex-wrap gap-3 items-center">
            <Button variant="outline" onClick={handleAcceptAll} disabled={!rows.length}>
              {t("synthetic.selectAllPage")}
            </Button>
            <Button variant="outline" onClick={handleUnselectAll} disabled={!rows.length}>
              {t("synthetic.unselectAllPage")}
            </Button>
            <Button variant="outline" onClick={handleGeminiReview} disabled={geminiReviewing || !hasAnySelected}>
              {geminiReviewing ? t("synthetic.geminiReviewing") : t("synthetic.geminiReview")}
            </Button>
            <Button
              variant="outline"
              onClick={() => void handleApplyGeminiSuggestions(visibleGeminiSuggestions)}
              disabled={geminiApplying || visibleGeminiSuggestions.length === 0}
            >
              {geminiApplying
                ? t("synthetic.geminiApplying")
                : t("synthetic.applyGeminiAll", { count: visibleGeminiSuggestions.length })}
            </Button>
            <Button variant="outline" onClick={handleSaveReview} disabled={saveLoading || !rows.length}>
              {saveLoading ? t("synthetic.saving") : t("synthetic.saveReview")}
            </Button>
            <Button variant="destructive" onClick={handleDeleteSelected} disabled={!hasAnySelected || deleteLoading}>
              {deleteLoading ? t("synthetic.deleting") : t("synthetic.deleteSelected")}
            </Button>
            <Button variant="outline" onClick={() => fetchPreview(1, pageSize)} disabled={loading}>
              {loading ? t("synthetic.processing") : t("synthetic.refresh")}
            </Button>
            {status && <span className="text-sm text-muted-foreground">{status}</span>}
          </div>
          {error && <p className="mt-3 text-sm text-destructive">{error}</p>}
          <p className="mt-2 text-xs text-muted-foreground">{t("synthetic.queueOnlyHint")}</p>
        </Card>

        <Card className="bg-card p-6 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div>
              <Label className="text-sm text-muted-foreground">{t("synthetic.batchId")}</Label>
              <input
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={batchIdFilter}
                onChange={(event) => {
                  setBatchIdFilter(event.target.value);
                  setPage(1);
                }}
                placeholder={t("synthetic.filterByBatch")}
              />
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">{t("synthetic.pageSize")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={pageSize}
                onChange={(event) => {
                  setPageSize(Number(event.target.value));
                  setPage(1);
                }}
              >
                {[10, 25, 50, 100].map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </div>
            <div className="text-sm text-muted-foreground">
              {t("synthetic.unreviewedPage", { selected: acceptedCountCurrentPage, total: rows.length })}
            </div>
          </div>
        </Card>

        <Card className="bg-card p-6 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3 mb-4 text-sm">
            <div className="border rounded-lg p-3">{t("synthetic.totalGenerated")}: <strong>{stats?.total_generated ?? 0}</strong></div>
            <div className="border rounded-lg p-3">{t("synthetic.accepted")}: <strong>{stats?.accepted ?? 0}</strong></div>
            <div className="border rounded-lg p-3">{t("synthetic.rejected")}: <strong>{stats?.rejected ?? 0}</strong></div>
            <div className="border rounded-lg p-3">{t("synthetic.acceptanceRate")}: <strong>{((stats?.acceptance_rate ?? 0) * 100).toFixed(1)}%</strong></div>
            <div className="border rounded-lg p-3">
              {t("synthetic.constructiveness")}: <strong>{constructivenessStats.included}</strong> / {constructivenessStats.masked} {t("synthetic.constructivenessMask")}
            </div>
          </div>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("synthetic.tableAccept")}</TableHead>
                <TableHead>{t("synthetic.tableTextEditable")}</TableHead>
                <TableHead>{t("synthetic.tableActions")}</TableHead>
                <TableHead>{t("synthetic.tableLabel")}</TableHead>
                <TableHead>{t("synthetic.tableConstructiveness")}</TableHead>
                <TableHead>{t("synthetic.tableDomain")}</TableHead>
                <TableHead>{t("synthetic.tableStyle")}</TableHead>
                <TableHead>{t("synthetic.tableBatch")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const nextLabel = editedLabelMap[row.id] ?? (row.label === 1 ? 1 : 0);
                const nextConstructiveness =
                  editedConstructivenessMap[row.id] === 1 ? 1 : editedConstructivenessMap[row.id] === 0 ? 0 : null;
                const selected = checkedMap[row.id] ?? row.is_accepted;
                const suggestion = geminiSuggestions[row.id];
                return (
                  <TableRow
                    key={row.id}
                    onClick={(event: MouseEvent<HTMLTableRowElement>) => handleRowToggle(event, row.id)}
                    className="cursor-pointer hover:bg-background-secondary"
                  >
                    <TableCell>
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleChecked(row.id)}
                      />
                    </TableCell>
                    <TableCell className="min-w-[360px]">
                      <textarea
                        className="w-full border rounded-md px-2 py-1 text-sm min-h-[72px]"
                        value={editedTextMap[row.id] ?? row.text}
                        onChange={(event) =>
                          setEditedTextMap((prev) => ({
                            ...prev,
                            [row.id]: event.target.value,
                          }))
                        }
                      />
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-col gap-2">
                        <Button variant="outline" size="sm" onClick={() => void handleCopyRowText(row.id)}>
                          {t("synthetic.copy")}
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => void handlePasteRowText(row.id)}>
                          {t("synthetic.paste")}
                        </Button>
                        {suggestion && (
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={geminiApplying}
                            onClick={() => void handleApplyGeminiSuggestions([suggestion])}
                          >
                            {t("synthetic.applyGemini")}
                          </Button>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <select
                        className={`border rounded-md px-2 py-1 text-sm font-medium ${nextLabel === 1 ? "bg-background-danger text-text-danger border-border-danger" : "bg-background-success text-text-success border-border-success"}`}
                        value={nextLabel}
                        onChange={(event) =>
                          setEditedLabelMap((prev) => ({
                            ...prev,
                            [row.id]: Number(event.target.value) as 0 | 1,
                          }))
                        }
                      >
                        <option value={1}>{t("synthetic.toxic")}</option>
                        <option value={0}>{t("synthetic.clean")}</option>
                      </select>
                      <div className="mt-2 flex flex-col gap-1">
                        <Badge variant={nextLabel === 1 ? "destructive" : "secondary"}>
                          {t("synthetic.currentLabel")}: {nextLabel === 1 ? "Toxic" : "Clean"}
                        </Badge>
                        {suggestion && (
                          <>
                            <Badge variant="outline">
                              Gemini: {suggestion.toxicity_label === 1 ? "Toxic" : "Clean"} · {suggestion.confidence}
                            </Badge>
                            {suggestion.reason && (
                              <span className="max-w-[220px] text-xs text-muted-foreground">{suggestion.reason}</span>
                            )}
                          </>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <select
                        className="border rounded-md px-2 py-1 text-sm font-medium"
                        value={nextConstructiveness === null ? "mask" : String(nextConstructiveness)}
                        onChange={(event) =>
                          setEditedConstructivenessMap((prev) => ({
                            ...prev,
                            [row.id]: event.target.value === "mask" ? null : (Number(event.target.value) as 0 | 1),
                          }))
                        }
                      >
                        <option value="mask">{t("synthetic.constructivenessMask")}</option>
                        <option value={1}>{t("synthetic.constructive")}</option>
                        <option value={0}>{t("synthetic.nonConstructive")}</option>
                      </select>
                      {suggestion && (
                        <Badge variant="outline" className="mt-2">
                          Gemini: {suggestion.constructiveness_label == null
                            ? "Masked"
                            : suggestion.constructiveness_label === 1
                              ? "Constructive"
                              : "Non-constructive"}
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>{row.domain}</TableCell>
                    <TableCell>{row.style}</TableCell>
                    <TableCell className="max-w-[180px] truncate" title={row.batch_id}>
                      {row.batch_id}
                    </TableCell>
                  </TableRow>
                );
              })}
              {!rows.length && !loading && (
                <TableRow>
                  <TableCell colSpan={8} className="text-center text-sm text-muted-foreground">
                    {t("synthetic.noData")}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>

          <div className="mt-4 flex items-center justify-between text-sm">
            <Button variant="outline" onClick={() => setPage((prev) => Math.max(1, prev - 1))} disabled={page <= 1}>
              {t("synthetic.previous")}
            </Button>
            <span>
              {t("synthetic.pageOf", { page, total: totalPages })}
            </span>
            <Button
              variant="outline"
              onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
              disabled={page >= totalPages}
            >
              {t("synthetic.next")}
            </Button>
          </div>
        </Card>

        <Dialog open={transferDialogOpen} onOpenChange={setTransferDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{t("synthetic.transferDialogTitle")}</DialogTitle>
              <DialogDescription>{t("synthetic.transferDialogDescription")}</DialogDescription>
            </DialogHeader>
            {transferSummary && (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="rounded-lg border p-3">
                    <p className="text-xs text-muted-foreground">{t("synthetic.transferTotal")}</p>
                    <p className="mt-1 text-2xl font-semibold">{transferSummary.eligible}</p>
                  </div>
                  <div className="rounded-lg border p-3">
                    <p className="text-xs text-muted-foreground">Toxic</p>
                    <p className="mt-1 text-2xl font-semibold text-destructive">{transferSummary.toxic}</p>
                  </div>
                  <div className="rounded-lg border p-3">
                    <p className="text-xs text-muted-foreground">Clean</p>
                    <p className="mt-1 text-2xl font-semibold text-text-success">{transferSummary.clean}</p>
                  </div>
                </div>
                <div className="rounded-lg bg-muted p-3 text-sm text-muted-foreground">
                  {t("synthetic.transferConstructiveness", {
                    constructive: transferSummary.constructive,
                    nonConstructive: transferSummary.non_constructive,
                    masked: transferSummary.constructiveness_masked,
                  })}
                  {transferSummary.already_transferred > 0 && (
                    <p className="mt-2">
                      {t("synthetic.transferAlreadySent", { count: transferSummary.already_transferred })}
                    </p>
                  )}
                </div>
              </div>
            )}
            <DialogFooter>
              <Button variant="outline" onClick={() => setTransferDialogOpen(false)} disabled={transferLoading}>
                {t("synthetic.cancel")}
              </Button>
              <Button
                onClick={handleConfirmTransfer}
                disabled={transferLoading || !transferSummary?.eligible}
              >
                {transferLoading ? t("synthetic.processing") : t("synthetic.adminConfirmTransfer")}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
