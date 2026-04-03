import { useEffect, useMemo, useState, type MouseEvent } from "react";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
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

interface SyntheticExportResponse {
  path: string;
  count: number;
}

interface SyntheticGenerationPageProps {
  onBack: () => void;
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

export function SyntheticGenerationPage({ onBack }: SyntheticGenerationPageProps) {
  const { t } = useI18n();
  const [domain, setDomain] = useState<"education" | "news" | "politic">("education");
  const [style, setStyle] = useState<"formal" | "informal">("formal");
  const [label, setLabel] = useState<0 | 1>(1);
  const [count, setCount] = useState(10);

  const [rows, setRows] = useState<SyntheticRow[]>([]);
  const [checkedMap, setCheckedMap] = useState<Record<number, boolean>>({});
  const [editedTextMap, setEditedTextMap] = useState<Record<number, string>>({});
  const [editedLabelMap, setEditedLabelMap] = useState<Record<number, 0 | 1>>({});
  const [stats, setStats] = useState<SyntheticStats | null>(null);
  const [batchIdFilter, setBatchIdFilter] = useState<string>("");
  const [acceptedFilter, setAcceptedFilter] = useState<"all" | "accepted" | "rejected">("all");
  const [viewMode, setViewMode] = useState<"queue" | "reviewed">("queue");

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [totalPages, setTotalPages] = useState(1);

  const [loading, setLoading] = useState(false);
  const [generateLoading, setGenerateLoading] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const acceptedCountCurrentPage = useMemo(() => {
    return rows.filter((row) => checkedMap[row.id] ?? row.is_accepted).length;
  }, [rows, checkedMap]);

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
      if (viewMode === "queue") {
        params.set("reviewed", "false");
      } else {
        params.set("reviewed", "true");
        if (acceptedFilter !== "all") params.set("accepted", acceptedFilter === "accepted" ? "true" : "false");
      }

      const response = await fetch(buildApiUrl(`/api/dataset/synthetic/preview?${params.toString()}`));
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
      (data.items || []).forEach((item) => {
        nextChecked[item.id] = item.is_accepted;
        nextEditedText[item.id] = item.text;
        nextEditedLabel[item.id] = item.label === 1 ? 1 : 0;
      });
      setCheckedMap(nextChecked);
      setEditedTextMap(nextEditedText);
      setEditedLabelMap(nextEditedLabel);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.loadingDataset");
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchPreview(page, pageSize);
  }, [page, pageSize, batchIdFilter, acceptedFilter, viewMode]);

  const handleGenerate = async () => {
    setGenerateLoading(true);
    setStatus(null);
    setError(null);
    try {
      const response = await fetch(buildApiUrl("/api/dataset/synthetic/generate"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ domain, style, label, count }),
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
      setViewMode("queue");
      setAcceptedFilter("all");
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

  const handleSaveReview = async () => {
    const updates = rows
      .map((row) => {
        const nextAccepted = checkedMap[row.id] ?? row.is_accepted;
        const nextText = (editedTextMap[row.id] ?? row.text).trim();
        const nextLabel = editedLabelMap[row.id] ?? (row.label === 1 ? 1 : 0);
        const changed = nextAccepted !== row.is_accepted || nextText !== row.text || nextLabel !== row.label;
        return {
          id: row.id,
          is_accepted: nextAccepted,
          text: nextText,
          label: nextLabel,
          changed,
        };
      })
      .filter((item) => item.changed)
      .map(({ id, is_accepted, text, label }) => ({ id, is_accepted, text, label }));

    if (!updates.length) {
      setStatus(t("synthetic.noChanges"));
      return;
    }

    setSaveLoading(true);
    setStatus(null);
    setError(null);
    try {
      const response = await fetch(buildApiUrl("/api/dataset/synthetic/review"), {
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
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.saveFailed");
      setError(message);
    } finally {
      setSaveLoading(false);
    }
  };

  const handleExport = async () => {
    setStatus(null);
    setError(null);
    try {
      const response = await fetch(buildApiUrl("/api/dataset/synthetic/export"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          batch_id: batchIdFilter.trim() || undefined,
          accepted_only: true,
        }),
      });
      const data = (await response.json()) as SyntheticExportResponse;
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }
      setStatus(t("synthetic.exportDone", { count: data.count, path: data.path }));
    } catch (err) {
      const message = err instanceof Error ? err.message : t("synthetic.exportFailed");
      setError(message);
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
      const response = await fetch(buildApiUrl("/api/dataset/synthetic/delete"), {
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

  return (
    <div className="min-h-screen bg-background py-12 px-4 sm:px-6 lg:px-8">
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
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 items-end">
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
            <Button variant="outline" onClick={handleSaveReview} disabled={saveLoading || !rows.length}>
              {saveLoading ? t("synthetic.saving") : t("synthetic.saveReview")}
            </Button>
            <Button variant="outline" onClick={handleExport} disabled={viewMode !== "reviewed"}>
              {t("synthetic.exportAccepted")}
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
          <p className="mt-2 text-xs text-muted-foreground">
            {viewMode === "queue" ? t("synthetic.queueHint") : t("synthetic.reviewedHint")}
          </p>
        </Card>

        <Card className="bg-card p-6 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 items-end">
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
              <Label className="text-sm text-muted-foreground">{t("synthetic.viewMode")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={viewMode}
                onChange={(event) => {
                  const nextMode = event.target.value as "queue" | "reviewed";
                  setViewMode(nextMode);
                  if (nextMode === "queue") {
                    setAcceptedFilter("all");
                  }
                  setPage(1);
                }}
              >
                <option value="queue">{t("synthetic.queueUnreviewed")}</option>
                <option value="reviewed">{t("synthetic.dbReviewed")}</option>
              </select>
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
            <div>
              <Label className="text-sm text-muted-foreground">{t("synthetic.acceptedFilter")}</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={acceptedFilter}
                onChange={(event) => {
                  setAcceptedFilter(event.target.value as "all" | "accepted" | "rejected");
                  setPage(1);
                }}
                disabled={viewMode !== "reviewed"}
              >
                <option value="all">{t("synthetic.filterAll")}</option>
                <option value="accepted">{t("synthetic.filterAccepted")}</option>
                <option value="rejected">{t("synthetic.filterRejected")}</option>
              </select>
            </div>
            <div className="text-sm text-muted-foreground">
              {t("synthetic.acceptedPage", { count: acceptedCountCurrentPage, total: rows.length })}
            </div>
          </div>
        </Card>

        <Card className="bg-card p-6 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4 text-sm">
            <div className="border rounded-lg p-3">{t("synthetic.totalGenerated")}: <strong>{stats?.total_generated ?? 0}</strong></div>
            <div className="border rounded-lg p-3">{t("synthetic.accepted")}: <strong>{stats?.accepted ?? 0}</strong></div>
            <div className="border rounded-lg p-3">{t("synthetic.rejected")}: <strong>{stats?.rejected ?? 0}</strong></div>
            <div className="border rounded-lg p-3">{t("synthetic.acceptanceRate")}: <strong>{((stats?.acceptance_rate ?? 0) * 100).toFixed(1)}%</strong></div>
          </div>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("synthetic.tableAccept")}</TableHead>
                <TableHead>{t("synthetic.tableTextEditable")}</TableHead>
                <TableHead>{t("synthetic.tableActions")}</TableHead>
                <TableHead>{t("synthetic.tableLabel")}</TableHead>
                <TableHead>{t("synthetic.tableDomain")}</TableHead>
                <TableHead>{t("synthetic.tableStyle")}</TableHead>
                <TableHead>{t("synthetic.tableBatch")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const nextLabel = editedLabelMap[row.id] ?? (row.label === 1 ? 1 : 0);
                const selected = checkedMap[row.id] ?? row.is_accepted;
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
                  <TableCell colSpan={7} className="text-center text-sm text-muted-foreground">
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
      </div>
    </div>
  );
}
