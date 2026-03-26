import { useEffect, useMemo, useState, type MouseEvent, type ChangeEvent } from "react";
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
      const message = err instanceof Error ? err.message : "Không thể tải synthetic dataset";
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
      setStatus(`Đã sinh ${data.generated_count}/${data.requested_count} mẫu. Batch: ${data.batch_id}. Length: ${bucketText}`);
      await fetchPreview(1, pageSize);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Sinh dữ liệu thất bại";
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
        throw new Error("Clipboard API không khả dụng");
      }
      await navigator.clipboard.writeText(text);
      setStatus(`Đã copy text của mẫu #${rowId}.`);
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Copy thất bại";
      setError(message);
    }
  };

  const handlePasteRowText = async (rowId: number) => {
    try {
      if (!navigator?.clipboard?.readText) {
        throw new Error("Clipboard API không khả dụng");
      }
      const pasted = await navigator.clipboard.readText();
      setEditedTextMap((prev) => ({
        ...prev,
        [rowId]: pasted,
      }));
      setStatus(`Đã paste text vào mẫu #${rowId}.`);
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Paste thất bại";
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
      setStatus("Không có thay đổi để lưu.");
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
      setStatus(`Đã cập nhật ${data.updated} mẫu review. Queue đã được làm mới.`);
      await fetchPreview(page, pageSize);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Lưu review thất bại";
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
      setStatus(`Đã export ${data.count} mẫu vào ${data.path}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Export thất bại";
      setError(message);
    }
  };

  const handleDeleteSelected = async () => {
    const selectedIds = rows
      .filter((row) => checkedMap[row.id] ?? row.is_accepted)
      .map((row) => row.id);

    if (!selectedIds.length) {
      setStatus("Chưa chọn row nào để xoá.");
      return;
    }

    const confirmed = window.confirm(`Xoá vĩnh viễn ${selectedIds.length} row đã chọn?`);
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
      setStatus(`Đã xoá ${data.deleted} row khỏi DB.`);
      await fetchPreview(page, pageSize);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Xoá thất bại";
      setError(message);
    } finally {
      setDeleteLoading(false);
    }
  };

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8" style={{ backgroundColor: "var(--viet-bg)" }}>
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-3xl" style={{ color: "var(--viet-primary)" }}>
              Synthetic Dataset Generation
            </h1>
            <p className="text-sm text-gray-600">Sinh dữ liệu mới và review trước khi export.</p>
          </div>
          <Button variant="outline" onClick={onBack}>
            Quay lại Dataset
          </Button>
        </div>

        <Card className="bg-white p-6 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 items-end">
            <div>
              <Label className="text-sm text-gray-600">Domain</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={domain}
                onChange={(event) => setDomain(event.target.value as "education" | "news" | "politic")}
              >
                <option value="education">education</option>
                <option value="news">news</option>
                <option value="politic">politic</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-gray-600">Style</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={style}
                onChange={(event) => setStyle(event.target.value as "formal" | "informal")}
              >
                <option value="formal">formal</option>
                <option value="informal">informal</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-gray-600">Label</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={label}
                onChange={(event) => setLabel(Number(event.target.value) as 0 | 1)}
              >
                <option value={1}>1 - toxic</option>
                <option value={0}>0 - clean</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-gray-600">Count</Label>
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
              {generateLoading ? "Đang sinh..." : "Generate"}
            </Button>
          </div>

          <div className="mt-4 flex flex-wrap gap-3 items-center">
            <Button variant="outline" onClick={handleAcceptAll} disabled={!rows.length}>
              Select all (page)
            </Button>
            <Button variant="outline" onClick={handleUnselectAll} disabled={!rows.length}>
              Unselect all (page)
            </Button>
            <Button variant="outline" onClick={handleSaveReview} disabled={saveLoading || !rows.length}>
              {saveLoading ? "Đang lưu..." : "Save review"}
            </Button>
            <Button variant="outline" onClick={handleExport} disabled={viewMode !== "reviewed"}>
              Export accepted
            </Button>
            <Button variant="destructive" onClick={handleDeleteSelected} disabled={!hasAnySelected || deleteLoading}>
              {deleteLoading ? "Đang xoá..." : "Delete selected"}
            </Button>
            <Button variant="outline" onClick={() => fetchPreview(1, pageSize)} disabled={loading}>
              {loading ? "Đang tải..." : "Refresh"}
            </Button>
            {status && <span className="text-sm text-gray-600">{status}</span>}
          </div>
          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
          <p className="mt-2 text-xs text-gray-500">
            {viewMode === "queue"
              ? "Queue chưa review: bấm Save review để ghi DB, sau đó các row này sẽ rời khỏi queue."
              : "DB đã review: dùng Accepted filter để chọn accepted/rejected và Export accepted để xuất JSONL."}
          </p>
        </Card>

        <Card className="bg-white p-6 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 items-end">
            <div>
              <Label className="text-sm text-gray-600">Batch ID</Label>
              <input
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={batchIdFilter}
                onChange={(event) => {
                  setBatchIdFilter(event.target.value);
                  setPage(1);
                }}
                placeholder="Lọc theo batch"
              />
            </div>
            <div>
              <Label className="text-sm text-gray-600">View mode</Label>
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
                <option value="queue">Queue chưa review</option>
                <option value="reviewed">DB đã review</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-gray-600">Page size</Label>
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
              <Label className="text-sm text-gray-600">Accepted filter</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={acceptedFilter}
                onChange={(event) => {
                  setAcceptedFilter(event.target.value as "all" | "accepted" | "rejected");
                  setPage(1);
                }}
                disabled={viewMode !== "reviewed"}
              >
                <option value="all">all</option>
                <option value="accepted">accepted</option>
                <option value="rejected">rejected</option>
              </select>
            </div>
            <div className="text-sm text-gray-600">
              Accepted page: <strong>{acceptedCountCurrentPage}</strong> / {rows.length}
            </div>
          </div>
        </Card>

        <Card className="bg-white p-6 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4 text-sm">
            <div className="border rounded-lg p-3">Total generated: <strong>{stats?.total_generated ?? 0}</strong></div>
            <div className="border rounded-lg p-3">Accepted: <strong>{stats?.accepted ?? 0}</strong></div>
            <div className="border rounded-lg p-3">Rejected: <strong>{stats?.rejected ?? 0}</strong></div>
            <div className="border rounded-lg p-3">Acceptance rate: <strong>{((stats?.acceptance_rate ?? 0) * 100).toFixed(1)}%</strong></div>
          </div>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Accept</TableHead>
                <TableHead>Text (editable)</TableHead>
                <TableHead>Actions</TableHead>
                <TableHead>Label</TableHead>
                <TableHead>Domain</TableHead>
                <TableHead>Style</TableHead>
                <TableHead>Batch</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const nextLabel = editedLabelMap[row.id] ?? (row.label === 1 ? 1 : 0);
                const selected = checkedMap[row.id] ?? row.is_accepted;
                return (
                  <TableRow
                    key={row.id}
                    onClick={(event) => handleRowToggle(event, row.id)}
                    className="cursor-pointer hover:bg-slate-50"
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
                          Copy
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => void handlePasteRowText(row.id)}>
                          Paste
                        </Button>
                      </div>
                    </TableCell>
                    <TableCell>
                      <select
                        className={`border rounded-md px-2 py-1 text-sm font-medium ${nextLabel === 1 ? "bg-red-200 text-red-900 border-red-400" : "bg-green-200 text-green-900 border-green-400"}`}
                        value={nextLabel}
                        onChange={(event) =>
                          setEditedLabelMap((prev) => ({
                            ...prev,
                            [row.id]: Number(event.target.value) as 0 | 1,
                          }))
                        }
                      >
                        <option value={1}>toxic</option>
                        <option value={0}>clean</option>
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
                  <TableCell colSpan={7} className="text-center text-sm text-gray-500">
                    Không có dữ liệu
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>

          <div className="mt-4 flex items-center justify-between text-sm">
            <Button variant="outline" onClick={() => setPage((prev) => Math.max(1, prev - 1))} disabled={page <= 1}>
              Previous
            </Button>
            <span>
              Page {page} / {totalPages}
            </span>
            <Button
              variant="outline"
              onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
              disabled={page >= totalPages}
            >
              Next
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}
