import { useEffect, useMemo, useState } from "react";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Label } from "@/app/components/ui/label";
import { Pagination, PaginationContent, PaginationItem, PaginationLink, PaginationNext, PaginationPrevious } from "@/app/components/ui/pagination";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/app/components/ui/table";
import { useI18n } from "@/app/i18n/context";
import type { MlflowCandidate, MlflowTrainingPlan, MlflowTrainingPreview } from "../../hooks/useMlflowStore";

interface DatasetRow { text: string; label: number; constructiveness?: number | string | null; meta?: { split?: string; created_at?: string; constructiveness?: number | string | null }; }
interface DatasetStats { total: number; by_source: Record<string, { total: number; clean: number; toxic: number }>; }
interface DatasetPreview { items: DatasetRow[]; total: number; total_pages: number; stats?: DatasetStats; }
interface DatasetExport { path: string; artifact_path?: string; manifest_path?: string; count: number; artifact_versions?: { dataset_version?: string; model_version?: string; policy_version?: string }; }
interface DatasetPageProps { adminToken?: string; onAdminUnauthorized?: () => void; }
type View = "gold" | "candidates";
type PlanFilter = "all" | "included" | "balance_dropped" | "duplicate" | "needs_review";

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");
const API_BASES = Array.from(new Set(["", API_BASE, "http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:8001", "http://localhost:8001"].filter(Boolean)));
let lastApiBase: string | null = null;
const apiUrl = (base: string, path: string) => base ? `${base}${path.startsWith("/") ? path : `/${path}`}` : path;
const isNetworkError = (error: unknown) => error instanceof TypeError && /failed to fetch|networkerror|load failed/i.test(error.message);
async function apiFetch(path: string, init?: RequestInit) {
  let lastError: unknown;
  for (const base of [lastApiBase || "", ...API_BASES].filter((value, index, values) => values.indexOf(value) === index)) {
    try {
      const response = await fetch(apiUrl(base, path), init);
      if (!base && response.status === 404) continue;
      lastApiBase = base;
      return response;
    } catch (error) { if (!isNetworkError(error)) throw error; lastError = error; }
  }
  throw new Error(`Cannot reach backend API for ${path}. ${(lastError as Error | undefined)?.message || ""}`.trim());
}
const pct = (value: number, total: number) => total ? `${((value / total) * 100).toFixed(1)}%` : "0.0%";

export function DatasetPage({ adminToken, onAdminUnauthorized }: DatasetPageProps) {
  const { t } = useI18n();
  const [view, setView] = useState<View>("gold");
  const [goldRows, setGoldRows] = useState<DatasetRow[]>([]);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [goldTotal, setGoldTotal] = useState(0);
  const [goldPages, setGoldPages] = useState(1);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [label, setLabel] = useState("all");
  const [split, setSplit] = useState("all");
  const [origin, setOrigin] = useState("all");
  const [planFilter, setPlanFilter] = useState<PlanFilter>("all");
  const [goldLoading, setGoldLoading] = useState(false);
  const [candidateLoading, setCandidateLoading] = useState(false);
  const [goldError, setGoldError] = useState<string | null>(null);
  const [candidateError, setCandidateError] = useState<string | null>(null);
  const [exportStatus, setExportStatus] = useState<string | null>(null);
  const [preview, setPreview] = useState<MlflowTrainingPreview | null>(null);
  const [plan, setPlan] = useState<MlflowTrainingPlan | null>(null);

  const goldStats = useMemo(() => {
    const values = Object.values(stats?.by_source || {});
    const clean = values.reduce((sum, value) => sum + value.clean, 0);
    const toxic = values.reduce((sum, value) => sum + value.toxic, 0);
    return { total: stats?.total || clean + toxic, clean, toxic };
  }, [stats]);
  const fetchGold = async (targetPage = page) => {
    setGoldLoading(true); setGoldError(null);
    try {
      const query = new URLSearchParams({ page: String(targetPage), page_size: String(pageSize), include_stats: "true", dataset_version: "latest" });
      if (label !== "all") query.set("label", label === "toxic" ? "1" : "0");
      if (split !== "all") query.set("split", split);
      const response = await apiFetch(`/api/dataset/preview?${query}`);
      const data = await response.json() as DatasetPreview;
      if (!response.ok) throw new Error(JSON.stringify(data));
      setGoldRows(data.items || []); setStats(data.stats || null); setGoldTotal(data.total || 0); setGoldPages(data.total_pages || 1);
    } catch (error) { setGoldError(error instanceof Error ? error.message : t("dataset.status.cannotLoadDataset")); }
    finally { setGoldLoading(false); }
  };
  const fetchCandidates = async () => {
    setCandidateLoading(true); setCandidateError(null);
    try {
      if (!adminToken) throw new Error(t("dataset.trainingCandidates.adminRequired"));
      const headers = { Authorization: `Bearer ${adminToken}` };
      const query = "page=1&page_size=300&scope=all_batches";
      const [previewResult, planResult] = await Promise.allSettled([
        apiFetch(`/api/mlflow/training-preview?${query}`, { headers }),
        apiFetch(`/api/mlflow/training-plan?${query}&balance_strategy=balanced_50_50`, { headers }),
      ]);
      if (previewResult.status === "rejected") throw previewResult.reason;
      if (previewResult.value.status === 401) { onAdminUnauthorized?.(); throw new Error(t("dataset.trainingCandidates.adminRequired")); }
      const previewData = await previewResult.value.json() as MlflowTrainingPreview;
      if (!previewResult.value.ok) throw new Error(JSON.stringify(previewData));
      setPreview(previewData);
      if (planResult.status === "fulfilled" && planResult.value.ok) setPlan(await planResult.value.json() as MlflowTrainingPlan);
      else { setPlan(null); setCandidateError(t("dataset.trainingCandidates.planUnavailable")); }
    } catch (error) { setPreview(null); setPlan(null); setCandidateError(error instanceof Error ? error.message : t("dataset.trainingCandidates.cannotLoad")); }
    finally { setCandidateLoading(false); }
  };
  const exportGold = async () => {
    setExportStatus(null);
    try {
      const body: Record<string, unknown> = { dataset_version: "latest", model_version: "phobert/baseline", policy_version: "policy-v1" };
      if (label !== "all") body.label = [label === "toxic" ? 1 : 0];
      if (split !== "all") body.split = [split];
      const response = await apiFetch("/api/dataset/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await response.json() as DatasetExport;
      if (!response.ok) throw new Error(JSON.stringify(data));
      setExportStatus(t("dataset.status.exportedRowsWithLineage", { count: data.count, path: data.artifact_path || data.path, manifest: data.manifest_path || t("dataset.common.na"), datasetVersion: data.artifact_versions?.dataset_version || t("dataset.common.na"), modelVersion: data.artifact_versions?.model_version || t("dataset.common.na"), policyVersion: data.artifact_versions?.policy_version || t("dataset.common.na") }));
    } catch (error) { setExportStatus(error instanceof Error ? error.message : t("dataset.status.exportFailed")); }
  };
  useEffect(() => { setPage(1); }, [view, label, split, origin, planFilter, pageSize]);
  useEffect(() => { if (view === "gold") void fetchGold(); }, [view, page, pageSize, label, split]);
  useEffect(() => { if (view === "candidates") void fetchCandidates(); }, [view, adminToken]);

  const rowPlanCode = (item: MlflowCandidate) => {
    if (item.review_reason === "model_conflict" || item.review_reason === "model_uncertain" || item.verification_status === "unverified") return "needs_review";
    return plan?.row_statuses[String(item.id)]?.reason_code || "other";
  };
  const candidates = useMemo(() => (preview?.items || []).filter((item) =>
    (label === "all" || item.pseudo_label === (label === "toxic" ? 1 : 0)) &&
    (origin === "all" || item.source_type === origin) &&
    (planFilter === "all" || rowPlanCode(item) === planFilter)), [preview, plan, label, origin, planFilter]);
  const candidatePages = Math.max(1, Math.ceil(candidates.length / pageSize));
  const visibleCandidates = candidates.slice((page - 1) * pageSize, page * pageSize);
  const activePages = view === "gold" ? goldPages : candidatePages;
  const visibleOrigins = useMemo(() => (preview?.items || []).reduce<Record<string, number>>((all, item) => { const key = item.source_type || "unknown"; all[key] = (all[key] || 0) + 1; return all; }, {}), [preview]);
  const originText = (value: string) => value === "crawl" ? t("dataset.trainingCandidates.websiteCollection") : value === "synthetic" ? t("dataset.trainingCandidates.synthetic") : value.replaceAll("_", " ");
  const planText = (item: MlflowCandidate) => ({ included: t("dataset.trainingCandidates.plannedAddition"), balance_dropped: t("dataset.trainingCandidates.excludedBalance"), duplicate: t("dataset.trainingCandidates.excludedDedup"), needs_review: t("dataset.trainingCandidates.needsReview") }[rowPlanCode(item)] || t("dataset.trainingCandidates.notEligible"));
  const reviewText = (item: MlflowCandidate) => item.verification_status === "manual_accepted" ? t("dataset.trainingCandidates.reviewedAccepted") : item.verification_status === "auto_accepted" ? t("dataset.trainingCandidates.autoAccepted") : item.verification_status === "unverified" ? t("dataset.trainingCandidates.needsReview") : item.verification_status.replaceAll("_", " ");
  const pageLinks = Array.from(new Set([1, activePages, page - 1, page, page + 1])).filter((value) => value >= 1 && value <= activePages).sort((a, b) => a - b);

  return <div className="dashboard-page"><div className="max-w-6xl mx-auto">
    <div className="mb-10 text-center"><h1 className="text-4xl mb-3 text-primary">{t("dataset.hero.title")}</h1><p className="text-lg text-muted-foreground">{t("dataset.hero.subtitle")}</p></div>
    <Card className="bg-card p-6 mb-8 shadow-lg"><div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
      <div><Label>{t("dataset.trainingCandidates.view")}</Label><select className="mt-2 w-full border rounded-lg px-3 py-2 text-sm" value={view} onChange={(event) => setView(event.target.value as View)}><option value="gold">{t("dataset.trainingCandidates.victsdGold")}</option><option value="candidates">{t("dataset.trainingCandidates.title")}</option></select></div>
      <div><Label>{t("dataset.filters.label")}</Label><select className="mt-2 w-full border rounded-lg px-3 py-2 text-sm" value={label} onChange={(event) => setLabel(event.target.value)}><option value="all">{t("dataset.filters.all")}</option><option value="clean">{t("dataset.filters.clean")}</option><option value="toxic">{t("dataset.filters.toxic")}</option></select></div>
      {view === "gold" ? <div><Label>{t("dataset.filters.split")}</Label><select className="mt-2 w-full border rounded-lg px-3 py-2 text-sm" value={split} onChange={(event) => setSplit(event.target.value)}><option value="all">{t("dataset.filters.all")}</option><option value="train">{t("dataset.filters.train")}</option><option value="validation">{t("dataset.filters.validation")}</option><option value="test">{t("dataset.filters.test")}</option></select></div> : <div><Label>{t("dataset.trainingCandidates.origin")}</Label><select className="mt-2 w-full border rounded-lg px-3 py-2 text-sm" value={origin} onChange={(event) => setOrigin(event.target.value)}><option value="all">{t("dataset.filters.all")}</option><option value="crawl">{t("dataset.trainingCandidates.websiteCollection")}</option><option value="synthetic">{t("dataset.trainingCandidates.synthetic")}</option></select></div>}
      {view === "gold" ? <div><Label>{t("dataset.filters.pageSize")}</Label><select className="mt-2 w-full border rounded-lg px-3 py-2 text-sm" value={pageSize} onChange={(event) => setPageSize(Number(event.target.value))}>{[10, 25, 50, 100].map((size) => <option key={size} value={size}>{size}</option>)}</select></div> : <div><Label>{t("dataset.trainingCandidates.bundlePlan")}</Label><select className="mt-2 w-full border rounded-lg px-3 py-2 text-sm" value={planFilter} onChange={(event) => setPlanFilter(event.target.value as PlanFilter)}><option value="all">{t("dataset.filters.all")}</option><option value="included">{t("dataset.trainingCandidates.plannedAddition")}</option><option value="balance_dropped">{t("dataset.trainingCandidates.excludedBalance")}</option><option value="duplicate">{t("dataset.trainingCandidates.excludedDedup")}</option><option value="needs_review">{t("dataset.trainingCandidates.needsReview")}</option></select></div>}
    </div><div className="mt-4 flex flex-wrap gap-3 items-center"><Button onClick={() => view === "gold" ? fetchGold(1) : fetchCandidates()} disabled={goldLoading || candidateLoading}>{goldLoading || candidateLoading ? t("dataset.status.loading") : t("dataset.actions.refresh")}</Button>{view === "gold" && <Button variant="outline" onClick={exportGold}>{t("dataset.actions.exportJsonl")}</Button>}{view === "gold" && exportStatus && <span className="text-sm text-muted-foreground">{exportStatus}</span>}</div>{view === "gold" && goldError && <p className="mt-3 text-sm text-destructive">{goldError}</p>}{view === "candidates" && candidateError && <p className="mt-3 text-sm text-destructive">{candidateError}</p>}</Card>
    {view === "gold" ? <>
      <Card className="bg-card p-6 mb-8 shadow-lg"><h2 className="text-2xl text-primary">{t("dataset.trainingCandidates.victsdGold")}</h2><p className="mt-1 text-sm text-muted-foreground">{t("dataset.analysis.baseDatasetNote")}</p><div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3"><Metric label={t("dataset.overview.totalSamples")} value={goldStats.total} /><Metric label={t("dataset.overview.cleanNonToxic")} value={goldStats.clean} /><Metric label={t("dataset.filters.toxic")} value={goldStats.toxic} /></div><p className="mt-3 text-xs text-muted-foreground">{t("dataset.trainingCandidates.goldFilterNote", { clean: pct(goldStats.clean, goldStats.total), toxic: pct(goldStats.toxic, goldStats.total) })}</p></Card>
      <Card className="bg-card p-6 shadow-lg"><h2 className="text-2xl text-primary">{t("dataset.currentView.title")}</h2><p className="mt-1 text-sm text-muted-foreground">{t("dataset.currentView.matchingRows", { count: goldTotal.toLocaleString() })}</p><Table className="mt-4"><TableHeader><TableRow><TableHead>{t("dataset.table.text")}</TableHead><TableHead>{t("dataset.filters.label")}</TableHead><TableHead>{t("dataset.filters.split")}</TableHead><TableHead>{t("dataset.table.constructiveness")}</TableHead><TableHead>{t("dataset.table.created")}</TableHead></TableRow></TableHeader><TableBody>{goldRows.map((row, index) => <TableRow key={`${row.text.slice(0, 24)}-${index}`}><TableCell className="max-w-[420px] truncate" title={row.text}>{row.text}</TableCell><TableCell>{row.label === 1 ? t("dataset.filters.toxic") : t("dataset.filters.clean")}</TableCell><TableCell>{row.meta?.split || "—"}</TableCell><TableCell>{row.constructiveness === 1 || row.meta?.constructiveness === 1 ? t("dataset.constructiveness.constructive") : row.constructiveness === 0 || row.meta?.constructiveness === 0 ? t("dataset.constructiveness.notConstructive") : "—"}</TableCell><TableCell>{row.meta?.created_at || "—"}</TableCell></TableRow>)}{!goldRows.length && !goldLoading && <EmptyRow span={5} text={t("dataset.common.noData")} />}</TableBody></Table></Card>
    </> : <>
      <Card className="bg-card p-6 mb-8 shadow-lg"><h2 className="text-2xl text-primary">{t("dataset.trainingCandidates.title")}</h2><p className="mt-1 text-sm text-muted-foreground">{t("dataset.trainingCandidates.description")}</p>{preview && <><div className="mt-4 grid grid-cols-1 md:grid-cols-6 gap-3"><Metric label={t("dataset.trainingCandidates.currentCandidates")} value={preview.total} /><Metric label={t("dataset.filters.clean")} value={preview.counts.selected_clean} /><Metric label={t("dataset.filters.toxic")} value={preview.counts.selected_toxic} /><Metric label={t("dataset.trainingCandidates.plannedAddition")} value={plan?.summary.mlflow_added ?? "—"} /><Metric label={t("dataset.trainingCandidates.excludedBalance")} value={plan ? plan.summary.eligible_mlflow - plan.summary.after_balance : "—"} /><Metric label={t("dataset.trainingCandidates.excludedDedup")} value={plan?.summary.duplicates_skipped ?? "—"} /></div><p className="mt-3 text-sm text-muted-foreground">{t("dataset.trainingCandidates.selectionNote")}</p><div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-sm"><strong>{t("dataset.trainingCandidates.origin")}</strong>{Object.entries(visibleOrigins).map(([key, count]) => <span key={key}>{originText(key)}: {count}</span>)}</div></>}</Card>
      {plan && <Card className="bg-card p-6 mb-8 shadow-lg"><h3 className="text-lg font-semibold text-primary">{t("dataset.trainingCandidates.planTitle")}</h3><p className="mt-2 text-sm text-muted-foreground">{t("dataset.trainingCandidates.planEquation", { goldTrain: plan.summary.gold_train, added: plan.summary.mlflow_added, finalTrain: plan.summary.final_train, validation: plan.summary.gold_validation, test: plan.summary.gold_test })}</p><p className="mt-2 text-sm text-muted-foreground">{t("dataset.trainingCandidates.dedup", { count: plan.summary.duplicates_skipped })}</p></Card>}
      <Card className="bg-card p-6 shadow-lg"><h2 className="text-2xl text-primary">{t("dataset.trainingCandidates.readOnlyTable")}</h2><p className="mt-1 text-sm text-muted-foreground">{t("dataset.currentView.matchingRows", { count: candidates.length.toLocaleString() })}</p><Table className="mt-4"><TableHeader><TableRow><TableHead>{t("dataset.table.text")}</TableHead><TableHead>{t("dataset.filters.label")}</TableHead><TableHead>{t("dataset.trainingCandidates.origin")}</TableHead><TableHead>{t("dataset.trainingCandidates.reviewState")}</TableHead><TableHead>{t("dataset.trainingCandidates.candidateState")}</TableHead><TableHead>{t("dataset.trainingCandidates.bundlePlan")}</TableHead></TableRow></TableHeader><TableBody>{visibleCandidates.map((item) => <TableRow key={item.id}><TableCell className="max-w-[360px] truncate" title={item.text}>{item.text}</TableCell><TableCell>{item.pseudo_label === 1 ? t("dataset.filters.toxic") : t("dataset.filters.clean")}</TableCell><TableCell>{originText(item.source_type || "unknown")}</TableCell><TableCell>{reviewText(item)}</TableCell><TableCell>{rowPlanCode(item) === "needs_review" ? t("dataset.trainingCandidates.needsReview") : t("dataset.trainingCandidates.eligible")}</TableCell><TableCell>{planText(item)}</TableCell></TableRow>)}{!visibleCandidates.length && !candidateLoading && <EmptyRow span={6} text={t("dataset.common.noData")} />}</TableBody></Table></Card>
    </>}
    <div className="mt-6"><Pagination><PaginationContent><PaginationItem><PaginationPrevious href="#" onClick={(event) => { event.preventDefault(); setPage((current) => Math.max(1, current - 1)); }} /></PaginationItem>{pageLinks.map((item) => <PaginationItem key={item}><PaginationLink href="#" isActive={item === page} onClick={(event) => { event.preventDefault(); setPage(item); }}>{item}</PaginationLink></PaginationItem>)}<PaginationItem><PaginationNext href="#" onClick={(event) => { event.preventDefault(); setPage((current) => Math.min(activePages, current + 1)); }} /></PaginationItem></PaginationContent></Pagination></div>
  </div></div>;
}

function Metric({ label, value }: { label: string; value: string | number }) { return <div className="rounded-lg border bg-muted/30 p-4"><p className="text-xs text-muted-foreground">{label}</p><p className="text-2xl font-semibold">{typeof value === "number" ? value.toLocaleString() : value}</p></div>; }
function EmptyRow({ span, text }: { span: number; text: string }) { return <TableRow><TableCell colSpan={span} className="text-center text-sm text-muted-foreground">{text}</TableCell></TableRow>; }
