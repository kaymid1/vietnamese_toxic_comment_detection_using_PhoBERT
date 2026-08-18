import { useCallback, useEffect, useMemo, useState } from "react";
import { CalendarClock, CheckCircle2, CircleAlert, Loader2, Play, RefreshCw, Save, Settings2, XCircle } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/app/components/ui/badge";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Input } from "@/app/components/ui/input";
import { Switch } from "@/app/components/ui/switch";
import { getModelLabel } from "@/app/modelCatalog";
import { fetchApiWithFallback } from "../../hooks/useMlflowStore";

interface ScheduledTask {
  id: string;
  name: string;
  task_type: string;
  enabled: boolean;
  interval_minutes: number;
  timezone: string;
  rss_url: string;
  max_articles_per_run: number;
  model_name?: string | null;
  next_run_at?: string | null;
  last_run_at?: string | null;
}

interface ScheduledTaskRun {
  id: string;
  task_id: string;
  scheduled_for: string;
  trigger_type: "scheduled" | "manual" | "recovery" | string;
  status: "queued" | "running" | "completed" | "failed" | "skipped" | string;
  started_at?: string | null;
  finished_at?: string | null;
  discovered_count: number;
  processed_count: number;
  failed_count: number;
  error?: string | null;
  metadata?: { articles?: Array<{ url?: string; status?: string; error?: string }> };
}

interface ScheduledTaskArticle {
  discovery_key: string;
  canonical_url: string;
  article_title?: string | null;
  stage: string;
  attempt_count: number;
  retry_after?: string | null;
  last_error?: string | null;
}

interface ScheduledTaskDetail {
  task: ScheduledTask;
  runs: ScheduledTaskRun[];
  articles: ScheduledTaskArticle[];
}

interface ScheduledTasksPageProps {
  adminToken: string;
  onAdminUnauthorized: () => void;
  availableModels: string[];
  modelLabels: Record<string, string>;
}

const parseJsonResponse = async <T,>(response: Response): Promise<T> => {
  const raw = await response.text();
  if (response.status === 401) throw new Error("UNAUTHORIZED");
  if (!response.ok) {
    let message = raw || "API request failed";
    try {
      const parsed = raw ? (JSON.parse(raw) as { detail?: unknown }) : null;
      if (typeof parsed?.detail === "string") message = parsed.detail;
    } catch {
      // Preserve the raw API message.
    }
    throw new Error(message);
  }
  return JSON.parse(raw) as T;
};

const formatTime = (value?: string | null) => {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

const statusVariant = (status?: string | null) => {
  if (status === "completed") return "default" as const;
  if (status === "failed") return "destructive" as const;
  if (status === "running") return "secondary" as const;
  return "outline" as const;
};

export function ScheduledTasksPage({ adminToken, onAdminUnauthorized, availableModels, modelLabels }: ScheduledTasksPageProps) {
  const [tasks, setTasks] = useState<ScheduledTask[]>([]);
  const [detail, setDetail] = useState<ScheduledTaskDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState(false);
  const [cancellingRunId, setCancellingRunId] = useState<string | null>(null);
  const [draftEnabled, setDraftEnabled] = useState(false);
  const [draftInterval, setDraftInterval] = useState("60");
  const [draftMaxArticles, setDraftMaxArticles] = useState("10");
  const [draftModel, setDraftModel] = useState("");

  const selectableModels = useMemo(
    () => availableModels.filter((model) => !model.toLowerCase().includes("deprecated")),
    [availableModels],
  );

  const defaultModel = selectableModels[0] || availableModels[0] || "";

  const authHeaders = useMemo(
    () => ({ Authorization: `Bearer ${adminToken}`, "Content-Type": "application/json" }),
    [adminToken],
  );

  const loadTasks = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    try {
      const payload = await parseJsonResponse<{ items: ScheduledTask[] }>(
        await fetchApiWithFallback("/api/admin/scheduled-tasks", { headers: { Authorization: `Bearer ${adminToken}` } }),
      );
      setTasks(payload.items || []);
      const selected = payload.items?.[0];
      if (selected) {
        const detailPayload = await parseJsonResponse<ScheduledTaskDetail>(
          await fetchApiWithFallback(`/api/admin/scheduled-tasks/${selected.id}`, { headers: { Authorization: `Bearer ${adminToken}` } }),
        );
        setDetail(detailPayload);
        setDraftEnabled(detailPayload.task.enabled);
        setDraftInterval(String(detailPayload.task.interval_minutes));
        setDraftMaxArticles(String(detailPayload.task.max_articles_per_run));
        setDraftModel(detailPayload.task.model_name || defaultModel);
      }
    } catch (error) {
      if (error instanceof Error && error.message === "UNAUTHORIZED") {
        onAdminUnauthorized();
        return;
      }
      if (!quiet) toast.error(error instanceof Error ? error.message : "Cannot load scheduled tasks");
    } finally {
      if (!quiet) setLoading(false);
    }
  }, [adminToken, defaultModel, onAdminUnauthorized]);

  const loadDetail = useCallback(async () => {
    if (!detail?.task.id) return;
    try {
      const payload = await parseJsonResponse<ScheduledTaskDetail>(
        await fetchApiWithFallback(`/api/admin/scheduled-tasks/${detail.task.id}`, { headers: { Authorization: `Bearer ${adminToken}` } }),
      );
      setDetail(payload);
      setTasks((current) => current.map((task) => task.id === payload.task.id ? payload.task : task));
    } catch (error) {
      if (error instanceof Error && error.message === "UNAUTHORIZED") onAdminUnauthorized();
    }
  }, [adminToken, detail?.task.id, onAdminUnauthorized]);

  useEffect(() => { void loadTasks(); }, [loadTasks]);
  useEffect(() => {
    const timer = window.setInterval(() => void loadDetail(), 15000);
    return () => window.clearInterval(timer);
  }, [loadDetail]);

  const saveTask = async () => {
    if (!detail) return;
    setSaving(true);
    try {
      const payload = await parseJsonResponse<{ task: ScheduledTask }>(
        await fetchApiWithFallback(`/api/admin/scheduled-tasks/${detail.task.id}`, {
          method: "PATCH",
          headers: authHeaders,
          body: JSON.stringify({
            enabled: draftEnabled,
            interval_minutes: Number(draftInterval),
            max_articles_per_run: Number(draftMaxArticles),
            model_name: draftModel || null,
          }),
        }),
      );
      setDetail((current) => current ? { ...current, task: payload.task } : current);
      setTasks((current) => current.map((task) => task.id === payload.task.id ? payload.task : task));
      toast.success("Scheduled Task đã được lưu.");
    } catch (error) {
      if (error instanceof Error && error.message === "UNAUTHORIZED") onAdminUnauthorized();
      else toast.error(error instanceof Error ? error.message : "Cannot save scheduled task");
    } finally {
      setSaving(false);
    }
  };

  const toggleTaskEnabled = async (enabled: boolean) => {
    if (!detail) return;
    const previous = draftEnabled;
    setDraftEnabled(enabled);
    setSaving(true);
    try {
      const payload = await parseJsonResponse<{ task: ScheduledTask }>(
        await fetchApiWithFallback(`/api/admin/scheduled-tasks/${detail.task.id}`, {
          method: "PATCH",
          headers: authHeaders,
          // Keep configuration drafts untouched: this action changes only the
          // scheduler switch and must take effect without pressing Save.
          body: JSON.stringify({ enabled }),
        }),
      );
      setDetail((current) => current ? { ...current, task: payload.task } : current);
      setTasks((current) => current.map((task) => task.id === payload.task.id ? payload.task : task));
      toast.success(enabled ? "Scheduled Task đã được bật ngay." : "Scheduled Task đã được tắt ngay.");
    } catch (error) {
      setDraftEnabled(previous);
      if (error instanceof Error && error.message === "UNAUTHORIZED") onAdminUnauthorized();
      else toast.error(error instanceof Error ? error.message : "Cannot update scheduled task status");
    } finally {
      setSaving(false);
    }
  };

  const runNow = async () => {
    if (!detail) return;
    setRunning(true);
    try {
      await parseJsonResponse<{ run: ScheduledTaskRun }>(
        await fetchApiWithFallback(`/api/admin/scheduled-tasks/${detail.task.id}/run-now`, {
          method: "POST",
          headers: authHeaders,
        }),
      );
      toast.success("Đã bắt đầu Run Now.");
      await loadDetail();
    } catch (error) {
      if (error instanceof Error && error.message === "UNAUTHORIZED") onAdminUnauthorized();
      else toast.error(error instanceof Error ? error.message : "Cannot start scheduled task");
    } finally {
      setRunning(false);
    }
  };

  const cancelRun = async (runId: string) => {
    setCancellingRunId(runId);
    try {
      await parseJsonResponse<{ run: ScheduledTaskRun }>(
        await fetchApiWithFallback(`/api/admin/scheduled-tasks/${detail?.task.id}/runs/${runId}/cancel`, {
          method: "POST",
          headers: authHeaders,
        }),
      );
      toast.success("Đã hủy run bị treo.");
      await loadDetail();
    } catch (error) {
      if (error instanceof Error && error.message === "UNAUTHORIZED") onAdminUnauthorized();
      else toast.error(error instanceof Error ? error.message : "Cannot cancel scheduled task run");
    } finally {
      setCancellingRunId(null);
    }
  };

  const latestRun = detail?.runs?.[0];
  const activeRun = detail?.runs?.find((run) => run.status === "running");
  const status = detail?.task.enabled ? latestRun?.status || "ready" : "disabled";

  return (
    <div className="dashboard-page mx-auto max-w-6xl space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wider text-muted-foreground">Admin / Scheduled Tasks</p>
          <h1 className="mt-1 text-2xl font-semibold text-foreground">Scheduled Tasks</h1>
          <p className="mt-1 text-sm text-muted-foreground">Persisted VnExpress RSS discovery and sequential collection.</p>
        </div>
        <Button variant="outline" onClick={() => void loadTasks()} disabled={loading}>
          <RefreshCw className="h-4 w-4" /> Refresh
        </Button>
      </header>

      {loading && !detail ? (
        <Card className="flex items-center gap-2 p-5 text-sm text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" /> Loading scheduled tasks…</Card>
      ) : detail ? (
        <>
          <Card className="p-5 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="flex items-start gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-background-info text-text-info"><CalendarClock className="h-5 w-5" /></div>
                <div>
                  <h2 className="text-lg font-semibold">{detail.task.name}</h2>
                  <p className="text-sm text-muted-foreground">Source: VnExpress RSS · Task type: {detail.task.task_type}</p>
                </div>
              </div>
              <Badge variant={statusVariant(status)}>{status === "ready" ? "Ready" : status}</Badge>
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-lg border border-border bg-background-secondary p-3"><p className="text-xs text-muted-foreground">Schedule</p><p className="mt-1 font-medium">Every {detail.task.interval_minutes} minutes</p><p className="text-xs text-muted-foreground">{detail.task.timezone}</p></div>
              <div className="rounded-lg border border-border bg-background-secondary p-3"><p className="text-xs text-muted-foreground">Max articles/run</p><p className="mt-1 font-medium">{detail.task.max_articles_per_run}</p></div>
              <div className="rounded-lg border border-border bg-background-secondary p-3"><p className="text-xs text-muted-foreground">Last run</p><p className="mt-1 font-medium">{formatTime(detail.task.last_run_at)}</p></div>
              <div className="rounded-lg border border-border bg-background-secondary p-3"><p className="text-xs text-muted-foreground">Next run</p><p className="mt-1 font-medium">{formatTime(detail.task.next_run_at)}</p></div>
            </div>

            <div className="mt-4 rounded-lg border border-border p-3 text-sm"><span className="text-muted-foreground">RSS: </span><code className="break-all">{detail.task.rss_url}</code></div>

            <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-border pt-4">
              <Button onClick={() => void runNow()} disabled={running || Boolean(activeRun)}>
                {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />} Run Now
              </Button>
              <div className="flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm">
                <Switch checked={draftEnabled} onCheckedChange={(enabled) => void toggleTaskEnabled(enabled)} disabled={saving} />
                <span>{draftEnabled ? "Enabled" : "Disabled"}</span>
                <span className="text-xs text-muted-foreground">Áp dụng ngay</span>
              </div>
            </div>
          </Card>

          <Card className="p-5 shadow-sm">
            <div className="mb-4 flex items-center gap-2"><Settings2 className="h-5 w-5 text-text-info" /><h2 className="text-lg font-semibold">Task configuration</h2></div>
            <div className="grid gap-4 md:grid-cols-4">
              <label className="text-sm"><span className="font-medium">Interval (minutes)</span><Input className="mt-1" type="number" min={1} max={10080} value={draftInterval} onChange={(event) => setDraftInterval(event.target.value)} /></label>
              <label className="text-sm"><span className="font-medium">Max articles per run</span><Input className="mt-1" type="number" min={1} max={100} value={draftMaxArticles} onChange={(event) => setDraftMaxArticles(event.target.value)} /></label>
              <label className="text-sm"><span className="font-medium">Model</span><select className="mt-1 h-10 w-full rounded-md border border-border bg-background px-3 text-sm" value={draftModel} onChange={(event) => setDraftModel(event.target.value)} disabled={saving || selectableModels.length === 0}><option value="">Use current default</option>{selectableModels.map((model) => <option key={model} value={model}>{getModelLabel(model, modelLabels)}</option>)}</select></label>
              <div className="text-sm"><span className="font-medium">Timezone</span><p className="mt-1 rounded-md border border-border bg-muted/30 px-3 py-2">{detail.task.timezone}</p></div>
            </div>
            <div className="mt-4 flex justify-end"><Button variant="outline" onClick={() => void saveTask()} disabled={saving}><Save className="h-4 w-4" /> {saving ? "Saving…" : "Save"}</Button></div>
          </Card>

          <Card className="p-5 shadow-sm">
            <div className="mb-4 flex items-center gap-2"><CheckCircle2 className="h-5 w-5 text-text-info" /><h2 className="text-lg font-semibold">Run history</h2></div>
            <div className="space-y-2">
              {!detail.runs.length && <p className="text-sm text-muted-foreground">No persisted runs yet.</p>}
              {detail.runs.map((run) => (
                <div key={run.id} className="flex items-start gap-2 rounded-lg border border-border p-3">
                  <details className="min-w-0 flex-1">
                    <summary className="cursor-pointer list-none">
                      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                        <div className="flex flex-wrap items-center gap-2"><Badge variant={statusVariant(run.status)}>{run.status}</Badge><span>{run.trigger_type}</span><span className="text-muted-foreground">scheduled {formatTime(run.scheduled_for)}</span></div>
                        <span className="text-muted-foreground">{run.processed_count} processed · {run.failed_count} failed</span>
                      </div>
                    </summary>
                    <div className="mt-3 grid gap-2 border-t border-border pt-3 text-xs text-muted-foreground sm:grid-cols-2">
                      <span>Started: {formatTime(run.started_at)}</span><span>Finished: {formatTime(run.finished_at)}</span>
                      <span>Discovered: {run.discovered_count}</span><span>Run ID: {run.id}</span>
                      {run.error && <span className="flex gap-1 text-destructive sm:col-span-2"><CircleAlert className="h-4 w-4 shrink-0" />{run.error}</span>}
                    </div>
                  </details>
                  {run.status === "running" && <Button size="sm" variant="destructive" className="shrink-0" onClick={() => void cancelRun(run.id)} disabled={cancellingRunId === run.id}><XCircle className="h-4 w-4" /> {cancellingRunId === run.id ? "Canceling…" : "Cancel run"}</Button>}
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-5 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Article discovery state</h2>
            <div className="space-y-2">
              {!detail.articles.length && <p className="text-sm text-muted-foreground">No discovered articles yet.</p>}
              {detail.articles.slice(0, 30).map((article) => (
                <div key={article.discovery_key} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border p-3 text-sm">
                  <div className="min-w-0"><p className="truncate font-medium">{article.article_title || article.canonical_url}</p><p className="truncate text-xs text-muted-foreground">{article.canonical_url}</p></div>
                  <div className="flex items-center gap-2"><Badge variant={article.stage === "completed" ? "default" : article.stage.startsWith("failed") ? "destructive" : "outline"}>{article.stage}</Badge><span className="text-xs text-muted-foreground">attempts {article.attempt_count}</span></div>
                </div>
              ))}
            </div>
          </Card>
        </>
      ) : (
        <Card className="p-5 text-sm text-muted-foreground">No scheduled task is available.</Card>
      )}
    </div>
  );
}
