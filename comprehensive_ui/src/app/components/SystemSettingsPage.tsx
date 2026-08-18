import { useEffect, useMemo, useState } from "react";
import { CircleHelp, Eye, EyeOff, RotateCcw, Save } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/app/components/ui/badge";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Input } from "@/app/components/ui/input";
import { Switch } from "@/app/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { Textarea } from "@/app/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/app/components/ui/tooltip";
import { fetchApiWithFallback } from "../../hooks/useMlflowStore";

interface SystemSettingItem {
  key: string;
  label: string;
  description?: string;
  type: "string" | "int" | "bool" | "enum";
  required: boolean;
  secret: boolean;
  has_value: boolean;
  source: "db" | "env" | "default";
  value: string | null;
  masked_value?: string | null;
  default?: string | null;
  min?: number | null;
  options?: string[];
  multiline?: boolean;
}

interface SystemSettingGroup {
  id: string;
  label: string;
  settings: SystemSettingItem[];
}

interface SystemSettingsResponse {
  groups: SystemSettingGroup[];
}

interface SystemSettingsPageProps {
  adminToken: string;
  onAdminUnauthorized: () => void;
}

const MLFLOW_CLEAR_ALL_CONFIRM_TOKEN = "DELETE_ALL_MLFLOW_DATA";

const parseJsonResponse = async <T,>(response: Response): Promise<T> => {
  const raw = await response.text();
  if (response.status === 401) {
    throw new Error("UNAUTHORIZED");
  }
  if (!response.ok) {
    let message = raw || "API request failed";
    try {
      const parsed = raw ? (JSON.parse(raw) as { detail?: unknown; message?: unknown }) : null;
      if (typeof parsed?.detail === "string") message = parsed.detail;
      if (typeof parsed?.message === "string") message = parsed.message;
    } catch {
      // Keep raw error.
    }
    throw new Error(message);
  }
  return JSON.parse(raw) as T;
};

const sourceVariant = (source: SystemSettingItem["source"]) => {
  if (source === "db") return "default" as const;
  if (source === "env") return "secondary" as const;
  return "outline" as const;
};

const initialDraftValue = (setting: SystemSettingItem): string | boolean => {
  if (setting.type === "bool") {
    return String(setting.value ?? setting.default ?? "false").toLowerCase() === "true";
  }
  if (setting.secret) return "";
  return String(setting.value ?? "");
};

export function SystemSettingsPage({ adminToken, onAdminUnauthorized }: SystemSettingsPageProps) {
  const [groups, setGroups] = useState<SystemSettingGroup[]>([]);
  const [drafts, setDrafts] = useState<Record<string, string | boolean>>({});
  const [dirtyKeys, setDirtyKeys] = useState<Set<string>>(new Set());
  const [revealedKeys, setRevealedKeys] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState("kaggle_account");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const automationDryRunSetting = groups
    .find((group) => group.id === "mlflow_automation")
    ?.settings.find((setting) => setting.key === "MLFLOW_AUTOMATION_DRY_RUN");
  const automationDryRunEffective = automationDryRunSetting
    ? String(automationDryRunSetting.value ?? automationDryRunSetting.default ?? "false").toLowerCase() === "true"
    : false;
  const automationDryRunDraft = automationDryRunSetting
    ? Boolean(drafts[automationDryRunSetting.key])
    : automationDryRunEffective;

  const authHeaders = useMemo(
    () => ({
      Authorization: `Bearer ${adminToken}`,
      "Content-Type": "application/json",
    }),
    [adminToken],
  );

  const loadSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await parseJsonResponse<SystemSettingsResponse>(
        await fetchApiWithFallback("/api/admin/system-settings", {
          headers: { Authorization: `Bearer ${adminToken}` },
        }),
      );
      setGroups(payload.groups || []);
      const nextDrafts: Record<string, string | boolean> = {};
      for (const group of payload.groups || []) {
        for (const setting of group.settings) {
          nextDrafts[setting.key] = initialDraftValue(setting);
        }
      }
      setDrafts(nextDrafts);
      setDirtyKeys(new Set());
      setRevealedKeys(new Set());
      if (payload.groups?.[0]?.id) setActiveTab((current) => current || payload.groups[0].id);
    } catch (err) {
      if (err instanceof Error && err.message === "UNAUTHORIZED") {
        onAdminUnauthorized();
        return;
      }
      setError(err instanceof Error ? err.message : "Cannot load system settings");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadSettings();
  }, [adminToken]);

  const markDraft = (key: string, value: string | boolean) => {
    setDrafts((prev) => ({ ...prev, [key]: value }));
    setDirtyKeys((prev) => new Set(prev).add(key));
  };

  const toggleSecretVisibility = async (setting: SystemSettingItem) => {
    const key = setting.key;
    if (revealedKeys.has(key)) {
      setRevealedKeys((prev) => {
        const copy = new Set(prev);
        copy.delete(key);
        return copy;
      });
      return;
    }

    try {
      const payload = await parseJsonResponse<{ key: string; value: string }>(
        await fetchApiWithFallback("/api/admin/system-settings/reveal-secret", {
          method: "POST",
          headers: authHeaders,
          body: JSON.stringify({ key }),
        }),
      );
      setDrafts((prev) => ({ ...prev, [key]: payload.value || "" }));
      setRevealedKeys((prev) => new Set(prev).add(key));
    } catch (err) {
      if (err instanceof Error && err.message === "UNAUTHORIZED") {
        onAdminUnauthorized();
        return;
      }
      toast.error(err instanceof Error ? err.message : "Cannot reveal secret");
    }
  };

  const saveGroup = async (group: SystemSettingGroup) => {
    const settings: Record<string, string | boolean> = {};
    for (const setting of group.settings) {
      if (!dirtyKeys.has(setting.key)) continue;
      settings[setting.key] = drafts[setting.key] ?? "";
    }
    if (Object.keys(settings).length === 0) {
      toast.info("Chưa có thay đổi để lưu.");
      return;
    }

    setSaving(true);
    try {
      const payload = await parseJsonResponse<SystemSettingsResponse>(
        await fetchApiWithFallback("/api/admin/system-settings", {
          method: "PATCH",
          headers: authHeaders,
          body: JSON.stringify({ settings }),
        }),
      );
      setGroups(payload.groups || []);
      const updatedByKey = new Map(
        (payload.groups || []).flatMap((payloadGroup) => payloadGroup.settings).map((setting) => [setting.key, setting]),
      );
      setDrafts((prev) => {
        const next = { ...prev };
        Object.keys(settings).forEach((key) => {
          const updated = updatedByKey.get(key);
          if (updated) next[key] = initialDraftValue(updated);
        });
        return next;
      });
      setRevealedKeys((prev) => {
        const next = new Set(prev);
        Object.keys(settings).forEach((key) => next.delete(key));
        return next;
      });
      setDirtyKeys((prev) => {
        const next = new Set(prev);
        Object.keys(settings).forEach((key) => next.delete(key));
        return next;
      });
      toast.success("Đã lưu cấu hình hệ thống.");
    } catch (err) {
      if (err instanceof Error && err.message === "UNAUTHORIZED") {
        onAdminUnauthorized();
        return;
      }
      toast.error(err instanceof Error ? err.message : "Cannot save settings");
    } finally {
      setSaving(false);
    }
  };

  const resetSetting = async (key: string) => {
    setSaving(true);
    try {
      const payload = await parseJsonResponse<SystemSettingsResponse>(
        await fetchApiWithFallback("/api/admin/system-settings", {
          method: "PATCH",
          headers: authHeaders,
          body: JSON.stringify({ settings: {}, clear: [key] }),
        }),
      );
      setGroups(payload.groups || []);
      const next = payload.groups.flatMap((group) => group.settings).find((setting) => setting.key === key);
      if (next) {
        setDrafts((prev) => ({ ...prev, [key]: initialDraftValue(next) }));
      }
      setDirtyKeys((prev) => {
        const copy = new Set(prev);
        copy.delete(key);
        return copy;
      });
      setRevealedKeys((prev) => {
        const copy = new Set(prev);
        copy.delete(key);
        return copy;
      });
      toast.success("Đã trả cấu hình về env/default.");
    } catch (err) {
      if (err instanceof Error && err.message === "UNAUTHORIZED") {
        onAdminUnauthorized();
        return;
      }
      toast.error(err instanceof Error ? err.message : "Cannot reset setting");
    } finally {
      setSaving(false);
    }
  };

  const clearAllMlflow = async () => {
    if (!window.confirm("Xóa toàn bộ dữ liệu MLflow? Hành động này không thể hoàn tác.")) return;

    const token = window.prompt(`Nhập ${MLFLOW_CLEAR_ALL_CONFIRM_TOKEN} để xác nhận clear all:`);
    if (token === null) return;
    if (token.trim() !== MLFLOW_CLEAR_ALL_CONFIRM_TOKEN) {
      toast.error("Sai confirm token. Đã hủy clear all.");
      return;
    }

    setSaving(true);
    try {
      const payload = await parseJsonResponse<{ deleted_rows: Record<string, number> }>(
        await fetchApiWithFallback("/api/mlflow/clear-all", {
          method: "POST",
          headers: authHeaders,
          body: JSON.stringify({ confirm_token: token.trim() }),
        }),
      );
      const rows = payload.deleted_rows;
      toast.success(
        `Đã clear MLflow: do_run=${rows.mlflow_do_run ?? 0}, artifacts=${rows.mlflow_training_artifact ?? 0}, predictions=${rows.mlflow_comment_prediction ?? 0}, items=${rows.mlflow_comment_item ?? 0}, batches=${rows.mlflow_crawl_batch ?? 0}.`,
      );
    } catch (err) {
      if (err instanceof Error && err.message === "UNAUTHORIZED") {
        onAdminUnauthorized();
        return;
      }
      toast.error(err instanceof Error ? err.message : "Clear all MLflow thất bại.");
    } finally {
      setSaving(false);
    }
  };

  const renderInput = (setting: SystemSettingItem) => {
    const value = drafts[setting.key];
    if (setting.type === "bool") {
      return (
        <Switch
          checked={Boolean(value)}
          onCheckedChange={(checked) => markDraft(setting.key, checked)}
          disabled={saving}
        />
      );
    }
    if (setting.type === "enum" && setting.options?.length) {
      return (
        <select
          className="h-9 w-full rounded-md border border-input bg-input-background px-3 text-sm outline-none focus:border-ring focus:ring-2 focus:ring-ring/40"
          value={String(value ?? "")}
          onChange={(event) => markDraft(setting.key, event.target.value)}
          disabled={saving}
        >
          {setting.options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      );
    }
    if (setting.multiline) {
      return (
        <Textarea
          value={String(value ?? "")}
          onChange={(event) => markDraft(setting.key, event.target.value)}
          disabled={saving}
          className="min-h-20"
        />
      );
    }
    return (
      <Input
        type={setting.secret && !revealedKeys.has(setting.key) ? "password" : setting.type === "int" ? "number" : "text"}
        min={setting.min ?? undefined}
        value={String(value ?? "")}
        placeholder={setting.secret ? setting.masked_value || "Not configured" : undefined}
        onChange={(event) => markDraft(setting.key, event.target.value)}
        disabled={saving}
      />
    );
  };

  return (
    <div className="dashboard-page mx-auto max-w-6xl space-y-5">
      <div>
        <p className="text-xs uppercase tracking-wider text-muted-foreground">Admin / Runtime Config</p>
        <h2 className="mt-1 text-2xl font-semibold text-foreground">Cấu hình hệ thống</h2>
      </div>

      {error && <Card className="border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">{error}</Card>}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="flex h-auto flex-wrap justify-start gap-2 rounded-lg border bg-muted/40 p-1.5">
          {groups.map((group) => (
            <TabsTrigger
              key={group.id}
              value={group.id}
              className="border border-transparent px-3 py-2 data-[state=active]:border-primary/30 data-[state=active]:bg-background data-[state=active]:font-semibold data-[state=active]:shadow-sm"
            >
              {group.label}
            </TabsTrigger>
          ))}
        </TabsList>

        {groups.map((group) => (
          <TabsContent key={group.id} value={group.id} className="space-y-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold text-foreground">{group.label}</h3>
                <p className="text-sm text-muted-foreground">
                  Giá trị trong DB ưu tiên hơn `.env.local`; để trống sẽ dùng env/default.
                </p>
              </div>
              <Button type="button" onClick={() => saveGroup(group)} disabled={loading || saving}>
                <Save className="mr-2 h-4 w-4" />
                Lưu panel này
              </Button>
            </div>

            {group.id === "mlflow_automation" && (
              <>
                <Card className="border-amber-500/40 bg-amber-500/10 p-4 text-sm text-muted-foreground">
                  Chỉ bật công tắc toàn cục khi Kaggle bundle endpoint đã public và kiểm tra preflight đạt. `train_only`
                  tạo candidate để admin duyệt; `full_auto` chỉ promote candidate do automation tạo và đã qua production gate.
                  Dữ liệu mới vẫn phải vượt ngưỡng số dòng và cooldown bên dưới.
                </Card>
                {automationDryRunSetting && (
                  <Card className="border-primary/30 bg-primary/5 p-4">
                    <div className="flex flex-wrap items-start justify-between gap-4">
                      <div className="space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <p className="font-medium text-foreground">Dry run</p>
                          <Badge variant={sourceVariant(automationDryRunSetting.source)}>{automationDryRunSetting.source}</Badge>
                          {dirtyKeys.has(automationDryRunSetting.key) && <Badge variant="outline">Unsaved</Badge>}
                        </div>
                        <p className="font-mono text-xs text-muted-foreground">MLFLOW_AUTOMATION_DRY_RUN</p>
                        <p className="text-sm text-muted-foreground">When enabled, automation validates and prepares the bundle but does not submit a real Kaggle job.</p>
                        <p className="text-xs text-muted-foreground">
                          Effective: <b>{automationDryRunEffective ? "On" : "Off"}</b>
                          {dirtyKeys.has(automationDryRunSetting.key) && <> · Will save: <b>{automationDryRunDraft ? "On" : "Off"}</b></>}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-foreground">{automationDryRunDraft ? "On" : "Off"}</span>
                        <Switch
                          checked={automationDryRunDraft}
                          onCheckedChange={(checked) => markDraft(automationDryRunSetting.key, checked)}
                          disabled={saving}
                          aria-label="Toggle automation dry run"
                        />
                      </div>
                    </div>
                  </Card>
                )}
              </>
            )}

            {group.id === "mlflow_dataset" && (
              <Card className="border-destructive/40 bg-destructive/5 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="font-medium text-destructive">Vùng dữ liệu nguy hiểm</p>
                    <p className="text-sm text-muted-foreground">
                      Clear all sẽ xóa toàn bộ batch, review item, bundle metadata và Kaggle run MLflow; không thể hoàn tác.
                    </p>
                  </div>
                  <Button type="button" variant="destructive" onClick={() => void clearAllMlflow()} disabled={loading || saving}>
                    Clear all MLflow
                  </Button>
                </div>
              </Card>
            )}

            <div className="grid gap-3">
              {group.settings.filter((setting) => setting.key !== "MLFLOW_AUTOMATION_DRY_RUN").map((setting) => (
                <Card key={setting.key} className="border bg-card p-4 shadow-sm">
                  <div className="grid gap-3 lg:grid-cols-[minmax(220px,0.8fr)_minmax(280px,1.2fr)] lg:items-start">
                    <div className="space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-medium text-foreground">{setting.label}</span>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              className="inline-flex rounded-sm text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
                              aria-label={`Giải thích ${setting.key}`}
                            >
                              <CircleHelp className="h-4 w-4" />
                            </button>
                          </TooltipTrigger>
                          <TooltipContent side="right" sideOffset={8}>
                            {setting.description || `Cấu hình runtime cho ${setting.key}.`}
                          </TooltipContent>
                        </Tooltip>
                        {setting.required && <Badge variant="outline">required</Badge>}
                        {setting.secret && <Badge variant="secondary">secret</Badge>}
                        <Badge variant={sourceVariant(setting.source)}>{setting.source}</Badge>
                      </div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <button
                            type="button"
                            className="break-all rounded-sm font-mono text-left text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline focus-visible:ring-2 focus-visible:ring-ring"
                          >
                            {setting.key}
                          </button>
                        </TooltipTrigger>
                        <TooltipContent side="right" sideOffset={8}>
                          {setting.description || `Cấu hình runtime cho ${setting.key}.`}
                        </TooltipContent>
                      </Tooltip>
                    </div>

                    <div className="space-y-2">
                      <div className="flex gap-2">
                        <div className="min-w-0 flex-1">{renderInput(setting)}</div>
                        {setting.secret && (
                          <Button
                            type="button"
                            variant="outline"
                            onClick={() => toggleSecretVisibility(setting)}
                            disabled={saving}
                            title={revealedKeys.has(setting.key) ? "Hide secret value" : "Reveal secret value"}
                            aria-label={revealedKeys.has(setting.key) ? "Hide secret value" : "Reveal secret value"}
                          >
                            {revealedKeys.has(setting.key) ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                          </Button>
                        )}
                        <Button
                          type="button"
                          variant="outline"
                          onClick={() => resetSetting(setting.key)}
                          disabled={saving}
                          title="Reset to env/default (remove DB override)"
                          aria-label="Reset to env/default (remove DB override)"
                        >
                          <RotateCcw className="h-4 w-4" />
                        </Button>
                      </div>
                      {setting.secret && setting.has_value && !revealedKeys.has(setting.key) && (
                        <p className="text-xs text-muted-foreground">Current value: {setting.masked_value}</p>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
