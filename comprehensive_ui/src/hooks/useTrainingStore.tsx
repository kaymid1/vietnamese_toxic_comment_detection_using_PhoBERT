import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type PhaseStatus = "not_started" | "in_progress" | "done";

export interface TrainingTaskDefinition {
  id: string;
  label: string;
  param?: string | null;
  sort_order: number;
  checked: boolean;
}

export interface TrainingTaskGroupDefinition {
  id: string;
  title: string;
  sort_order: number;
  tasks: TrainingTaskDefinition[];
}

export interface TrainingPhaseDefinition {
  id: string;
  title: string;
  sort_order: number;
  tasks: TrainingTaskDefinition[];
  groups: TrainingTaskGroupDefinition[];
}

export interface TrainingResultRecord {
  id: string;
  scenario_name: string;
  macro_f1: number;
  f1_toxic: number;
  precision_toxic: number;
  recall_toxic: number;
  val_loss: number | null;
  best_threshold_macro_f1: number | null;
  best_threshold_f1_toxic: number | null;
  notes: string;
  created_at: string;
}

export interface TrainingResultInput {
  scenario_name: string;
  phase_id?: string | null;
  macro_f1: number;
  f1_toxic: number;
  precision_toxic: number;
  recall_toxic: number;
  val_loss: number | null;
  best_threshold_macro_f1: number | null;
  best_threshold_f1_toxic: number | null;
  notes: string;
}

interface TrainingTrackerPayload {
  phases: TrainingPhaseDefinition[];
  results: TrainingResultRecord[];
}

interface TrainingStoreValue {
  phases: TrainingPhaseDefinition[];
  results: TrainingResultRecord[];
  loading: boolean;
  error: string | null;
  phaseOptions: Array<{ value: string; label: string }>;
  refresh: () => Promise<void>;
  getPhaseProgress: (phaseId: string) => { checkedCount: number; totalCount: number; status: PhaseStatus };
  toggleTask: (taskId: string, checked: boolean) => Promise<void>;
  createPhase: (title: string) => Promise<void>;
  updatePhase: (phaseId: string, title: string) => Promise<void>;
  deletePhase: (phaseId: string) => Promise<void>;
  reorderPhases: (phaseIds: string[]) => Promise<void>;
  createGroup: (phaseId: string, title: string) => Promise<void>;
  updateGroup: (groupId: string, title: string) => Promise<void>;
  deleteGroup: (groupId: string) => Promise<void>;
  reorderGroups: (phaseId: string, groupIds: string[]) => Promise<void>;
  createTask: (phaseId: string, groupId: string | null, label: string, param?: string) => Promise<void>;
  updateTask: (taskId: string, label: string, param?: string) => Promise<void>;
  deleteTask: (taskId: string) => Promise<void>;
  reorderTasks: (phaseId: string, groupId: string | null, taskIds: string[]) => Promise<void>;
  addResult: (input: TrainingResultInput) => Promise<void>;
  deleteResult: (id: string) => Promise<void>;
  exportResultsJson: () => void;
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

const emptyPayload: TrainingTrackerPayload = {
  phases: [],
  results: [],
};

const TrainingStoreContext = createContext<TrainingStoreValue | null>(null);

async function requestPayload(path: string, init?: RequestInit): Promise<TrainingTrackerPayload> {
  const response = await fetch(buildApiUrl(path), {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    ...init,
  });

  const rawText = await response.text();
  let body: unknown = null;
  if (rawText) {
    try {
      body = JSON.parse(rawText);
    } catch {
      body = rawText;
    }
  }

  if (!response.ok) {
    let detail = `${response.status}`;
    if (body && typeof body === "object" && "detail" in body) {
      const apiDetail = (body as { detail: unknown }).detail;
      if (typeof apiDetail === "string") {
        detail = apiDetail;
      } else if (apiDetail !== undefined && apiDetail !== null) {
        detail = JSON.stringify(apiDetail);
      }
    }
    throw new Error(detail);
  }

  if (!body || typeof body !== "object") {
    return emptyPayload;
  }

  const payload = body as Partial<TrainingTrackerPayload>;
  return {
    phases: Array.isArray(payload.phases) ? (payload.phases as TrainingPhaseDefinition[]) : [],
    results: Array.isArray(payload.results) ? (payload.results as TrainingResultRecord[]) : [],
  };
}

function getTasksForPhase(phase: TrainingPhaseDefinition): TrainingTaskDefinition[] {
  return [...phase.tasks, ...phase.groups.flatMap((group) => group.tasks)];
}

export function TrainingStoreProvider({ children }: { children: ReactNode }) {
  const [payload, setPayload] = useState<TrainingTrackerPayload>(emptyPayload);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await requestPayload("/api/training-tracker", { method: "GET" });
      setPayload(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load training tracker");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const mutate = useCallback(async (path: string, init: RequestInit) => {
    setError(null);
    try {
      const next = await requestPayload(path, init);
      setPayload(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training tracker request failed");
      throw err;
    }
  }, []);

  const getPhaseProgress = useCallback(
    (phaseId: string) => {
      const phase = payload.phases.find((item) => item.id === phaseId);
      if (!phase) {
        return { checkedCount: 0, totalCount: 0, status: "not_started" as PhaseStatus };
      }
      const tasks = getTasksForPhase(phase);
      const checkedCount = tasks.filter((task) => task.checked).length;
      const totalCount = tasks.length;
      let status: PhaseStatus = "not_started";
      if (totalCount > 0 && checkedCount === totalCount) {
        status = "done";
      } else if (checkedCount > 0) {
        status = "in_progress";
      }
      return { checkedCount, totalCount, status };
    },
    [payload.phases],
  );

  const createPhase = useCallback(
    async (title: string) => {
      await mutate("/api/training-tracker/phases", {
        method: "POST",
        body: JSON.stringify({ title }),
      });
    },
    [mutate],
  );

  const updatePhase = useCallback(
    async (phaseId: string, title: string) => {
      await mutate(`/api/training-tracker/phases/${phaseId}`, {
        method: "PATCH",
        body: JSON.stringify({ title }),
      });
    },
    [mutate],
  );

  const deletePhase = useCallback(
    async (phaseId: string) => {
      await mutate(`/api/training-tracker/phases/${phaseId}`, { method: "DELETE" });
    },
    [mutate],
  );

  const reorderPhases = useCallback(
    async (phaseIds: string[]) => {
      if (!phaseIds.length) return;
      await mutate("/api/training-tracker/phases/reorder", {
        method: "POST",
        body: JSON.stringify({ phase_ids: phaseIds }),
      });
    },
    [mutate],
  );
  const createGroup = useCallback(
    async (phaseId: string, title: string) => {
      await mutate("/api/training-tracker/groups", {
        method: "POST",
        body: JSON.stringify({ phase_id: phaseId, title }),
      });
    },
    [mutate],
  );

  const updateGroup = useCallback(
    async (groupId: string, title: string) => {
      await mutate(`/api/training-tracker/groups/${groupId}`, {
        method: "PATCH",
        body: JSON.stringify({ title }),
      });
    },
    [mutate],
  );

  const deleteGroup = useCallback(
    async (groupId: string) => {
      await mutate(`/api/training-tracker/groups/${groupId}`, { method: "DELETE" });
    },
    [mutate],
  );

  const reorderGroups = useCallback(
    async (phaseId: string, groupIds: string[]) => {
      if (!groupIds.length) return;
      await mutate("/api/training-tracker/groups/reorder", {
        method: "POST",
        body: JSON.stringify({ phase_id: phaseId, group_ids: groupIds }),
      });
    },
    [mutate],
  );
  const createTask = useCallback(
    async (phaseId: string, groupId: string | null, label: string, param?: string) => {
      await mutate("/api/training-tracker/tasks", {
        method: "POST",
        body: JSON.stringify({ phase_id: phaseId, group_id: groupId, label, param: param || null }),
      });
    },
    [mutate],
  );

  const updateTask = useCallback(
    async (taskId: string, label: string, param?: string) => {
      await mutate(`/api/training-tracker/tasks/${taskId}`, {
        method: "PATCH",
        body: JSON.stringify({ label, param: param || null }),
      });
    },
    [mutate],
  );

  const deleteTask = useCallback(
    async (taskId: string) => {
      await mutate(`/api/training-tracker/tasks/${taskId}`, { method: "DELETE" });
    },
    [mutate],
  );

  const reorderTasks = useCallback(
    async (phaseId: string, groupId: string | null, taskIds: string[]) => {
      if (!taskIds.length) return;
      await mutate("/api/training-tracker/tasks/reorder", {
        method: "POST",
        body: JSON.stringify({ phase_id: phaseId, group_id: groupId, task_ids: taskIds }),
      });
    },
    [mutate],
  );
  const toggleTask = useCallback(
    async (taskId: string, checked: boolean) => {
      await mutate(`/api/training-tracker/tasks/${taskId}/check`, {
        method: "POST",
        body: JSON.stringify({ checked }),
      });
    },
    [mutate],
  );

  const addResult = useCallback(
    async (input: TrainingResultInput) => {
      await mutate("/api/training-tracker/results", {
        method: "POST",
        body: JSON.stringify(input),
      });
    },
    [mutate],
  );

  const deleteResult = useCallback(
    async (id: string) => {
      await mutate(`/api/training-tracker/results/${id}`, { method: "DELETE" });
    },
    [mutate],
  );

  const exportResultsJson = useCallback(() => {
    const now = new Date();
    const two = (value: number) => String(value).padStart(2, "0");
    const fileName = `training_results_${now.getFullYear()}${two(now.getMonth() + 1)}${two(now.getDate())}_${two(now.getHours())}${two(now.getMinutes())}${two(now.getSeconds())}.json`;

    const blob = new Blob([JSON.stringify(payload.results, null, 2)], { type: "application/json" });
    const url = window.URL.createObjectURL(blob);
    const link = window.document.createElement("a");
    link.href = url;
    link.download = fileName;
    link.click();
    window.URL.revokeObjectURL(url);
  }, [payload.results]);

  const phaseOptions = useMemo(
    () => payload.phases.map((phase) => ({ value: phase.id, label: phase.title })),
    [payload.phases],
  );

  const value: TrainingStoreValue = {
    phases: payload.phases,
    results: payload.results,
    loading,
    error,
    phaseOptions,
    refresh,
    getPhaseProgress,
    toggleTask,
    createPhase,
    updatePhase,
    deletePhase,
    reorderPhases,
    createGroup,
    updateGroup,
    deleteGroup,
    reorderGroups,
    createTask,
    updateTask,
    deleteTask,
    reorderTasks,
    addResult,
    deleteResult,
    exportResultsJson,
  };

  return <TrainingStoreContext.Provider value={value}>{children}</TrainingStoreContext.Provider>;
}

export const useTrainingStore = () => {
  const context = useContext(TrainingStoreContext);
  if (!context) {
    throw new Error("useTrainingStore must be used within TrainingStoreProvider");
  }
  return context;
};
