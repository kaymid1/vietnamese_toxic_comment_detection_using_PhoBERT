import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CheckCircle2, LoaderCircle, X, XCircle } from "lucide-react";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Progress } from "@/app/components/ui/progress";

type ProgressState = "running" | "success" | "error";

interface ProgressItem {
  id: string;
  title: string;
  message?: string;
  value?: number;
  state: ProgressState;
}

interface ProgressUpdate {
  title?: string;
  message?: string;
  value?: number;
}

interface ProgressNotificationContextValue {
  start: (id: string, update: Required<Pick<ProgressUpdate, "title">> & Omit<ProgressUpdate, "title">) => void;
  update: (id: string, update: ProgressUpdate) => void;
  succeed: (id: string, update?: ProgressUpdate) => void;
  fail: (id: string, update?: ProgressUpdate) => void;
  dismiss: (id: string) => void;
}

const ProgressNotificationContext = createContext<ProgressNotificationContextValue | null>(null);
const AUTO_DISMISS_MS = 5_000;

export function ProgressNotificationProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ProgressItem[]>([]);
  const dismissTimers = useRef<Record<string, number>>({});

  const clearDismissTimer = useCallback((id: string) => {
    const timer = dismissTimers.current[id];
    if (timer) window.clearTimeout(timer);
    delete dismissTimers.current[id];
  }, []);

  const dismiss = useCallback(
    (id: string) => {
      clearDismissTimer(id);
      setItems((current) => current.filter((item) => item.id !== id));
    },
    [clearDismissTimer],
  );

  const scheduleDismiss = useCallback(
    (id: string) => {
      clearDismissTimer(id);
      dismissTimers.current[id] = window.setTimeout(() => dismiss(id), AUTO_DISMISS_MS);
    },
    [clearDismissTimer, dismiss],
  );

  const start = useCallback(
    (id: string, update: Required<Pick<ProgressUpdate, "title">> & Omit<ProgressUpdate, "title">) => {
      clearDismissTimer(id);
      setItems((current) => {
        const next: ProgressItem = { id, title: update.title, message: update.message, value: update.value, state: "running" };
        const existingIndex = current.findIndex((item) => item.id === id);
        return existingIndex < 0 ? [...current, next] : current.map((item) => (item.id === id ? next : item));
      });
    },
    [clearDismissTimer],
  );

  const update = useCallback((id: string, update: ProgressUpdate) => {
    setItems((current) => current.map((item) => (item.id === id ? { ...item, ...update } : item)));
  }, []);

  const settle = useCallback(
    (id: string, state: Extract<ProgressState, "success" | "error">, update?: ProgressUpdate) => {
      setItems((current) =>
        current.map((item) => {
          if (item.id !== id) return item;
          return { ...item, ...update, value: state === "success" ? 100 : item.value, state };
        }),
      );
      scheduleDismiss(id);
    },
    [scheduleDismiss],
  );

  useEffect(() => () => Object.values(dismissTimers.current).forEach((timer) => window.clearTimeout(timer)), []);

  const value = useMemo(
    () => ({ start, update, succeed: (id: string, update?: ProgressUpdate) => settle(id, "success", update), fail: (id: string, update?: ProgressUpdate) => settle(id, "error", update), dismiss }),
    [dismiss, settle, start, update],
  );

  return (
    <ProgressNotificationContext.Provider value={value}>
      {children}
      <div className="fixed bottom-4 right-4 z-[70] flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-2" aria-live="polite">
        {items.map((item) => (
          <Card key={item.id} className="border bg-background/95 p-3 shadow-xl backdrop-blur">
            <div className="flex items-start gap-2">
              {item.state === "running" ? <LoaderCircle className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-primary" /> : item.state === "success" ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" /> : <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />}
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">{item.title}</p>
                {item.message && <p className="mt-0.5 text-xs text-muted-foreground">{item.message}</p>}
                {item.state === "running" && <Progress value={item.value ?? 12} className="mt-2 h-1.5" />}
              </div>
              <Button type="button" variant="ghost" size="icon" className="-mr-1 -mt-1 h-7 w-7" onClick={() => dismiss(item.id)} aria-label={`Đóng thông báo ${item.title}`}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </ProgressNotificationContext.Provider>
  );
}

export function useProgressNotification(): ProgressNotificationContextValue {
  const context = useContext(ProgressNotificationContext);
  if (!context) throw new Error("useProgressNotification must be used inside ProgressNotificationProvider");
  return context;
}
