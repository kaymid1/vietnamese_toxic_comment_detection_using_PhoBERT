import { useMemo, useState } from "react";
import { Button } from "../app/components/ui/button";
import { Card } from "../app/components/ui/card";
import { useTrainingStore, type TrainingResultInput, type TrainingResultRecord } from "../hooks/useTrainingStore";

const formatMetric = (value: number) => value.toFixed(3);

const getMetricClass = (value: number): string => {
  if (value > 0.8) {
    return "text-emerald-600 dark:text-emerald-400";
  }
  if (value >= 0.6) {
    return "text-amber-600 dark:text-amber-400";
  }
  return "text-red-600 dark:text-red-400";
};

const parseMetricInput = (value: string): number | null | "invalid" => {
  if (!value.trim()) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : "invalid";
};

const toInputString = (value: number | null): string => {
  if (value === null || Number.isNaN(value)) {
    return "";
  }
  return `${value}`;
};

const parseOptionalNumber = (value: string): number | null => {
  if (!value.trim()) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const formatTimestamp = (iso: string) => {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return date.toLocaleString();
};

const renderOptionalMetric = (label: string, value: number | null) => {
  if (value === null) {
    return (
      <div className="rounded border border-dashed px-3 py-2 text-sm text-muted-foreground">
        {label}: N/A
      </div>
    );
  }

  return (
    <div className="rounded border px-3 py-2 text-sm">
      {label}: <span className={getMetricClass(value)}>{formatMetric(value)}</span>
    </div>
  );
};

interface FormState {
  scenarioName: string;
  macroF1: string;
  f1Toxic: string;
  precisionToxic: string;
  recallToxic: string;
  valLoss: string;
  bestThresholdMacroF1: string;
  bestThresholdF1Toxic: string;
  notes: string;
}

const createInitialFormState = (): FormState => ({
  scenarioName: "",
  macroF1: "",
  f1Toxic: "",
  precisionToxic: "",
  recallToxic: "",
  valLoss: "",
  bestThresholdMacroF1: "",
  bestThresholdF1Toxic: "",
  notes: "",
});

const SavedResultCard = ({ record, onDelete }: { record: TrainingResultRecord; onDelete: (id: string) => Promise<void> }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card className="border p-3">
      <div className="flex items-start justify-between gap-3">
        <button
          type="button"
          className="flex-1 text-left"
          onClick={() => setExpanded((previous) => !previous)}
          aria-expanded={expanded}
        >
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <h4 className="text-base" style={{ color: "var(--primary)" }}>
              {record.scenario_name}
            </h4>
            <span className="text-xs text-muted-foreground">{formatTimestamp(record.created_at)}</span>
            <span className="rounded border px-2 py-0.5 text-xs">
              macro_f1: <span className={getMetricClass(record.macro_f1)}>{formatMetric(record.macro_f1)}</span>
            </span>
            <span className="rounded border px-2 py-0.5 text-xs">
              f1_toxic: <span className={getMetricClass(record.f1_toxic)}>{formatMetric(record.f1_toxic)}</span>
            </span>
            <span className="rounded border px-2 py-0.5 text-xs">
              precision: <span className={getMetricClass(record.precision_toxic)}>{formatMetric(record.precision_toxic)}</span>
            </span>
            <span className="rounded border px-2 py-0.5 text-xs">
              recall: <span className={getMetricClass(record.recall_toxic)}>{formatMetric(record.recall_toxic)}</span>
            </span>
            <span className="text-xs text-muted-foreground">{expanded ? "Thu gọn" : "Mở chi tiết"}</span>
          </div>
        </button>

        <Button variant="outline" size="sm" onClick={() => void onDelete(record.id)}>
          Xoá
        </Button>
      </div>

      {expanded && (
        <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-2">
          {renderOptionalMetric("val_loss", record.val_loss)}
          {renderOptionalMetric("best_threshold_macro_f1", record.best_threshold_macro_f1)}
          {renderOptionalMetric("best_threshold_f1_toxic", record.best_threshold_f1_toxic)}
          {record.notes.trim() && (
            <div className="md:col-span-3 rounded border border-dashed p-3 text-sm text-foreground whitespace-pre-wrap">
              <span className="text-muted-foreground">Notes / config snapshot:</span>
              <div className="mt-1">{record.notes}</div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
};

export function TrainingResults() {
  const { results, addResult, deleteResult, exportResultsJson, loading, phaseOptions } = useTrainingStore();

  const [form, setForm] = useState<FormState>(() => createInitialFormState());
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const canSave = useMemo(() => {
    return form.scenarioName.trim().length > 0;
  }, [form.scenarioName]);

  const updateField = (field: keyof FormState, value: string) => {
    setForm((previous) => ({ ...previous, [field]: value }));
  };

  const handleSave = async () => {
    if (!canSave) {
      setErrorMessage("Vui lòng nhập Tên kịch bản.");
      return;
    }

    const parsedMacro = parseMetricInput(form.macroF1);
    const parsedF1Toxic = parseMetricInput(form.f1Toxic);
    const parsedPrecision = parseMetricInput(form.precisionToxic);
    const parsedRecall = parseMetricInput(form.recallToxic);

    if ([parsedMacro, parsedF1Toxic, parsedPrecision, parsedRecall].some((value) => value === "invalid")) {
      setErrorMessage("Metric phải là số hợp lệ.");
      return;
    }

    const payload: TrainingResultInput = {
      scenario_name: form.scenarioName.trim(),
      phase_id: phaseOptions[0]?.value ?? "manual",
      macro_f1: parsedMacro === null ? 0 : parsedMacro === "invalid" ? 0 : parsedMacro,
      f1_toxic: parsedF1Toxic === null ? 0 : parsedF1Toxic === "invalid" ? 0 : parsedF1Toxic,
      precision_toxic: parsedPrecision === null ? 0 : parsedPrecision === "invalid" ? 0 : parsedPrecision,
      recall_toxic: parsedRecall === null ? 0 : parsedRecall === "invalid" ? 0 : parsedRecall,
      val_loss: parseOptionalNumber(form.valLoss),
      best_threshold_macro_f1: parseOptionalNumber(form.bestThresholdMacroF1),
      best_threshold_f1_toxic: parseOptionalNumber(form.bestThresholdF1Toxic),
      notes: form.notes.trim(),
    };

    try {
      await addResult(payload);
      setErrorMessage(null);
      setForm(createInitialFormState());
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Lưu thất bại.");
    }
  };

  return (
    <div className="space-y-4">
      <Card className="border p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="text-lg" style={{ color: "var(--primary)" }}>
            Lưu kết quả kịch bản
          </h3>
          <Button variant="outline" onClick={exportResultsJson} disabled={!results.length}>
            Export JSON
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className="text-sm text-muted-foreground">
            Tên kịch bản
            <input
              className="mt-1 w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground"
              value={form.scenarioName}
              onChange={(event) => updateField("scenarioName", event.target.value)}
              placeholder="LoRA r=16, LR=5e-5, gamma=2.0"
            />
          </label>

          <label className="text-sm text-muted-foreground">
            macro_f1
            <input
              type="number"
              step="0.001"
              className="mt-1 w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground"
              value={form.macroF1}
              onChange={(event) => updateField("macroF1", event.target.value)}
            />
          </label>

          <label className="text-sm text-muted-foreground">
            f1_toxic
            <input
              type="number"
              step="0.001"
              className="mt-1 w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground"
              value={form.f1Toxic}
              onChange={(event) => updateField("f1Toxic", event.target.value)}
            />
          </label>

          <label className="text-sm text-muted-foreground">
            precision_toxic
            <input
              type="number"
              step="0.001"
              className="mt-1 w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground"
              value={form.precisionToxic}
              onChange={(event) => updateField("precisionToxic", event.target.value)}
            />
          </label>

          <label className="text-sm text-muted-foreground">
            recall_toxic
            <input
              type="number"
              step="0.001"
              className="mt-1 w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground"
              value={form.recallToxic}
              onChange={(event) => updateField("recallToxic", event.target.value)}
            />
          </label>

          <label className="text-sm text-muted-foreground">
            val_loss (optional)
            <input
              type="number"
              step="0.001"
              className="mt-1 w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground"
              value={toInputString(parseOptionalNumber(form.valLoss))}
              onChange={(event) => updateField("valLoss", event.target.value)}
            />
          </label>

          <label className="text-sm text-muted-foreground">
            best_threshold_macro_f1 (optional)
            <input
              type="number"
              step="0.001"
              className="mt-1 w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground"
              value={toInputString(parseOptionalNumber(form.bestThresholdMacroF1))}
              onChange={(event) => updateField("bestThresholdMacroF1", event.target.value)}
            />
          </label>

          <label className="text-sm text-muted-foreground md:col-span-2">
            best_threshold_f1_toxic (optional)
            <input
              type="number"
              step="0.001"
              className="mt-1 w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground"
              value={toInputString(parseOptionalNumber(form.bestThresholdF1Toxic))}
              onChange={(event) => updateField("bestThresholdF1Toxic", event.target.value)}
            />
          </label>

          <label className="text-sm text-muted-foreground md:col-span-2">
            Notes / config snapshot
            <textarea
              rows={4}
              className="mt-1 w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground"
              value={form.notes}
              onChange={(event) => updateField("notes", event.target.value)}
              placeholder="Free text"
            />
          </label>
        </div>

        {errorMessage && <p className="text-sm text-destructive mt-3">{errorMessage}</p>}

        <div className="mt-4">
          <Button onClick={() => void handleSave()} disabled={!canSave || loading}>
            Lưu kịch bản
          </Button>
        </div>
      </Card>

      <Card className="border p-5">
        <h3 className="text-lg mb-4" style={{ color: "var(--primary)" }}>
          Kết quả đã lưu
        </h3>

        {results.length === 0 && <p className="text-sm text-muted-foreground">Chưa có kết quả nào được lưu.</p>}

        <div className="space-y-3">
          {results.map((record) => (
            <SavedResultCard key={record.id} record={record} onDelete={deleteResult} />
          ))}
        </div>
      </Card>
    </div>
  );
}
