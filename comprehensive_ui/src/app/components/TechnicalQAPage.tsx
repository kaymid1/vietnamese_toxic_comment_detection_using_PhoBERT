import { useState, type ReactNode } from "react";
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Calculator,
  CheckCircle2,
  GitBranch,
  Image as ImageIcon,
  Info,
  ListChecks,
  ShieldAlert,
  SlidersHorizontal,
} from "lucide-react";
import { Badge } from "@/app/components/ui/badge";
import { Card } from "@/app/components/ui/card";
import { useI18n } from "@/app/i18n/context";

const EvidenceBadge = ({ children, className = "" }: { children: ReactNode; className?: string }) => (
  <Badge variant="outline" className={`bg-background-secondary ${className}`}>
    {children}
  </Badge>
);

const SectionIcon = ({ children }: { children: ReactNode }) => (
  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-background-info text-text-info">
    {children}
  </div>
);

const THRESHOLD_CHART_ASSETS = {
  macroF1: "/qa/threshold/threshold_vs_macro_f1.png",
  precisionRecall: "/qa/threshold/threshold_vs_precision_recall.png",
  optional: ["/qa/threshold/threshold_vs_f1_toxic.png", "/qa/threshold/threshold_vs_fpr_fnr.png"],
} as const;

export function TechnicalQAPage() {
  const { t } = useI18n();

  return (
    <div className="dashboard-page">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="mb-8 text-center">
          <div className="mb-3 flex items-center justify-center gap-2">
            <Badge className="bg-background-info text-text-info">{t("technicalQa.referenceBadge")}</Badge>
          </div>
          <h1 className="mb-3 text-4xl text-primary">{t("technicalQa.title")}</h1>
          <p className="mb-3 text-xl text-muted-foreground">{t("technicalQa.subtitle")}</p>
          <p className="mx-auto max-w-3xl text-sm leading-6 text-muted-foreground">{t("technicalQa.intro")}</p>
        </header>

        <Card className="bg-card p-5 shadow-lg md:p-6">
          <div className="flex gap-3">
            <SectionIcon><SlidersHorizontal className="h-5 w-5" /></SectionIcon>
            <div className="min-w-0 flex-1">
              <QHeader number="Q1" question={t("technicalQa.q1.question")} />
              <p className="mt-3 leading-6 text-foreground">{t("technicalQa.q1.answer")}</p>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <Takeaway tone="info" label={t("technicalQa.q1.lowerLabel")} text={t("technicalQa.q1.lowerText")} />
                <Takeaway tone="warning" label={t("technicalQa.q1.higherLabel")} text={t("technicalQa.q1.higherText")} />
              </div>
            </div>
          </div>
        </Card>

        <Card className="bg-card p-5 shadow-lg md:p-6">
          <div className="flex gap-3">
            <SectionIcon><BarChart3 className="h-5 w-5" /></SectionIcon>
            <div className="min-w-0 flex-1">
              <QHeader number="Q2" question={t("technicalQa.q2.question")} />
              <p className="mt-3 leading-6 text-foreground">{t("technicalQa.q2.answer")}</p>

              <div className="mt-5 grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(280px,0.85fr)]">
                <div className="rounded-xl border border-border bg-background-secondary p-4">
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                    <h3 className="font-semibold text-primary">{t("technicalQa.q2.evidenceTitle")}</h3>
                    <EvidenceBadge>{t("technicalQa.badges.validationOptimized")}</EvidenceBadge>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <EvidenceStat label={t("technicalQa.q2.searchRange")} value="0.05 – 0.95" />
                    <EvidenceStat label={t("technicalQa.q2.step")} value="0.01" />
                    <EvidenceStat label={t("technicalQa.q2.objective")} value="Macro-F1" />
                    <EvidenceStat label={t("technicalQa.q2.bestThreshold")} value="0.49" />
                    <EvidenceStat label={t("technicalQa.q2.validationMacro")} value="0.7423" />
                    <EvidenceStat label={t("technicalQa.q2.toxicF1")} value="0.5495" />
                  </div>
                  <p className="mt-4 text-sm leading-6 text-muted-foreground">{t("technicalQa.q2.explanation")}</p>
                  <p className="mt-3 flex gap-2 border-t border-border pt-3 text-xs leading-5 text-muted-foreground">
                    <Info className="mt-0.5 h-4 w-4 shrink-0 text-text-info" />
                    <span>{t("technicalQa.q2.provenanceNote")}</span>
                  </p>
                </div>

                <div className="rounded-xl border border-border bg-background-secondary p-4">
                  <div className="mb-3 flex items-center gap-2">
                    <ImageIcon className="h-4 w-4 text-text-info" />
                    <h3 className="font-semibold text-primary">{t("technicalQa.q2.chart.title")}</h3>
                  </div>
                  <ThresholdChart
                    src={THRESHOLD_CHART_ASSETS.macroF1}
                    alt={t("technicalQa.q2.chart.alt")}
                    placeholderTitle={t("technicalQa.q2.chart.placeholderTitle")}
                    pending={t("technicalQa.q2.chart.pending")}
                  />
                </div>
              </div>

              <HistoricalThresholdTable title={t("technicalQa.table.title")} interpretation={t("technicalQa.table.interpretation")} t={t} />
            </div>
          </div>
        </Card>

        <Card className="bg-card p-5 shadow-lg md:p-6">
          <div className="flex gap-3">
            <SectionIcon><BarChart3 className="h-5 w-5" /></SectionIcon>
            <div className="min-w-0 flex-1">
              <QHeader number="Q3" question={t("technicalQa.q3.question")} />
              <p className="mt-3 leading-6 text-foreground">{t("technicalQa.q3.answer")}</p>
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <ComparisonCard title={t("technicalQa.q3.historicalTitle")} value="0.7423" details={t("technicalQa.q3.historicalDetails")} badge={t("technicalQa.badges.historicalExample")} />
                <ComparisonCard title={t("technicalQa.q3.controlledTitle")} value="0.7477" details={t("technicalQa.q3.controlledDetails")} badge={t("technicalQa.badges.controlledEvaluation")} />
              </div>
              <p className="mt-4 rounded-lg border border-border bg-background-secondary p-3 text-sm text-muted-foreground">{t("technicalQa.q3.note")}</p>
            </div>
          </div>
        </Card>

        <Card className="bg-card p-5 shadow-lg md:p-6">
          <div className="flex gap-3">
            <SectionIcon><GitBranch className="h-5 w-5" /></SectionIcon>
            <div className="min-w-0 flex-1">
              <QHeader number="Q4" question={t("technicalQa.q4.question")} />
              <p className="mt-3 leading-6 text-foreground">{t("technicalQa.q4.answer")}</p>
              <div className="mt-5 flex flex-col items-stretch gap-2 text-sm md:flex-row md:items-center md:justify-between">
                {["training", "checkpoint", "validation", "sweep", "operating"].map((key, index, items) => (
                  <div key={key} className="flex items-center gap-2">
                    <span className="rounded-lg border border-border bg-background-secondary px-3 py-2 font-medium">{t(`technicalQa.q4.flow.${key}`)}</span>
                    {index < items.length - 1 && <ArrowRight className="hidden h-4 w-4 text-muted-foreground md:block" />}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>

        <Card className="bg-card p-5 shadow-lg md:p-6">
          <div className="flex gap-3">
            <SectionIcon><ShieldAlert className="h-5 w-5" /></SectionIcon>
            <div className="min-w-0 flex-1">
              <QHeader number="Q5" question={t("technicalQa.q5.question")} />
              <p className="mt-3 leading-6 text-foreground">{t("technicalQa.q5.answer")}</p>
              <p className="mt-3 text-sm font-medium text-foreground">{t("technicalQa.q5.tradeoffNote")}</p>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">{t("technicalQa.q5.evidenceNote")}</p>
              <div className="mt-4 flex flex-wrap items-center gap-2">
                <Badge className="bg-background-warning text-text-warning">{t("technicalQa.q5.runtimePolicyBadge")}</Badge>
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {[
                  ["news", "0.72"],
                  ["social", "0.50"],
                  ["forum", "0.60"],
                  ["unknown", "0.62"],
                ].map(([key, value]) => <EvidenceStat key={key} label={t(`technicalQa.q5.${key}`)} value={value} />)}
              </div>
              <p className="mt-4 text-sm text-muted-foreground"><b className="text-foreground">{t("technicalQa.q5.safetyRangeLabel")}:</b> 0.40 – 0.85</p>
              <div className="mt-5">
                <HistoricalThresholdTable title={t("technicalQa.q5.comparisonTitle")} interpretation={t("technicalQa.q5.comparisonInterpretation")} t={t} />
              </div>
              <div className="mt-5 rounded-xl border border-border bg-background-secondary p-4">
                <div className="mb-3 flex items-center gap-2">
                  <ImageIcon className="h-4 w-4 text-text-info" />
                  <h3 className="font-semibold text-primary">{t("technicalQa.q5.chart.title")}</h3>
                </div>
                <ThresholdChart
                  src={THRESHOLD_CHART_ASSETS.precisionRecall}
                  alt={t("technicalQa.q5.chart.alt")}
                  placeholderTitle={t("technicalQa.q5.chart.placeholderTitle")}
                  pending={t("technicalQa.q5.chart.pending")}
                />
              </div>
              <div className="mt-5 rounded-xl border border-primary/30 bg-background-info/40 p-4">
                <p className="font-semibold text-primary">{t("technicalQa.q5.takeawayTitle")}</p>
                <p className="mt-2 text-sm leading-6 text-foreground">{t("technicalQa.q5.takeaway")}</p>
              </div>
            </div>
          </div>
        </Card>

        <Card className="bg-card p-5 shadow-lg md:p-6">
          <div className="flex gap-3">
            <SectionIcon><ListChecks className="h-5 w-5" /></SectionIcon>
            <div className="min-w-0 flex-1">
              <QHeader number="Q6" question={t("technicalQa.q6.question")} />
              <p className="mt-3 leading-6 text-foreground">{t("technicalQa.q6.answer")}</p>
              <div className="mt-4 flex items-center gap-2 rounded-lg border border-border bg-background-secondary p-3 text-sm font-medium">
                <Info className="h-4 w-4 shrink-0 text-text-info" />
                {t("technicalQa.q6.note")}
              </div>
            </div>
          </div>
        </Card>

        <Card className="bg-card p-5 shadow-lg md:p-6">
          <div className="flex gap-3">
            <SectionIcon><Calculator className="h-5 w-5" /></SectionIcon>
            <div className="min-w-0 flex-1">
              <QHeader number="Q7" question={t("technicalQa.q7.question")} />
              <p className="mt-3 leading-6 text-foreground">{t("technicalQa.q7.answer")}</p>
              <div className="mt-5 grid gap-3 sm:grid-cols-4 sm:items-center">
                <EvidenceStat label={t("technicalQa.q7.extracted")} value="20" />
                <EvidenceStat label={t("technicalQa.q7.toxic")} value="6" />
                <EvidenceStat label={t("technicalQa.q7.rate")} value="6 / 20 = 30%" />
                <EvidenceStat label={t("technicalQa.q7.aggregateThreshold")} value="25%" />
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-2 rounded-lg border border-border bg-background-secondary p-3 text-sm">
                <CheckCircle2 className="h-4 w-4 text-text-success" />
                <span>{t("technicalQa.q7.result")}</span>
                <Badge className="bg-background-success text-text-success">{t("technicalQa.q7.alert")}</Badge>
              </div>
              <p className="mt-3 text-xs text-muted-foreground">{t("technicalQa.q7.terms")}</p>
            </div>
          </div>
        </Card>

        <Card className="bg-card p-5 shadow-lg md:p-6">
          <div className="flex gap-3">
            <SectionIcon><AlertTriangle className="h-5 w-5" /></SectionIcon>
            <div className="min-w-0 flex-1">
              <QHeader number="Q8" question={t("technicalQa.q8.question")} />
              <p className="mt-3 leading-6 text-foreground">{t("technicalQa.q8.answer")}</p>
              <div className="mt-5 grid gap-3 md:grid-cols-3">
                <GateCard range="≤ 0.20" label={t("technicalQa.q8.cleanCandidate")} tone="success" />
                <GateCard range="0.20 – 0.80" label={t("technicalQa.q8.reviewCandidate")} tone="warning" />
                <GateCard range="≥ 0.80" label={t("technicalQa.q8.toxicCandidate")} tone="danger" />
              </div>
              <div className="mt-4">
                <EvidenceBadge>{t("technicalQa.badges.mlflowGate")}</EvidenceBadge>
              </div>
            </div>
          </div>
        </Card>

        <Card className="bg-card p-5 shadow-lg md:p-6">
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <Info className="h-5 w-5 text-text-info" />
            <h2 className="text-xl font-semibold text-primary">{t("technicalQa.distinctionTitle")}</h2>
          </div>
          <div className="grid gap-3 md:grid-cols-4">
            {[
              ["validationOptimized", "technicalQa.badges.validationOptimized"],
              ["runtimePolicy", "technicalQa.badges.runtimePolicy"],
              ["urlAggregate", "technicalQa.badges.urlAggregate"],
              ["mlflowGate", "technicalQa.badges.mlflowGate"],
            ].map(([key, labelKey]) => <div key={key} className="rounded-lg border border-border bg-background-secondary p-3 text-sm"><Badge variant="outline" className="mb-2">{t(labelKey)}</Badge><p className="text-muted-foreground">{t(`technicalQa.distinctions.${key}`)}</p></div>)}
          </div>
        </Card>

        <details className="rounded-xl border border-border bg-card p-5 shadow-lg md:p-6">
          <summary className="cursor-pointer font-semibold text-primary">{t("technicalQa.technicalDetails.title")}</summary>
          <ul className="mt-4 list-disc space-y-2 pl-5 text-sm leading-6 text-muted-foreground">
            {(["sweep", "range", "objective", "temperature", "rawThreshold", "registry", "artifact", "futureEvaluation"] as const).map((key) => <li key={key}>{t(`technicalQa.technicalDetails.${key}`)}</li>)}
          </ul>
        </details>
      </div>
    </div>
  );
}

function QHeader({ number, question }: { number: string; question: string }) {
  return (
    <div className="flex flex-wrap items-start gap-2">
      <Badge variant="outline" className="font-semibold">{number}</Badge>
      <h2 className="min-w-0 flex-1 text-xl font-semibold leading-7 text-primary">{question}</h2>
    </div>
  );
}

function EvidenceStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 font-semibold tabular-nums text-foreground">{value}</p>
    </div>
  );
}

function ThresholdChart({ src, alt, placeholderTitle, pending }: { src: string; alt: string; placeholderTitle: string; pending: string }) {
  const [available, setAvailable] = useState(true);
  const filename = src.split("/").pop() || src;
  return available ? (
    <img src={src} alt={alt} className="h-auto max-h-[320px] w-full rounded-lg border border-border object-contain" onError={() => setAvailable(false)} />
  ) : (
    <div className="flex min-h-[250px] flex-col items-center justify-center rounded-lg border border-dashed border-muted-foreground/40 bg-card px-5 text-center">
      <ImageIcon className="mb-3 h-8 w-8 text-muted-foreground" />
      <p className="font-medium text-foreground">{placeholderTitle}</p>
      <code className="mt-1 text-xs text-muted-foreground">{filename}</code>
      <p className="mt-3 text-sm text-muted-foreground">{pending}</p>
    </div>
  );
}

function HistoricalThresholdTable({
  title,
  interpretation,
  t,
}: {
  title: string;
  interpretation: string;
  t: (key: string) => string;
}) {
  return (
    <>
      <div className="overflow-x-auto rounded-xl border border-border">
        <table className="w-full min-w-[620px] text-sm">
          <caption className="border-b border-border bg-background-secondary px-4 py-3 text-left font-semibold text-primary">{title}</caption>
          <thead className="bg-background-secondary text-left text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              {["threshold", "precision", "recall", "fpr", "macroF1"].map((key) => <th key={key} className="px-4 py-3">{t(`technicalQa.table.${key}`)}</th>)}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {[
              ["0.49", "0.517", "0.586", "0.073", "0.742"],
              ["0.60", "0.553", "0.522", "0.057", "0.738"],
              ["0.72", "0.577", "0.418", "0.041", "0.713"],
            ].map((row) => <tr key={row[0]}>{row.map((value, index) => <td key={`${row[0]}-${index}`} className="px-4 py-3 tabular-nums">{value}</td>)}</tr>)}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-sm text-muted-foreground">{interpretation}</p>
    </>
  );
}

function Takeaway({ tone, label, text }: { tone: "info" | "warning"; label: string; text: string }) {
  return (
    <div className={`rounded-lg border p-3 ${tone === "info" ? "border-border bg-background-info/40" : "border-border bg-background-warning/40"}`}>
      <p className="font-semibold text-foreground">{label}</p>
      <p className="mt-1 text-sm text-muted-foreground">{text}</p>
    </div>
  );
}

function ComparisonCard({ title, value, details, badge }: { title: string; value: string; details: string; badge: string }) {
  return (
    <div className="rounded-xl border border-border bg-background-secondary p-4">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2"><h3 className="font-semibold text-primary">{title}</h3><EvidenceBadge>{badge}</EvidenceBadge></div>
      <p className="text-3xl font-semibold tabular-nums text-foreground">{value} <span className="text-sm font-normal text-muted-foreground">Macro-F1</span></p>
      <p className="mt-2 whitespace-pre-line text-sm leading-6 text-muted-foreground">{details}</p>
    </div>
  );
}

function GateCard({ range, label, tone }: { range: string; label: string; tone: "success" | "warning" | "danger" }) {
  const styles = {
    success: "bg-background-success text-text-success",
    warning: "bg-background-warning text-text-warning",
    danger: "bg-background-danger text-text-danger",
  };
  return <div className="rounded-lg border border-border bg-background-secondary p-4"><p className={`inline-flex rounded-md px-2 py-1 font-semibold tabular-nums ${styles[tone]}`}>{range}</p><p className="mt-3 text-sm text-muted-foreground">{label}</p></div>;
}
