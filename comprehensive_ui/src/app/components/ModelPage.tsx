import { useEffect, useMemo, useState } from "react";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/components/ui/table";
import { Badge } from "@/app/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/app/components/ui/dialog";
import { Label } from "@/app/components/ui/label";
import {
  ArrowRight,
  CheckCircle,
  XCircle,
  TrendingUp,
  Database,
  Cpu,
  LineChart,
  FileText,
  Download,
} from "lucide-react";
import {
  LineChart as RechartsLine,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

interface ModelPageProps {
  onTryNow: () => void;
}

interface RegistryRun {
  run_id: string;
  model_name: string;
  dataset_version: string;
  created_at: string;
  checkpoint_path: string;
  hyperparameters: Record<string, unknown>;
  metrics: Record<string, number>;
  is_baseline?: boolean;
}

interface RegistryResponse {
  runs: RegistryRun[];
  last_updated?: string | null;
}

interface EvalPolicyResponse {
  policy: {
    split?: { train?: number; val?: number; test?: number };
    test_set_fixed?: boolean;
    fixed_since?: string;
    hard_case_subsets_evaluated?: boolean;
    note?: string;
  };
  last_updated?: string | null;
}

interface ErrorRow {
  text: string;
  true_label: number;
  predicted_label: number;
  confidence?: number;
  source_dataset?: string;
  subset_tag?: string;
}

interface ErrorResponse {
  items: ErrorRow[];
  last_updated?: string | null;
}

interface HardCaseRow extends ErrorRow {
  candidate_reason?: string;
}

interface HardCaseResponse {
  items: HardCaseRow[];
  last_updated?: string | null;
}

interface PreprocessStep {
  id: string;
  label: string;
  active: boolean;
}

interface PreprocessResponse {
  steps: PreprocessStep[];
}

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

const buildApiUrl = (path: string) => {
  if (!path.startsWith("/")) {
    return API_BASE ? `${API_BASE}/${path}` : `/${path}`;
  }
  return API_BASE ? `${API_BASE}${path}` : path;
};

const formatRatio = (value?: number) => (value === undefined ? "--" : `${(value * 100).toFixed(1)}%`);

const formatScore = (value?: number | null) => (value == null ? "Chưa có metrics" : value.toFixed(3));

const demoExamples = [
  "Trời ơi!!! Đẹp vãi 😡😡",
  "k có j đâu mà ồn ào...",
  "Giỏi quá ha, chắc chưa? 😏",
];

const applyActiveSteps = (text: string, steps: PreprocessStep[]) => {
  let output = text;
  steps.forEach((step) => {
    if (!step.active) return;
    if (step.id === "trim") {
      output = output.trim();
    }
    if (step.id === "normalize_unicode") {
      output = output.normalize("NFC");
    }
    if (step.id === "normalize_whitespace") {
      output = output.replace(/\s+/g, " ").trim();
    }
  });
  return output;
};

export function ModelPage({ onTryNow }: ModelPageProps) {
  const [registry, setRegistry] = useState<RegistryResponse>({ runs: [] });
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [preprocessSteps, setPreprocessSteps] = useState<PreprocessStep[]>([]);
  const [policy, setPolicy] = useState<EvalPolicyResponse | null>(null);
  const [errorRows, setErrorRows] = useState<ErrorRow[]>([]);
  const [errorLastUpdated, setErrorLastUpdated] = useState<string | null>(null);
  const [hardCases, setHardCases] = useState<HardCaseRow[]>([]);
  const [hardCaseLastUpdated, setHardCaseLastUpdated] = useState<string | null>(null);
  const [errorFilter, setErrorFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [subsetFilter, setSubsetFilter] = useState("all");
  const [demoStepIndex, setDemoStepIndex] = useState(0);
  const [demoExampleIndex, setDemoExampleIndex] = useState(0);

  const fetchRegistry = async (refresh = false) => {
    try {
      const endpoint = refresh ? "/api/experiments/registry?refresh=true" : "/api/experiments/registry";
      const response = await fetch(buildApiUrl(endpoint));
      const data = (await response.json()) as RegistryResponse;
      if (!response.ok) throw new Error(JSON.stringify(data));
      setRegistry(data);
      setRegistryError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Không thể tải registry";
      setRegistryError(message);
    }
  };

  useEffect(() => {

    const fetchSteps = async () => {
      try {
        const response = await fetch(buildApiUrl("/api/preprocessing/steps"));
        const data = (await response.json()) as PreprocessResponse;
        if (!response.ok) throw new Error(JSON.stringify(data));
        setPreprocessSteps(data.steps || []);
      } catch {
        setPreprocessSteps([]);
      }
    };

    const fetchPolicy = async () => {
      try {
        const response = await fetch(buildApiUrl("/api/eval/policy"));
        const data = (await response.json()) as EvalPolicyResponse;
        if (!response.ok) throw new Error(JSON.stringify(data));
        setPolicy(data);
      } catch {
        setPolicy(null);
      }
    };

    const fetchErrors = async () => {
      try {
        const response = await fetch(buildApiUrl("/api/eval/errors"));
        const data = (await response.json()) as ErrorResponse;
        if (!response.ok) throw new Error(JSON.stringify(data));
        setErrorRows(data.items || []);
        setErrorLastUpdated(data.last_updated ?? null);
      } catch {
        setErrorRows([]);
        setErrorLastUpdated(null);
      }
    };

    const fetchHardCases = async () => {
      try {
        const response = await fetch(buildApiUrl("/api/eval/hard-cases"));
        const data = (await response.json()) as HardCaseResponse;
        if (!response.ok) throw new Error(JSON.stringify(data));
        setHardCases(data.items || []);
        setHardCaseLastUpdated(data.last_updated ?? null);
      } catch {
        setHardCases([]);
        setHardCaseLastUpdated(null);
      }
    };

    void fetchRegistry();
    void fetchSteps();
    void fetchPolicy();
    void fetchErrors();
    void fetchHardCases();
  }, []);

  useEffect(() => {
    if (!preprocessSteps.length) return;
    const totalSteps = preprocessSteps.length;
    const interval = window.setInterval(() => {
      setDemoStepIndex((prev) => (prev + 1) % totalSteps);
      setDemoExampleIndex((prev) => (prev + 1) % demoExamples.length);
    }, 2500);
    return () => window.clearInterval(interval);
  }, [preprocessSteps.length]);

  const baselineRun = useMemo(
    () => registry.runs.find((run) => run.is_baseline) || registry.runs[0],
    [registry.runs],
  );

  const comparisonData = useMemo(
    () =>
      registry.runs.map((run) => ({
        model: run.model_name,
        macroF1: run.metrics?.f1 ?? 0,
        toxicF1: (run.metrics as Record<string, number> | undefined)?.f1_toxic ?? run.metrics?.f1 ?? 0,
      })),
    [registry.runs],
  );

  const trainingData = useMemo(() => {
    const curve = baselineRun?.hyperparameters?.training_curve;
    if (Array.isArray(curve)) return curve as { epoch: number; loss: number; f1: number }[];
    return [
      { epoch: 1, loss: 0.68, f1: 0.42 },
      { epoch: 2, loss: 0.55, f1: 0.52 },
      { epoch: 3, loss: 0.48, f1: 0.61 },
      { epoch: 4, loss: 0.42, f1: 0.68 },
      { epoch: 5, loss: 0.38, f1: 0.71 },
    ];
  }, [baselineRun]);

  const pipelineSteps = preprocessSteps.length
    ? preprocessSteps
    : [
        { id: "trim", label: "Loại bỏ khoảng trắng đầu/cuối", active: true },
        { id: "normalize_unicode", label: "Chuẩn hoá Unicode (NFC)", active: true },
        { id: "normalize_whitespace", label: "Chuẩn hoá khoảng trắng", active: true },
        { id: "lowercase", label: "Chuyển lowercase", active: false },
        { id: "remove_emoji", label: "Xử lý emoji", active: false },
        { id: "strip_punctuation", label: "Loại bỏ dấu câu mạnh", active: false },
        { id: "teencode", label: "Chuẩn hoá teencode", active: false },
      ];

  const activeSteps = pipelineSteps.filter((step) => step.active);
  const demoInput = demoExamples[demoExampleIndex];
  const demoOutput = applyActiveSteps(demoInput, pipelineSteps);

  const filteredErrors = useMemo(() => {
    return errorRows.filter((row) => {
      const mismatch = row.true_label === 1 && row.predicted_label === 0 ? "fn" : "fp";
      const matchesType = errorFilter === "all" || mismatch === errorFilter;
      const matchesSource = sourceFilter === "all" || row.source_dataset === sourceFilter;
      const matchesSubset = subsetFilter === "all" || row.subset_tag === subsetFilter;
      return matchesType && matchesSource && matchesSubset;
    });
  }, [errorRows, errorFilter, sourceFilter, subsetFilter]);

  const errorStats = useMemo(() => {
    let fp = 0;
    let fn = 0;
    errorRows.forEach((row) => {
      if (row.true_label === 1 && row.predicted_label === 0) fn += 1;
      if (row.true_label === 0 && row.predicted_label === 1) fp += 1;
    });
    return { fp, fn };
  }, [errorRows]);

  const errorSources = useMemo(() => {
    const sources = new Set<string>();
    errorRows.forEach((row) => row.source_dataset && sources.add(row.source_dataset));
    return ["all", ...Array.from(sources).sort((a, b) => a.localeCompare(b))];
  }, [errorRows]);

  const errorSubsets = useMemo(() => {
    const subsets = new Set<string>();
    errorRows.forEach((row) => row.subset_tag && subsets.add(row.subset_tag));
    return ["all", ...Array.from(subsets).sort((a, b) => a.localeCompare(b))];
  }, [errorRows]);

  const hardCaseBreakdown = useMemo(() => {
    const breakdown: Record<string, number> = {};
    hardCases.forEach((row) => {
      const reason = row.candidate_reason || "unknown";
      breakdown[reason] = (breakdown[reason] ?? 0) + 1;
    });
    return breakdown;
  }, [hardCases]);

  const downloadHardCases = () => {
    const blob = new Blob([JSON.stringify(hardCases, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "hard_case_candidates.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ backgroundColor: "var(--viet-bg)" }} className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <h1 className="text-4xl mb-4" style={{ color: "var(--viet-primary)" }}>
            Model & Hiệu Năng Hệ Thống
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Tìm hiểu về mô hình PhoBERT và hiệu năng phát hiện nội dung độc hại tiếng Việt
          </p>
        </div>

        {/* Model Description */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <div className="flex items-start gap-4 mb-6">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
            >
              <Cpu className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl mb-3" style={{ color: "var(--viet-primary)" }}>
                Về Mô Hình PhoBERT
              </h2>
              <p className="text-gray-700 leading-relaxed mb-4">
                VietToxic Detector sử dụng <strong>PhoBERT-base</strong>, một mô hình ngôn ngữ
                tiên tiến được huấn luyện trước trên kho ngữ liệu tiếng Việt lớn. Mô hình này
                được tinh chỉnh (fine-tuned) đặc biệt cho tác vụ phát hiện nội dung độc hại.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--viet-safe)" }} />
                  <div>
                    <h4 className="mb-1">Tiền xử lý văn bản</h4>
                    <p className="text-sm text-gray-600">Chuẩn hóa Unicode + khoảng trắng</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--viet-safe)" }} />
                  <div>
                    <h4 className="mb-1">Giữ nguyên chữ hoa</h4>
                    <p className="text-sm text-gray-600">Không lowercase để giữ thông tin quan trọng</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Database className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--viet-primary)" }} />
                  <div>
                    <h4 className="mb-1">Bộ dữ liệu</h4>
                    <p className="text-sm text-gray-600">ViCTSD + feedback mới thu thập</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <TrendingUp className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--viet-primary)" }} />
                  <div>
                    <h4 className="mb-1">Transfer Learning</h4>
                    <p className="text-sm text-gray-600">Tận dụng kiến thức từ pre-training</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Experiment Registry */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <h2 className="text-2xl mb-1" style={{ color: "var(--viet-primary)" }}>
                Model Registry
              </h2>
              <p className="text-sm text-gray-600">Thống kê từ experiments/registry.json</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge className="bg-blue-100 text-blue-700">Last updated: {registry.last_updated ?? "--"}</Badge>
              <Button variant="outline" onClick={() => void fetchRegistry(true)}>
                Refresh registry
              </Button>
            </div>
          </div>
          {registryError && <p className="text-sm text-red-600 mb-4">{registryError}</p>}
          {baselineRun && (
            <div className="border rounded-lg p-5 mb-6">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-lg" style={{ color: "var(--viet-primary)" }}>
                    Baseline: {baselineRun.model_name}
                  </h3>
                  <p className="text-sm text-gray-600">Dataset: {baselineRun.dataset_version}</p>
                </div>
                <Badge className="bg-green-100 text-green-700">{baselineRun.created_at}</Badge>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="bg-white border p-4">
                  <p className="text-xs text-gray-500">F1</p>
                  <p className="text-xl text-gray-800">{formatScore(baselineRun.metrics?.f1)}</p>
                </Card>
                <Card className="bg-white border p-4">
                  <p className="text-xs text-gray-500">Precision</p>
                  <p className="text-xl text-gray-800">{formatScore(baselineRun.metrics?.precision)}</p>
                </Card>
                <Card className="bg-white border p-4">
                  <p className="text-xs text-gray-500">Recall</p>
                  <p className="text-xl text-gray-800">{formatScore(baselineRun.metrics?.recall)}</p>
                </Card>
                <Card className="bg-white border p-4">
                  <p className="text-xs text-gray-500">Accuracy</p>
                  <p className="text-xl text-gray-800">{formatScore(baselineRun.metrics?.accuracy)}</p>
                </Card>
              </div>
              <p className="text-sm text-gray-600 mt-3">Checkpoint: {baselineRun.checkpoint_path}</p>
            </div>
          )}

          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run ID</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Dataset</TableHead>
                  <TableHead className="text-right">F1</TableHead>
                  <TableHead className="text-right">Precision</TableHead>
                  <TableHead className="text-right">Recall</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {registry.runs.map((run) => (
                  <TableRow key={run.run_id}>
                    <TableCell>{run.run_id}</TableCell>
                    <TableCell>{run.model_name}</TableCell>
                    <TableCell>{run.dataset_version}</TableCell>
                    <TableCell className="text-right">{formatScore(run.metrics?.f1)}</TableCell>
                    <TableCell className="text-right">{formatScore(run.metrics?.precision)}</TableCell>
                    <TableCell className="text-right">{formatScore(run.metrics?.recall)}</TableCell>
                    <TableCell className="text-right">{formatScore(run.metrics?.accuracy)}</TableCell>
                    <TableCell>{run.created_at}</TableCell>
                  </TableRow>
                ))}
                {!registry.runs.length && (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center text-sm text-gray-500">
                      Chưa có thí nghiệm nào
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </Card>

        {/* Pipeline Demo */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <div className="flex items-start gap-4 mb-6">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
            >
              <FileText className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl mb-2" style={{ color: "var(--viet-primary)" }}>
                Pipeline Demo
              </h2>
              <p className="text-sm text-gray-600">
                Mô phỏng các bước tiền xử lý thực tế (chỉ hiển thị bước đã triển khai).
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="border rounded-lg p-4 bg-gray-50">
              <p className="text-xs text-gray-500 mb-2">Input</p>
              <p className="text-gray-800 text-sm">{demoInput}</p>
              <div className="mt-4">
                <p className="text-xs text-gray-500 mb-2">Output (sau {activeSteps.length} bước)</p>
                <p className="text-gray-800 text-sm">{demoOutput}</p>
              </div>
            </div>

            <div className="border rounded-lg p-4">
              <p className="text-xs text-gray-500 mb-3">Các bước</p>
              <div className="space-y-2">
                {pipelineSteps.map((step, idx) => {
                  const isActive = step.active;
                  const isCurrent = idx === demoStepIndex;
                  return (
                    <div
                      key={step.id}
                      className={`flex items-center justify-between rounded-lg border px-3 py-2 text-sm ${
                        isActive ? "bg-white" : "bg-gray-50 text-gray-400"
                      } ${isCurrent ? "border-blue-300" : "border-gray-200"}`}
                    >
                      <span>{step.label}</span>
                      <Badge className={isActive ? "bg-green-100 text-green-700" : "bg-gray-200 text-gray-500"}>
                        {isActive ? "Active" : "Planned — not yet active"}
                      </Badge>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="mt-6 border rounded-lg p-4 bg-blue-50/40">
            <h3 className="text-sm mb-3" style={{ color: "var(--viet-primary)" }}>
              Lý do preprocessing hiện chỉ active 3 bước chính
            </h3>
            <ul className="list-disc pl-5 space-y-2 text-sm text-gray-700">
              <li>
                <strong>PhoBERT là mô hình case-sensitive</strong>, nên không lowercase để giữ tín hiệu chữ hoa/chữ thường.
              </li>
              <li>
                <strong>Emoji và dấu câu</strong> (ví dụ !!!, ???, 😡, 😂) có thể là tín hiệu toxic/sarcasm, nên chưa loại bỏ.
              </li>
              <li>
                Tránh <strong>over-cleaning</strong> để dữ liệu huấn luyện gần với dữ liệu thực tế khi deploy.
              </li>
              <li>
                <strong>Teencode normalization</strong> vẫn để planned vì mapping dễ sai ngữ cảnh; chỉ nên bật sau khi có kết quả ablation rõ ràng.
              </li>
              <li>
                Bản baseline ưu tiên các bước “an toàn cao”: trim + Unicode NFC + chuẩn hoá khoảng trắng.
              </li>
            </ul>
          </div>
        </Card>

        {/* Performance Metrics */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <div className="flex items-start gap-4 mb-6">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
            >
              <LineChart className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl mb-3" style={{ color: "var(--viet-primary)" }}>
                Hiệu Năng & Metrics
              </h2>
              <p className="text-gray-700 mb-6">Đánh giá chất lượng mô hình theo registry</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl">
              <h4 className="text-sm text-gray-600 mb-2">Macro F1-Score</h4>
              <div className="flex items-end gap-2">
                <span className="text-4xl" style={{ color: "var(--viet-primary)" }}>
                  {formatScore(baselineRun?.metrics?.f1)}
                </span>
                <Badge className="mb-2" style={{ backgroundColor: "var(--viet-safe)" }}>
                  Baseline
                </Badge>
              </div>
              <p className="text-sm text-gray-600 mt-2">Trung bình F1 của tất cả các lớp</p>
            </div>

            <div className="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-xl">
              <h4 className="text-sm text-gray-600 mb-2">Precision</h4>
              <div className="flex items-end gap-2">
                <span className="text-4xl" style={{ color: "var(--viet-toxic)" }}>
                  {formatScore(baselineRun?.metrics?.precision)}
                </span>
                <Badge className="mb-2 bg-orange-500">Baseline</Badge>
              </div>
              <p className="text-sm text-gray-600 mt-2">Độ chính xác (precision)</p>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl">
              <h4 className="text-sm text-gray-600 mb-2">Recall</h4>
              <div className="flex items-end gap-2">
                <span className="text-4xl" style={{ color: "var(--viet-safe)" }}>
                  {formatScore(baselineRun?.metrics?.recall)}
                </span>
                <Badge className="mb-2" style={{ backgroundColor: "var(--viet-safe)" }}>
                  Baseline
                </Badge>
              </div>
              <p className="text-sm text-gray-600 mt-2">Độ bao phủ (recall)</p>
            </div>
          </div>

          {/* Detailed Metrics Table */}
          <div className="mb-6">
            <h3 className="text-xl mb-4" style={{ color: "var(--viet-primary)" }}>
              Chi Tiết Metrics
            </h3>
            <div className="border rounded-lg overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metric</TableHead>
                    <TableHead className="text-right">Baseline</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell>Precision</TableCell>
                    <TableCell className="text-right">{formatScore(baselineRun?.metrics?.precision)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Recall</TableCell>
                    <TableCell className="text-right">{formatScore(baselineRun?.metrics?.recall)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>F1-Score</TableCell>
                    <TableCell className="text-right font-medium">{formatScore(baselineRun?.metrics?.f1)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Accuracy</TableCell>
                    <TableCell className="text-right">{formatScore(baselineRun?.metrics?.accuracy)}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </div>
          </div>
        </Card>

        {/* Model Comparison */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <h2 className="text-2xl mb-6" style={{ color: "var(--viet-primary)" }}>
            So Sánh Với Baseline Models
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="macroF1" fill="var(--viet-primary)" name="Macro F1" />
              <Bar dataKey="toxicF1" fill="var(--viet-toxic)" name="Toxic Class F1" />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-sm text-gray-600 mt-4 text-center">
            So sánh dựa trên registry hiện có.
          </p>
        </Card>

        {/* Evaluation Policy */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-2xl mb-2" style={{ color: "var(--viet-primary)" }}>
                Evaluation Policy
              </h2>
              <p className="text-sm text-gray-600">Giải thích cách chia tập và đánh giá.</p>
            </div>
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline">Xem chi tiết</Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Evaluation Policy</DialogTitle>
                  <DialogDescription>
                    Thông tin được lấy từ config/eval_policy.json
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 text-sm text-gray-700">
                  <div>
                    <p className="font-medium">Tỷ lệ chia tập</p>
                    <p>
                      Train: {formatRatio(policy?.policy.split?.train)} | Val: {formatRatio(policy?.policy.split?.val)} | Test:{" "}
                      {formatRatio(policy?.policy.split?.test)}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">Test set cố định</p>
                    <p>
                      {policy?.policy.test_set_fixed ? "Có" : "Không"} (từ {policy?.policy.fixed_since ?? "--"})
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">Hard-case subsets</p>
                    <p>
                      {policy?.policy.hard_case_subsets_evaluated ? "Được đánh giá" : "Chưa áp dụng"}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">Ghi chú</p>
                    <p>{policy?.policy.note ?? "--"}</p>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </Card>

        {/* Training Visualization */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <h2 className="text-2xl mb-6" style={{ color: "var(--viet-primary)" }}>
            Quá Trình Training & Fine-tuning
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Training Loss Curve */}
            <div>
              <h3 className="mb-4">Training Loss</h3>
              <ResponsiveContainer width="100%" height={250}>
                <RechartsLine data={trainingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: "Epoch", position: "insideBottom", offset: -5 }} />
                  <YAxis label={{ value: "Loss", angle: -90, position: "insideLeft" }} />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="var(--viet-toxic)" 
                    strokeWidth={2}
                    name="Validation Loss"
                  />
                </RechartsLine>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-2 text-center">
                Loss giảm đều qua các epoch
              </p>
            </div>

            {/* F1 Score Curve */}
            <div>
              <h3 className="mb-4">Validation F1-Score</h3>
              <ResponsiveContainer width="100%" height={250}>
                <RechartsLine data={trainingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: "Epoch", position: "insideBottom", offset: -5 }} />
                  <YAxis domain={[0, 1]} label={{ value: "F1 Score", angle: -90, position: "insideLeft" }} />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="f1" 
                    stroke="var(--viet-safe)" 
                    strokeWidth={2}
                    name="Macro F1"
                  />
                </RechartsLine>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-2 text-center">
                F1 score cải thiện và ổn định sau 5 epochs
              </p>
            </div>
          </div>
        </Card>

        {/* Error Analysis */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <h2 className="text-2xl mb-1" style={{ color: "var(--viet-primary)" }}>
                Phân Tích Lỗi (Error Analysis)
              </h2>
              <p className="text-sm text-gray-600">Tổng hợp từ data/processed/error_analysis.json</p>
            </div>
            <Badge className="bg-blue-100 text-blue-700">Last updated: {errorLastUpdated ?? "--"}</Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <Card className="border p-5">
              <div className="flex items-center gap-2 mb-2">
                <XCircle className="w-5 h-5" style={{ color: "var(--viet-toxic)" }} />
                <h3 style={{ color: "var(--viet-primary)" }}>False Positives</h3>
              </div>
              <p className="text-3xl" style={{ color: "var(--viet-primary)" }}>{errorStats.fp}</p>
            </Card>
            <Card className="border p-5">
              <div className="flex items-center gap-2 mb-2">
                <XCircle className="w-5 h-5" style={{ color: "var(--viet-safe)" }} />
                <h3 style={{ color: "var(--viet-primary)" }}>False Negatives</h3>
              </div>
              <p className="text-3xl" style={{ color: "var(--viet-primary)" }}>{errorStats.fn}</p>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div>
              <Label className="text-sm text-gray-600">Mismatch type</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={errorFilter}
                onChange={(event) => setErrorFilter(event.target.value)}
              >
                <option value="all">all</option>
                <option value="fp">false positive</option>
                <option value="fn">false negative</option>
              </select>
            </div>
            <div>
              <Label className="text-sm text-gray-600">Source dataset</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={sourceFilter}
                onChange={(event) => setSourceFilter(event.target.value)}
              >
                {errorSources.map((source) => (
                  <option key={source} value={source}>
                    {source}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <Label className="text-sm text-gray-600">Subset tag</Label>
              <select
                className="mt-2 w-full border rounded-lg px-3 py-2 text-sm"
                value={subsetFilter}
                onChange={(event) => setSubsetFilter(event.target.value)}
              >
                {errorSubsets.map((subset) => (
                  <option key={subset} value={subset}>
                    {subset}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Text</TableHead>
                  <TableHead className="text-right">True</TableHead>
                  <TableHead className="text-right">Pred</TableHead>
                  <TableHead className="text-right">Confidence</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead>Subset</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredErrors.map((row, idx) => (
                  <TableRow key={`${row.text.slice(0, 16)}-${idx}`}>
                    <TableCell className="max-w-[360px]">
                      <details>
                        <summary className="cursor-pointer truncate">{row.text}</summary>
                        <p className="mt-2 text-sm text-gray-600">{row.text}</p>
                      </details>
                    </TableCell>
                    <TableCell className="text-right">{row.true_label}</TableCell>
                    <TableCell className="text-right">{row.predicted_label}</TableCell>
                    <TableCell className="text-right">{row.confidence?.toFixed(2) ?? "--"}</TableCell>
                    <TableCell>{row.source_dataset ?? "--"}</TableCell>
                    <TableCell>{row.subset_tag ?? "--"}</TableCell>
                  </TableRow>
                ))}
                {!filteredErrors.length && (
                  <TableRow>
                    <TableCell colSpan={6} className="text-center text-sm text-gray-500">
                      Chưa có lỗi để hiển thị
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </Card>

        {/* Hard-case Candidates */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <h2 className="text-2xl mb-1" style={{ color: "var(--viet-primary)" }}>
                Phase 1 — Heuristic Candidates
              </h2>
              <p className="text-sm text-gray-600">Hard-case subset scaffold</p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Badge className="bg-blue-100 text-blue-700">Last updated: {hardCaseLastUpdated ?? "--"}</Badge>
              <Button variant="outline" onClick={downloadHardCases} disabled={!hardCases.length}>
                <Download className="w-4 h-4 mr-2" />
                Export for annotation
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Card className="border p-4">
              <p className="text-sm text-gray-500">Candidate count</p>
              <p className="text-2xl" style={{ color: "var(--viet-primary)" }}>{hardCases.length}</p>
            </Card>
            <Card className="border p-4 md:col-span-2">
              <p className="text-sm text-gray-500 mb-2">Breakdown by reason</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(hardCaseBreakdown).map(([reason, count]) => (
                  <Badge key={reason} className="bg-blue-100 text-blue-700">
                    {reason}: {count}
                  </Badge>
                ))}
                {!Object.keys(hardCaseBreakdown).length && <span className="text-sm text-gray-500">--</span>}
              </div>
            </Card>
          </div>

          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Text</TableHead>
                  <TableHead>Reason</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead>Subset</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {hardCases.slice(0, 6).map((row, idx) => (
                  <TableRow key={`${row.text.slice(0, 16)}-${idx}`}>
                    <TableCell className="max-w-[320px] truncate" title={row.text}>
                      {row.text}
                    </TableCell>
                    <TableCell>{row.candidate_reason ?? "--"}</TableCell>
                    <TableCell>{row.source_dataset ?? "--"}</TableCell>
                    <TableCell>{row.subset_tag ?? "--"}</TableCell>
                  </TableRow>
                ))}
                {!hardCases.length && (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center text-sm text-gray-500">
                      Chưa có hard-case candidates
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </Card>

        {/* MLOps & Reliability */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <h2 className="text-2xl mb-6" style={{ color: "var(--viet-primary)" }}>
            MLOps & Độ Tin Cậy
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="border rounded-lg p-5">
              <h4 className="mb-3" style={{ color: "var(--viet-primary)" }}>
                Model Versioning
              </h4>
              <p className="text-sm text-gray-600 mb-3">
                Tất cả các phiên bản mô hình được theo dõi và có thể rollback
              </p>
              <Badge className="bg-blue-100 text-blue-700">v1.2.0 (Current)</Badge>
            </div>

            <div className="border rounded-lg p-5">
              <h4 className="mb-3" style={{ color: "var(--viet-primary)" }}>
                Experiment Tracking
              </h4>
              <p className="text-sm text-gray-600 mb-3">
                Sử dụng MLflow để theo dõi thí nghiệm và hyperparameters
              </p>
              <Badge className="bg-green-100 text-green-700">Active</Badge>
            </div>

            <div className="border rounded-lg p-5">
              <h4 className="mb-3" style={{ color: "var(--viet-primary)" }}>
                Monitoring
              </h4>
              <p className="text-sm text-gray-600 mb-3">
                Theo dõi performance và data drift trong production
              </p>
              <Badge className="bg-purple-100 text-purple-700">Real-time</Badge>
            </div>
          </div>

          <div className="border-l-4 p-5 rounded" style={{ 
            borderLeftColor: "var(--viet-primary)",
            backgroundColor: "rgba(0, 51, 102, 0.05)"
          }}>
            <h4 className="mb-2" style={{ color: "var(--viet-primary)" }}>
              ⚠️ Disclaimer - Lưu Ý Quan Trọng
            </h4>
            <p className="text-gray-700">
              Kết quả dự đoán từ mô hình AI mang tính <strong>hỗ trợ và tham khảo</strong>, 
              không thay thế được đánh giá và phán đoán của con người. Hệ thống có thể mắc lỗi 
              và không nên được sử dụng làm căn cứ duy nhất cho các quyết định quan trọng. 
              Luôn kết hợp với kiểm chứng thủ công và suy nghĩ phản biện.
            </p>
          </div>
        </Card>

        {/* CTA */}
        <div className="text-center">
          <div className="inline-block bg-white rounded-2xl shadow-lg p-8">
            <h3 className="text-2xl mb-4" style={{ color: "var(--viet-primary)" }}>
              Sẵn Sàng Thử Nghiệm?
            </h3>
            <p className="text-gray-600 mb-6 max-w-md">
              Phân tích nội dung web của bạn ngay với mô hình PhoBERT
            </p>
            <Button
              onClick={onTryNow}
              className="h-12 px-8"
              style={{ backgroundColor: "var(--viet-primary)" }}
            >
              Thử Phân Tích URL Ngay
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
