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
import { ArrowRight, CheckCircle, XCircle, TrendingUp, Database, Cpu, LineChart } from "lucide-react";
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

export function ModelPage({ onTryNow }: ModelPageProps) {
  // Mock data for training curve
  const trainingData = [
    { epoch: 1, loss: 0.68, f1: 0.42 },
    { epoch: 2, loss: 0.55, f1: 0.52 },
    { epoch: 3, loss: 0.48, f1: 0.61 },
    { epoch: 4, loss: 0.42, f1: 0.68 },
    { epoch: 5, loss: 0.38, f1: 0.71 },
  ];

  // Model comparison data
  const comparisonData = [
    { model: "TF-IDF + Logistic Regression", macroF1: 0.62, toxicF1: 0.35 },
    { model: "PhoBERT (Fine-tuned)", macroF1: 0.71, toxicF1: 0.49 },
  ];

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
            <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
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
                    <p className="text-sm text-gray-600">
                      NFC normalization để chuẩn hóa ký tự tiếng Việt
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--viet-safe)" }} />
                  <div>
                    <h4 className="mb-1">Giữ nguyên chữ hoa</h4>
                    <p className="text-sm text-gray-600">
                      Không lowercase để giữ thông tin quan trọng
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Database className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--viet-primary)" }} />
                  <div>
                    <h4 className="mb-1">Bộ dữ liệu</h4>
                    <p className="text-sm text-gray-600">
                      Vietnamese Toxic Speech Dataset với 10,000+ mẫu
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <TrendingUp className="w-5 h-5 mt-1 flex-shrink-0" style={{ color: "var(--viet-primary)" }} />
                  <div>
                    <h4 className="mb-1">Transfer Learning</h4>
                    <p className="text-sm text-gray-600">
                      Tận dụng kiến thức từ pre-training
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Performance Metrics */}
        <Card className="bg-white p-8 mb-8 shadow-lg">
          <div className="flex items-start gap-4 mb-6">
            <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
            >
              <LineChart className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl mb-3" style={{ color: "var(--viet-primary)" }}>
                Hiệu Năng & Metrics
              </h2>
              <p className="text-gray-700 mb-6">
                Đánh giá chất lượng mô hình trên tập validation
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl">
              <h4 className="text-sm text-gray-600 mb-2">Macro F1-Score</h4>
              <div className="flex items-end gap-2">
                <span className="text-4xl" style={{ color: "var(--viet-primary)" }}>
                  0.71
                </span>
                <Badge className="mb-2" style={{ backgroundColor: "var(--viet-safe)" }}>
                  Tốt
                </Badge>
              </div>
              <p className="text-sm text-gray-600 mt-2">
                Trung bình F1 của tất cả các lớp
              </p>
            </div>

            <div className="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-xl">
              <h4 className="text-sm text-gray-600 mb-2">Toxic Class F1</h4>
              <div className="flex items-end gap-2">
                <span className="text-4xl" style={{ color: "var(--viet-toxic)" }}>
                  0.49
                </span>
                <Badge className="mb-2 bg-orange-500">Trung Bình</Badge>
              </div>
              <p className="text-sm text-gray-600 mt-2">
                F1 cho lớp độc hại (challenging)
              </p>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl">
              <h4 className="text-sm text-gray-600 mb-2">Non-Toxic Class F1</h4>
              <div className="flex items-end gap-2">
                <span className="text-4xl" style={{ color: "var(--viet-safe)" }}>
                  0.93
                </span>
                <Badge className="mb-2" style={{ backgroundColor: "var(--viet-safe)" }}>
                  Xuất Sắc
                </Badge>
              </div>
              <p className="text-sm text-gray-600 mt-2">
                F1 cho lớp không độc hại
              </p>
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
                    <TableHead className="text-right">Toxic Class</TableHead>
                    <TableHead className="text-right">Non-Toxic Class</TableHead>
                    <TableHead className="text-right">Macro Average</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell>Precision</TableCell>
                    <TableCell className="text-right">0.52</TableCell>
                    <TableCell className="text-right">0.91</TableCell>
                    <TableCell className="text-right">0.72</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Recall</TableCell>
                    <TableCell className="text-right">0.46</TableCell>
                    <TableCell className="text-right">0.95</TableCell>
                    <TableCell className="text-right">0.71</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>F1-Score</TableCell>
                    <TableCell className="text-right font-medium">0.49</TableCell>
                    <TableCell className="text-right font-medium">0.93</TableCell>
                    <TableCell className="text-right font-medium">0.71</TableCell>
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
            PhoBERT vượt trội so với phương pháp truyền thống TF-IDF + Logistic Regression
          </p>
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
          <h2 className="text-2xl mb-6" style={{ color: "var(--viet-primary)" }}>
            Phân Tích Lỗi (Error Analysis)
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* False Positives */}
            <div className="border rounded-lg p-5">
              <div className="flex items-center gap-2 mb-4">
                <XCircle className="w-5 h-5" style={{ color: "var(--viet-toxic)" }} />
                <h3 style={{ color: "var(--viet-primary)" }}>False Positives</h3>
              </div>
              <div className="space-y-3">
                <div className="bg-red-50 p-3 rounded border-l-4 border-red-300">
                  <p className="text-sm mb-1">
                    "Bài viết này phê phán mạnh mẽ chính sách mới"
                  </p>
                  <p className="text-xs text-gray-600">
                    Dự đoán: Độc hại | Thực tế: Không độc hại
                  </p>
                </div>
                <div className="bg-red-50 p-3 rounded border-l-4 border-red-300">
                  <p className="text-sm mb-1">
                    "Diễn viên này thật sự tệ trong vai diễn đó"
                  </p>
                  <p className="text-xs text-gray-600">
                    Dự đoán: Độc hại | Thực tế: Phê bình hợp lý
                  </p>
                </div>
              </div>
              <p className="text-sm text-gray-600 mt-4">
                Mô hình đôi khi nhầm lẫn phê bình chính đáng với ngôn ngữ độc hại
              </p>
            </div>

            {/* False Negatives */}
            <div className="border rounded-lg p-5">
              <div className="flex items-center gap-2 mb-4">
                <XCircle className="w-5 h-5" style={{ color: "var(--viet-safe)" }} />
                <h3 style={{ color: "var(--viet-primary)" }}>False Negatives</h3>
              </div>
              <div className="space-y-3">
                <div className="bg-orange-50 p-3 rounded border-l-4 border-orange-300">
                  <p className="text-sm mb-1">
                    "Một số câu độc hại ngầm định hoặc châm biếm"
                  </p>
                  <p className="text-xs text-gray-600">
                    Dự đoán: Không độc hại | Thực tế: Độc hại
                  </p>
                </div>
                <div className="bg-orange-50 p-3 rounded border-l-4 border-orange-300">
                  <p className="text-sm mb-1">
                    "Ngôn ngữ độc hại được che giấu bằng từ ngữ tế nhị"
                  </p>
                  <p className="text-xs text-gray-600">
                    Dự đoán: Không độc hại | Thực tế: Độc hại ngầm
                  </p>
                </div>
              </div>
              <p className="text-sm text-gray-600 mt-4">
                Mô hình khó phát hiện các hình thức độc hại tinh vi hoặc châm biếm
              </p>
            </div>
          </div>

          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h4 className="mb-2" style={{ color: "var(--viet-primary)" }}>
              Hướng Cải Thiện
            </h4>
            <ul className="space-y-2 text-sm text-gray-700">
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>Thu thập thêm dữ liệu cho các trường hợp biên và ngôn ngữ châm biếm</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>Tích hợp context awareness để hiểu ngữ cảnh tốt hơn</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>Sử dụng ensemble methods kết hợp nhiều mô hình</span>
              </li>
            </ul>
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
