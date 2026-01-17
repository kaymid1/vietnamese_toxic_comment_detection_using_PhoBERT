import { Card } from "@/app/components/ui/card";
import { Mail, Github, Twitter, MessageCircle } from "lucide-react";

export function ContactPage() {
  return (
    <div style={{ backgroundColor: "var(--viet-bg)" }} className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <h1 className="text-4xl mb-4" style={{ color: "var(--viet-primary)" }}>
            Liên Hệ & Hỗ Trợ
          </h1>
          <p className="text-xl text-gray-600">
            Chúng tôi luôn sẵn sàng lắng nghe phản hồi và hỗ trợ của bạn
          </p>
        </div>

        {/* Contact Methods */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          <Card className="bg-white p-6 shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
              >
                <Mail className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
              </div>
              <div>
                <h3 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                  Email Hỗ Trợ
                </h3>
                <p className="text-gray-600 mb-2">
                  Gửi câu hỏi hoặc báo lỗi qua email
                </p>
                <a 
                  href="mailto:support@viettoxic.ai" 
                  className="text-blue-600 hover:underline"
                >
                  support@viettoxic.ai
                </a>
              </div>
            </div>
          </Card>

          <Card className="bg-white p-6 shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
              >
                <Github className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
              </div>
              <div>
                <h3 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                  GitHub
                </h3>
                <p className="text-gray-600 mb-2">
                  Xem source code và đóng góp
                </p>
                <a 
                  href="https://github.com/viettoxic/detector" 
                  className="text-blue-600 hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  github.com/viettoxic
                </a>
              </div>
            </div>
          </Card>

          <Card className="bg-white p-6 shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
              >
                <Twitter className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
              </div>
              <div>
                <h3 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                  Twitter/X
                </h3>
                <p className="text-gray-600 mb-2">
                  Theo dõi cập nhật mới nhất
                </p>
                <a 
                  href="https://twitter.com/viettoxic" 
                  className="text-blue-600 hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  @viettoxic
                </a>
              </div>
            </div>
          </Card>

          <Card className="bg-white p-6 shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: "rgba(0, 51, 102, 0.1)" }}
              >
                <MessageCircle className="w-6 h-6" style={{ color: "var(--viet-primary)" }} />
              </div>
              <div>
                <h3 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                  Discord Community
                </h3>
                <p className="text-gray-600 mb-2">
                  Tham gia cộng đồng người dùng
                </p>
                <a 
                  href="https://discord.gg/viettoxic" 
                  className="text-blue-600 hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Join Discord
                </a>
              </div>
            </div>
          </Card>
        </div>

        {/* FAQ */}
        <Card className="bg-white p-8 shadow-lg mb-8">
          <h2 className="text-2xl mb-6" style={{ color: "var(--viet-primary)" }}>
            Câu Hỏi Thường Gặp
          </h2>

          <div className="space-y-6">
            <div>
              <h4 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                Hệ thống hoạt động như thế nào?
              </h4>
              <p className="text-gray-600">
                VietToxic Detector sử dụng mô hình PhoBERT để phân tích văn bản tiếng Việt. 
                Khi bạn nhập URL, hệ thống sẽ cào nội dung, tiền xử lý văn bản, và chạy qua 
                mô hình AI để dự đoán mức độ độc hại.
              </p>
            </div>

            <div>
              <h4 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                Dữ liệu của tôi có được lưu trữ không?
              </h4>
              <p className="text-gray-600">
                Không. Chúng tôi không lưu trữ URL hoặc nội dung bạn phân tích. Tất cả dữ 
                liệu chỉ được xử lý tạm thời trong bộ nhớ và bị xóa ngay sau khi phân tích 
                hoàn tất.
              </p>
            </div>

            <div>
              <h4 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                Độ chính xác của mô hình là bao nhiêu?
              </h4>
              <p className="text-gray-600">
                Mô hình hiện tại đạt Macro F1-Score khoảng 0.71 (~71%). Tuy nhiên, với lớp 
                "độc hại" cụ thể, F1 là 0.49 do tính chất imbalanced của dữ liệu. Chúng tôi 
                liên tục cải thiện mô hình.
              </p>
            </div>

            <div>
              <h4 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                Tôi có thể đóng góp vào dự án không?
              </h4>
              <p className="text-gray-600">
                Có! Dự án là mã nguồn mở. Bạn có thể đóng góp code, báo cáo lỗi, đề xuất 
                tính năng mới, hoặc giúp cải thiện dataset trên GitHub của chúng tôi.
              </p>
            </div>

            <div>
              <h4 className="mb-2" style={{ color: "var(--viet-primary)" }}>
                Hệ thống có API không?
              </h4>
              <p className="text-gray-600">
                Hiện tại chúng tôi đang phát triển API để các developer có thể tích hợp vào 
                ứng dụng của mình. Vui lòng theo dõi trên GitHub để cập nhật thông tin mới 
                nhất.
              </p>
            </div>
          </div>
        </Card>

        {/* Research & Collaboration */}
        <Card className="bg-white p-8 shadow-lg">
          <h2 className="text-2xl mb-4" style={{ color: "var(--viet-primary)" }}>
            Nghiên Cứu & Hợp Tác
          </h2>
          <p className="text-gray-600 mb-4">
            Nếu bạn là nhà nghiên cứu, sinh viên, hoặc tổ chức quan tâm đến việc hợp tác 
            phát triển công nghệ phát hiện nội dung độc hại tiếng Việt, chúng tôi rất 
            hoan nghênh!
          </p>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="mb-2">Các lĩnh vực hợp tác:</h4>
            <ul className="space-y-2 text-sm text-gray-700">
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>Thu thập và gán nhãn dữ liệu tiếng Việt</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>Phát triển mô hình AI mới với độ chính xác cao hơn</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>Nghiên cứu về Explainable AI cho NLP tiếng Việt</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>Tích hợp vào các nền tảng truyền thông xã hội</span>
              </li>
            </ul>
          </div>
          <p className="text-sm text-gray-600 mt-4">
            Liên hệ: <a href="mailto:research@viettoxic.ai" className="text-blue-600 hover:underline">
              research@viettoxic.ai
            </a>
          </p>
        </Card>
      </div>
    </div>
  );
}
