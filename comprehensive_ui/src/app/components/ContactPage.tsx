import { Card } from "@/app/components/ui/card";
import { Mail, Github, Twitter, MessageCircle } from "lucide-react";

export function ContactPage() {
  return (
    <div className="min-h-screen bg-background py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-12 text-center">
          <h1 className="text-4xl mb-4 text-primary">Liên Hệ & Hỗ Trợ</h1>
          <p className="text-xl text-muted-foreground">
            Chúng tôi luôn sẵn sàng lắng nghe phản hồi và hỗ trợ của bạn
          </p>
        </div>

        <div className="mb-12 grid grid-cols-1 gap-6 md:grid-cols-2">
          <Card className="bg-card p-6 shadow-lg transition-shadow hover:shadow-xl">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-background-info">
                <Mail className="h-6 w-6 text-text-info" />
              </div>
              <div>
                <h3 className="mb-2 text-primary">Email Hỗ Trợ</h3>
                <p className="mb-2 text-muted-foreground">Gửi câu hỏi hoặc báo lỗi qua email</p>
                <a href="mailto:mittech.official@gmail.com" className="text-text-info hover:underline">
                  mittech.official@gmail.com
                </a>
              </div>
            </div>
          </Card>

          <Card className="bg-card p-6 shadow-lg transition-shadow hover:shadow-xl">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-background-info">
                <Github className="h-6 w-6 text-text-info" />
              </div>
              <div>
                <h3 className="mb-2 text-primary">GitHub</h3>
                <p className="mb-2 text-muted-foreground">Xem source code và đóng góp</p>
                <a
                  href="https://github.com/kaymid1/vietnamese_toxic_comment_detection_using_PhoBERT"
                  className="text-text-info hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  github.com/vietnamese_toxic_comment_detection_using_PhoBERT
                </a>
              </div>
            </div>
          </Card>

          <Card className="bg-card p-6 shadow-lg transition-shadow hover:shadow-xl">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-background-info">
                <Twitter className="h-6 w-6 text-text-info" />
              </div>
              <div>
                <h3 className="mb-2 text-primary">Twitter/X</h3>
                <p className="mb-2 text-muted-foreground">Theo dõi cập nhật mới nhất</p>
                <a
                  href="https://twitter.com/viettoxic"
                  className="text-text-info hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  @viettoxic
                </a>
              </div>
            </div>
          </Card>

          <Card className="bg-card p-6 shadow-lg transition-shadow hover:shadow-xl">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-background-info">
                <MessageCircle className="h-6 w-6 text-text-info" />
              </div>
              <div>
                <h3 className="mb-2 text-primary">Discord Community</h3>
                <p className="mb-2 text-muted-foreground">Tham gia cộng đồng người dùng</p>
                <a
                  href="https://discord.gg/viettoxic"
                  className="text-text-info hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Join Discord
                </a>
              </div>
            </div>
          </Card>
        </div>

        <Card className="mb-8 bg-card p-8 shadow-lg">
          <h2 className="mb-6 text-2xl text-primary">Câu Hỏi Thường Gặp</h2>

          <div className="space-y-6">
            <div>
              <h4 className="mb-2 text-primary">Hệ thống hoạt động như thế nào?</h4>
              <p className="text-muted-foreground">
                VietToxic Detector sử dụng mô hình PhoBERT để phân tích văn bản tiếng Việt.
                Khi bạn nhập URL, hệ thống sẽ cào nội dung, tiền xử lý văn bản, và chạy qua
                mô hình AI để dự đoán mức độ độc hại.
              </p>
            </div>

            <div>
              <h4 className="mb-2 text-primary">Dữ liệu của tôi có được lưu trữ không?</h4>
              <p className="text-muted-foreground">
                Không. Chúng tôi không lưu trữ URL hoặc nội dung bạn phân tích. Tất cả dữ
                liệu chỉ được xử lý tạm thời trong bộ nhớ và bị xóa ngay sau khi phân tích
                hoàn tất.
              </p>
            </div>

            <div>
              <h4 className="mb-2 text-primary">Độ chính xác của mô hình là bao nhiêu?</h4>
              <p className="text-muted-foreground">
                Mô hình hiện tại đạt Macro F1-Score khoảng 0.71 (~71%). Tuy nhiên, với lớp
                "độc hại" cụ thể, F1 là 0.49 do tính chất imbalanced của dữ liệu. Chúng tôi
                liên tục cải thiện mô hình.
              </p>
            </div>

            <div>
              <h4 className="mb-2 text-primary">Tôi có thể đóng góp vào dự án không?</h4>
              <p className="text-muted-foreground">
                Có! Dự án là mã nguồn mở. Bạn có thể đóng góp code, báo cáo lỗi, đề xuất
                tính năng mới, hoặc giúp cải thiện dataset trên GitHub của chúng tôi.
              </p>
            </div>

            <div>
              <h4 className="mb-2 text-primary">Hệ thống có API không?</h4>
              <p className="text-muted-foreground">
                Hiện tại chúng tôi đang phát triển API để các developer có thể tích hợp vào
                ứng dụng của mình. Vui lòng theo dõi trên GitHub để cập nhật thông tin mới
                nhất.
              </p>
            </div>
          </div>
        </Card>

        <Card className="bg-card p-8 shadow-lg">
          <h2 className="mb-4 text-2xl text-primary">Nghiên Cứu & Hợp Tác</h2>
          <p className="mb-4 text-muted-foreground">
            Nếu bạn là nhà nghiên cứu, sinh viên, hoặc tổ chức quan tâm đến việc hợp tác
            phát triển công nghệ phát hiện nội dung độc hại tiếng Việt, chúng tôi rất
            hoan nghênh!
          </p>
          <div className="rounded-lg bg-background-secondary p-4">
            <h4 className="mb-2">Các lĩnh vực hợp tác:</h4>
            <ul className="space-y-2 text-sm text-foreground">
              <li className="flex items-start gap-2">
                <span className="mt-1 text-text-info">•</span>
                <span>Thu thập và gán nhãn dữ liệu tiếng Việt</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-text-info">•</span>
                <span>Phát triển mô hình AI mới với độ chính xác cao hơn</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-text-info">•</span>
                <span>Nghiên cứu về Explainable AI cho NLP tiếng Việt</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-text-info">•</span>
                <span>Tích hợp vào các nền tảng truyền thông xã hội</span>
              </li>
            </ul>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            Liên hệ: <a href="mailto:research@viettoxic.ai" className="text-text-info hover:underline">research@viettoxic.ai</a>
          </p>
        </Card>
      </div>
    </div>
  );
}
