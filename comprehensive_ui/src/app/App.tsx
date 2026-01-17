import { useState } from "react";
import { Navigation } from "@/app/components/Navigation";
import { HomePage } from "@/app/components/HomePage";
import { ResultsPage } from "@/app/components/ResultsPage";
import { ModelPage } from "@/app/components/ModelPage";
import { ContactPage } from "@/app/components/ContactPage";

// Mock data generator for demonstration
const generateMockResults = (urls: string[]) => {
  const domains = urls.map((url) => {
    try {
      const urlObj = new URL(url.startsWith("http") ? url : `https://${url}`);
      return urlObj.hostname;
    } catch {
      return "example.com";
    }
  });

  return urls.map((url, idx) => {
    const toxicityScore = Math.floor(Math.random() * 70) + 15; // 15-85%
    const numSegments = Math.floor(Math.random() * 5) + 3; // 3-7 segments

    const sampleTexts = [
      "Bài viết này cung cấp thông tin hữu ích về chủ đề đang được quan tâm.",
      "Theo nguồn tin đáng tin cậy, sự kiện này đã thu hút sự chú ý của cộng đồng.",
      "Nội dung này thật là rác rưởi và vô giá trị, không nên đọc.",
      "Chúng tôi khuyến nghị người đọc nên tham khảo thêm từ nhiều nguồn khác nhau.",
      "Đây là một phân tích khách quan về tình hình hiện tại.",
      "Những lời bình luận độc hại và thiếu văn hóa như thế này cần được lọc bỏ.",
      "Tác giả đưa ra quan điểm rõ ràng và có căn cứ.",
      "Thật kinh khủng khi lại có những người như vậy trong xã hội.",
    ];

    const segments = Array.from({ length: numSegments }, (_, i) => {
      const isToxic = Math.random() < toxicityScore / 100;
      return {
        text: sampleTexts[Math.floor(Math.random() * sampleTexts.length)],
        confidence: Math.random() * 0.3 + (isToxic ? 0.6 : 0.5),
        isToxic,
      };
    });

    return {
      url,
      title: `Bài Viết Phân Tích ${idx + 1}`,
      domain: domains[idx],
      toxicityScore,
      toxicSegments: segments,
    };
  });
};

export default function App() {
  const [currentPage, setCurrentPage] = useState("home");
  const [analysisResults, setAnalysisResults] = useState<any[]>([]);

  const handleNavigate = (page: string) => {
    setCurrentPage(page);
  };

  const handleAnalyze = (urls: string[]) => {
    const results = generateMockResults(urls);
    setAnalysisResults(results);
    setCurrentPage("results");
  };

  const handleScanAgain = () => {
    setCurrentPage("home");
    setAnalysisResults([]);
  };

  const handleTryNow = () => {
    setCurrentPage("home");
  };

  return (
    <div className="min-h-screen">
      <Navigation currentPage={currentPage} onNavigate={handleNavigate} />

      {currentPage === "home" && <HomePage onAnalyze={handleAnalyze} />}
      
      {currentPage === "results" && (
        <ResultsPage results={analysisResults} onScanAgain={handleScanAgain} />
      )}
      
      {currentPage === "model" && <ModelPage onTryNow={handleTryNow} />}
      
      {currentPage === "contact" && <ContactPage />}
    </div>
  );
}
