import type { ComponentProps } from "react";
import { Badge } from "@/app/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/app/components/ui/tooltip";

export interface MlflowPresentation {
  label: string;
  tooltip: string;
  className: string;
}

export interface MlflowGateThresholds {
  discard: number;
  accept: number;
}

export const DEFAULT_MLFLOW_GATE_THRESHOLDS: MlflowGateThresholds = {
  discard: 0.2,
  accept: 0.8,
};

const COLORS = {
  toxic:
    "border-red-300 bg-red-100 text-red-800 dark:border-red-800 dark:bg-red-950/60 dark:text-red-200",
  clean:
    "border-green-300 bg-green-100 text-green-800 dark:border-green-800 dark:bg-green-950/60 dark:text-green-200",
  automatic:
    "border-blue-300 bg-blue-100 text-blue-800 dark:border-blue-800 dark:bg-blue-950/60 dark:text-blue-200",
  manual:
    "border-yellow-300 bg-yellow-100 text-yellow-900 dark:border-yellow-700 dark:bg-yellow-950/60 dark:text-yellow-200",
  warning:
    "border-amber-300 bg-amber-100 text-amber-900 dark:border-amber-700 dark:bg-amber-950/60 dark:text-amber-200",
  orange:
    "border-orange-300 bg-orange-100 text-orange-900 dark:border-orange-700 dark:bg-orange-950/60 dark:text-orange-200",
  constructive:
    "border-teal-300 bg-teal-100 text-teal-900 dark:border-teal-700 dark:bg-teal-950/60 dark:text-teal-200",
  neutral:
    "border-slate-300 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-900/70 dark:text-slate-300",
  muted:
    "border-border bg-muted/50 text-muted-foreground",
  locked:
    "border-violet-300 bg-violet-100 text-violet-800 dark:border-violet-800 dark:bg-violet-950/60 dark:text-violet-200",
};

export function makeMlflowTooltip(title: string, text: string): string {
  return `${title}\n${text}`;
}

export function MlflowTooltipBody({ text }: { text: string }) {
  const [title, ...bodyParts] = text.split("\n");
  const body = bodyParts.join("\n").trim();
  return (
    <div className="max-w-[320px] whitespace-normal break-words text-left leading-snug">
      <div className="font-medium">{title}</div>
      {body && <div className="mt-1 font-normal opacity-90">{body}</div>}
    </div>
  );
}

export function getToxicityPresentation(label?: number | null): MlflowPresentation {
  if (label === 1) {
    return {
      label: "Độc hại",
      tooltip: makeMlflowTooltip(
        "Nhãn Độc hại",
        "Mẫu đang được gắn nhãn Độc hại cho dữ liệu review/export. Nhãn này có thể đến từ hệ thống, thao tác thủ công hoặc trợ lý review.",
      ),
      className: COLORS.toxic,
    };
  }
  if (label === 0) {
    return {
      label: "Sạch",
      tooltip: makeMlflowTooltip(
        "Nhãn Sạch",
        "Mẫu đang được gắn nhãn Sạch cho dữ liệu review/export. Nhãn này có thể đến từ hệ thống, thao tác thủ công hoặc trợ lý review.",
      ),
      className: COLORS.clean,
    };
  }
  return {
    label: "Chưa có nhãn độc hại",
    tooltip: makeMlflowTooltip(
      "Chưa có nhãn cuối cùng",
      "Mẫu này chưa có nhãn Độc hại hoặc Sạch hợp lệ. Hãy review nhãn trước khi dùng cho tập training.",
    ),
    className: COLORS.muted,
  };
}

export function getConstructivenessPresentation(label?: number | null): MlflowPresentation {
  if (label === 1) {
    return {
      label: "Có tính xây dựng",
      tooltip: makeMlflowTooltip(
        "Đánh dấu có tính xây dựng",
        "Bấm để ghi nhận mẫu có đóng góp hoặc giúp cuộc trao đổi tốt hơn. Dùng khi nội dung có hướng xây dựng rõ ràng.",
      ),
      className: COLORS.constructive,
    };
  }
  if (label === 0) {
    return {
      label: "Không rõ hoặc không đóng góp",
      tooltip: makeMlflowTooltip(
        "Đánh dấu chưa xây dựng",
        "Bấm khi mẫu không thể hiện đóng góp rõ ràng hoặc cần giữ ở nhóm không xây dựng. Bạn vẫn có thể đổi lại sau khi review.",
      ),
      className: COLORS.neutral,
    };
  }
  return {
    label: "Ẩn hoặc chưa có nhãn tính xây dựng",
    tooltip: makeMlflowTooltip(
      "Ẩn nhãn tính xây dựng",
      "Bấm để ẩn hoặc xóa nhãn tính xây dựng khỏi mẫu này. Trạng thái trống chỉ cho biết chưa hiển thị nhãn cuối cùng.",
    ),
    className: COLORS.muted,
  };
}

export function getDataSourcePresentation(sourceType?: string | null): MlflowPresentation {
  if (sourceType === "synthetic") {
    return {
      label: "Tạo sinh bằng Gemini",
      tooltip: makeMlflowTooltip(
        "Nguồn dữ liệu tạo sinh",
        "Comment được Gemini tạo sinh, sau đó được admin review và chuyển vào Training Preview. Trạng thái review được hiển thị ở badge Review riêng.",
      ),
      className: COLORS.constructive,
    };
  }
  return {
    label: "Thu thập từ website",
    tooltip: makeMlflowTooltip(
      "Nguồn dữ liệu thu thập",
      "Comment được lấy từ website qua quy trình crawl và ingestion, sau đó được mô hình local chấm điểm và đưa qua gate.",
    ),
    className: COLORS.neutral,
  };
}

export function getReviewStatusPresentation(status?: string | null): MlflowPresentation {
  if (status === "auto_gemini") {
    return {
      label: "Tự động + Gemini",
      tooltip: makeMlflowTooltip(
        "Tự động, có Gemini hỗ trợ",
        "Mẫu ban đầu được gate tự động chấp nhận, sau đó đã áp dụng gợi ý từ Gemini.",
      ),
      className: COLORS.automatic,
    };
  }
  if (status === "manual_gemini") {
    return {
      label: "Thủ công + Gemini",
      tooltip: makeMlflowTooltip(
        "Thủ công, có Gemini hỗ trợ",
        "Mẫu đã có thao tác review thủ công trước khi gợi ý Gemini được áp dụng.",
      ),
      className: COLORS.manual,
    };
  }
  if (status === "auto") {
    return {
      label: "Tự động",
      tooltip: makeMlflowTooltip(
        "Được hệ thống tự động đánh giá",
        "Nhãn được gán dựa trên điểm mô hình và ngưỡng hiện tại, chưa có thao tác xác nhận thủ công.",
      ),
      className: COLORS.automatic,
    };
  }
  if (status === "manual_approved") {
    return {
      label: "Đã duyệt thủ công",
      tooltip: makeMlflowTooltip(
        "Đã được xử lý thủ công",
        "Mẫu đã có thao tác thủ công liên quan đến nhãn hoặc việc chọn cho training. Trạng thái này không có nghĩa mẫu đã được huấn luyện.",
      ),
      className: COLORS.manual,
    };
  }
  if (status === "pending") {
    return {
      label: "Chờ duyệt",
      tooltip: makeMlflowTooltip(
        "Đang chờ review",
        "Mẫu cần được xem lại trước khi trở thành dữ liệu đáng tin cậy cho export. Hãy kiểm tra nhãn và quyết định chọn training nếu phù hợp.",
      ),
      className: COLORS.neutral,
    };
  }
  if (status === "manual_removed") {
    return {
      label: "Đã bỏ chọn thủ công",
      tooltip: makeMlflowTooltip(
        "Đã bỏ chọn thủ công",
        "Mẫu đã được người dùng gỡ khỏi danh sách có thể dùng khi export. Mẫu vẫn còn trong danh sách review.",
      ),
      className: COLORS.muted,
    };
  }
  if (status === "manual") {
    return {
      label: "Admin đã chỉnh",
      tooltip: makeMlflowTooltip(
        "Đã có chỉnh sửa thủ công",
        "Mẫu từng được người dùng cập nhật trong quy trình review. Hãy xem chi tiết kỹ thuật nếu cần biết trạng thái lưu trữ.",
      ),
      className: COLORS.manual,
    };
  }
  if (status === "removed") {
    return {
      label: "Đã bỏ khỏi selection",
      tooltip: makeMlflowTooltip(
        "Đã bỏ khỏi danh sách training",
        "Mẫu hiện không nằm trong danh sách dữ liệu có thể dùng khi export. Bạn có thể chọn lại nếu mẫu đủ điều kiện.",
      ),
      className: COLORS.muted,
    };
  }
  if (status === "gemini_assist") {
    return {
      label: "Gemini hỗ trợ",
      tooltip: makeMlflowTooltip(
        "Đã áp dụng đề xuất trợ lý",
        "Một đề xuất review đã được áp dụng cho mẫu này. Trạng thái này không có nghĩa mẫu đã được huấn luyện.",
      ),
      className: COLORS.warning,
    };
  }
  return {
    label: status || "Chưa có trạng thái review",
    tooltip: makeMlflowTooltip(
      status ? "Trạng thái review khác" : "Chưa có trạng thái review",
      status
        ? "UI chưa có mô tả nghiệp vụ riêng cho trạng thái này. Mở Chi tiết kỹ thuật nếu cần kiểm tra giá trị lưu trữ."
        : "Mẫu này chưa có trạng thái review rõ ràng. Hãy review nhãn nếu bạn muốn dùng cho training.",
    ),
    className: COLORS.muted,
  };
}

export function getVerificationStatusPresentation(status?: string | null): MlflowPresentation {
  if (status === "auto_accepted") {
    return {
      label: "Tự động chấp nhận",
      tooltip: makeMlflowTooltip(
        "Được hệ thống tự động đánh giá",
        "Nhãn được gán dựa trên điểm mô hình và ngưỡng hiện tại, chưa có thao tác xác nhận thủ công.",
      ),
      className: COLORS.automatic,
    };
  }
  if (status === "unverified") {
    return {
      label: "Chưa xác minh",
      tooltip: makeMlflowTooltip(
        "Cần review thêm",
        "Mẫu chưa có quyết định chấp nhận hoặc loại bỏ thủ công. Hãy kiểm tra nội dung trước khi dùng cho training.",
      ),
      className: COLORS.neutral,
    };
  }
  if (status === "manual_accepted") {
    return {
      label: "Đã chấp nhận thủ công",
      tooltip: makeMlflowTooltip(
        "Đã được xử lý thủ công",
        "Mẫu đã có thao tác thủ công liên quan đến nhãn hoặc việc chọn cho training. Trạng thái này không có nghĩa mẫu đã được huấn luyện.",
      ),
      className: COLORS.manual,
    };
  }
  if (status === "manual_rejected") {
    return {
      label: "Đã loại thủ công",
      tooltip: makeMlflowTooltip(
        "Đã loại khỏi review",
        "Mẫu đã bị loại khỏi nhóm có thể đưa vào export. Nội dung vẫn còn để theo dõi trong lịch sử.",
      ),
      className: COLORS.muted,
    };
  }
  return {
    label: status || "Chưa có verification status",
    tooltip: makeMlflowTooltip(
      status ? "Trạng thái xác minh khác" : "Chưa có trạng thái xác minh",
      status
        ? "UI chưa có mô tả nghiệp vụ riêng cho trạng thái này. Mở Chi tiết kỹ thuật nếu cần kiểm tra giá trị lưu trữ."
        : "Mẫu này chưa có kết quả xác minh rõ ràng. Hãy review trước khi dùng cho training.",
    ),
    className: COLORS.muted,
  };
}

export function getGateBucketPresentation(bucket?: string | null): MlflowPresentation {
  if (bucket === "accepted") {
    return {
      label: "Đã chấp nhận",
      tooltip: makeMlflowTooltip(
        "Có thể xét để export",
        "Mẫu đã qua bước chấp nhận. Để vào tập training, mẫu vẫn cần được chọn và có nhãn Độc hại hoặc Sạch hợp lệ.",
      ),
      className: COLORS.automatic,
    };
  }
  if (bucket === "candidate") {
    return {
      label: "Cần xem lại",
      tooltip: makeMlflowTooltip(
        "Chưa đủ điều kiện export",
        "Mẫu cần được chấp nhận và có nhãn Độc hại hoặc Sạch hợp lệ trước khi có thể đưa vào tập training.",
      ),
      className: COLORS.neutral,
    };
  }
  if (bucket === "discarded") {
    return {
      label: "Đã loại",
      tooltip: makeMlflowTooltip(
        "Không dùng cho export",
        "Mẫu đã bị loại khỏi nhóm có thể đưa vào tập training. Nội dung vẫn còn để đối chiếu khi cần.",
      ),
      className: COLORS.muted,
    };
  }
  return {
    label: bucket || "Chưa có bucket",
    tooltip: makeMlflowTooltip(
      bucket ? "Trạng thái export khác" : "Chưa có trạng thái export",
      bucket
        ? "UI chưa có mô tả nghiệp vụ riêng cho trạng thái này. Mở Chi tiết kỹ thuật nếu cần kiểm tra giá trị lưu trữ."
        : "Mẫu chưa có trạng thái export rõ ràng. Hãy review trước khi dùng cho training.",
    ),
    className: COLORS.muted,
  };
}

export function getScorePresentation(
  score?: number | null,
  thresholds: MlflowGateThresholds = DEFAULT_MLFLOW_GATE_THRESHOLDS,
): MlflowPresentation {
  const formatted = typeof score === "number" && Number.isFinite(score) ? score.toFixed(3) : "-";
  const title = `Điểm độc hại: ${formatted}`;
  const finalLabelNote = "Điểm này không thay thế nhãn cuối cùng sau review.";

  if (typeof score !== "number" || !Number.isFinite(score)) {
    return {
      label: `Điểm độc hại ${formatted}`,
      tooltip: makeMlflowTooltip(title, `Chưa có điểm hợp lệ để đánh giá mức độ độc hại. ${finalLabelNote}`),
      className: COLORS.muted,
    };
  }
  if (score <= thresholds.discard) {
    return {
      label: `Điểm độc hại ${formatted}`,
      tooltip: makeMlflowTooltip(title, `Mức thấp: nội dung có ít tín hiệu độc hại theo mô hình hiện tại. ${finalLabelNote}`),
      className: COLORS.clean,
    };
  }
  if (score < 0.5) {
    return {
      label: `Điểm độc hại ${formatted}`,
      tooltip: makeMlflowTooltip(title, `Mức trung bình thấp: nên xem lại trước khi quyết định. ${finalLabelNote}`),
      className: COLORS.warning,
    };
  }
  if (score < thresholds.accept) {
    return {
      label: `Điểm độc hại ${formatted}`,
      tooltip: makeMlflowTooltip(title, `Mức cao: có tín hiệu độc hại đáng chú ý và cần review kỹ. ${finalLabelNote}`),
      className: COLORS.orange,
    };
  }
  return {
    label: `Điểm độc hại ${formatted}`,
    tooltip: makeMlflowTooltip(title, `Mức rất cao: mô hình nhận thấy tín hiệu độc hại mạnh. ${finalLabelNote}`),
    className: COLORS.toxic,
  };
}

export function getTrainingSelectionPresentation(
  selected?: number | boolean | null,
  exportEligible = true,
): MlflowPresentation {
  if (Boolean(selected)) {
    return {
      label: "Đã chọn cho training",
      tooltip: makeMlflowTooltip(
        "Đã chọn cho training",
        "Mẫu này có thể được sử dụng khi export dữ liệu. Trạng thái này không có nghĩa mẫu đã được huấn luyện.",
      ),
      className: COLORS.automatic,
    };
  }
  if (!exportEligible) {
    return {
      label: "Chưa đủ điều kiện export",
      tooltip: makeMlflowTooltip(
        "Chưa đủ điều kiện export",
        "Mẫu cần được chấp nhận và có nhãn Độc hại hoặc Sạch hợp lệ trước khi có thể đưa vào tập training.",
      ),
      className: COLORS.warning,
    };
  }
  return {
    label: "Chưa chọn cho training",
    tooltip: makeMlflowTooltip(
      "Chưa chọn cho training",
      "Bấm để thêm mẫu này vào danh sách dữ liệu có thể dùng khi export.",
    ),
    className: COLORS.muted,
  };
}

export function getLockPresentation(locked?: number | boolean | null): MlflowPresentation {
  if (Boolean(locked)) {
    return {
      label: "Đã khóa",
      tooltip: makeMlflowTooltip(
        "Đã khóa",
        "Không thể bỏ mẫu khỏi training khi đang khóa. Thao tác Remove trong Manual Verify cũng sẽ bỏ qua mẫu này.",
      ),
      className: COLORS.locked,
    };
  }
  return {
    label: "Chưa khóa",
    tooltip: makeMlflowTooltip(
      "Chưa khóa",
      "Bấm để khóa mẫu nếu bạn muốn giữ mẫu trong danh sách training. Khi chưa khóa, mẫu có thể được bỏ chọn hoặc Remove theo thao tác hiện tại.",
    ),
    className: COLORS.muted,
  };
}

export function formatMlflowConfidence(confidence?: string | null): string {
  if (confidence === "high") return "Tin cậy cao";
  if (confidence === "medium") return "Tin cậy vừa";
  if (confidence === "low") return "Tin cậy thấp";
  return "Chưa rõ độ tin cậy";
}

export function formatGeminiAction(action?: string | null): string {
  if (action === "apply") return "Có thể áp dụng";
  if (action === "review_more") return "Cần xem thêm";
  return action || "-";
}

interface MlflowBadgeProps extends Omit<ComponentProps<typeof Badge>, "children"> {
  presentation: MlflowPresentation;
  prefix?: string;
}

export function MlflowBadge({ presentation, prefix, className = "", ...badgeProps }: MlflowBadgeProps) {
  const visibleLabel = prefix ? `${prefix}: ${presentation.label}` : presentation.label;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          variant="outline"
          tabIndex={0}
          aria-label={`${visibleLabel}. ${presentation.tooltip}`}
          className={`${presentation.className} cursor-help focus-visible:outline-none ${className}`.trim()}
          {...badgeProps}
        >
          {visibleLabel}
        </Badge>
      </TooltipTrigger>
      <TooltipContent sideOffset={6}>
        <MlflowTooltipBody text={presentation.tooltip} />
      </TooltipContent>
    </Tooltip>
  );
}
