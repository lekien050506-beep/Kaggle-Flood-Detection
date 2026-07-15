import pandas as pd

# 1. Đọc 2 file kết quả chứa xác suất
df_effnet = pd.read_csv("submission_5fold_ensemble.csv")
df_convnext = pd.read_csv("submission_convnext_ensemble.csv")

# 2. CÀI ĐẶT TỈ LỆ TỐI ƯU (Trọng số)
# Tổng 2 số này phải luôn bằng 1.0
# Mẹo: Mô hình nào F1-Score cao hơn trên Kaggle thì cho nó trọng số lớn hơn (VD: 0.6 và 0.4)
WEIGHT_EFFNET = 0.25
WEIGHT_CONVNEXT = 0.75

print(f"Đang trộn mô hình với tỉ lệ: EfficientNet ({WEIGHT_EFFNET*100}%) - ConvNeXt ({WEIGHT_CONVNEXT*100}%)")

# 3. Trộn xác suất (Trung bình cộng có trọng số)
blended_probs = (df_effnet['prob'] * WEIGHT_EFFNET) + (df_convnext['prob'] * WEIGHT_CONVNEXT)

# 4. Áp dụng ngưỡng (Threshold) để chốt nhãn cuối cùng
# Ngưỡng 0.45 thường rất tốt cho bài toán ngập lụt (ưu tiên bắt được nhiều ảnh ngập hơn)
OPTIMAL_THRESHOLD = 0.45
final_labels = (blended_probs >= OPTIMAL_THRESHOLD).astype(int)

# 5. Xuất ra file Submission cuối cùng để nộp bài
sub_df = pd.DataFrame({
    "id": df_effnet["id"],
    "label": final_labels
})

sub_df.to_csv("submission_ultimate_ensemble.csv", index=False)
print("=== THÀNH CÔNG! Đã tạo file submission_ultimate_ensemble.csv ===")