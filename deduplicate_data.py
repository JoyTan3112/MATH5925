import pandas as pd

# 读取当前的 CSV
df = pd.read_csv("processed_icu_data_causal.csv")

print("原始数据形状:", df.shape)

# 检查关键标识列（比如 subject_id 或 icustay_id）
if 'subject_id' in df.columns:
    print("唯一 subject_id 数量:", df['subject_id'].nunique())
if 'icustay_id' in df.columns:
    print("唯一 icustay_id 数量:", df['icustay_id'].nunique())
# 在文件最后加上这些检查：
if 'subject_id' in df.columns:
    print("唯一 subject_id 数量:", df['subject_id'].nunique())
if 'icustay_id' in df.columns:
    print("唯一 icustay_id 数量:", df['icustay_id'].nunique())

# 检查是否存在重复行
dup_count = df.duplicated().sum()
print(f"检测到 {dup_count} 行重复记录")

# 删除重复行
df = df.drop_duplicates()

print("去重后数据形状:", df.shape)

# 保存新的 CSV
df.to_csv("processed_icu_data_causal_dedup.csv", index=False)
print("✅ 已保存去重后的数据 → processed_icu_data_causal_dedup.csv")
