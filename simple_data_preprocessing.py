import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ICUDataPreprocessor:
    def __init__(self, outcomes_file, features_dir):
        """
        初始化数据预处理器
        
        Parameters:
        outcomes_file: 结果文件路径
        features_dir: 特征文件目录路径
        """
        self.outcomes_file = outcomes_file
        self.features_dir = features_dir
        self.outcomes_df = None
        self.features_df = None
        self.merged_df = None
        self.selected_features = None
        
    def load_outcomes(self):
        """加载结果数据"""
        print("正在加载结果数据...")
        self.outcomes_df = pd.read_csv(self.outcomes_file)
        print(f"结果数据形状: {self.outcomes_df.shape}")
        print(f"死亡率分布: {self.outcomes_df['In-hospital_death'].value_counts()}")
        print(f"死亡率: {self.outcomes_df['In-hospital_death'].mean():.3f}")
        return self.outcomes_df
    
    def load_features(self, sample_size=None):
        """
        加载特征数据
        
        Parameters:
        sample_size: 如果指定，只加载部分样本用于测试
        """
        print("正在加载特征数据...")
        
        # 获取所有特征文件
        feature_files = [f for f in os.listdir(self.features_dir) if f.endswith('.txt')]
        
        if sample_size:
            feature_files = feature_files[:sample_size]
            print(f"使用样本数据，加载 {len(feature_files)} 个文件")
        else:
            print(f"加载全部数据，共 {len(feature_files)} 个文件")
        
        all_features = []
        
        for i, file in enumerate(feature_files):
            if i % 100 == 0:
                print(f"已处理 {i}/{len(feature_files)} 个文件")
            
            file_path = os.path.join(self.features_dir, file)
            try:
                # 读取单个病人的时间序列数据
                df = pd.read_csv(file_path)
                record_id = int(file.split('.')[0])
                
                # 提取特征
                features = self.extract_features_from_timeseries(df, record_id)
                all_features.append(features)
                
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue
        
        self.features_df = pd.DataFrame(all_features)
        print(f"特征数据形状: {self.features_df.shape}")
        return self.features_df
    
    def extract_features_from_timeseries(self, df, record_id):
        """
        从时间序列数据中提取特征
        
        Parameters:
        df: 单个病人的时间序列数据
        record_id: 病人ID
        
        Returns:
        features: 提取的特征字典
        """
        features = {'RecordID': record_id}
        
        # 基本信息特征（这些通常是固定值）
        basic_params = ['Age', 'Gender', 'Height', 'Weight', 'ICUType']
        for param in basic_params:
            param_data = df[df['Parameter'] == param]['Value']
            if not param_data.empty:
                # 取第一个非-1的值，如果都是-1则保持-1
                valid_values = param_data[param_data != -1]
                features[param] = valid_values.iloc[0] if not valid_values.empty else -1
            else:
                features[param] = -1
        
        # 生理参数特征统计
        vital_params = ['HR', 'SysABP', 'DiasABP', 'MAP', 'RespRate', 'Temp', 'GCS']
        lab_params = ['pH', 'PaCO2', 'PaO2', 'SaO2', 'FiO2', 'HCO3', 'HCT', 'BUN', 
                     'Creatinine', 'Glucose', 'Na', 'K', 'Mg', 'WBC', 'Platelets',
                     'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin']
        
        # 非侵入性血压参数
        ni_params = ['NISysABP', 'NIDiasABP', 'NIMAP']
        
        all_params = vital_params + lab_params + ni_params + ['Urine', 'MechVent', 'Lactate']
        
        for param in all_params:
            param_data = df[df['Parameter'] == param]['Value']
            param_data = param_data[param_data != -1]  # 过滤缺失值
            
            if not param_data.empty:
                # 统计特征
                features[f'{param}_mean'] = param_data.mean()
                features[f'{param}_median'] = param_data.median()
                features[f'{param}_std'] = param_data.std()
                features[f'{param}_min'] = param_data.min()
                features[f'{param}_max'] = param_data.max()
                features[f'{param}_count'] = len(param_data)
                
                # 趋势特征（如果有多个值）
                if len(param_data) > 1:
                    features[f'{param}_trend'] = param_data.iloc[-1] - param_data.iloc[0]
                    features[f'{param}_range'] = param_data.max() - param_data.min()
                else:
                    features[f'{param}_trend'] = 0
                    features[f'{param}_range'] = 0
            else:
                # 如果没有数据，设置为缺失值
                for suffix in ['_mean', '_median', '_std', '_min', '_max', '_count', '_trend', '_range']:
                    features[f'{param}{suffix}'] = np.nan
        
        # 时间特征
        features['total_records'] = len(df)
        features['time_span'] = df['Time'].apply(self.parse_time).max() if not df.empty else 0
        
        return features
    
    def parse_time(self, time_str):
        """解析时间字符串为小时数"""
        try:
            if ':' in str(time_str):
                hours, minutes = map(int, str(time_str).split(':'))
                return hours + minutes / 60.0
            else:
                return float(time_str)
        except:
            return 0
    
    def merge_data(self):
        """合并特征数据和结果数据"""
        print("正在合并数据...")
        
        if self.outcomes_df is None or self.features_df is None:
            raise ValueError("请先加载结果数据和特征数据")
        
        self.merged_df = pd.merge(self.features_df, self.outcomes_df, on='RecordID', how='inner')
        print(f"合并后数据形状: {self.merged_df.shape}")
        print(f"合并后死亡率: {self.merged_df['In-hospital_death'].mean():.3f}")
        
        return self.merged_df
    
    def handle_missing_values(self, strategy='median'):
        """处理缺失值"""
        print("正在处理缺失值...")
        
        if self.merged_df is None:
            raise ValueError("请先合并数据")
        
        # 统计缺失值
        missing_stats = self.merged_df.isnull().sum()
        missing_percent = (missing_stats / len(self.merged_df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_stats,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Percent', ascending=False)
        
        print(f"缺失值最多的前10个特征:")
        print(missing_df.head(10))
        
        # 移除缺失值过多的特征（超过70%缺失）
        high_missing_features = missing_df[missing_df['Missing_Percent'] > 70].index.tolist()
        print(f"移除缺失值超过70%的特征: {len(high_missing_features)} 个")
        
        self.merged_df = self.merged_df.drop(columns=high_missing_features)
        
        # 处理剩余缺失值
        numeric_cols = self.merged_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['RecordID', 'In-hospital_death']]
        
        if strategy == 'median':
            self.merged_df[numeric_cols] = self.merged_df[numeric_cols].fillna(
                self.merged_df[numeric_cols].median()
            )
        elif strategy == 'mean':
            self.merged_df[numeric_cols] = self.merged_df[numeric_cols].fillna(
                self.merged_df[numeric_cols].mean()
            )
        elif strategy == 'zero':
            self.merged_df[numeric_cols] = self.merged_df[numeric_cols].fillna(0)
        
        print(f"处理缺失值后数据形状: {self.merged_df.shape}")
        return self.merged_df
    
    def simple_univariate_analysis(self, top_k=20):
        """简单的单因素分析 - 使用相关性分析"""
        print("正在进行单因素分析...")
        
        if self.merged_df is None:
            raise ValueError("请先合并并处理数据")
        
        # 准备特征和标签
        feature_cols = [col for col in self.merged_df.columns 
                       if col not in ['RecordID', 'SAPS-I', 'SOFA', 'Length_of_stay', 
                                    'Survival', 'In-hospital_death']]
        
        # 移除常数特征
        constant_features = []
        for col in feature_cols:
            if self.merged_df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"移除常数特征: {len(constant_features)} 个")
            feature_cols = [col for col in feature_cols if col not in constant_features]
        
        # 计算与目标变量的相关性
        correlations = []
        for col in feature_cols:
            try:
                corr = self.merged_df[col].corr(self.merged_df['In-hospital_death'])
                if not np.isnan(corr):
                    correlations.append({
                        'Feature': col,
                        'Correlation': abs(corr),
                        'Raw_Correlation': corr
                    })
            except:
                continue
        
        # 转换为DataFrame并排序
        correlation_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
        
        print(f"单因素分析完成，共分析 {len(correlation_df)} 个特征")
        print(f"前{top_k}个相关性最高的特征:")
        print(correlation_df.head(top_k))
        
        try:
            # 可视化top特征
            plt.figure(figsize=(12, 8))
            top_features = correlation_df.head(top_k)
            colors = ['red' if x < 0 else 'blue' for x in top_features['Raw_Correlation']]
            plt.barh(range(len(top_features)), top_features['Raw_Correlation'], color=colors)
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Correlation with In-hospital Death')
            plt.title(f'Top {top_k} Features - Univariate Analysis')
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('univariate_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("单因素分析图表已保存为 'univariate_analysis.png'")
        except Exception as e:
            print(f"保存图表时出错: {e}")
        
        return correlation_df
    
    def feature_statistics(self):
        """特征统计分析"""
        print("正在进行特征统计分析...")
        
        if self.merged_df is None:
            raise ValueError("请先合并并处理数据")
        
        # 按死亡结果分组分析
        dead_group = self.merged_df[self.merged_df['In-hospital_death'] == 1]
        alive_group = self.merged_df[self.merged_df['In-hospital_death'] == 0]
        
        print(f"死亡组样本数: {len(dead_group)}")
        print(f"存活组样本数: {len(alive_group)}")
        
        # 分析主要特征的差异
        key_features = ['Age', 'Gender', 'SAPS-I', 'SOFA', 'Length_of_stay']
        
        stats_results = []
        for feature in key_features:
            if feature in self.merged_df.columns:
                dead_mean = dead_group[feature].mean()
                alive_mean = alive_group[feature].mean()
                dead_std = dead_group[feature].std()
                alive_std = alive_group[feature].std()
                
                stats_results.append({
                    'Feature': feature,
                    'Dead_Mean': dead_mean,
                    'Dead_Std': dead_std,
                    'Alive_Mean': alive_mean,
                    'Alive_Std': alive_std,
                    'Difference': dead_mean - alive_mean
                })
        
        stats_df = pd.DataFrame(stats_results)
        print("\n主要特征统计分析:")
        print(stats_df.round(3))
        
        return stats_df
    
    def generate_final_dataset(self, top_features=50):
        """生成最终的预处理数据集"""
        print("正在生成最终数据集...")
        
        # 基本特征
        basic_features = ['RecordID', 'Age', 'Gender', 'Height', 'Weight', 'ICUType']
        
        # 如果有特征分析结果，选择最重要的特征
        if hasattr(self, 'feature_analysis_results'):
            selected_features = self.feature_analysis_results.head(top_features)['Feature'].tolist()
        else:
            # 否则选择所有数值特征
            numeric_cols = self.merged_df.select_dtypes(include=[np.number]).columns.tolist()
            selected_features = [col for col in numeric_cols 
                               if col not in ['RecordID', 'SAPS-I', 'SOFA', 'Length_of_stay', 
                                            'Survival', 'In-hospital_death']][:top_features]
        
        # 结果特征
        outcome_features = ['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death']
        
        # 合并所有特征
        final_features = basic_features + selected_features + outcome_features
        
        # 确保所有特征都存在
        final_features = [col for col in final_features if col in self.merged_df.columns]
        
        # 去重
        final_features = list(dict.fromkeys(final_features))
        
        final_df = self.merged_df[final_features].copy()
        
        print(f"最终数据集形状: {final_df.shape}")
        print(f"选择的特征数量: {len(final_features) - len(basic_features) - len(outcome_features)}")
        
        # 保存数据集
        final_df.to_csv('processed_icu_data.csv', index=False)
        print("最终数据集已保存为 'processed_icu_data.csv'")
        
        # 保存特征列表
        feature_info = pd.DataFrame({
            'Feature': final_features,
            'Type': ['Basic' if f in basic_features else 'Outcome' if f in outcome_features else 'Selected' for f in final_features]
        })
        feature_info.to_csv('feature_list.csv', index=False)
        print("特征列表已保存为 'feature_list.csv'")
        
        return final_df
    
    def data_summary(self):
        """数据摘要统计"""
        if self.merged_df is None:
            return
        
        print("\n=== 数据摘要 ===")
        print(f"总样本数: {len(self.merged_df)}")
        print(f"总特征数: {self.merged_df.shape[1] - 5}")  # 减去ID和结果变量
        print(f"死亡病例数: {self.merged_df['In-hospital_death'].sum()}")
        print(f"存活病例数: {len(self.merged_df) - self.merged_df['In-hospital_death'].sum()}")
        print(f"死亡率: {self.merged_df['In-hospital_death'].mean():.3f}")
        
        # 基本统计
        print(f"\n年龄统计:")
        age_data = self.merged_df['Age'][self.merged_df['Age'] != -1]
        if not age_data.empty:
            print(f"  平均年龄: {age_data.mean():.1f}")
            print(f"  年龄范围: {age_data.min():.0f} - {age_data.max():.0f}")
        
        print(f"\n性别分布:")
        gender_counts = self.merged_df['Gender'].value_counts()
        print(f"  男性(1): {gender_counts.get(1, 0)}")
        print(f"  女性(0): {gender_counts.get(0, 0)}")
        
        print(f"\nICU类型分布:")
        icu_counts = self.merged_df['ICUType'].value_counts()
        for icu_type, count in icu_counts.items():
            print(f"  类型{icu_type}: {count}")
        
        # SAPS-I和SOFA评分统计
        if 'SAPS-I' in self.merged_df.columns:
            saps_data = self.merged_df['SAPS-I'][self.merged_df['SAPS-I'] != -1]
            if not saps_data.empty:
                print(f"\nSAPS-I评分统计:")
                print(f"  平均值: {saps_data.mean():.1f}")
                print(f"  范围: {saps_data.min():.0f} - {saps_data.max():.0f}")
        
        if 'SOFA' in self.merged_df.columns:
            sofa_data = self.merged_df['SOFA'][self.merged_df['SOFA'] != -1]
            if not sofa_data.empty:
                print(f"\nSOFA评分统计:")
                print(f"  平均值: {sofa_data.mean():.1f}")
                print(f"  范围: {sofa_data.min():.0f} - {sofa_data.max():.0f}")


def main():
    """主函数"""
    print("开始ICU数据预处理...")
    
    # 文件路径
    outcomes_file = r"c:\Users\dell\Desktop\DM文件夹\6月6\202510月\不爱喝小米粥爱喝冬瓜汤\Outcomes-group_10.txt"
    features_dir = r"c:\Users\dell\Desktop\DM文件夹\6月6\202510月\不爱喝小米粥爱喝冬瓜汤\Features_group_10"
    
    # 初始化预处理器
    preprocessor = ICUDataPreprocessor(outcomes_file, features_dir)
    
    # 步骤1: 加载数据
    outcomes_df = preprocessor.load_outcomes()
    
    # 询问用户是否要处理全部数据
    print("\n选择处理模式:")
    print("1. 小样本测试 (100个文件)")
    print("2. 中等样本 (500个文件)")
    print("3. 处理全部数据 (4000个文件)")
    choice = input("请输入选择 (1, 2 或 3, 默认为1): ").strip()
    
    if choice == "3":
        print("正在处理全部数据，这可能需要较长时间...")
        features_df = preprocessor.load_features(sample_size=None)
    elif choice == "2":
        print("正在处理中等样本数据...")
        features_df = preprocessor.load_features(sample_size=500)
    else:
        print("使用小样本数据进行测试...")
        features_df = preprocessor.load_features(sample_size=100)
    
    # 步骤2: 合并数据
    merged_df = preprocessor.merge_data()
    
    # 步骤3: 处理缺失值
    processed_df = preprocessor.handle_missing_values()
    
    # 步骤4: 数据摘要
    preprocessor.data_summary()
    
    # 步骤5: 特征统计分析
    stats_results = preprocessor.feature_statistics()
    
    # 步骤6: 简单单因素分析
    correlation_features = preprocessor.simple_univariate_analysis()
    preprocessor.feature_analysis_results = correlation_features
    
    # 步骤7: 生成最终数据集
    final_df = preprocessor.generate_final_dataset()
    
    print("\n数据预处理完成!")
    print(f"处理了 {len(features_df)} 个样本的时间序列数据")
    print(f"最终数据集包含 {len(final_df)} 个样本和 {final_df.shape[1]-6} 个预测特征")
    
    # 保存分析结果
    correlation_features.to_csv('feature_analysis_results.csv', index=False)
    stats_results.to_csv('feature_statistics_results.csv', index=False)
    
    print("\n生成的文件:")
    print("- processed_icu_data.csv: 预处理后的数据集")
    print("- feature_analysis_results.csv: 特征分析结果")
    print("- feature_statistics_results.csv: 特征统计结果")
    print("- feature_list.csv: 特征列表")
    print("- univariate_analysis.png: 单因素分析图表")


if __name__ == "__main__":
    main()