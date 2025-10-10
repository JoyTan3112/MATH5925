import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
                     'Creatinine', 'Glucose', 'Na', 'K', 'Mg', 'WBC', 'Platelets']
        
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
        
        # 移除缺失值过多的特征（超过50%缺失）
        high_missing_features = missing_df[missing_df['Missing_Percent'] > 50].index.tolist()
        print(f"移除缺失值超过50%的特征: {len(high_missing_features)} 个")
        
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
    
    def univariate_analysis(self, top_k=20):
        """单因素分析"""
        print("正在进行单因素分析...")
        
        if self.merged_df is None:
            raise ValueError("请先合并并处理数据")
        
        # 准备特征和标签
        feature_cols = [col for col in self.merged_df.columns 
                       if col not in ['RecordID', 'SAPS-I', 'SOFA', 'Length_of_stay', 
                                    'Survival', 'In-hospital_death']]
        
        X = self.merged_df[feature_cols]
        y = self.merged_df['In-hospital_death']
        
        # 移除常数特征
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"移除常数特征: {len(constant_features)} 个")
            X = X.drop(columns=constant_features)
        
        # 使用F统计量进行特征选择
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        # 获取特征得分
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_,
            'P_value': selector.pvalues_
        }).sort_values('Score', ascending=False)
        
        # 选择显著特征 (p < 0.05)
        significant_features = feature_scores[feature_scores['P_value'] < 0.05]
        
        print(f"单因素分析完成，共 {len(significant_features)} 个显著特征")
        print(f"前{top_k}个最重要的特征:")
        print(significant_features.head(top_k))
        
        # 可视化top特征
        plt.figure(figsize=(12, 8))
        top_features = significant_features.head(top_k)
        plt.barh(range(len(top_features)), top_features['Score'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('F-Score')
        plt.title(f'Top {top_k} Features - Univariate Analysis')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('univariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return significant_features
    
    def feature_correlation_analysis(self, selected_features=None, top_k=20):
        """特征相关性分析 - 替代多因素分析"""
        print("正在进行特征相关性分析...")
        
        if self.merged_df is None:
            raise ValueError("请先合并并处理数据")
        
        # 准备特征
        if selected_features is not None:
            feature_cols = selected_features['Feature'].head(50).tolist()
        else:
            feature_cols = [col for col in self.merged_df.columns 
                           if col not in ['RecordID', 'SAPS-I', 'SOFA', 'Length_of_stay', 
                                        'Survival', 'In-hospital_death']]
        
        # 移除不存在的特征
        feature_cols = [col for col in feature_cols if col in self.merged_df.columns]
        
        # 计算与目标变量的相关性
        target_correlations = []
        for col in feature_cols:
            try:
                corr = self.merged_df[col].corr(self.merged_df['In-hospital_death'])
                if not np.isnan(corr):
                    target_correlations.append({
                        'Feature': col,
                        'Correlation': abs(corr),
                        'Raw_Correlation': corr
                    })
            except:
                continue
        
        # 转换为DataFrame并排序
        correlation_df = pd.DataFrame(target_correlations).sort_values('Correlation', ascending=False)
        
        print(f"与死亡率相关性最高的前{top_k}个特征:")
        print(correlation_df.head(top_k))
        
        # 可视化相关性
        plt.figure(figsize=(12, 8))
        top_features = correlation_df.head(top_k)
        colors = ['red' if x < 0 else 'blue' for x in top_features['Raw_Correlation']]
        plt.barh(range(len(top_features)), top_features['Raw_Correlation'], color=colors)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Correlation with In-hospital Death')
        plt.title(f'Top {top_k} Features - Correlation Analysis')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.selected_features = correlation_df.head(30)
        return correlation_df
    
    def generate_final_dataset(self, top_features=30):
        """生成最终的预处理数据集"""
        print("正在生成最终数据集...")
        
        if self.selected_features is None:
            raise ValueError("请先进行特征选择")
        
        # 选择最终特征
        final_features = ['RecordID'] + self.selected_features.head(top_features)['Feature'].tolist()
        final_features.extend(['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death'])
        
        # 确保所有特征都存在
        final_features = [col for col in final_features if col in self.merged_df.columns]
        
        final_df = self.merged_df[final_features].copy()
        
        print(f"最终数据集形状: {final_df.shape}")
        print(f"最终特征数量: {len(final_features) - 5}")  # 减去ID和目标变量
        
        # 保存数据集
        final_df.to_csv('processed_icu_data.csv', index=False)
        print("最终数据集已保存为 'processed_icu_data.csv'")
        
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
    print("2. 处理全部数据 (4000个文件)")
    choice = input("请输入选择 (1 或 2, 默认为1): ").strip()
    
    if choice == "2":
        print("正在处理全部数据，这可能需要较长时间...")
        features_df = preprocessor.load_features(sample_size=None)
    else:
        print("使用小样本数据进行测试...")
        features_df = preprocessor.load_features(sample_size=100)
    
    # 步骤2: 合并数据
    merged_df = preprocessor.merge_data()
    
    # 步骤3: 处理缺失值
    processed_df = preprocessor.handle_missing_values()
    
    # 步骤4: 数据摘要
    preprocessor.data_summary()
    
    # 步骤5: 单因素分析
    significant_features = preprocessor.univariate_analysis()
    
    # 步骤6: 特征相关性分析
    correlation_features = preprocessor.feature_correlation_analysis(significant_features)
    
    # 步骤7: 生成最终数据集
    final_df = preprocessor.generate_final_dataset()
    
    print("\n数据预处理完成!")
    print(f"处理了 {len(features_df)} 个样本的时间序列数据")
    print(f"最终数据集包含 {len(final_df)} 个样本和 {final_df.shape[1]-5} 个特征")
    print("预处理后的数据已保存为 'processed_icu_data.csv'")
    
    # 保存特征分析结果
    significant_features.to_csv('univariate_analysis_results.csv', index=False)
    correlation_features.to_csv('correlation_analysis_results.csv', index=False)
    print("特征分析结果已保存为 'univariate_analysis_results.csv' 和 'correlation_analysis_results.csv'")


if __name__ == "__main__":
    main()