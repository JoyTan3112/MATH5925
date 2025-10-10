import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class NoLeakageICUPreprocessor:
    """
    防止数据泄露的ICU数据预处理器
    """
    
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
        self.raw_data = None
        
    def load_outcomes(self):
        """加载结果数据"""
        print("正在加载结果数据...")
        self.outcomes_df = pd.read_csv(self.outcomes_file)
        print(f"结果数据形状: {self.outcomes_df.shape}")
        print(f"死亡率分布: {self.outcomes_df['In-hospital_death'].value_counts()}")
        print(f"总体死亡率: {self.outcomes_df['In-hospital_death'].mean():.3f}")
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
        
        # 基本信息特征
        basic_params = ['Age', 'Gender', 'Height', 'Weight', 'ICUType']
        for param in basic_params:
            param_data = df[df['Parameter'] == param]['Value']
            if not param_data.empty:
                valid_values = param_data[param_data != -1]
                features[param] = valid_values.iloc[0] if not valid_values.empty else -1
            else:
                features[param] = -1
        
        # 生理参数和实验室指标
        vital_params = ['HR', 'SysABP', 'DiasABP', 'MAP', 'RespRate', 'Temp', 'GCS']
        lab_params = ['pH', 'PaCO2', 'PaO2', 'SaO2', 'FiO2', 'HCO3', 'HCT', 'BUN', 
                     'Creatinine', 'Glucose', 'Na', 'K', 'Mg', 'WBC', 'Platelets',
                     'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin']
        ni_params = ['NISysABP', 'NIDiasABP', 'NIMAP']
        other_params = ['Urine', 'MechVent', 'Lactate']
        
        all_params = vital_params + lab_params + ni_params + other_params
        
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
                
                # 趋势和范围特征
                if len(param_data) > 1:
                    features[f'{param}_trend'] = param_data.iloc[-1] - param_data.iloc[0]
                    features[f'{param}_range'] = param_data.max() - param_data.min()
                else:
                    features[f'{param}_trend'] = 0
                    features[f'{param}_range'] = 0
            else:
                # 设置为缺失值
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
    
    def merge_and_prepare_data(self):
        """合并数据并准备用于分割"""
        print("正在合并数据...")
        
        if self.outcomes_df is None or self.features_df is None:
            raise ValueError("请先加载结果数据和特征数据")
        
        # 合并数据
        self.raw_data = pd.merge(self.features_df, self.outcomes_df, on='RecordID', how='inner')
        print(f"合并后数据形状: {self.raw_data.shape}")
        print(f"合并后死亡率: {self.raw_data['In-hospital_death'].mean():.3f}")
        
        return self.raw_data
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """
        进行训练-测试分割，避免数据泄露
        """
        print("正在进行训练-测试分割...")
        
        if self.raw_data is None:
            raise ValueError("请先合并数据")
        
        # 分离特征和标签
        feature_cols = [col for col in self.raw_data.columns 
                       if col not in ['RecordID', 'SAPS-I', 'SOFA', 'Length_of_stay', 
                                    'Survival', 'In-hospital_death']]
        
        X = self.raw_data[feature_cols].copy()
        y = self.raw_data['In-hospital_death'].copy()
        
        # 进行分层分割
        from sklearn.model_selection import train_test_split
        
        # 添加RecordID用于跟踪
        X_with_id = self.raw_data[['RecordID'] + feature_cols].copy()
        
        X_train_with_id, X_test_with_id, y_train, y_test = train_test_split(
            X_with_id, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 分离ID和特征
        train_ids = X_train_with_id['RecordID']
        test_ids = X_test_with_id['RecordID']
        X_train = X_train_with_id.drop('RecordID', axis=1)
        X_test = X_test_with_id.drop('RecordID', axis=1)
        
        print(f"训练集大小: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
        print(f"测试集大小: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征")
        print(f"训练集死亡率: {y_train.mean():.3f}")
        print(f"测试集死亡率: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test, train_ids, test_ids
    
    def preprocess_train_data(self, X_train, y_train, missing_threshold=0.7):
        """
        只在训练集上进行预处理，学习参数
        """
        print("正在处理训练集...")
        
        X_train_processed = X_train.copy()
        
        # 1. 计算缺失值统计（只在训练集上）
        missing_stats = X_train_processed.isnull().sum()
        missing_percent = (missing_stats / len(X_train_processed)) * 100
        
        # 2. 移除高缺失率特征
        high_missing_features = missing_stats[missing_percent > missing_threshold * 100].index.tolist()
        print(f"移除缺失值超过{missing_threshold*100}%的特征: {len(high_missing_features)} 个")
        
        X_train_processed = X_train_processed.drop(columns=high_missing_features)
        
        # 3. 移除常数特征（只在训练集上判断）
        constant_features = []
        for col in X_train_processed.columns:
            if X_train_processed[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"移除常数特征: {len(constant_features)} 个")
            X_train_processed = X_train_processed.drop(columns=constant_features)
        
        # 4. 计算填充值（只在训练集上）
        numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns
        train_medians = X_train_processed[numeric_cols].median()
        
        # 5. 填充缺失值
        X_train_processed[numeric_cols] = X_train_processed[numeric_cols].fillna(train_medians)
        
        # 6. 进行单变量特征选择（只在训练集上）
        feature_scores = self.calculate_univariate_scores(X_train_processed, y_train)
        
        print(f"训练集预处理完成，形状: {X_train_processed.shape}")
        
        # 保存预处理参数
        self.preprocessing_params = {
            'removed_features': high_missing_features + constant_features,
            'fill_values': train_medians,
            'feature_scores': feature_scores,
            'selected_columns': X_train_processed.columns.tolist()
        }
        
        return X_train_processed, feature_scores
    
    def preprocess_test_data(self, X_test):
        """
        使用训练集学到的参数预处理测试集
        """
        print("正在处理测试集...")
        
        if not hasattr(self, 'preprocessing_params'):
            raise ValueError("请先处理训练集以学习预处理参数")
        
        X_test_processed = X_test.copy()
        
        # 1. 移除相同的特征
        features_to_remove = [col for col in self.preprocessing_params['removed_features'] 
                             if col in X_test_processed.columns]
        X_test_processed = X_test_processed.drop(columns=features_to_remove)
        
        # 2. 确保列顺序一致
        selected_columns = [col for col in self.preprocessing_params['selected_columns'] 
                           if col in X_test_processed.columns]
        X_test_processed = X_test_processed[selected_columns]
        
        # 3. 使用训练集的填充值
        numeric_cols = X_test_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in self.preprocessing_params['fill_values']:
                fill_value = self.preprocessing_params['fill_values'][col]
                X_test_processed[col] = X_test_processed[col].fillna(fill_value)
        
        print(f"测试集预处理完成，形状: {X_test_processed.shape}")
        
        return X_test_processed
    
    def calculate_univariate_scores(self, X, y, top_k=20):
        """
        计算单变量特征分数
        """
        print("正在进行单变量特征分析...")
        
        correlations = []
        for col in X.columns:
            try:
                # 计算相关系数
                corr = X[col].corr(y)
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
        
        print(f"单变量分析完成，共分析 {len(correlation_df)} 个特征")
        print(f"前{top_k}个相关性最高的特征:")
        print(correlation_df.head(top_k))
        
        return correlation_df
    
    def select_top_features(self, X_train, X_test, feature_scores, top_k=50):
        """
        选择top特征
        """
        print(f"选择前{top_k}个重要特征...")
        
        # 获取top特征名称
        top_features = feature_scores.head(top_k)['Feature'].tolist()
        
        # 确保特征在数据中存在
        available_features = [f for f in top_features if f in X_train.columns and f in X_test.columns]
        
        print(f"实际选择了 {len(available_features)} 个特征")
        
        X_train_selected = X_train[available_features]
        X_test_selected = X_test[available_features]
        
        return X_train_selected, X_test_selected, available_features
    
    def generate_summary_report(self, X_train, X_test, y_train, y_test, feature_scores, selected_features):
        """
        生成处理摘要报告
        """
        print("\n" + "="*50)
        print("数据预处理摘要报告")
        print("="*50)
        
        print(f"\n数据集大小:")
        print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
        print(f"  测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征")
        
        print(f"\n目标变量分布:")
        print(f"  训练集死亡率: {y_train.mean():.3f} ({y_train.sum()}/{len(y_train)})")
        print(f"  测试集死亡率: {y_test.mean():.3f} ({y_test.sum()}/{len(y_test)})")
        
        print(f"\n特征工程结果:")
        print(f"  原始特征数: {self.raw_data.shape[1] - 6}")  # 减去ID和结果变量
        print(f"  最终特征数: {len(selected_features)}")
        
        print(f"\n前10个重要特征:")
        for i, feature in enumerate(selected_features[:10], 1):
            score = feature_scores[feature_scores['Feature'] == feature]['Raw_Correlation'].iloc[0]
            print(f"  {i:2d}. {feature:30s} (相关性: {score:6.3f})")
        
        print(f"\n数据泄露预防措施:")
        print(f"  ✓ 先分割数据，再进行预处理")
        print(f"  ✓ 填充值只从训练集计算")
        print(f"  ✓ 特征选择只在训练集上进行")
        print(f"  ✓ 测试集使用训练集的预处理参数")
        
        # 保存处理参数
        summary_data = {
            'preprocessing_params': self.preprocessing_params,
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'data_shapes': {
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'train_death_rate': float(y_train.mean()),
                'test_death_rate': float(y_test.mean())
            }
        }
        
        return summary_data


def main():
    """主函数"""
    print("开始无数据泄露的ICU数据预处理...")
    
    # 文件路径
    outcomes_file = r"c:\Users\dell\Desktop\DM文件夹\6月6\202510月\不爱喝小米粥爱喝冬瓜汤\Outcomes-group_10.txt"
    features_dir = r"c:\Users\dell\Desktop\DM文件夹\6月6\202510月\不爱喝小米粥爱喝冬瓜汤\Features_group_10"
    
    # 初始化预处理器
    preprocessor = NoLeakageICUPreprocessor(outcomes_file, features_dir)
    
    # 步骤1: 加载数据
    preprocessor.load_outcomes()
    
    # 选择处理模式
    print("\n选择处理模式:")
    print("1. 小样本测试 (100个文件)")
    print("2. 中等样本 (500个文件)")
    print("3. 处理全部数据 (4000个文件)")
    choice = input("请输入选择 (1, 2 或 3, 默认为1): ").strip()
    
    if choice == "3":
        preprocessor.load_features(sample_size=None)
    elif choice == "2":
        preprocessor.load_features(sample_size=500)
    else:
        preprocessor.load_features(sample_size=100)
    
    # 步骤2: 合并数据
    preprocessor.merge_and_prepare_data()
    
    # 步骤3: 关键 - 先分割数据
    X_train, X_test, y_train, y_test, train_ids, test_ids = preprocessor.train_test_split()
    
    # 步骤4: 只在训练集上进行预处理
    X_train_processed, feature_scores = preprocessor.preprocess_train_data(X_train, y_train)
    
    # 步骤5: 使用训练集参数处理测试集
    X_test_processed = preprocessor.preprocess_test_data(X_test)
    
    # 步骤6: 特征选择
    X_train_final, X_test_final, selected_features = preprocessor.select_top_features(
        X_train_processed, X_test_processed, feature_scores, top_k=50
    )
    
    # 步骤7: 生成报告
    summary = preprocessor.generate_summary_report(
        X_train_final, X_test_final, y_train, y_test, feature_scores, selected_features
    )
    
    # 保存结果（现在是安全的）
    print("\n正在保存处理结果...")
    
    # 保存训练集
    train_data = X_train_final.copy()
    train_data['RecordID'] = train_ids.values
    train_data['In-hospital_death'] = y_train.values
    train_data.to_csv('train_data_no_leakage.csv', index=False)
    
    # 保存测试集
    test_data = X_test_final.copy()
    test_data['RecordID'] = test_ids.values
    test_data['In-hospital_death'] = y_test.values
    test_data.to_csv('test_data_no_leakage.csv', index=False)
    
    # 保存特征分析结果
    feature_scores.to_csv('feature_importance_no_leakage.csv', index=False)
    
    # 保存处理参数
    import json
    with open('preprocessing_params.json', 'w') as f:
        # 转换numpy类型为Python类型
        params_to_save = {
            'selected_features': selected_features,
            'removed_features': preprocessor.preprocessing_params['removed_features'],
            'fill_values': {k: float(v) if not np.isnan(v) else None 
                           for k, v in preprocessor.preprocessing_params['fill_values'].items()},
            'data_shapes': summary['data_shapes']
        }
        json.dump(params_to_save, f, indent=2)
    
    print("\n处理完成！生成的文件:")
    print("- train_data_no_leakage.csv: 训练集数据")
    print("- test_data_no_leakage.csv: 测试集数据")
    print("- feature_importance_no_leakage.csv: 特征重要性")
    print("- preprocessing_params.json: 预处理参数")
    
    print(f"\n✓ 无数据泄露的预处理完成！")
    print(f"✓ 训练集: {X_train_final.shape[0]} 样本, {X_train_final.shape[1]} 特征")
    print(f"✓ 测试集: {X_test_final.shape[0]} 样本, {X_test_final.shape[1]} 特征")


if __name__ == "__main__":
    # 检查是否有scikit-learn
    try:
        from sklearn.model_selection import train_test_split
        main()
    except ImportError:
        print("错误: 需要安装 scikit-learn")
        print("请运行: pip install scikit-learn")