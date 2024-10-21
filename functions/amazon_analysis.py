import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import os
import pickle
import json

class AmazonDataframe:
    def __init__(self, file_path):
        self.file_path = file_path
        self._df = None
        self.verify_file_path()

    def verify_file_path(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        if not os.path.isfile(self.file_path):
            raise IsADirectoryError(f"The path {self.file_path} is a directory, not a file.")
        if not os.access(self.file_path, os.R_OK):
            raise PermissionError(f"You don't have permission to read the file {self.file_path}")
        print(f"File path verified: {self.file_path}")

    @property
    def df(self):
        if self._df is None:
            try:
                self._df = pd.read_csv(self.file_path)
                print(f"Successfully loaded CSV from {self.file_path}")
            except pd.errors.EmptyDataError:
                print(f"Warning: The file {self.file_path} is empty. Creating an empty DataFrame.")
                self._df = pd.DataFrame()
            except Exception as e:
                print(f"Error loading CSV from {self.file_path}: {str(e)}")
                raise
        return self._df

    def info(self):
        return self.df.info()

    def basic_stats(self):
        return self.df.describe()

    def column_names(self):
        return list(self.df.columns)

    # Add more EDA methods as needed

class AmazonAnalyzer:
    def __init__(self, cache_dir='./cache'):
        self.dataframes = {}
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def add_dataframe(self, name, file_path):
        cache_path = os.path.join(self.cache_dir, f"{name}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.dataframes[name] = pickle.load(f)
                print(f"Loaded {name} from cache")
            except (pickle.PickleError, EOFError):
                print(f"Cache for {name} is corrupted. Loading from CSV.")
                self._load_and_cache(name, file_path, cache_path)
            except FileNotFoundError:
                print(f"Error: File not found - {file_path}")
        else:
            self._load_and_cache(name, file_path, cache_path)

    def _load_and_cache(self, name, file_path, cache_path):
        try:
            df = AmazonDataframe(file_path)
            self.dataframes[name] = df
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            print(f"Loaded {name} from CSV and cached")
        except Exception as e:
            print(f"Error loading {name}: {str(e)}")

    def clear_cache(self):
        for file in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, file))
        print("Cache cleared")

    def get_dataframe(self, name):
        return self.dataframes[name].df

    def compare_columns_presence(self):
        try:
            if not self.dataframes:
                print("No dataframes loaded.")
                return None

            all_columns = set()
            for df in self.dataframes.values():
                all_columns.update(df.column_names())
            
            presence_dict = {col: [] for col in all_columns}
            
            for name, df in self.dataframes.items():
                df_columns = set(df.column_names())
                for col in all_columns:
                    presence_dict[col].append(col in df_columns)
            
            presence_df = pd.DataFrame(presence_dict, index=self.dataframes.keys())
            return presence_df.T.astype(bool)
        except Exception as e:
            print(f"Error in compare_columns_presence: {str(e)}")
            return None

    def get_common_columns(self):
        presence_df = self.compare_columns_presence()
        return presence_df[presence_df.all(axis=1)].index.tolist()


    def top_items(self, df_name, column, n=10, sort_by=None):
        df = self.get_dataframe(df_name)
        if sort_by is None:
            return df[column].value_counts().nlargest(n)
        else:
            return df.groupby(column)[sort_by].sum().nlargest(n)

    def top_items_filtered(self, df_name, column, filter_column=None, filter_value=None, sort_by=None, n=10):
        """
        Get top items with optional filtering and sorting
        
        Parameters:
        df_name (str): Name of the dataframe to analyze
        column (str): Column to show in results (e.g., 'product_name')
        filter_column (str): Column to filter on (e.g., 'category')
        filter_value (str): Value to filter by (e.g., 'suitcase')
        sort_by (str): Column to sort by (e.g., 'discounted_price')
        n (int): Number of results to return
        
        Returns:
        pandas.Series: Top n items meeting the criteria
        """
        df = self.get_dataframe(df_name)
        
        # Apply filter if specified
        if filter_column is not None and filter_value is not None:
            df = df[df[filter_column] == filter_value]
        
        # Sort and group
        if sort_by is None:
            return df[column].value_counts().nlargest(n)
        else:
            # Return both the sort column and the display column
            result = df.sort_values(by=sort_by, ascending=False)[['product_name', sort_by]].head(n)
            return result

    def top_items_multi_filter(self, df_name, column, filters=None, sort_by=None, n=10):
        """
        filters = {
            'category': 'suitcase',
            'isBestSeller': True,
            'rating': 4.5
        }
        """
        df = self.get_dataframe(df_name)
        
        # Apply multiple filters
        if filters:
            for col, value in filters.items():
                df = df[df[col] == value]
        
        if sort_by is None:
            return df[column].value_counts().nlargest(n)
        else:
            result = df.sort_values(by=sort_by, ascending=False)[[column, sort_by]].head(n)
            return result

    def top_items_price_range(self, df_name, column, price_column, min_price=None, max_price=None, n=10):
        df = self.get_dataframe(df_name)
        
        if min_price is not None:
            df = df[df[price_column] >= min_price]
        if max_price is not None:
            df = df[df[price_column] <= max_price]
        
        return df.sort_values(by=price_column, ascending=False)[[column, price_column]].head(n)

    def plot_distribution(self, df_name, column):
        try:
            if df_name not in self.dataframes:
                print(f"Error: Dataframe '{df_name}' not found.")
                return

            df = self.dataframes[df_name].df
            if column not in df.columns:
                print(f"Error: Column '{column}' not found in dataframe '{df_name}'.")
                return

            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], kde=True)
            plt.title(f'Distribution of {column} in {df_name}')
            plt.show()
        except Exception as e:
            print(f"Error in plot_distribution: {str(e)}")

    def analyze_categories(self, df_name, top_n=20, show_percent=True):
        """
        카테고리 분석을 수행하는 메서드
        
        Parameters:
        df_name (str): 분석할 데이터프레임 이름
        top_n (int): 상위 몇 개의 카테고리를 볼 것인지
        show_percent (bool): 백분율 표시 여부
        
        Returns:
        pandas.DataFrame: 카테고리 분석 결과
        """
        df = self.get_dataframe(df_name)
        
        # 카테고리별 상품 수 계산
        category_counts = df['category'].value_counts()
        total_items = len(df)
        
        # 상위 N개 카테고리 선택
        top_categories = category_counts.head(top_n)
        
        # 결과 데이터프레임 생성
        results = pd.DataFrame({
            'category': top_categories.index,
            'item_count': top_categories.values,
            'percentage': (top_categories.values / total_items * 100).round(2)
        })
        
        # 결과 출력
        print(f"\n=== {df_name} 카테고리 분석 ===")
        print(f"총 카테고리 수: {len(category_counts):,}개")
        print(f"총 상품 수: {total_items:,}개\n")
        
        print(f"상위 {top_n}개 카테고리:")
        for idx, row in results.iterrows():
            if show_percent:
                print(f"{idx+1:2d}. {row['category']:<40} {row['item_count']:,}개 ({row['percentage']}%)")
            else:
                print(f"{idx+1:2d}. {row['category']:<40} {row['item_count']:,}개")
        
        # 기타 카테고리 정보 출력
        other_items = total_items - results['item_count'].sum()
        other_percent = (other_items / total_items * 100).round(2)
        print(f"\n기타 카테고리: {other_items:,}개 ({other_percent}%)")
        
        return results

    def analyze_category_metrics(self, df_name, metric_column, top_n=20):
        """
        카테고리별 특정 지표 분석
        
        Parameters:
        df_name (str): 데이터프레임 이름
        metric_column (str): 분석할 지표 컬럼 (예: 'discounted_price', 'rating')
        top_n (int): 상위 몇 개의 카테고리를 볼 것인지
        """
        df = self.get_dataframe(df_name)
        
        # 카테고리별 평균값 계산
        category_metrics = df.groupby('category')[metric_column].agg(['mean', 'count', 'std']).round(2)
        category_metrics = category_metrics.sort_values('mean', ascending=False).head(top_n)
        
        print(f"\n=== 카테고리별 평균 {metric_column} 분석 (상위 {top_n}개) ===")
        for idx, (category, row) in enumerate(category_metrics.iterrows(), 1):
            if metric_column.lower().find('price') >= 0:
                mean_value = format_korean_number(row['mean'])  # 이전에 만든 가격 포맷팅 함수 사용
            else:
                mean_value = f"{row['mean']:.2f}"
            
            print(f"{idx:2d}. {category:<40} {mean_value} (상품수: {row['count']:,}개)")

    @classmethod
    def from_config(cls, config_filename, project_root):
        # config_path = os.path.join(project_root, 'functions', config_filename)
        config_path = os.path.join(project_root, config_filename)  
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            analyzer = cls()
            for name, path in config['dataframes'].items():
                full_path = os.path.abspath(os.path.join(project_root, path))
                print(f"Attempting to access: {full_path}")
                
                try:
                    analyzer.add_dataframe(name, full_path)
                    print(f"Successfully added dataframe: {name}")
                except FileNotFoundError:
                    print(f"Warning: File not found - {full_path}")
                except Exception as e:
                    print(f"Error adding dataframe {name}: {str(e)}")
            
            return analyzer
        except json.JSONDecodeError:
            print(f"Error: {config_path} is not a valid JSON file.")
        except FileNotFoundError:
            print(f"Error: {config_path} not found.")
        except KeyError:
            print(f"Error: 'dataframes' key not found in {config_path}.")
        return None
    
def get_common_columns_by_threshold(analyzer, min_true_count=4):
    """
    Get columns that are present (True) in at least min_true_count dataframes
    
    Parameters:
    analyzer (AmazonAnalyzer): The analyzer instance
    min_true_count (int): Minimum number of True values required (default=4)
    
    Returns:
    list: Column names that meet the threshold criteria
    """
    presence_df = analyzer.compare_columns_presence()
    if presence_df is None:
        return []
        
    # Sum True values across each row
    true_counts = presence_df.sum(axis=1)
    
    # Filter rows where count >= min_true_count
    filtered_columns = presence_df[true_counts >= min_true_count].index.tolist()
    
    return filtered_columns

# Add this method to your AmazonAnalyzer class
def get_columns_by_presence(self, min_true_count=4):
    """
    Wrapper method for the analyzer class
    """
    return get_common_columns_by_threshold(self, min_true_count)

    # Add more EDA methods as needed

def format_korean_number(number):
    if number == 0:
        return "0원"
    
    units = ['원', '만', '억', '조']
    result = []
    
    # 음수 체크
    is_negative = number < 0
    number = abs(int(number))
    
    for i, unit in enumerate(units):
        unit_value = number % 10000
        if unit_value > 0:
            if i == 0:  # '원' 단위일 때는 쉼표 포함
                result.append(f"{unit_value:,}{unit}")
            else:
                result.append(f"{unit_value}{unit}")
        number //= 10000
        if number == 0:
            break
    
    # 결과를 역순으로 조합
    final_result = ' '.join(reversed(result))
    
    # 음수면 앞에 마이너스 표시
    if is_negative:
        final_result = f"-{final_result}"
        
    return final_result

# Example usage
if __name__ == "__main__":
    print("Amazon Analysis Class imported successfully")