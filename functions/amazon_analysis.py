import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import os
import pickle
import json
from typing import Dict, List, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import itertools
from functools import lru_cache

class AmazonDataframe:
    def __init__(self, file_path):
        self.file_path = file_path
        self._df = None
        self._category_converted = False
        self.verify_file_path()

    def verify_file_path(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        if not os.path.isfile(self.file_path):
            raise IsADirectoryError(f"The path {self.file_path} is a directory, not a file.")
        if not os.access(self.file_path, os.R_OK):
            raise PermissionError(f"You don't have permission to read the file {self.file_path}")
        print(f"File path verified: {self.file_path}")

    def _convert_category_column(self):
            """Convert category column to categorical type if possible"""
            if 'category' not in self._df.columns:
                print(f"Warning: No 'category' column found in {self.file_path}")
                return
            
            # Check if category column contains boolean values
            if self._df['category'].dtype == bool:
                print(f"Warning: 'category' column in {self.file_path} contains boolean values. "
                    "Skipping conversion to categorical type. Please check the data.")
                return
            
            try:
                self._df['category'] = self._df['category'].astype('category')
                self._category_converted = True
                print(f"Successfully converted 'category' column to categorical type in {self.file_path}")
            except Exception as e:
                print(f"Error converting 'category' column in {self.file_path}: {str(e)}")

    @property
    def df(self):
        if self._df is None:
            try:
                self._df = pd.read_csv(self.file_path)
                print(f"Successfully loaded CSV from {self.file_path}")
                # Convert category column after loading
                if not self._category_converted:
                    self._convert_category_column()

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

    def category_summary(self):
        """Print summary of category column"""
        if self._df is None:
            self.df  # This will trigger loading the dataframe
        
        if 'category' not in self._df.columns:
            print("No 'category' column found in the dataframe")
            return
        
        print(f"\nCategory Summary for {self.file_path}:")
        print(f"Data type: {self._df['category'].dtype}")
        print(f"Number of unique categories: {self._df['category'].nunique()}")
        print(f"Top 5 most common categories:")
        print(self._df['category'].value_counts().head())

    # Add more EDA methods as needed

class AmazonAnalyzer:
    def __init__(self, cache_dir='./cache'):
        self.dataframes = {}
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def add_dataframe(self, name: str, file_path):
        """Add a dataframe to the analyzer"""
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
    
    # Add this method to your AmazonAnalyzer class
    def get_columns_by_presence(self, min_true_count=4):
        """
        Wrapper method for the analyzer class
        """
        return get_common_columns_by_threshold(self, min_true_count)

        # Add more EDA methods as needed

    def _process_product_pair(self, data: tuple) -> dict:
        """Process a single pair of dataframes for product overlap"""
        df1_name, df2_name, df1, df2 = data
        
        # Convert to sets for faster intersection
        df1_ids = set(df1['product_id'].unique())
        df2_ids = set(df2['product_id'].unique())
        
        common_ids = df1_ids & df2_ids
        
        if not common_ids:
            return None
            
        # Create dictionaries for faster lookup
        df1_names = df1[df1['product_id'].isin(common_ids)].set_index('product_id')['product_name']
        df2_names = df2[df2['product_id'].isin(common_ids)].set_index('product_id')['product_name']
        
        name_conflicts = []
        for pid in common_ids:
            if df1_names[pid] != df2_names[pid]:
                name_conflicts.append({
                    'product_id': pid,
                    f'{df1_name}_name': df1_names[pid],
                    f'{df2_name}_name': df2_names[pid]
                })

        return {
            'pair': f"{df1_name}_vs_{df2_name}",
            'common_ids': common_ids,
            'name_conflicts': name_conflicts,
            'stats': {
                'common_products': len(common_ids),
                'name_conflicts': len(name_conflicts),
                'overlap_percentage': len(common_ids) / len(df1_ids) * 100
            }
        }

    def analyze_product_overlaps(self) -> Dict:
        """
        Optimized analysis of product ID overlaps and name consistency
        """
        # Check cache first
        cache_path = os.path.join(self.cache_dir, 'product_overlap_analysis.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)

        results = {
            'overlap_summary': {},
            'name_conflicts': {},
            'timestamp': datetime.now().isoformat()
        }

        # Prepare data for parallel processing
        df_pairs = []
        df_names = list(self.dataframes.keys())
        for i, j in itertools.combinations(range(len(df_names)), 2):
            df1_name = df_names[i]
            df2_name = df_names[j]
            df_pairs.append((
                df1_name,
                df2_name,
                self.dataframes[df1_name].df,
                self.dataframes[df2_name].df
            ))

        # Process in parallel
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
            futures = [executor.submit(self._process_product_pair, pair) 
                      for pair in df_pairs]
            
            for future in futures:
                result = future.result()
                if result:
                    pair_key = result['pair']
                    results['overlap_summary'][pair_key] = result['stats']
                    if result['name_conflicts']:
                        results['name_conflicts'][pair_key] = result['name_conflicts']

        # Cache results
        with open(cache_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _process_category_pair(self, data: tuple) -> dict:
        """Process a single pair of dataframes for category analysis"""
        df1_name, df2_name, df1_cats, df2_cats = data
        
        cats1 = set(df1_cats)
        cats2 = set(df2_cats)
        
        return {
            'pair': f"{df1_name}_vs_{df2_name}",
            'common_categories': list(cats1 & cats2),
            f'unique_to_{df1_name}': list(cats1 - cats2),
            f'unique_to_{df2_name}': list(cats2 - cats1)
        }

    def analyze_categories(self) -> Dict:
        """
        Optimized analysis of category differences and distributions
        """
        # Check cache first
        cache_path = os.path.join(self.cache_dir, 'category_analysis.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)

        results = {
            'category_summary': {},
            'category_mapping': {},
            'timestamp': datetime.now().isoformat()
        }

        # Process individual dataframe summaries
        for name, df_obj in self.dataframes.items():
            # Convert to categorical for better performance
            df_obj.df['category'] = df_obj.df['category'].astype('category')
            category_dist = df_obj.df['category'].value_counts().to_dict()
            unique_cats = set(df_obj.df['category'].unique())
            
            results['category_summary'][name] = {
                'total_categories': len(unique_cats),
                'category_distribution': category_dist
            }

        # Prepare data for parallel processing
        df_pairs = []
        df_names = list(self.dataframes.keys())
        for i, j in itertools.combinations(range(len(df_names)), 2):
            df1_name = df_names[i]
            df2_name = df_names[j]
            df_pairs.append((
                df1_name,
                df2_name,
                self.dataframes[df1_name].df['category'].unique(),
                self.dataframes[df2_name].df['category'].unique()
            ))

        # Process pairs in parallel
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
            futures = [executor.submit(self._process_category_pair, pair) 
                      for pair in df_pairs]
            
            for future in futures:
                result = future.result()
                results['category_mapping'][result['pair']] = {
                    'common_categories': result['common_categories'],
                    f'unique_to_{result["pair"].split("_vs_")[0]}': result[f'unique_to_{result["pair"].split("_vs_")[0]}'],
                    f'unique_to_{result["pair"].split("_vs_")[1]}': result[f'unique_to_{result["pair"].split("_vs_")[1]}']
                }

        # Cache results
        with open(cache_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def print_analysis_summary(self, min_overlap_percent: float = 5.0):
        """
        Print a comprehensive summary of the analysis
        """
        # Product overlaps
        product_results = self.analyze_product_overlaps()
        print("\n=== Product Overlap Analysis ===")
        for pair, stats in product_results['overlap_summary'].items():
            if stats['overlap_percentage'] >= min_overlap_percent:
                print(f"\n{pair}:")
                print(f"Common products: {stats['common_products']}")
                print(f"Name conflicts: {stats['name_conflicts']}")
                print(f"Overlap percentage: {stats['overlap_percentage']:.1f}%")

        # Category analysis
        category_results = self.analyze_categories()
        print("\n=== Category Analysis ===")
        for name, summary in category_results['category_summary'].items():
            print(f"\n{name}:")
            print(f"Total categories: {summary['total_categories']}")
            print("Top 5 categories by product count:")
            top_cats = dict(sorted(summary['category_distribution'].items(), 
                                 key=lambda x: x[1], reverse=True)[:5])
            for cat, count in top_cats.items():
                print(f"- {cat}: {count} products")

    @lru_cache(maxsize=128)
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Cached similarity calculation"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _process_category_pair(self, pair_data: tuple) -> tuple:
        """Process a single pair of categories"""
        cat1, cat2 = pair_data
        score = self._calculate_similarity(cat1, cat2)
        return (cat1, cat2, score) if score > 0.6 else None

    def get_category_mapping_suggestions(self, batch_size: int = 1000) -> Dict:
        """
        Generate suggestions for mapping categories between dataframes
        using batched processing and parallel execution.
        Skips dataframes that contain boolean values in their category column.
        """
        results = {
            'mapping_suggestions': {},
            'timestamp': datetime.now().isoformat(),
            'skipped_dataframes': []  # Track skipped dataframes
        }
        
        df_names = list(self.dataframes.keys())
        valid_df_names = []
        
        # Check each dataframe for boolean values in category column
        for df_name in df_names:
            categories = self.dataframes[df_name].df['category']
            if categories.dtype == bool or categories.apply(lambda x: isinstance(x, bool)).any():
                results['skipped_dataframes'].append({
                    'name': df_name,
                    'reason': 'Contains boolean values in category column'
                })
            else:
                valid_df_names.append(df_name)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
            for i, j in itertools.combinations(range(len(valid_df_names)), 2):
                df1_name = valid_df_names[i]
                df2_name = valid_df_names[j]
                
                # Get unique categories
                cats1 = list(set(self.dataframes[df1_name].df['category'].unique()))
                cats2 = list(set(self.dataframes[df2_name].df['category'].unique()))
                
                # Create batches of category pairs
                category_pairs = list(itertools.product(cats1, cats2))
                
                # Process in batches
                mapping_suggestions = {}
                for batch_start in range(0, len(category_pairs), batch_size):
                    batch_end = min(batch_start + batch_size, len(category_pairs))
                    batch = category_pairs[batch_start:batch_end]
                    
                    # Process batch in parallel
                    futures = [executor.submit(self._process_category_pair, pair)
                            for pair in batch]
                    
                    # Collect results
                    for future in futures:
                        result = future.result()
                        if result:
                            cat1, cat2, score = result
                            if cat1 not in mapping_suggestions:
                                mapping_suggestions[cat1] = []
                            mapping_suggestions[cat1].append({
                                'category': cat2,
                                'similarity_score': score
                            })
                
                # Sort suggestions by similarity score
                for cat in mapping_suggestions:
                    mapping_suggestions[cat] = sorted(
                        mapping_suggestions[cat],
                        key=lambda x: x['similarity_score'],
                        reverse=True
                    )
                
                pair_key = f"{df1_name}_vs_{df2_name}"
                results['mapping_suggestions'][pair_key] = mapping_suggestions
        
        # Cache results
        cache_path = os.path.join(self.cache_dir, 'category_mapping_suggestions.json')
        with open(cache_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

    def print_mapping_suggestions(self, min_similarity: float = 0.7, max_suggestions: int = 3):
        """
        Print mapping suggestions with control over output size
        """
        mapping_suggestions = self.get_category_mapping_suggestions()
        
        for pair, suggestions in mapping_suggestions['mapping_suggestions'].items():
            print(f"\nMapping suggestions for {pair}:")
            for cat, matches in suggestions.items():
                # Filter matches by similarity threshold
                good_matches = [m for m in matches if m['similarity_score'] >= min_similarity]
                if good_matches:
                    print(f"\n{cat} matches:")
                    # Limit number of suggestions printed
                    for match in good_matches[:max_suggestions]:
                        print(f"- {match['category']} "
                              f"(similarity: {match['similarity_score']:.2f})")

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