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

# Example usage
if __name__ == "__main__":
    print("Amazon Analysis Class imported successfully")