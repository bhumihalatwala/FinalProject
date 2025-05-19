import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

sns.set_style("whitegrid")

class SalesDataAnalyzer:
    def __init__(self, file_path=None):
        self.data = None
        self.plot = None  # Initialize plot attribute
        if file_path:
            self.load_data(file_path)
    
    def __del__(self):
        print("SalesDataAnalyzer instance is being destroyed. Data cleared.")

    def load_data(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'])
            print("Dataset loaded successfully!")
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except Exception as e:
            print(f"Error loading data: {str(e)}")

    def explore_data(self, choice):
        if self.data is None:
            print("No dataset loaded!")
            return
        if choice == 1:
            print("First 5 rows:\n", self.data.head())
        elif choice == 2:
            print("Last 5 rows:\n", self.data.tail())
        elif choice == 3:
            print("Column names:", list(self.data.columns))
        elif choice == 4:
            print("Data types:\n", self.data.dtypes)
        elif choice == 5:
            print("Basic info:")
            self.data.info()

    def clean_data(self, choice):
        if self.data is None:
            print("No dataset loaded!")
            return
        if choice == 1:
            missing = self.data[self.data.isnull().any(axis=1)]
            if missing.empty:
                print("No missing values found!")
            else:
                print("Rows with missing values:\n", missing)
        elif choice == 2:
            for col in ['Sales', 'Profit']:
                if col in self.data.columns and self.data[col].isnull().any():
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
            print("Missing values in numerical columns filled with mean.")
        elif choice == 3:
            initial_rows = len(self.data)
            self.data.dropna(inplace=True)
            print(f"Dropped {initial_rows - len(self.data)} rows with missing values.")
        elif choice == 4:
            value = input("Enter value to replace missing values: ")
            self.data.fillna(value, inplace=True)
            print(f"Missing values replaced with {value}.")

    def mathematical_operations(self):
        if self.data is None:
            print("No dataset loaded!")
            return
        sales_array = self.data['Sales'].to_numpy()
        profit_array = self.data['Profit'].to_numpy()
        
        print("First 5 sales values:", sales_array[:5])
        print("Last 5 profit values:", profit_array[-5:])
        
        print("Sales doubled:", sales_array * 2)
        print("Profit increased by 10%:", profit_array * 1.1)
        
        print("Total Sales:", np.nansum(sales_array))
        print("Average Profit:", np.nanmean(profit_array))

    def combine_data(self, other_file_path):
        if self.data is None:
            print("No dataset loaded!")
            return
        try:
            other_df = pd.read_csv(other_file_path)
            self.data = pd.concat([self.data, other_df], ignore_index=True)
            print("Data combined successfully!")
        except Exception as e:
            print(f"Error combining data: {str(e)}")

    def split_data(self, criterion='Region'):
        if self.data is None:
            print("No dataset loaded!")
            return
        if criterion in self.data.columns:
            grouped = self.data.groupby(criterion)
            split_dfs = {name: group for name, group in grouped}
            print(f"Data split into {len(split_dfs)} DataFrames based on {criterion}.")
            return split_dfs
        else:
            print(f"Column {criterion} not found!")
            return None

    def search_sort_filter(self, operation):
        if self.data is None:
            print("No dataset loaded!")
            return
        if operation == 1:  
            column = input("Enter column to search (e.g., Product): ")
            value = input("Enter value to search for: ")
            result = self.data[self.data[column].str.contains(value, case=False, na=False)]
            print("Search results:\n", result)
        elif operation == 2:  
            column = input("Enter column to sort by (e.g., Sales): ")
            ascending = input("Sort ascending? (y/n): ").lower() == 'y'
            self.data.sort_values(by=column, ascending=ascending, inplace=True)
            print("Data sorted:\n", self.data.head())
        elif operation == 3: 
            column = input("Enter column to filter (e.g., Region): ")
            value = input("Enter value to filter by: ")
            filtered = self.data[self.data[column] == value]
            print("Filtered data:\n", filtered)

    def aggregate_functions(self):
        if self.data is None:
            print("No dataset loaded!")
            return
        print("Aggregated statistics:")
        print("Total Sales:", self.data['Sales'].sum(skipna=True))
        print("Average Sales:", self.data['Sales'].mean(skipna=True))
        print("Total Profit:", self.data['Profit'].sum(skipna=True))
        print("Count of Records:", self.data['SalesID'].count())

    def statistical_analysis(self):
        if self.data is None:
            print("No dataset loaded!")
            return
        print("Statistical Analysis:")
        print("Sales Description:\n", self.data['Sales'].describe())
        print("Profit Standard Deviation:", self.data['Profit'].std())
        print("Sales Variance:", self.data['Sales'].var())
        print("25th Percentile of Sales:", self.data['Sales'].quantile(0.25))

    def create_pivot_table(self):
        if self.data is None:
            print("No dataset loaded!")
            return
        pivot = pd.pivot_table(self.data, values='Sales', index='Region', columns='Year', aggfunc='sum')
        print("Pivot Table (Sales by Region and Year):\n", pivot)
        return pivot

    def visualize_data(self, plot_type):
        if self.data is None:
            print("No dataset loaded!")
            return
        self.plot = plt.figure(figsize=(10, 6))
        
        if plot_type == 1:  
            sns.barplot(x='Region', y='Sales', data=self.data)
            plt.title('Sales by Region')
        elif plot_type == 2:  
            self.data.groupby('Date')['Sales'].sum().plot(kind='line')
            plt.title('Sales Trend Over Time')
        elif plot_type == 3:  
            x_col = input("Enter x-axis column name: ")
            y_col = input("Enter y-axis column name: ")
            plt.scatter(self.data[x_col], self.data[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'{y_col} vs {x_col}')
        elif plot_type == 4: 
            region_sales = self.data.groupby('Region')['Sales'].sum()
            plt.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%')
            plt.title('Sales Distribution by Region')
        elif plot_type == 5: 
            sns.boxplot(x='Region', y='Sales', data=self.data)
            plt.title('Sales Distribution by Region')
        elif plot_type == 6:  
            plt.hist(self.data['Sales'].dropna(), bins=20) 
            plt.title('Sales Distribution')
            plt.xlabel('Sales')
        elif plot_type == 7:  
            sns.violinplot(x='Region', y='Sales', data=self.data)
            plt.title('Sales Distribution by Region')
        elif plot_type == 8:  
            grouped = self.data.groupby(['Year', 'Region'])['Sales'].sum().unstack().fillna(0)
            plt.stackplot(grouped.index, grouped.values.T, labels=grouped.columns)
            plt.legend(loc='upper left')
            plt.title('Sales by Region Over Years')
        elif plot_type == 9:  
            self.data.groupby('Date')['Sales'].sum().plot(kind='line', drawstyle='steps-post')
            plt.title('Sales Trend (Step Chart)')
        
        plt.tight_layout()
        plt.show()
        print("Plot displayed successfully!")
        

    def save_visualization(self, file_name):
        if self.plot is None:
            print("No plot to save! Generate a plot first using option 6.")
            return
        try:
            directory = os.path.dirname(file_name)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            self.plot.savefig(file_name, format=file_name.split('.')[-1], dpi=300, bbox_inches='tight')
            print(f"Visualization saved as {file_name} successfully!")
            plt.close(self.plot)
            self.plot = None  
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")

def main():
    analyzer = None
    while True:
        print("\nData Analysis & Visualization Program")
        print("Please select an option:")
        print("1. Load Dataset")
        print("2. Explore Data")
        print("3. Perform DataFrame Operations")
        print("4. Handle Missing Data")
        print("5. Generate Descriptive Statistics")
        print("6. Data Visualization")
        print("7. Save Visualization")
        print("8. Exit")
        
        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Invalid input! Please enter a number.")
            continue

        if choice == 1:
            file_path = input("Enter the path of the dataset (CSV file): ")
            analyzer = SalesDataAnalyzer(file_path)
        
        elif choice == 2:
            if analyzer is None or analyzer.data is None:
                print("No dataset loaded!")
                continue
            print("Explore Data")
            print("1. Display the first 5 rows")
            print("2. Display the last 5 rows")
            print("3. Display column names")
            print("4. Display data types")
            print("5. Display basic info")
            try:
                sub_choice = int(input("Enter your choice: "))
                analyzer.explore_data(sub_choice)
            except ValueError:
                print("Invalid input! Please enter a number.")
        
        elif choice == 3:
            if analyzer is None or analyzer.data is None:
                print("No dataset loaded!")
                continue
            print("DataFrame Operations")
            print("1. Mathematical Operations")
            print("2. Combine Data")
            print("3. Split Data")
            print("4. Search, Sort, Filter")
            print("5. Aggregate Functions")
            print("6. Create Pivot Table")
            try:
                sub_choice = int(input("Enter your choice: "))
                if sub_choice == 1:
                    analyzer.mathematical_operations()
                elif sub_choice == 2:
                    other_file = input("Enter path of another CSV file to combine: ")
                    analyzer.combine_data(other_file)
                elif sub_choice == 3:
                    criterion = input("Enter column to split by (e.g., Region): ")
                    analyzer.split_data(criterion)
                elif sub_choice == 4:
                    print("1. Search")
                    print("2. Sort")
                    print("3. Filter")
                    try:
                        op_choice = int(input("Enter operation: "))
                        analyzer.search_sort_filter(op_choice)
                    except ValueError:
                        print("Invalid input! Please enter a number.")
                elif sub_choice == 5:
                    analyzer.aggregate_functions()
                elif sub_choice == 6:
                    analyzer.create_pivot_table()
                else:
                    print("Invalid choice! Please select a number between 1 and 6.")
            except ValueError:
                print("Invalid input! Please enter a number.")
        
        elif choice == 4:
            if analyzer is None or analyzer.data is None:
                print("No dataset loaded!")
                continue
            print("Handle Missing Data")
            print("1. Display rows with missing values")
            print("2. Fill missing values with mean")
            print("3. Drop rows with missing values")
            print("4. Replace missing values with a specific value")
            try:
                sub_choice = int(input("Enter your choice: "))
                analyzer.clean_data(sub_choice)
            except ValueError:
                print("Invalid input! Please enter a number.")
        
        elif choice == 5:
            if analyzer is None or analyzer.data is None:
                print("No dataset loaded!")
                continue
            analyzer.statistical_analysis()
        
        elif choice == 6:
            if analyzer is None or analyzer.data is None:
                print("No dataset loaded!")
                continue
            print("Data Visualization")
            print("1. Bar Plot")
            print("2. Line Plot")
            print("3. Scatter Plot")
            print("4. Pie Chart")
            print("5. Box Plot")
            print("6. Histogram")
            print("7. Violin Plot")
            print("8. Stack Plot")
            print("9. Step Chart")
            try:
                sub_choice = int(input("Enter your choice: "))
                analyzer.visualize_data(sub_choice)
            except ValueError:
                print("Invalid input! Please enter a number.")
        
        elif choice == 7:
            if analyzer is None:
                print("No analyzer instance created! Load a dataset first.")
                continue
            file_name = input("Enter file name to save the plot (e.g., scatter_plot.png): ")
            analyzer.save_visualization(file_name)
        
        elif choice == 8:
            print("Exiting the program. Goodbye!")
            break
        
        else:
            print("Invalid choice! Please select a number between 1 and 8.")

if __name__ == "__main__":
    main()