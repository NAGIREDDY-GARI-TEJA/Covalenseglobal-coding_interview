import pandas as pd
import json
from datetime import datetime
import numpy as np

class FitnessDataAggregator:
    def __init__(self):
        self.merged_data = None
        self.cleaned_data = None
        self.user_stats = []
        self.daily_top_users = []
    
    def read_csv_data(self, csv_file_path):
        """Read CSV data"""
        try:
            csv_data = pd.read_csv(csv_file_path)
            print(f"CSV data loaded: {len(csv_data)} records")
            return csv_data
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return pd.DataFrame()
    
    def read_json_data(self, json_file_path):
        """Read JSON data"""
        try:
        # Read JSON file
            with open(json_file_path, 'r') as file:
                json_data = json.load(file)
        
        # Convert to DataFrame
            df = pd.DataFrame(json_data)
            print(f"JSON data loaded: {len(df)} records")
            return df
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return pd.DataFrame()
    
    def merge_data(self, csv_data, json_data):
        """Merge CSV and JSON data"""
        try:
        # Ensure both DataFrames have the same columns
            all_columns = ['date', 'user_id', 'steps', 'calories', 'sleep_minutes']
        
        # Add missing columns to each DataFrame if needed
            for col in all_columns:
                if col not in csv_data.columns:
                    csv_data[col] = np.nan
                if col not in json_data.columns:
                    json_data[col] = np.nan
        
        # Reorder columns to match
            csv_data = csv_data[all_columns]
            json_data = json_data[all_columns]
        
        # Merge the DataFrames
            merged_data = pd.concat([csv_data, json_data], ignore_index=True)
        
            print(f"Data merged successfully: {len(merged_data)} total records")
            print(f"CSV records: {len(csv_data)}, JSON records: {len(json_data)}")
        
            return merged_data
        except Exception as e:
            print(f"Error merging data: {e}")
            return pd.DataFrame()
    
    def normalize_dates(self, data):
        """Normalize date formats to YYYY-MM-DD"""
        try:
            normalized_data = data.copy()
        
            def parse_date(date_str):
                """Parse different date formats and return YYYY-MM-DD"""
                if pd.isna(date_str):
                    return date_str
            
                date_str = str(date_str).strip()
            
            # Try different date formats
                formats = [
                    '%Y-%m-%d',      # 2025-09-01
                    '%d/%m/%Y',      # 01/09/2025
                    '%d-%m-%Y',      # 01-09-2025
                    '%m/%d/%Y',      # 09/01/2025 (fallback)
                    '%m-%d-%Y'       # 09-01-2025 (fallback)
                ]
            
                for fmt in formats:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        return parsed_date.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
            
            # If no format works, return original
                print(f"Warning: Could not parse date: {date_str}")
                return date_str
        
            # Apply date normalization
            normalized_data['date'] = normalized_data['date'].apply(parse_date)
        
            print("Date normalization completed")
            return normalized_data
        
        except Exception as e:
            print(f"Error normalizing dates: {e}")
            return data
    
    def remove_duplicates(self, data):
        """Remove duplicate entries based on user_id + date"""
        try:
            original_count = len(data)
        
                # Remove duplicates based on user_id and date combination
            # Keep the first occurrence
            cleaned_data = data.drop_duplicates(subset=['user_id', 'date'], keep='first')
        
            removed_count = original_count - len(cleaned_data)
        
            if removed_count > 0:
                print(f"Removed {removed_count} duplicate records")
                print("Duplicate combinations found:")
            
            # Show which duplicates were found
                duplicates = data[data.duplicated(subset=['user_id', 'date'], keep=False)]
                if not duplicates.empty:
                    duplicate_summary = duplicates.groupby(['user_id', 'date']).size()
                    for (user, date), count in duplicate_summary.items():
                        if count > 1:
                            print(f"  - {user} on {date}: {count} entries")
            else:
                print("No duplicate records found")
        
            print(f"Records after duplicate removal: {len(cleaned_data)}")
            return cleaned_data.reset_index(drop=True)
        
        except Exception as e:
            print(f"Error removing duplicates: {e}")
            return data
    
    def handle_missing_data(self, data):
        """Handle missing fields by filling with user averages"""
        try:
            filled_data = data.copy()
        
            print("--- Missing Data Handling ---")
            print("Missing data before handling:")
            missing_before = filled_data.isnull().sum()
            print(missing_before)
        
         # Calculate user averages for numeric columns
            numeric_columns = ['steps', 'calories', 'sleep_minutes']
        
            for col in numeric_columns:
                if col in filled_data.columns:
                # Calculate user averages (excluding NaN values)
                    user_averages = filled_data.groupby('user_id')[col].mean()
                
                # Fill missing values with user's average
                    for user_id in filled_data['user_id'].unique():
                        user_mask = filled_data['user_id'] == user_id
                        missing_mask = filled_data[col].isnull()
                    
                    # Fill missing values for this user with their average
                        if user_id in user_averages and not pd.isna(user_averages[user_id]):
                            filled_data.loc[user_mask & missing_mask, col] = round(user_averages[user_id])
                        else:
                        # If user has no valid data for this column, use overall average
                            overall_avg = filled_data[col].mean()
                            if not pd.isna(overall_avg):
                                filled_data.loc[user_mask & missing_mask, col] = round(overall_avg)
        
            print("\nMissing data after handling:")
            missing_after = filled_data.isnull().sum()
            print(missing_after)
        
        # Show what was filled
            for col in numeric_columns:
                filled_count = missing_before[col] - missing_after[col]
                if filled_count > 0:
                    print(f"Filled {filled_count} missing values in {col}")
        
            return filled_data
        
        except Exception as e:
            print(f"Error handling missing data: {e}")
            return data
    
    def clean_data(self):
        """Main data cleaning function - combines all cleaning steps"""
        try:
            if self.merged_data is None or self.merged_data.empty:
                print("❌ No merged data available for cleaning")
                return
            
            print("=== DATA CLEANING PIPELINE ===")
            print(f"Starting with {len(self.merged_data)} records")
            
            # Step 1: Normalize dates
            print("\nStep 1: Normalizing dates...")
            normalized_data = self.normalize_dates(self.merged_data)
            
            # Step 2: Remove duplicates  
            print("\nStep 2: Removing duplicates...")
            deduplicated_data = self.remove_duplicates(normalized_data)
            
            # Step 3: Handle missing data
            print("\nStep 3: Handling missing data...")
            cleaned_data = self.handle_missing_data(deduplicated_data)
            
            # Store the cleaned data
            self.cleaned_data = cleaned_data
            
            print(f"\n✅ Data cleaning completed!")
            print(f"Final clean dataset: {len(self.cleaned_data)} records")
            print(f"Unique users: {len(self.cleaned_data['user_id'].unique())}")
            
            return self.cleaned_data
            
        except Exception as e:
            print(f"❌ Error in data cleaning pipeline: {e}")
            return None
        
    def calculate_user_stats(self):
        """Calculate user statistics - total steps, calories, and weekly averages"""
        try:
            if self.cleaned_data is None or self.cleaned_data.empty:
                print("❌ No cleaned data available for statistics calculation")
                return
            
            print("\n=== CALCULATING USER STATISTICS ===")
            
            # Convert date column to datetime for easier manipulation
            self.cleaned_data['date'] = pd.to_datetime(self.cleaned_data['date'])
            
            # Add week information for weekly averages
            self.cleaned_data['year_week'] = self.cleaned_data['date'].dt.strftime('%Y-week-%U')
            
            user_stats = []
            
            # Process each user
            for user_id in sorted(self.cleaned_data['user_id'].unique()):
                user_data = self.cleaned_data[self.cleaned_data['user_id'] == user_id]
                
                # Calculate totals
                total_steps = int(user_data['steps'].sum())
                total_calories = int(user_data['calories'].sum())
                
                # Calculate weekly averages
                weekly_stats = user_data.groupby('year_week')['steps'].mean()
                weekly_avg_steps = {}
                
                for week, avg_steps in weekly_stats.items():
                    weekly_avg_steps[week] = round(avg_steps, 1)
                
                # Create user statistics
                user_stat = {
                    "user_id": user_id,
                    "total_steps": total_steps,
                    "total_calories": total_calories,
                    "weekly_avg_steps": weekly_avg_steps
                }
                
                user_stats.append(user_stat)
                
                print(f"✓ {user_id}: {total_steps:,} steps, {total_calories:,} calories, {len(weekly_avg_steps)} weeks")
            
            self.user_stats = user_stats
            print(f"\n✅ User statistics calculated for {len(user_stats)} users")
            
            return self.user_stats
            
        except Exception as e:
            print(f"❌ Error calculating user statistics: {e}")
            return []
    
    def find_daily_top_users(self):
        """Find daily top users by steps for each date"""
        try:
            if self.cleaned_data is None or self.cleaned_data.empty:
                print("❌ No cleaned data available for daily top users calculation")
                return
            
            print("\n=== FINDING DAILY TOP USERS ===")
            
            daily_top_users = []
            
            # Group by date and find the user with maximum steps for each day
            daily_max = self.cleaned_data.groupby('date')['steps'].transform('max')
            top_users_data = self.cleaned_data[self.cleaned_data['steps'] == daily_max]
            
            # Process each date
            for _, row in top_users_data.iterrows():
                daily_top = {
                    "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    "user_id": row['user_id'],
                    "steps": int(row['steps'])
                }
                daily_top_users.append(daily_top)
            
            # Sort by date
            daily_top_users.sort(key=lambda x: x['date'])
            
            # Remove any duplicate dates (in case of ties, keep first occurrence)
            seen_dates = set()
            unique_daily_tops = []
            for entry in daily_top_users:
                if entry['date'] not in seen_dates:
                    unique_daily_tops.append(entry)
                    seen_dates.add(entry['date'])
            
            self.daily_top_users = unique_daily_tops
            
            print(f"✅ Daily top users identified for {len(unique_daily_tops)} days")
            
            # Show some sample daily tops
            print("\nSample daily top users:")
            for i, top in enumerate(unique_daily_tops[:5]):
                print(f"  {top['date']}: {top['user_id']} with {top['steps']:,} steps")
            
            if len(unique_daily_tops) > 5:
                print(f"  ... and {len(unique_daily_tops) - 5} more days")
            
            return self.daily_top_users
            
        except Exception as e:
            print(f"❌ Error finding daily top users: {e}")
            return []
    def generate_output(self):
        """Generate final JSON output with user stats and daily top users"""
        try:
            if not self.user_stats or not self.daily_top_users:
                print("❌ Missing user statistics or daily top users data")
                return None
            
            print("\n=== GENERATING FINAL JSON OUTPUT ===")
            
            # Create the final output structure
            output = {
                "user_stats": self.user_stats,
                "daily_top_user": self.daily_top_users
            }
            
            print(f"✅ JSON output generated successfully!")
            print(f"   - User stats: {len(output['user_stats'])} users")
            print(f"   - Daily top users: {len(output['daily_top_user'])} days")
            
            return output
            
        except Exception as e:
            print(f"❌ Error generating output: {e}")
            return None
    
    def run_pipeline(self):
        """Run the complete data processing pipeline"""
        # Read data
        csv_data = self.read_csv_data(r"C:\Users\NAGIREDDY GARI TEJA\Downloads\fitness_data.csv")
        json_data = self.read_json_data(r"C:\Users\NAGIREDDY GARI TEJA\Downloads\fitness_data.json")
        
        # Merge data
        self.merged_data = self.merge_data(csv_data, json_data)
        
        # Clean data
        self.clean_data()
        
        # Calculate statistics
        self.calculate_user_stats()
        self.find_daily_top_users()
        
        # Generate output
        return self.generate_output()

# Test complete pipeline
if __name__ == "__main__":
    aggregator = FitnessDataAggregator()
    
    # Run the complete pipeline
    result = aggregator.run_pipeline()
    
    if result:
        print("\n=== FINAL OUTPUT SAMPLE ===")
        
        # Show sample user stats (first 2 users)
        print("\nSample User Statistics:")
        for i, user in enumerate(result['user_stats'][:2]):
            print(f"\n{i+1}. User: {user['user_id']}")
            print(f"   Total Steps: {user['total_steps']:,}")
            print(f"   Total Calories: {user['total_calories']:,}")
            print(f"   Weekly Averages: {user['weekly_avg_steps']}")
        
        # Show sample daily tops (first 5 days)  
        print(f"\nSample Daily Top Users (first 5 days):")
        for i, day in enumerate(result['daily_top_user'][:5]):
            print(f"{i+1}. {day['date']}: {day['user_id']} ({day['steps']:,} steps)")
        
        # Save to JSON file
        try:
            with open("C:\\Users\\NAGIREDDY GARI TEJA\\Downloads\\fitness_results.json", 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to 'C:\\Users\\NAGIREDDY GARI TEJA\\Downloads\\fitness_results.json' ")
        except Exception as e:
            print(f"\n⚠️  Could not save to file: {e}")
        
        print(f"\n✅ COMPLETE SUCCESS!")
        print(f"   📊 Processed {len(result['user_stats'])} users")
        print(f"   📅 Analyzed {len(result['daily_top_user'])} days")
        print(f"   🏆 Pipeline completed successfully!")
        
    else:
        print("\n❌ PIPELINE FAILED!")