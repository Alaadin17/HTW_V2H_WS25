"""
Script to build input timeseries file for oemof models from Emobpy results
Reads CSV files from results/emobpy-V2H-WS25 and combines them into
input files for oemof models.
"""
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TimeseriesBuilder:
    def __init__(self):
        # Define paths relative to script location
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        
        # Input: Emobpy results
        self.emobpy_dir = self.project_root / "results" / "emobpy-V2H-WS25"
        
        # Input: PV data from pvlib results
        self.pvlib_dir = self.project_root / "results" / "pvlib-V2H-WS25"
        
        # Input: Load profile data
        self.load_profile_dir = self.project_root / "results" / "load-profile-15min"
        
        # Output: oemof input timeseries
        self.output_dir = self.project_root / "models" / "oemof" / "oemof-V2H-WS25" / "Input_timeseries"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage variables
        self.pv_data = None
        self.load_data = None
    
    def read_emobpy_results(self, filename=None):
        """
        Read Emobpy results from CSV file
        
        Parameters:
        -----------
        filename : str, optional
            Name of the CSV file to read. If None, searches for CSV files in directory.
        
        Returns:
        --------
        list : List of tuples (DataFrame, filename)
        """
        if filename:
            file_path = self.emobpy_dir / filename
            csv_files = [file_path]
        else:
            # Search for CSV files in Emobpy directory
            csv_files = list(self.emobpy_dir.glob("*.csv"))
            
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.emobpy_dir}")
            
            if len(csv_files) > 1:
                logging.info(f"Multiple CSV files found ({len(csv_files)}). Processing all files...")
        
        # Return list of all files to process
        results = []
        for file_path in csv_files:
            logging.info(f"Reading Emobpy data from: {file_path}")
            
            try:
                df = pd.read_csv(file_path)
                logging.info(f"Successfully loaded {len(df)} rows from {file_path.name}")
                results.append((df, file_path.name))
            except Exception as e:
                logging.error(f"Error reading Emobpy file {file_path.name}: {e}")
                continue
        
        if not results:
            raise RuntimeError("No files could be successfully loaded")
        
        return results
    
    def build_oemof_input(self, emobpy_df, pv_data=None, load_data=None, price_data=None, 
                          output_filename="input_timeseries_emobpy.csv"):
        """
        Build oemof input timeseries from Emobpy results and additional data
        
        Parameters:
        -----------
        emobpy_df : pd.DataFrame
            DataFrame with Emobpy results
        pv_data : pd.Series or array-like, optional
            PV generation time series (kW)
        load_data : pd.Series or array-like, optional
            Household load time series (kW)
        price_data : pd.Series or array-like, optional
            Day-ahead prices (€/MWh)
        output_filename : str
            Name of the output CSV file
        
        Returns:
        --------
        pd.DataFrame : Combined timeseries
        """
        logging.info("Building oemof input timeseries...")
        
        # Create base dataframe with timestamp
        if 'date' in emobpy_df.columns:
            df_output = pd.DataFrame({'datetime': emobpy_df['date']})
        elif 'datetime' in emobpy_df.columns:
            df_output = pd.DataFrame({'datetime': emobpy_df['datetime']})
        elif 'timestamp' in emobpy_df.columns:
            df_output = pd.DataFrame({'datetime': emobpy_df['timestamp']})
        else:
            # Generate timestamp if not available
            logging.warning("No timestamp column found, generating timestamps...")
            df_output = pd.DataFrame({
                'datetime': pd.date_range(
                    start='2025-01-01', 
                    periods=len(emobpy_df), 
                    freq='15min'
                )
            })
        
        # Add PV data
        if pv_data is not None:
            df_output['PV_kW'] = pv_data
        elif 'PV_kW' in emobpy_df.columns:
            df_output['PV_kW'] = emobpy_df['PV_kW']
        else:
            # Try to load PV data from pvlib results
            logging.info("Attempting to load PV data from pvlib results...")
            
            # Determine frequency from datetime column
            if len(df_output) > 1:
                time_diff = pd.to_datetime(df_output['datetime'].iloc[1]) - pd.to_datetime(df_output['datetime'].iloc[0])
                freq = '15min' if time_diff.total_seconds() <= 900 else '1H'
            else:
                freq = '15min'  # default
            
            pv_data_loaded = self.load_pv_data(len(df_output), freq=freq)
            
            if pv_data_loaded is not None:
                df_output['PV_kW'] = pv_data_loaded
            else:
                logging.warning("No PV data available, filling with zeros")
                df_output['PV_kW'] = 0.0
        
        # Add Load data
        if load_data is not None:
            df_output['Load_kW'] = load_data
        elif 'Load_kW' in emobpy_df.columns:
            df_output['Load_kW'] = emobpy_df['Load_kW']
        else:
            # Try to load Load data from load-profile-15min results
            logging.info("Attempting to load Load data from load-profile-15min results...")
            
            # Determine frequency from datetime column
            if len(df_output) > 1:
                time_diff = pd.to_datetime(df_output['datetime'].iloc[1]) - pd.to_datetime(df_output['datetime'].iloc[0])
                freq = '15min' if time_diff.total_seconds() <= 900 else '1H'
            else:
                freq = '15min'  # default
            
            load_data_loaded = self.load_load_profile(len(df_output), freq=freq)
            
            if load_data_loaded is not None:
                df_output['Load_kW'] = load_data_loaded
            else:
                logging.warning("No Load data available, filling with zeros")
                df_output['Load_kW'] = 0.0
        
        # Add BEV data from Emobpy
        # Add state column
        for col_variant in ['state', 'location', 'status']:
            if col_variant in emobpy_df.columns:
                df_output['state'] = emobpy_df[col_variant]
                break
        else:
            logging.warning("No state/location column found")
            df_output['state'] = 'home'
        
        # Add charging power (average power in W)
        for col_variant in ['average power in W', 'avg_power_W', 'charging_power_W']:
            if col_variant in emobpy_df.columns:
                df_output['average power in W'] = emobpy_df[col_variant]
                break
        else:
            df_output['average power in W'] = 0.0
        
        # Add BEV at home indicator
        if 'BEV_at_home' in emobpy_df.columns:
            df_output['BEV_at_home'] = emobpy_df['BEV_at_home']
        elif 'state' in df_output.columns:
            # Create from state column (1 if home, 0 otherwise)
            df_output['BEV_at_home'] = (df_output['state'] == 'home').astype(int)
        else:
            df_output['BEV_at_home'] = 1
        
        # Add consumption
        for col_variant in ['consumption', 'energy_consumption', 'consumption_kW']:
            if col_variant in emobpy_df.columns:
                df_output['consumption'] = emobpy_df[col_variant]
                break
        else:
            df_output['consumption'] = 0.0
        
        # Add charging power (kW)
        for col_variant in ['charging_power_kW', 'charging_power', 'charge_power']:
            if col_variant in emobpy_df.columns:
                df_output['charging_power_kW'] = emobpy_df[col_variant]
                break
        else:
            df_output['charging_power_kW'] = 11.0  # default wallbox power
        
        # Add price data
        if price_data is not None:
            df_output['day_ahead_price[€/MWh]'] = price_data
        elif 'day_ahead_price[€/MWh]' in emobpy_df.columns:
            df_output['day_ahead_price[€/MWh]'] = emobpy_df['day_ahead_price[€/MWh]']
        elif 'price' in emobpy_df.columns:
            df_output['day_ahead_price[€/MWh]'] = emobpy_df['price']
        else:
            # Try to load prices from local files
            logging.info("Attempting to load electricity prices from local files...")
            
            # Determine frequency from datetime column
            if len(df_output) > 1:
                time_diff = pd.to_datetime(df_output['datetime'].iloc[1]) - pd.to_datetime(df_output['datetime'].iloc[0])
                freq = '15min' if time_diff.total_seconds() <= 900 else '1H'
            else:
                freq = '15min'  # default
            
            prices = self.load_electricity_prices(len(df_output), freq=freq)
            
            if prices is not None:
                df_output['day_ahead_price[€/MWh]'] = prices
            else:
                logging.warning("No price data available, filling with default value 0.1")
                df_output['day_ahead_price[€/MWh]'] = 0.1
        
        # Save to CSV
        output_path = self.output_dir / output_filename
        df_output.to_csv(output_path, index=False)
        
        logging.info(f"✓ Successfully created input timeseries with {len(df_output)} rows")
        logging.info(f"✓ Saved to: {output_path}")
        logging.info(f"✓ Columns: {', '.join(df_output.columns)}")
        
        # Print summary statistics
        logging.info("\nData Summary:")
        logging.info(f"  Time range: {df_output['datetime'].iloc[0]} to {df_output['datetime'].iloc[-1]}")
        if df_output['PV_kW'].sum() > 0:
            logging.info(f"  PV: {df_output['PV_kW'].min():.2f} - {df_output['PV_kW'].max():.2f} kW")
        if df_output['Load_kW'].sum() > 0:
            logging.info(f"  Load: {df_output['Load_kW'].min():.2f} - {df_output['Load_kW'].max():.2f} kW")
        if df_output['consumption'].sum() > 0:
            logging.info(f"  BEV consumption: {df_output['consumption'].sum():.2f} kWh total")
        logging.info(f"  BEV at home: {df_output['BEV_at_home'].sum()} / {len(df_output)} timesteps")
        
        return df_output
    
    def load_pv_data(self, num_timesteps, freq='15min'):
        """
        Load PV generation data from pvlib results
        
        Parameters:
        -----------
        num_timesteps : int
            Number of timesteps needed
        freq : str
            Time frequency ('15min' or 'H'/'1H' for hourly)
        
        Returns:
        --------
        pd.Series : PV data or None if file not found
        """
        # Determine which file to use based on frequency
        if '15' in freq or freq == '15min' or freq == '15T':
            pv_file = "pv_timeseries_15min.csv"
        else:
            pv_file = "pv_timeseries_hourly.csv"
        
        pv_path = self.pvlib_dir / pv_file
        
        if pv_path.exists():
            try:
                # Try reading with different delimiters (pvlib uses semicolon)
                try:
                    df_pv = pd.read_csv(pv_path, sep=';')
                except:
                    df_pv = pd.read_csv(pv_path)
                
                # Look for PV power column
                pv_cols = [col for col in df_pv.columns if 'pv' in col.lower() or 'power' in col.lower() or 'p_' in col.lower() or 'ac' in col.lower()]
                
                if pv_cols:
                    pv_data = df_pv[pv_cols[0]]
                else:
                    # Use second column if no obvious PV column found
                    pv_data = df_pv.iloc[:, 1] if len(df_pv.columns) > 1 else df_pv.iloc[:, 0]
                
                # Convert to numeric in case it's strings
                pv_data = pd.to_numeric(pv_data, errors='coerce').fillna(0)
                
                # Convert W to kW if necessary (check if values are very large)
                if pv_data.max() > 100:  # likely in Watts
                    pv_data = pv_data / 1000.0
                    logging.info("Converted PV data from W to kW")
                
                # Match length to num_timesteps
                if len(pv_data) > num_timesteps:
                    pv_data = pv_data.iloc[:num_timesteps]
                elif len(pv_data) < num_timesteps:
                    # Repeat the pattern if needed
                    repeats = (num_timesteps // len(pv_data)) + 1
                    pv_data = pd.concat([pv_data] * repeats, ignore_index=True).iloc[:num_timesteps]
                
                logging.info(f"Loaded PV data from: {pv_file}")
                logging.info(f"  PV range: {pv_data.min():.2f} - {pv_data.max():.2f} kW")
                self.pv_data = pv_data.reset_index(drop=True)
                return self.pv_data
                
            except Exception as e:
                logging.warning(f"Error loading PV file {pv_file}: {e}")
                return None
        else:
            logging.warning(f"PV file not found: {pv_path}")
            return None
    
    def load_load_profile(self, num_timesteps, freq='15min'):
        """
        Load household load profile data from load-profile-15min results
        
        Parameters:
        -----------
        num_timesteps : int
            Number of timesteps needed
        freq : str
            Time frequency ('15min' or 'H'/'1H' for hourly)
        
        Returns:
        --------
        pd.Series : Load data or None if file not found
        """
        # Look for load profile file
        load_file = "load_profile_15min.csv"
        load_path = self.load_profile_dir / load_file
        
        if load_path.exists():
            try:
                # Read the CSV file (default comma separator)
                df_load = pd.read_csv(load_path)
                
                # Look for load/power column
                load_cols = [col for col in df_load.columns if 'load' in col.lower() or 'power' in col.lower() or 'demand' in col.lower()]
                
                if load_cols:
                    load_data = df_load[load_cols[0]]
                else:
                    # Use second column if no obvious load column found
                    load_data = df_load.iloc[:, 1] if len(df_load.columns) > 1 else df_load.iloc[:, 0]
                
                # Convert to numeric in case it's strings
                load_data = pd.to_numeric(load_data, errors='coerce').fillna(0)
                
                # Convert W to kW if necessary (check if values are very large)
                if load_data.max() > 100:  # likely in Watts
                    load_data = load_data / 1000.0
                    logging.info("Converted Load data from W to kW")
                
                # Match length to num_timesteps
                if len(load_data) > num_timesteps:
                    load_data = load_data.iloc[:num_timesteps]
                elif len(load_data) < num_timesteps:
                    # Repeat the pattern if needed
                    repeats = (num_timesteps // len(load_data)) + 1
                    load_data = pd.concat([load_data] * repeats, ignore_index=True).iloc[:num_timesteps]
                
                logging.info(f"Loaded Load data from: {load_file}")
                logging.info(f"  Load range: {load_data.min():.2f} - {load_data.max():.2f} kW")
                self.load_data = load_data.reset_index(drop=True)
                return self.load_data
                
            except Exception as e:
                logging.warning(f"Error loading Load file {load_file}: {e}")
                return None
        else:
            logging.warning(f"Load file not found: {load_path}")
            return None
    
    def load_electricity_prices(self, num_timesteps, freq='15min'):
        """
        Load electricity prices from local files based on time resolution
        
        Parameters:
        -----------
        num_timesteps : int
            Number of timesteps needed
        freq : str
            Time frequency ('15min' or 'H'/'1H' for hourly)
        
        Returns:
        --------
        pd.Series : Price data or None if file not found
        """
        # Determine which file to use based on frequency
        if '15' in freq or freq == '15min' or freq == '15T':
            price_file = "strompreise_dayahead_2024_entsoe_DE_LU_15min.csv"
        else:
            price_file = "strompreise_dayahead_2024_entsoe_DE_LU_hourly.csv"
        
        price_path = self.output_dir / price_file
        
        # Also check in data directory
        if not price_path.exists():
            price_path = self.project_root / "data" / price_file
        
        if price_path.exists():
            try:
                df_prices = pd.read_csv(price_path)
                # Look for price column
                price_cols = [col for col in df_prices.columns if 'price' in col.lower() and 'mwh' in col.lower()]
                
                if price_cols:
                    prices = df_prices[price_cols[0]]
                elif 'price_eur_per_mwh' in df_prices.columns:
                    prices = df_prices['price_eur_per_mwh']
                else:
                    prices = df_prices.iloc[:, 1] if len(df_prices.columns) > 1 else df_prices.iloc[:, 0]
                
                # Match length to num_timesteps
                if len(prices) > num_timesteps:
                    prices = prices.iloc[:num_timesteps]
                elif len(prices) < num_timesteps:
                    # Repeat the pattern if needed
                    repeats = (num_timesteps // len(prices)) + 1
                    prices = pd.concat([prices] * repeats, ignore_index=True).iloc[:num_timesteps]
                
                logging.info(f"Loaded electricity prices from: {price_file}")
                logging.info(f"  Price range: {prices.min():.2f} - {prices.max():.2f} €/MWh")
                return prices.reset_index(drop=True)
                
            except Exception as e:
                logging.warning(f"Error loading price file {price_file}: {e}")
                return None
        else:
            logging.warning(f"Price file not found: {price_path}")
            return None
    
    def load_additional_timeseries(self, timeseries_list, num_timesteps=None):
        """
        Load multiple additional timeseries data from files with custom column names
        
        Parameters:
        -----------
        timeseries_list : list of dict
            List of dictionaries with keys:
            - 'file_path': str, path to CSV file (relative to project root or absolute)
            - 'column_name': str, name to give the column in output
            - 'source_column': str or int, optional, column name or index to extract (default: auto-detect)
            - 'unit_conversion': float, optional, multiply values by this factor (e.g., 0.001 for W to kW)
        num_timesteps : int, optional
            Number of timesteps needed (will trim or repeat data to match)
        
        Returns:
        --------
        dict : Dictionary with column_name as keys and pd.Series as values
        
        Example:
        --------
        timeseries_data = builder.load_additional_timeseries([
            {'file_path': 'data/wind_power.csv', 'column_name': 'Wind_kW', 'source_column': 'power_W', 'unit_conversion': 0.001},
            {'file_path': 'data/temperature.csv', 'column_name': 'Temp_C', 'source_column': 1}
        ], num_timesteps=35040)
        """
        result = {}
        
        for ts_config in timeseries_list:
            file_path = ts_config.get('file_path')
            column_name = ts_config.get('column_name')
            source_column = ts_config.get('source_column', None)
            unit_conversion = ts_config.get('unit_conversion', 1.0)
            
            if not file_path or not column_name:
                logging.warning(f"Skipping timeseries: missing file_path or column_name in {ts_config}")
                continue
            
            # Handle relative and absolute paths
            path = Path(file_path)
            if not path.is_absolute():
                path = self.project_root / file_path
            
            if not path.exists():
                logging.warning(f"File not found: {path}")
                continue
            
            try:
                # Read CSV file
                df = pd.read_csv(path)
                
                # Extract the appropriate column
                if source_column is not None:
                    if isinstance(source_column, str):
                        # Column name specified
                        if source_column in df.columns:
                            data = df[source_column]
                        else:
                            logging.warning(f"Column '{source_column}' not found in {path.name}, using first data column")
                            data = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
                    elif isinstance(source_column, int):
                        # Column index specified
                        data = df.iloc[:, source_column]
                    else:
                        data = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
                else:
                    # Auto-detect: skip timestamp columns, use first data column
                    data = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
                
                # Convert to numeric
                data = pd.to_numeric(data, errors='coerce').fillna(0)
                
                # Apply unit conversion
                if unit_conversion != 1.0:
                    data = data * unit_conversion
                    logging.info(f"Applied unit conversion factor {unit_conversion} to {column_name}")
                
                # Match length to num_timesteps if specified
                if num_timesteps is not None:
                    if len(data) > num_timesteps:
                        data = data.iloc[:num_timesteps]
                    elif len(data) < num_timesteps:
                        # Repeat the pattern if needed
                        repeats = (num_timesteps // len(data)) + 1
                        data = pd.concat([data] * repeats, ignore_index=True).iloc[:num_timesteps]
                
                data = data.reset_index(drop=True)
                result[column_name] = data
                
                logging.info(f"Loaded '{column_name}' from {path.name}")
                logging.info(f"  Range: {data.min():.2f} - {data.max():.2f}")
                
            except Exception as e:
                logging.error(f"Error loading timeseries from {path.name}: {e}")
                continue
        
        return result


def main():
    """Main execution function"""
    print("="*60)
    print("Timeseries Input Builder for oemof")
    print("="*60)
    print()
    
    builder = TimeseriesBuilder()
    
    # Check if Emobpy results directory exists
    if not builder.emobpy_dir.exists():
        logging.error(f"Emobpy results directory not found: {builder.emobpy_dir}")
        logging.info("Please ensure Emobpy results are saved in results/emobpy-V2H-WS25/")
        return
    
    try:
        # Read Emobpy results (returns list of tuples: [(df, filename), ...])
        emobpy_files = builder.read_emobpy_results()
        
        # Optional: Load additional timeseries data
        # Example usage - uncomment and customize as needed:
        # additional_timeseries = builder.load_additional_timeseries([
        #     {
        #         'file_path': 'data/wind_power.csv',
        #         'column_name': 'Wind_kW',
        #         'source_column': 'power_W',
        #         'unit_conversion': 0.001  # Convert W to kW
        #     },
        #     {
        #         'file_path': 'results/some-folder/temperature.csv',
        #         'column_name': 'Temperature_C',
        #         'source_column': 1  # Use column index 1
        #     },
        #     {
        #         'file_path': 'data/custom_load.csv',
        #         'column_name': 'Custom_Load_kW'
        #         # source_column auto-detects (uses first data column)
        #     }
        # ], num_timesteps=35040)
        
        # Process each file
        print(f"\nProcessing {len(emobpy_files)} file(s)...\n")
        
        for i, (emobpy_df, source_file) in enumerate(emobpy_files, 1):
            print(f"\n[{i}/{len(emobpy_files)}] Processing: {source_file}")
            print("-" * 60)
            
            # Generate output filename based on source file
            output_filename = f"input_timeseries_{source_file.replace('.csv', '')}.csv"
            
            # Build oemof input file
            df_output = builder.build_oemof_input(
                emobpy_df,
                output_filename=output_filename
            )
            
            # Optional: Add additional timeseries columns after building
            # if additional_timeseries:
            #     for col_name, col_data in additional_timeseries.items():
            #         df_output[col_name] = col_data
            #     # Re-save with additional columns
            #     output_path = builder.output_dir / output_filename
            #     df_output.to_csv(output_path, index=False)
            #     logging.info(f"Added {len(additional_timeseries)} additional timeseries columns")
        
        print("\n" + "="*60)
        print(f"✓ Successfully created {len(emobpy_files)} timeseries input file(s)!")
        print("="*60)
        
    except Exception as e:
        logging.error(f"Error building timeseries: {e}")
        raise


if __name__ == "__main__":
    main()
