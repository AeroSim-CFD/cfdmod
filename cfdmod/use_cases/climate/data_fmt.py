import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from cfdmod.use_cases.analytical.wind_profile import WindProfile

def break_INMET_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Clean wind data from INMET (Instituto Nacional de Meteorologia) source and separate by year"""
    data["Hora Medicao"] = data["Hora Medicao"].astype("Int64") #handles NaN problems
    data['datetime'] = data.apply(lambda row: f"{row['Data Medicao']}T{row['Hora Medicao']//100:02}:00:00", axis=1)
    data['u_mean_raw'] = data["VENTO, VELOCIDADE HORARIA(m/s)"]
    data['u_gust_raw'] = data["VENTO, RAJADA MAXIMA(m/s)"]
    data['wind_direction'] = data["VENTO, DIRECAO HORARIA (gr)(° (gr))"]  % 360
    data = data.drop(columns=['Data Medicao', 'Hora Medicao', "VENTO, VELOCIDADE HORARIA(m/s)", "VENTO, RAJADA MAXIMA(m/s)", "VENTO, DIRECAO HORARIA (gr)(° (gr))", "Unnamed: 5"])
    return data
    
def break_NCEI_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Clean wind data from NCEI (National Centers for Environmental Information) source and separate stations"""
    data['station'] = data['STATION'].astype(str).str[0:5]
    data['datetime'] = data['DATE']
    columns_to_drop = ['STATION','DATE']
    
    def break_col(df, old_col, new_col, col_start, col_end, astype, multiplier, null_indicator):
        df[new_col] = df[old_col].str[col_start:col_end+1].astype(astype) * multiplier
        df[f'{new_col}_quality'] = df[old_col].str[col_end+2].astype("Int64") #there is always a ',' character between the value and its quality indicator
        mask_invalid_row = (df[new_col]==null_indicator)
        df.loc[mask_invalid_row, new_col] = np.nan
        return df

    for old_col, new_col, col_start, col_end, astype, multiplier, null_indicator in zip(
        ['WND', 'wind_direction', 0, 2, 'Int64', 1, 999],
        ['WND', 'u_mean_raw', 8, 11, 'float', 1/10, 999.9],
        ['OC1', 'u_gust_raw', 0, 3, 'float', 1/10, 999.9],
        ['TMP', 'temperature', 0, 4, 'float', 1/10, 999.9],
        ['DEW', 'dew_point', 0, 4,'float', 1/10, 999.9],
    ):
        if old_col in data.columns:
            data = break_col(data, old_col, new_col, col_start, col_end, astype, multiplier, null_indicator)
            columns_to_drop.append(old_col)
        else:
            data[[new_col,f'{new_col}_quality']] = np.nan
    data = data.drop(columns=list(set(columns_to_drop)))
    return data
    
    
def validate_table(data: pd.DataFrame):
    error = False
    # check if gust > mean
    if data[data['u_mean_raw']>data['u_gust_raw']].shape[0] > 0:
        print("There are mean values greater than gust values")
        error = True
    # check if there are invalid angles:
    if data[ (data['wind_direction']<0) | (data['wind_direction']>360) ].shape[0] > 0:
        print("There are angles outside the range [0,360]")
        error = True
    # check if there are absurd velocities:
    if data[ (data['u_mean_raw']<0) | (data['u_mean_raw']>70) | (data['u_gust_raw']<0) | (data['u_gust_raw']>70) ].shape[0] > 0:
        print("There are velocities outside the range [0,70]m/s")
        error = True
    if not error:
        print("The table has no invalid rows")
    return not error
        
def plot_series(data: pd.DataFrame):      
    # visualize mean
    fig, ax = plt.subplots(3,1,figsize=(30,15))
    if 'station' in data.columns:
        fig.suptitle(f"station: {data['station'].unique()[0]}")
    markersize=1.75
    ax[0].plot(pd.to_datetime(data['datetime']), data['u_mean_raw'],'.', markersize=markersize)
    ax[0].set_title('Mean velocity')
    ax[1].plot(pd.to_datetime(data['datetime']), data['u_gust_raw'],'.', markersize=markersize)
    ax[1].set_title('Gust velocity')
    ax[2].plot(pd.to_datetime(data['datetime']), data['wind_direction'],'.', markersize=markersize)
    ax[2].set_title('Wind direction')
    return fig, ax

  

def plot_pdfs(data: pd.DataFrame, bins_mean: str|int = 'auto', bins_gust: str|int = 'auto'):
    fig, ax = plt.subplots(1,3,figsize=(30,10))
    ax[0].hist(data['u_mean_raw'], bins=bins_mean,density=True,)
    ax[0].set_title('Mean velocity')
    ax[1].hist(data['u_gust_raw'], bins=bins_gust, density=True,)
    ax[1].set_title('Gust velocity')
    ax[2].hist(data['wind_direction'],bins=bins_mean, density=True,)
    ax[2].set_title('Wind direction')
    return fig, ax
    
    
def remove_date_ranges(data: pd.DataFrame, ranges_to_remove: list[str, pd.Timestamp, tuple[str,str]]|list[tuple[pd.Timestamp,pd.Timestamp]]) -> pd.DataFrame:
    ranges_to_remove = [pd.to_datetime(range) for range in ranges_to_remove]
    datetime = pd.to_datetime(data['datetime'])
    remove_mask = pd.Series(False, index=data.index)
    for range_to_remove in ranges_to_remove:
        if isinstance(range_to_remove, tuple) or isinstance(range_to_remove, list):
            start, end = range_to_remove
            remove_mask |= ((datetime >= start) & (datetime < end))
        else:
            remove_mask |= (datetime == range_to_remove)
    print("Number of values removed: ", remove_mask.sum())
    return data[~ remove_mask]

def select_date_ranges(data: pd.DataFrame, ranges_to_select: list[tuple[str,str]]|list[tuple[pd.Timestamp,pd.Timestamp]]) -> pd.DataFrame:
    ranges_to_select = [pd.to_datetime(range) for range in ranges_to_select]
    datetime = pd.to_datetime(data['datetime'])
    select_mask = pd.Series(False, index=data.index)
    for start, end in ranges_to_select:
        select_mask |= ((datetime >= start) & (datetime < end))
    print("Number of values selected: ", select_mask.sum())
    return data[select_mask]

def remove_wind_direction(data: pd.DataFrame, wind_directions_to_remove: list[int]) -> pd.DataFrame:
    remove_mask = pd.Series(False, index=data.index)
    for wind_direction in wind_directions_to_remove:
        remove_mask |= data['wind_direction']==wind_direction
    return data[~ remove_mask]

def remove_invalid_rows(data: pd.DataFrame, column: str) -> pd.DataFrame:
    remove_mask = data[f'{column}_quality'].isin([2,3,6,7])
    return data[~ remove_mask]


def add_season_and_period_columns(data: pd.DataFrame) -> pd.DataFrame:
    # add season
    seasons = {
        'autumn': [(pd.Timestamp('2000-03-21'), pd.Timestamp('2000-06-21'))],
        'winter': [(pd.Timestamp('2000-06-21'), pd.Timestamp('2000-09-21'))],
        'spring': [(pd.Timestamp('2000-09-21'), pd.Timestamp('2000-12-21'))],
        'summer': [(pd.Timestamp('2000-12-21'), pd.Timestamp('2000-12-31T23:59:59')), (pd.Timestamp('2000-01-01'), pd.Timestamp('2000-03-21'))]
    }
    normalized_datetime = pd.to_datetime(data['datetime']).apply(lambda x: x.replace(year=2000))
    for season in seasons:
        datetime = pd.to_datetime(normalized_datetime)
        for interval in seasons[season]:
            (start, end) = interval
            mask_season = (datetime>=start) & (datetime<end) 
            data.loc[mask_season, 'season'] = season
    # add period
    dt = pd.to_datetime(data['datetime']).dt
    mask_day = (dt.hour>=6) & (dt.hour<18)
    data.loc[mask_day, 'period'] = 'day'
    data.loc[~mask_day, 'period'] = 'night'

def add_rescaled_velocities_columns(data: pd.DataFrame, station_wind_profile: WindProfile, station_mast_height: float=10, filter_time_mean: float=3600, filter_time_gust: float=3):
    data["u_mean"] = np.nan
    data["u_gust"] = np.nan
    profile_opencountry = station_wind_profile.get_opencountry_profile()
    directions = list(station_wind_profile.directional_data['wind_direction'])
    
    direction_cuts = [(d_0+d_1)/2 for d_0, d_1 in zip(directions[:-1], directions[1:])]
    direction_cuts.append(((directions[0]+360)+directions[-1])/2 % 360)
    direction_cuts = sorted(direction_cuts)

    for i in range(len(direction_cuts)):
        direction = directions[i]
        d_0, d_1 = direction_cuts[i], direction_cuts[(i+1)%len(directions)]
        if(i < len(directions)-1):
            dir_selection = (data['wind_direction']>=d_0) & (data['wind_direction']<d_1)
        else:
            dir_selection = (data['wind_direction']>=d_0) | (data['wind_direction']<d_1)

        if(dir_selection.sum() == 0):
            continue

        multiplier_gust = station_wind_profile.get_U_H(height=station_mast_height, direction=direction, recurrence_period=50, time_filter_seconds=filter_time_gust)
        multiplier_OC_gust = profile_opencountry.get_U_H(height=10, direction=direction, recurrence_period=50, time_filter_seconds=filter_time_gust)
        multiplier_mean = station_wind_profile.get_U_H(height=station_mast_height, direction=direction, recurrence_period=50, time_filter_seconds=filter_time_mean)
        multiplier_OC_mean = profile_opencountry.get_U_H(height=10, direction=direction, recurrence_period=50, time_filter_seconds=3600)

        data.loc[dir_selection, 'u_mean'] = data[dir_selection]['u_mean_raw']/multiplier_mean*multiplier_OC_mean
        data.loc[dir_selection, 'u_gust'] = data[dir_selection]['u_gust_raw']/multiplier_gust*multiplier_OC_gust

def separate_by_year(data: pd.DataFrame) -> dict[int, pd.DataFrame]:
    years_col = pd.to_datetime(data['datetime']).dt.year
    results = {}
    for year in years_col.unique():
        year_mask = years_col == year
        results[int(year)] = data[year_mask].copy().reset_index(drop=True)
    return results

def separate_by_station(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    stations_col = data['station']
    results = {}
    for station in stations_col.unique():
        station_mask = stations_col == station
        results[station] = data[station_mask].copy().reset_index(drop=True)
    return results