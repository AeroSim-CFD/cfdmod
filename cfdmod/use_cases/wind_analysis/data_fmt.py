import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

from cfdmod.use_cases.wind_analysis.profile import ProfileCalculator_NBR

...
# Pandas DataFrame must have:
# Vavg
# dir
# Vgust
# time

def get_clean_INMET_file(csv_path: pathlib.Path) -> pd.DataFrame:
    """Clean wind data from INMET (Instituto Nacional de Meteorologia) source and separate by year"""
    df = pd.read_csv(csv_path, header=9, sep=';')
    df['datetime'] = df.apply(lambda row: f"{row['Data Medicao']}T{int(row['Hora Medicao'])//100:02}:00:00", axis=1)
    df['u_mean_raw'] = df["VENTO, VELOCIDADE HORARIA(m/s)"]
    df['u_gust_raw'] = df["VENTO, RAJADA MAXIMA(m/s)"]
    df['wind_direction'] = df["VENTO, DIRECAO HORARIA (gr)(° (gr))"]  % 360
    df = df.drop(columns=['Data Medicao', 'Hora Medicao', "VENTO, VELOCIDADE HORARIA(m/s)", "VENTO, RAJADA MAXIMA(m/s)", "VENTO, DIRECAO HORARIA (gr)(° (gr))", "Unnamed: 5"])
    df = df.dropna(subset=['u_mean_raw', 'u_gust_raw'], how='all')
    df[['u_mean_raw', 'u_gust_raw']] = df[['u_mean_raw', 'u_gust_raw']].fillna(0)
    return df
    
def get_clean_NCEI_file(csv_path: pathlib.Path) -> pd.DataFrame:
    """Clean wind data from NCEI (National Centers for Environmental Information) source and separate stations"""
    df = pd.read_csv(csv_path)
    df['station'] = df['STATION'].astype(str).str[0:5]
    df['datetime'] = df['DATE']
    df['u_mean_raw'] = df["WND"].str[8:12].astype(float) / 10
    df['u_gust_raw'] = df["OC1"].str[0:4].astype(float) / 10
    df['wind_direction'] = df["WND"].str[0:3].astype(int) % 360
    df = df.drop(columns=['STATION', 'DATE', "SOURCE", "REPORT_TYPE", "CALL_SIGN", "QUALITY_CONTROL", "OC1", "WND"])
    df = df.dropna(subset=['u_mean_raw', 'u_gust_raw'], how='all')
    # df[['u_mean_raw', 'u_gust_raw']] = df[['u_mean_raw', 'u_gust_raw']].fillna(0)
    mask_invalid_mean = (df['u_mean_raw']==999.9)
    df.loc[mask_invalid_mean, 'u_mean_raw'] = np.nan
    mask_invalid_gust = (df['u_gust_raw']==999.9)
    df.loc[mask_invalid_gust, 'u_gust_raw'] = np.nan
    return df
    
    
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
    # visualize mean
    fig, ax = plt.subplots(3,1,figsize=(30,15))
    if 'station' in data.columns:
        fig.suptitle(f"station: {data['station'].unique()[0]}")
    ax[0].plot(pd.to_datetime(data['datetime']), data['u_mean_raw'],'.')
    ax[0].set_title('Mean velocity')
    ax[1].plot(pd.to_datetime(data['datetime']), data['u_gust_raw'],'.')
    ax[1].set_title('Gust velocity')
    ax[2].plot(pd.to_datetime(data['datetime']), data['wind_direction'],'.')
    ax[2].set_title('Wind direction')
    plt.show()

def plot_pdfs(data: pd.DataFrame):
    fig, ax = plt.subplots(1,3,figsize=(30,10))
    ax[0].hist(data['u_mean_raw'], bins='auto')
    ax[0].set_title('Mean velocity')
    ax[1].hist(data['u_gust_raw'], bins='auto')
    ax[1].set_title('Gust velocity')
    ax[2].hist(data['wind_direction'],bins='auto')
    ax[2].set_title('Wind direction')
    plt.show()
    
    
def remove_date_ranges(data: pd.DataFrame, ranges_to_remove: list[tuple[str,str]]|list[tuple[pd.Timestamp,pd.Timestamp]]) -> pd.DataFrame:
    ranges_to_remove = [pd.to_datetime(range) for range in ranges_to_remove]
    datetime = pd.to_datetime(data['datetime'])
    remove_mask = pd.Series(False, index=data.index)
    for start, end in ranges_to_remove:
        remove_mask |= ((datetime >= start) & (datetime < end))
    print("Number of values removed: ", remove_mask.sum())
    return data[~ remove_mask].copy()

def remove_wind_direction(data: pd.DataFrame, wind_directions_to_remove: list[int]) -> pd.DataFrame:
    eps = 1
    remove_mask = pd.Series(False, index=data.index)
    for wind_direction in wind_directions_to_remove:
        remove_mask |= (data['wind_direction']-wind_direction).abs() <= eps
    return data[~ remove_mask].copy()


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

def add_rescaled_velocities_columns(data: pd.DataFrame, station_roughness_path: pathlib.Path, station_mast_height: float=10, filter_time_mean: float=3600, filter_time_gust: float=3):
    if not station_roughness_path.exists():
        station_roughness = _get_roughness_table_cat2([30,90,150,210,270,330])
        profile_station = ProfileCalculator_NBR(directional_data=station_roughness, V0=1)
    else:
        station_roughness = pd.read_csv(station_roughness_path)
        profile_station = ProfileCalculator_NBR.build(data_csv=station_roughness_path, V0=1)

    data["u_mean"] = np.nan
    data["u_gust"] = np.nan
    ref_cat2 = _get_roughness_table_cat2(station_roughness['wind_direction'].to_numpy())
    profile_cat2 = ProfileCalculator_NBR(directional_data=ref_cat2, V0=1)
    directions = list(station_roughness['wind_direction'])
    
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

        multiplier_gust = profile_station.get_U_H(height=station_mast_height, direction=direction, recurrence_period=50, time_filter_seconds=filter_time_gust)
        multiplier_mean = profile_station.get_U_H(height=station_mast_height, direction=direction, recurrence_period=50, time_filter_seconds=filter_time_mean)
        multiplier_cat2_mean = profile_cat2.get_U_H(height=10, direction=direction, recurrence_period=50, time_filter_seconds=3600)

        data.loc[dir_selection, 'u_mean'] = data[dir_selection]['u_mean_raw']/multiplier_mean*multiplier_cat2_mean
        data.loc[dir_selection, 'u_gust'] = data[dir_selection]['u_gust_raw']/multiplier_gust


def _get_roughness_table_cat2(wind_directions: list) -> pd.DataFrame:
    ref_cat2 = pd.DataFrame()
    ref_cat2['wind_direction'] = wind_directions
    for cat in ['I', 'III','IV','V']:
        ref_cat2[cat] = 0    
    ref_cat2['II'] = 1
    ref_cat2['Kd'] = 1
    return ref_cat2

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