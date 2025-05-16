'''
    Calculate Alpha from Azimuth and Elevation
    Azimuth and Elevation calculated from Ephemeris
'''
import os
import pandas as pd
import numpy as np
import re
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

list_satellite = ['G01', 'G02', 'G03', 'G04','G05', 'G06', 'G07', 'G08', 'G09', 'G10', 
                  'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'G19', 'G20', 
                  'G21', 'G22', 'G23', 'G24', 'G25', 'G26', 'G27', 'G28', 'G29', 'G30', 'G31', 'G32']

# Constants
mu = 3.986005e14  # Earth's gravitational constant (m^3/s^2)
omega_e = 7.2921151467e-5  # Earth's rotation rate (rad/s)

LINE_COUNT = 20000
c = 299792458
SMOOTH_WINDOW = 300

IS_CLOCK = 1
if IS_CLOCK == 0:
    # DATA_TYPE = 'NOCLOCK'
    # # No Clock
    # FOLDER_PATH = './Pseudorange_smooth_internal_clock/'  # Đường dẫn tới file 1
    # file1 = './Pseudorange_smooth_internal_clock/89683563/IGS000USA_R_20243560229_01D_01S_MO.rnx'  # Đường dẫn tới file 1
    # file2 = './Pseudorange_smooth_internal_clock/96813563/96813563.24O'  # Đường dẫn tới file 2
    NORMALIZE_VALUE = 0
else:
    DATA_TYPE = 'CLOCK'
    # Clock
    FOLDER_PATH = './Pseudorange_smooth_external_clock/'  # Đường dẫn tới file 2
    
    rx_name_01 = 'rx_01'
    # nav_file_01 = "F:\\PhDStudy\\Data\\Pseudorange_smooth_external_clock\\96813562\\96813562.24N"
    # observation_file_01 = "F:\\PhDStudy\\Data\\Pseudorange_smooth_external_clock\\96813562\\96813562.24O"


    nav_file_01 = "Data_Clock_Smooth\Pseudorange_smooth_external_clock\96813562\96813562.24N"
    observation_file_01 = "Data_Clock_Smooth\Pseudorange_smooth_external_clock\96813562\96813562.24O"

    
    # Tọa độ bộ thu trong hệ ECEF (Đã biết)
    receiver_ecef_01 = np.array([-1626343.5660, 5730606.2916, 2271881.4271])
    
    rx_name_02 = 'rx_02'
    # nav_file_02 = "F:\\PhDStudy\\Data\\Pseudorange_smooth_external_clock\\89683562\\IGS000USA_R_20243560202_01D_01S_GN.rnx"
    # observation_file_02 = "F:\\PhDStudy\\Data\\Pseudorange_smooth_external_clock\\89683562\\IGS000USA_R_20243560202_01D_01S_MO.rnx"
    nav_file_02 = "Data_Clock_Smooth\Pseudorange_smooth_external_clock\89683562\IGS000USA_R_20243560202_01D_01S_GN.rnx"
    observation_file_02 = "Data_Clock_Smooth\Pseudorange_smooth_external_clock\89683562\IGS000USA_R_20243560202_01D_01S_MO.rnx"
    
    
    # Tọa độ bộ thu trong hệ ECEF (Đã biết)
    receiver_ecef_02 = np.array([-1626584.7059, 5730519.4577, 2271864.3917])
    
    
    NORMALIZE_VALUE = 85897

# Leap seconds giữa GPS Time và UTC (18 giây tính đến năm 2024)
GPS_UTC_LEAP_SECONDS = 18 
def compute_dt_seconds(t_rx_utc_str):
    # Chuyển string sang datetime UTC
    #t_rx_utc = datetime.strptime(t_rx_utc_str, "%Y/%m/%d %H:%M:%S")
    
    t_rx_utc = datetime.strptime(t_rx_utc_str, "%Y %m %d %H %M %S.%f")

    # Chuyển t_rx từ UTC sang GPS Time
    t_rx_gps = t_rx_utc + timedelta(seconds=GPS_UTC_LEAP_SECONDS)

    # Xác định ngày đầu tuần GPS (Chủ Nhật trước đó)
    w_start = t_rx_gps - timedelta(days=t_rx_gps.weekday() + 1)
    gps_week_start = datetime(w_start.year, w_start.month, w_start.day, 0, 0, 0)
    # Chuyển thời điểm thu tín hiệu sang giây của tuần GPS (TOE format)
    t_rx_toe = (t_rx_gps - gps_week_start).total_seconds()

    return {
        "t_rx_utc": t_rx_utc,
        "t_rx_gps": t_rx_gps,
        "gps_week_start": gps_week_start,
        "t_rx_toe": t_rx_toe,
    }

def solve_kepler(M, e, tolerance=1e-10):
    """ Solve Kepler's Equation for E (Eccentric Anomaly). """
    E = M  # Initial guess
    while True:
        E_new = E + (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        if abs(E_new - E) < tolerance:
            break
        E = E_new
    return E

# Function to convert ECEF to geodetic coordinates
def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates to geodetic (latitude, longitude, height).

    Args:
        x, y, z (float): ECEF coordinates.

    Returns:
        tuple: Latitude (degrees), Longitude (degrees), Height (meters).
    """
    a = 6378137.0
    e2 = 6.69437999014e-3
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    return np.degrees(lat), np.degrees(lon), h

# Function to calculate satellite ECEF coordinates
def calculate_satellite_ecef(parameters, t):
    """
    Calculate satellite ECEF coordinates from orbital parameters.

    Args:
        parameters (list): Orbital parameters from NAV RINEX.
        t (float): Time since ephemeris reference epoch.

    Returns:
        tuple: ECEF coordinates (x, y, z).
    """
    sqrtA = parameters[7]
    e = parameters[5]
    M0 = parameters[3]
    omega = parameters[14]
    OMEGA0 = parameters[10]
    i0 = parameters[12]
    Delta_n = parameters[2]
    OMEGA_DOT = parameters[15]
    Toe = parameters[8]

    A = sqrtA**2
    n0 = np.sqrt(mu / A**3)
    n = n0 + Delta_n
    t_corr = t - Toe
    M = M0 + n * t_corr

    E = solve_kepler(M % (2 * np.pi), e)
    v = np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)
    phi = v + omega

    r = A * (1 - e * np.cos(E))
    x_orb = r * np.cos(phi)
    y_orb = r * np.sin(phi)

    OMEGA_corrected = OMEGA0 + (OMEGA_DOT - omega_e) * t_corr - omega_e * t

    x_ecef = x_orb * np.cos(OMEGA_corrected) - y_orb * np.cos(i0) * np.sin(OMEGA_corrected)
    y_ecef = x_orb * np.sin(OMEGA_corrected) + y_orb * np.cos(i0) * np.cos(OMEGA_corrected)
    z_ecef = y_orb * np.sin(i0)

    return x_ecef, y_ecef, z_ecef

# Function to calculate azimuth and elevation
def calculate_azimuth_elevation(sat_pos, receiver_pos):
    """
    Calculate azimuth and elevation angles between satellite and receiver.

    Args:
        sat_pos (list): Satellite position in ECEF [X, Y, Z].
        receiver_pos (list): Receiver position in ECEF [X, Y, Z].

    Returns:
        tuple: Azimuth (degrees), Elevation (degrees).
    """
    sat_pos = np.array(sat_pos)
    receiver_pos = np.array(receiver_pos)
    los_vector = sat_pos - receiver_pos
    
    euclid_dis = np.linalg.norm(los_vector)
    
    los_unit = los_vector / euclid_dis
    lat, lon, _ = ecef_to_geodetic(*receiver_pos)
    lat, lon = np.radians(lat), np.radians(lon)
    transform_matrix = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)],
        [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)],
    ])
    los_enu = np.dot(transform_matrix, los_unit)
    azimuth = np.degrees(np.arctan2(los_enu[0], los_enu[1]))
    elevation = np.degrees(np.arcsin(los_enu[2]))
    if azimuth < 0:
        azimuth += 360
        
    # tinh goc alpha 
    # cos alpha = -(cos Az) * (cos El)
    cos_alpha = - (math.cos(math.radians(azimuth)) * math.cos(math.radians(elevation)))
    alpha_angle = math.degrees(math.acos(cos_alpha))
    
    return azimuth, elevation, alpha_angle, cos_alpha

# Function to parse NAV RINEX file
def parse_nav_rinex(nav_file):
    """
    Parse satellite orbital parameters from NAV RINEX file.

    Args:
        nav_file (str): Path to NAV RINEX file.

    Returns:
        list: List of satellite data with orbital parameters.
    """
    satellite_data = []
    with open(nav_file, 'r') as file:
        reading_data = False
        current_satellite = None

        for line in file:
            if "END OF HEADER" in line:
                reading_data = True
                continue

            if reading_data:
                if line.startswith('G'):  # Start of new satellite data
                    if current_satellite:
                        satellite_data.append(current_satellite)

                    current_satellite = {
                        "PRN": line[:3].strip(),
                        "epoch": re.split(r'\s+', line[3:23].strip()),
                        "parameters": []
                    }
                elif current_satellite:
                    values = line.split()
                    if values:
                        current_satellite["parameters"].extend(
                            [float(v.replace('D', 'E')) for v in values]
                        )
        if current_satellite:
            satellite_data.append(current_satellite)

    return satellite_data

def read_rinex_observation(file_path, isClock):
    """
    Đọc file RINEX và trích xuất dữ liệu cần thiết.
    Trả về DataFrame chứa: thời gian (epoch), số hiệu vệ tinh, và pseudorange.
    """
    data = []
    reading_data = False
    epoch_time = ''
    clock_bias = 0
    count_line = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()[200:LINE_COUNT]
        for line in lines:

            if line.startswith('>'):  # Epoch bắt đầu bằng ký tự '>'
                epoch_time = line[1:25].strip()  # Lấy thời gian epoch
                parts = line.split()
                clock_bias = parts[9]

            if not reading_data:
                if line.startswith('>'):  # Epoch bắt đầu bằng ký tự '>'
                    reading_data = True
            else:
                if line.strip() == '':
                    continue

                # Split dong bang khoang trang hoac tab
                parts = line.split()
                if len(parts) < 2:
                    continue

                satellite_id = parts[0]  # Lấy số hiệu vệ tinh
                if satellite_id in list_satellite:
                    pseudorange = parts[1]  # Lấy pseudorange
                    #carrier_phase_str = parts[3] # lay carrier phase
                    carrier_phase_str = line[21:34]
                    if carrier_phase_str.isspace():
                        carrier_phase = 0
                    else:
                        carrier_phase = float(carrier_phase_str)

                    if pseudorange and carrier_phase != 0:
                        newPseudorange = float(pseudorange) - (c*float(clock_bias))
                        # if isClock == 1:
                        #     newPseudorange = float(pseudorange)
                        # else:
                        #     newPseudorange = float(pseudorange) - (c*float(clock_bias))
                        data.append((epoch_time, satellite_id, newPseudorange, carrier_phase))
    return pd.DataFrame(data, columns=['epoch', 'PRN', 'pseudorange', 'carrier_phase'])

def cal2gpstime(date_str: str):
    """
    Chuyển đổi thời gian dương lịch (UTC) dạng chuỗi sang GPS week và GPS seconds.

    Args:
        date_str (str): Chuỗi định dạng "YYYY MM DD HH MM SS.sss"
        
    Returns:
        tuple: (gps_week, gps_seconds)
    """
    # GPS epoch bắt đầu từ ngày 06/01/1980
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)

    # Chuyển đổi chuỗi đầu vào thành datetime
    dt = datetime.strptime(date_str, "%Y %m %d %H %M %S.%f")

    # Số giây từ epoch đến thời điểm hiện tại
    delta_seconds = (dt - gps_epoch).total_seconds()

    # Tính tuần GPS và số giây trong tuần
    gps_week = int(delta_seconds // 604800)  # Một tuần có 604800 giây
    gps_seconds = round(delta_seconds % 604800, 3)  # Làm tròn đến 3 chữ số thập phân

    return gps_week, gps_seconds

def plotAngle(obs_data, folder_path, angle_name, angle_column, rx_name):
    for satellite in obs_data['PRN'].unique():
        plt.figure(figsize=(15, 6))
        alpha_data = obs_data[obs_data['PRN'] == satellite]
        # plt.plot(satellite_data['epoch'], satellite_data['normalize_difference'], label=f'Satellite {satellite}')

        plt.plot(alpha_data['epoch'], alpha_data[angle_column], label=f'Satellite {satellite}')

        plt.xlabel('Epoch')
        plt.xticks(rotation=45)
        plt.ylabel(f'{angle_name} Angle (degree)')
        plt.title(rx_name + ' PRN-' + satellite + ' ' + f'{angle_name} Angle Time Series (' + DATA_TYPE + ')')
        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(folder_path, rx_name + '_' + satellite + '_' + DATA_TYPE + f'_{angle_name}Manual.png'))
        plt.close()
    #plt.show()
    
# Debug: Print receiver position
print("Receiver 01 Position:", receiver_ecef_01)

obs_data_01 = read_rinex_observation(observation_file_01, IS_CLOCK)
nav_data_01 = parse_nav_rinex(nav_file_01)

# Duyệt qua từng epoch trong obs_data
azimuths_01 = []
elevations_01 = []
alpha_angles_01 = []
cos_alpha_01 = []

for index, row in obs_data_01.iterrows():
    epoch = row['epoch']
    prn = row['PRN']
    
    temp_time_input = compute_dt_seconds(epoch)
    t = temp_time_input["t_rx_toe"]

    # Tìm vệ tinh tương ứng trong nav_data
    # sat_info = nav_data[nav_data['PRN'] == prn]
    sat_info = next((item for item in nav_data_01 if item['PRN'] == prn), None)
    if not sat_info:
        azimuths_01.append(np.nan)
        elevations_01.append(np.nan)
        continue

    # Lấy tọa độ ECEF của vệ tinh
    sat_ecef = calculate_satellite_ecef(sat_info["parameters"], t)
    #sat_ecef = np.array([sat_info['X'], sat_info['Y'], sat_info['Z']])
    
    # Tính góc azimuth & elevation
    az_01, el_01, alpha_01, cosAlpha_01 = calculate_azimuth_elevation(sat_ecef, receiver_ecef_01)
    
    # Lưu kết quả
    azimuths_01.append(az_01)
    elevations_01.append(el_01)
    alpha_angles_01.append(alpha_01)
    cos_alpha_01.append(cosAlpha_01)

# Thêm vào obs_data_01
obs_data_01['Azimuth'] = azimuths_01
obs_data_01['Elevation'] = elevations_01
obs_data_01['Alpha'] = alpha_angles_01
obs_data_01['cosAlpha'] = cos_alpha_01

# 
obs_data_02 = read_rinex_observation(observation_file_02, IS_CLOCK)
nav_data_02 = parse_nav_rinex(nav_file_02)
azimuths_02 = []
elevations_02 = []
alpha_angles_02 = []
cos_alpha_02 = []
for index, row in obs_data_02.iterrows():
    epoch = row['epoch']
    prn = row['PRN']
    
    temp_time_input = compute_dt_seconds(epoch)
    t = temp_time_input["t_rx_toe"]

    # Tìm vệ tinh tương ứng trong nav_data
    # sat_info = nav_data[nav_data['PRN'] == prn]
    sat_info = next((item for item in nav_data_02 if item['PRN'] == prn), None)
    if not sat_info:
        azimuths_02.append(np.nan)
        elevations_02.append(np.nan)
        continue

    # Lấy tọa độ ECEF của vệ tinh
    sat_ecef = calculate_satellite_ecef(sat_info["parameters"], t)
    #sat_ecef = np.array([sat_info['X'], sat_info['Y'], sat_info['Z']])
    
    # Tính góc azimuth & elevation
    az, el, alpha, cosAlpha = calculate_azimuth_elevation(sat_ecef, receiver_ecef_02)

    
    # Lưu kết quả
    azimuths_02.append(az)
    elevations_02.append(el)
    alpha_angles_02.append(alpha)
    cos_alpha_02.append(cosAlpha)


# Thêm vào obs_data_02
obs_data_02['Azimuth'] = azimuths_02
obs_data_02['Elevation'] = elevations_02
obs_data_02['Alpha'] = alpha_angles_02
obs_data_02['cosAlpha'] = cos_alpha_02

print(obs_data_01)
print(obs_data_02)

# Đồng bộ hóa các epoch dựa trên thời gian
merged_data = pd.merge(obs_data_01, obs_data_02, on=['epoch', 'PRN'], suffixes=('_file1', '_file2'))

# merged_smoothed_data = pd.merge(smoothed_data1, smoothed_data2, on=['epoch', 'satellite'], suffixes=('_file1', '_file2'))

# Tính toán hiệu alpha
merged_data['deltaAlpha'] = merged_data['Alpha_file1'] - merged_data['Alpha_file2']

# merged_smoothed_data['difference'] = merged_smoothed_data['smoothed_pseudorange_file1'] - merged_smoothed_data['smoothed_pseudorange_file2']

# Normalize (Min-Max)
# merged_data['normalize_difference'] = min_max_normalize(merged_data['difference'])

# Vẽ đồ thị sự khác biệt pseudorange
# plotAngle(obs_data, FOLDER_PATH, "Azimuth", "Azimuth")
# plotAngle(obs_data, FOLDER_PATH, "Elevation", "Elevation")
# plotAngle(obs_data_01, FOLDER_PATH, "Alpha", "Alpha")

# obs_data_01.sort_values(by=['epoch','PRN'], ascending=[True, True], inplace=True)
# plotAngle(obs_data_01, FOLDER_PATH, "Alpha", "deltaAlpha")

merged_data.sort_values(by=['epoch','PRN'], ascending=[True, True], inplace=True)

# In kết quả
(merged_data[['epoch', 'PRN', 'Azimuth_file1', 'Elevation_file1', 'Alpha_file1']]).to_csv("file01_result.txt", sep='\t', index=False)
(merged_data[['epoch', 'PRN', 'Azimuth_file2', 'Elevation_file2', 'Alpha_file2']]).to_csv("file02_result.txt", sep='\t', index=False)
(merged_data[['epoch', 'PRN', 'Alpha_file1', 'Alpha_file2', 'deltaAlpha']]).to_csv("diff_result.txt", sep='\t', index=False)
plotAngle(merged_data, FOLDER_PATH, "Alpha", "Alpha_file1", rx_name_01)
plotAngle(merged_data, FOLDER_PATH, "Alpha", "Alpha_file2", rx_name_02)

plotAngle(merged_data, FOLDER_PATH, "Diff", "deltaAlpha", 'Diff')
plotAngle(merged_data, FOLDER_PATH, "cosAlpha", "cosAlpha_file1", 'cosAlpha_file1')
plotAngle(merged_data, FOLDER_PATH, "cosAlpha", "cosAlpha_file2", 'cosAlpha_file2')

# plotClockBias()
# plotSingleDifference(merged_data)
# plotSmoothedSingleDiff(merged_smoothed_data)
