'''
    Calculate Alpha from Carrier Smoothing
    Data smoothed export from Receiver
'''
import os
import pandas as pd
import numpy as np
import re
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyfit

from datetime import datetime, timedelta

list_satellite = ['G01', 'G02', 'G03', 'G04','G05', 'G06', 'G07', 'G08', 'G09', 'G10', 
                  'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'G19', 'G20', 
                  'G21', 'G22', 'G23', 'G24', 'G25', 'G26', 'G27', 'G28', 'G29', 'G30', 'G31', 'G32']

LINE_COUNT = 20000
c = 299792458
L1 = 1575.42e6
Lamda_L1 = c/L1
D_rx1_rx2 = 3.2 # mét
angleAB = 58 # độ


# Làm tròn theo quy tắc bình thường
def normal_round(number):
    if number % 1 >= 0.5:
        return math.ceil(number)
    else:
        return math.floor(number)

IS_CLOCK2 = 1
if IS_CLOCK2 == 0:
    # DATA_TYPE = 'NOCLOCK'
    # # No Clock
    # FOLDER_PATH = './Pseudorange_smooth_internal_clock/'  # Đường dẫn tới file 1
    # file1 = './Pseudorange_smooth_internal_clock/89683563/IGS000USA_R_20243560229_01D_01S_MO.rnx'  # Đường dẫn tới file 1
    # file2 = './Pseudorange_smooth_internal_clock/96813563/96813563.24O'  # Đường dẫn tới file 2
    NORMALIZE_VALUE = 0
else:
    DATA_TYPE = 'CLOCK'
    # Clock
    FOLDER_PATH = r'./Pseudorange_smooth_external_clock/'  # Đường dẫn tới file 2

    rx_name_01 = 'rx_01'

    observation_file1 = r"hai\ngoc\spoof\SP-2025-05-14_07\raw_data_1.obs"
    observation_file2 = r"hai\ngoc\spoof\SP-2025-05-14_07\raw_data_2.obs"
    
    nav_file1= r"hai\ngoc\spoof\SP-2025-05-14_07\raw_data_1.nav"
    nav_file2= r"hai\ngoc\spoof\SP-2025-05-14_07\raw_data_2.nav"


    
    # Tọa độ bộ thu trong hệ ECEF (Đã biết)
    receiver_ecef_01 = np.array([  -1626589.5179 , 5730541.5951 , 2271880.7118 ])
    
    rx_name_02 = 'rx_02'   
    # Tọa độ bộ thu trong hệ ECEF (Đã biết)
    receiver_ecef_02 = np.array([  -1626589.7104 , 5730542.7933 , 2271881.0223 ])
    
    
    NORMALIZE_VALUE = 85897


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
    gps_seconds = round(delta_seconds % 604800, 3) +18  # Làm tròn đến 3 chữ số thập phân

    return gps_week, gps_seconds
def calc_gps_seconds(date_str: str):
    """bạn
    Chuyển đổi thời gian dương lịch (UTC) dạng chuỗi sang GPS week và GPS seconds.

    Args:
        date_str (str): Chuỗi định dạng "YYYY MM DD HH MM SS.sss"
        
    Returns:
        tuple: ( gps_seconds)
    """
    # GPS epoch bắt đầu từ ngày 06/01/1980
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)

    # Chuyển đổi chuỗi đầu vào thành datetime
    dt = datetime.strptime(date_str, "%Y %m %d %H %M %S.%f")

    # Số giây từ epoch đến thời điểm hiện tại
    delta_seconds = (dt - gps_epoch).total_seconds()

    # Tính tuần GPS và số giây trong tuần
    gps_seconds = round(delta_seconds % 604800, 3) +18  # Làm tròn đến 3 chữ số thập phân

    return gps_seconds



def cal2gpstime_v211(date_str: str):
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
    dt = datetime.strptime(date_str, "%y %m %d %H %M %S.%f")

    # Số giây từ epoch đến thời điểm hiện tại
    delta_seconds = (dt - gps_epoch).total_seconds()

    # Tính tuần GPS và số giây trong tuần
    gps_week = int(delta_seconds // 604800)  # Một tuần có 604800 giây
    gps_seconds = round(delta_seconds % 604800, 6) + 18  # Làm tròn đến 6 chữ số thập phân

    return gps_week, gps_seconds
GPS_UTC_LEAP_SECONDS = 18 
def  compute_dt_seconds(t_rx_utc_str):
    # Chuyển string sang datetime UTC
    t_rx_utc = datetime.strptime(t_rx_utc_str, "%Y %m %d %H %M %S.%f")

    # Chuyển t_rx từ UTC sang GPS Time
    t_rx_gps = t_rx_utc + timedelta(seconds=GPS_UTC_LEAP_SECONDS)
    # t_rx_gps = t_rx_utc 


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


def read_rinex_observation_Doppler(file_path, isClock):
    """
    Đọc file RINEX và trích xuất dữ liệu cần thiết.
    Trả về DataFrame chứa: thời gian (epoch), số hiệu vệ tinh, và pseudorange.
    """
    data = []
    reading_data = False
    epoch_time = ''
    clock_bias = 0



    with open(file_path, 'r') as file:
        lines = file.readlines()[20:LINE_COUNT]

        for line in lines:

            if line.startswith('>'):  # Epoch bắt đầu bằng ký tự '>'
                epoch_time = line[1:25].strip()  # Lấy thời gian epoch

                new_epoch = calc_gps_seconds(epoch_time)

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
                    
                    doppler_str = line[133:146]
                    if doppler_str.isspace():
                        doppler = 0
                    else:
                        doppler = float(doppler_str)
                    data.append(( new_epoch, epoch_time, satellite_id, doppler))
    return pd.DataFrame(data, columns=['epoch','epoch_time','PRN','doppler'])


def read_rinex_observation_NguyenDS2(file_path,observation_file_doppler, isClock):
    """
    Đọc file RINEX và trích xuất dữ liệu cần thiết.
    Trả về DataFrame chứa: thời gian (epoch), số hiệu vệ tinh, và pseudorange.
    """
    obs_data_doppler = read_rinex_observation_Doppler(observation_file_doppler, IS_CLOCK2)
    data = []
    reading_data = False
    epoch_time = ''
    clock_bias = 0
    count_line = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()[20:LINE_COUNT]

        for line in lines:

            if line.startswith('>'):  # Epoch bắt đầu bằng ký tự '>'
                epoch_time = line[1:24].strip()  # Lấy thời gian epoch
                _, gps_seconds = cal2gpstime(epoch_time) 

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
                    carrier_phase_str = parts[3] # lay carrier phase
                    
                    if carrier_phase_str.isspace():
                        carrier_phase = 0
                    else:
                        carrier_phase = float(carrier_phase_str)

                    if pseudorange and carrier_phase != 0:
                        newPseudorange = float(pseudorange) - (-c*float(clock_bias))
                        # if isClock == 1:
                        #     newPseudorange = float(pseudorange)
                        # else:
                        #     newPseudorange = float(pseudorange) - (c*float(clock_bias))
                        new_epoch = round( gps_seconds - (-float(clock_bias)))

                        # Lọc Doppler từ obs_data_doppler dựa trên epoch_time và PRN

                        # doppler_value = obs_data_doppler.loc[(calc_gps_seconds(obs_data_doppler['epoch_time']) == calc_gps_seconds(epoch_time)) & 
                        #              (obs_data_doppler['PRN'] == satellite_id), 'doppler'].values
                        # Lọc giá trị Doppler từ obs_data_doppler theo epoch và PRN
                        gps_seconds
                        doppler_value = obs_data_doppler.loc[
                            (obs_data_doppler['epoch'] == gps_seconds) & 
                            (obs_data_doppler['PRN'] == satellite_id), 'doppler'
                        ].values



                        if len(doppler_value) > 0:
                            doppler = doppler_value[0]  # Lấy giá trị Doppler từ dữ liệu đã đọc
                        else:
                            doppler = 0  # Nếu không có Doppler, gán giá trị mặc định

                        

                        # data.append((epoch_time, satellite_id, newPseudorange, carrier_phase))
                        new_carrier_phase = carrier_phase - (-float(clock_bias))*doppler



                        data.append((new_epoch, epoch_time, satellite_id, newPseudorange, new_carrier_phase, doppler))
    return pd.DataFrame(data, columns=['epoch', 'epoch_time', 'PRN', 'pseudorange', 'carrier_phase','doppler'])


# doc file rinex 3.04
def read_rinex_observation_haidv(file_path):
    """
    Đọc file RINEX và trích xuất dữ liệu cần thiết.
    Trả về DataFrame chứa: thời gian (epoch), số hiệu vệ tinh, và carrier phase.
    """
    data = []
    reading_data = False
    epoch_time = ''
    clock_bias = 0.0
    
    with open(file_path, 'r') as file:
        # Đọc toàn bộ file
        lines = file.readlines()
        
        # Tìm vị trí dòng kết thúc header
        header_end = -1
        for i, line in enumerate(lines):
            if "END OF HEADER" in line:
                header_end = i
                break
        
        # Nếu tìm thấy dòng kết thúc header, bắt đầu xử lý từ dòng tiếp theo
        if header_end >= 0:
            for line in lines[header_end + 1:LINE_COUNT]:
                if line.startswith('>'):  # Epoch bắt đầu bằng ký tự '>'
                    epoch_time = line[1:24].strip()  # Lấy thời gian epoch
                    _, gps_seconds = cal2gpstime(epoch_time) 
                    reading_data = True
                elif reading_data:
                    if line.strip() == '':
                        continue
                    
                    # Split dong bang khoang trang hoac tab
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    
                    satellite_id = parts[0]  # Lấy số hiệu vệ tinh
                    if satellite_id in list_satellite:
                        pseudorange = parts[1]  # Lấy pseudorange
                        carrier_phase_str = parts[2] # lay carrier phase
                        doppler_str = parts[3] # lay doppler
                        
                        if carrier_phase_str.isspace():
                            carrier_phase = 0
                        else:
                            carrier_phase = float(carrier_phase_str)
                            
                        if doppler_str.isspace():
                            doppler = 0.0  # Gán là float nếu trống
                        else:
                            doppler = float(doppler_str)
                            
                        if pseudorange and carrier_phase != 0:
                            newPseudorange = float(pseudorange) - (-c*float(clock_bias))
                            new_epoch = round(gps_seconds - (-float(clock_bias)))
                            new_carrier_phase = carrier_phase - (-float(clock_bias))*doppler
                            data.append((new_epoch, epoch_time, satellite_id, newPseudorange, new_carrier_phase, doppler))
    
    return pd.DataFrame(data, columns=['epoch', 'epoch_time', 'PRN', 'pseudorange', 'carrier_phase', 'doppler'])


obs_data1 = read_rinex_observation_haidv(observation_file1)
obs_data2 = read_rinex_observation_haidv(observation_file2)


#TÍNH GÓC

# Constants
mu = 3.986005e14  # Earth's gravitational constant (m^3/s^2)
omega_e = 7.2921151467e-5  # Earth's rotation rate (rad/s)

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
                    # Chỉ xử lý các dòng có khả năng chứa tham số (thường bắt đầu bằng dấu cách)
                    # và bỏ qua các dòng bắt đầu của vệ tinh khác (E, R, ...)
                    if line.startswith(' '): 
                        values = line.split()
                        if values:
                            try:
                                current_satellite["parameters"].extend(
                                    [float(v.replace('D', 'E')) for v in values]
                                )
                            except ValueError:
                                # Tùy chọn: Xử lý các dòng có dữ liệu không mong muốn trong khối GPS
                                print(f"Cảnh báo: Bỏ qua dòng có dữ liệu không mong muốn trong khối GPS: {line.strip()}")
                                pass
                    else: 
                        # Nếu dòng không bắt đầu bằng dấu cách (ví dụ: bắt đầu bằng 'R', 'E')
                        # tức là khối dữ liệu của vệ tinh GPS hiện tại đã kết thúc.
                        # Lưu dữ liệu GPS đã thu thập và đặt lại.
                        if current_satellite: # Chỉ lưu nếu có dữ liệu GPS
                             satellite_data.append(current_satellite)
                        current_satellite = None # Đặt lại, bỏ qua khối dữ liệu của vệ tinh không phải GPS
        if current_satellite: # Đảm bảo vệ tinh GPS cuối cùng cũng được lưu
            satellite_data.append(current_satellite)

    return satellite_data

from datetime import datetime, timedelta


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

# Leap seconds giữa GPS Time và UTC (18 giây tính đến năm 2024)


def solve_kepler(M, e, tolerance=1e-10):
    """ Solve Kepler's Equation for E (Eccentric Anomaly). """
    E = M  # Initial guess
    while True:
        E_new = E + (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        if abs(E_new - E) < tolerance:
            break
        E = E_new
    return E
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
    """
    Ví dụ: 
    G20 2025 04 04 10 00 00  .365244224668D-03 -.227373675443D-12  .000000000000D+00
      .180000000000D+02 -.795625000000D+02  .444661379074D-08 -.286673919817D+01
     -.403635203838D-05  .356791226659D-02  .389292836189D-05  .515374469757D+04
      .468000000000D+06  .931322574615D-08 -.262723586491D+01 -.242143869400D-07
      .958091353379D+00  .302000000000D+03 -.226999424114D+01 -.817212611614D-08
      .685742849656D-10  .100000000000D+01  .236000000000D+04  .000000000000D+00
      .200000000000D+01  .000000000000D+00 -.838190317154D-08  .180000000000D+02
      .464976000000D+06  .400000000000D+01
    
    Sai số đồng hồ vệ tinh:
        a_f0 = 0.365244224668×10^-3 s (clock bias).

        a_f1 = -0.227373675443×10^-12 s/s (clock drift).

        a_f2 = 0.000000000000×10^0 s/s^2 (clock drift rate).

        Tham số quỹ đạo:
        IODE = 18 (Issue of Data Ephemeris).

        Crs = -79.5625 m (hiệu chỉnh bán kính quỹ đạo, hướng radial).

        Δn = 0.444661379074×10^-8 rad/s (hiệu chỉnh chuyển động trung bình).

        M0 = -2.86673919817 rad (góc trung bình tại Toe).

        Cuc = -0.403635203838×10^-5 rad (hiệu chỉnh góc, cos).

        e = 0.0356791226659 (độ lệch tâm).

        Cus = 0.389292836189×10^-5 rad (hiệu chỉnh góc, sin).

        √a = 51537.4469757 m^1/2 (căn bậc hai trục bán chính).

        Toe = 468,000 s (thời điểm tham chiếu ephemeris).

        Cic = 0.931322574615×10^-8 rad (hiệu chỉnh góc nghiêng, cos).

        Ω0 = -2.62723586491 rad (kinh độ điểm đầu quỹ đạo tại thời điểm tuần).

        Cis = -0.242143869400×10^-7 rad (hiệu chỉnh góc nghiêng, sin).

        i0 = 0.958091353379 rad (độ nghiêng quỹ đạo tại Toe).

        Crc = 302 m (hiệu chỉnh bán kính quỹ đạo, hướng cross-track).

        ω = -2.26999424114 rad (góc cận điểm).

        Ω_dot = -0.817212611614×10^-8 rad/s (tốc độ quay kinh độ điểm đầu).

        i_dot = 0.685742849656×10^-10 rad/s (tốc độ thay đổi độ nghiêng).
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
    # chuyển động trung bình góc
    n0 = np.sqrt(mu / A**3)
    # chuyển động trung bình góc hiệu chỉnh
    n = n0 + Delta_n
    # thời gian hiệu chỉnh
    t_corr = t - Toe
    # góc trung bình
    M = M0 + n * t_corr

    E = solve_kepler(M % (2 * np.pi), e)
    # góc thực
    v = np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)

    # góc thực hiệu chỉnh
    phi = v + omega
    # bán kính
    r = A * (1 - e * np.cos(E))
    # tọa độ x
    x_orb = r * np.cos(phi)
    # tọa độ y
    y_orb = r * np.sin(phi)
    # góc kinh độ điểm đầu quỹ đạo tại thời điểm tuần
    OMEGA_corrected = OMEGA0 + (OMEGA_DOT - omega_e) * t_corr - omega_e * t
    # tọa độ 
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
    #angleAB = 58 độ khi đo ở lab tầng 11
    cos_alpha = -(math.cos(math.radians(azimuth - angleAB)) * math.cos(math.radians(elevation)))
    alpha_angle = math.degrees(math.acos(cos_alpha))
    
    return azimuth, elevation, alpha_angle, cos_alpha


nav_data_01 = parse_nav_rinex(nav_file1)
# Duyệt qua từng epoch trong obs_data
azimuths_01 = []
elevations_01 = []
alpha_angles_01 = []
cos_alpha_01 = []

for index, row in obs_data1.iterrows():
    epoch_time = row['epoch_time']
    prn = row['PRN']
    
    temp_time_input = compute_dt_seconds(epoch_time)
    t = temp_time_input["t_rx_toe"]

    # Tìm vệ tinh tương ứng trong nav_data
    sat_info = next((item for item in nav_data_01 if item['PRN'] == prn), None)
    if not sat_info:
        azimuths_01.append(np.nan)
        elevations_01.append(np.nan)
        alpha_angles_01.append(np.nan)
        cos_alpha_01.append(np.nan)
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

obs_data1['cosAlpha'] = cos_alpha_01

nav_data_02 = parse_nav_rinex(nav_file2)
azimuths_02 = []
elevations_02 = []
alpha_angles_02 = []
cos_alpha_02 = []
for index, row in obs_data2.iterrows():
    epoch_time = row['epoch_time']
    prn = row['PRN']
    
    temp_time_input = compute_dt_seconds(epoch_time)
    t = temp_time_input["t_rx_toe"]

    # Tìm vệ tinh tương ứng trong nav_data
    sat_info = next((item for item in nav_data_02 if item['PRN'] == prn), None)
    if not sat_info:
        azimuths_02.append(np.nan)
        elevations_02.append(np.nan)
        alpha_angles_02.append(np.nan)
        cos_alpha_02.append(np.nan)
        continue

    # Lấy tọa độ ECEF của vệ tinh

    sat_ecef = calculate_satellite_ecef(sat_info["parameters"], t)
    
    az, el, alpha, cosAlpha = calculate_azimuth_elevation(sat_ecef, receiver_ecef_02)
 
    # Lưu kết quả
    azimuths_02.append(az)
    elevations_02.append(el)
    alpha_angles_02.append(alpha)
    cos_alpha_02.append(cosAlpha)

obs_data2['cosAlpha'] = cos_alpha_02

print("obs_data1")
print(obs_data1)
print("obs_data2")
print(obs_data2)

merged_data = pd.merge(obs_data1, obs_data2, on=['epoch', 'PRN'], suffixes=('_file1', '_file2'))
merged_data['difference_pseudorange'] = merged_data['pseudorange_file1'] - merged_data['pseudorange_file2'] - NORMALIZE_VALUE
merged_data['difference_carrier_phase'] = merged_data['carrier_phase_file1'] - merged_data['carrier_phase_file2']
merged_data['cosAlpha_avg'] =  ( merged_data['cosAlpha_file1'] + merged_data['cosAlpha_file2'] ) /2

# Lọc 200 epoch đầu tiên
unique_epochs = merged_data['epoch'].unique()
if len(unique_epochs) > 200:
    epochs_to_use = sorted(unique_epochs)[:200]  # Lấy 200 epoch đầu tiên (đã sắp xếp)
    merged_data = merged_data[merged_data['epoch'].isin(epochs_to_use)]
    print(f"Chỉ sử dụng 200 epoch đầu tiên từ {epochs_to_use[0]} đến {epochs_to_use[-1]}")
else:
    print(f"Sử dụng tất cả {len(unique_epochs)} epoch có sẵn")

def calculate_double_difference(merged_smoothed_data):
    
    double_diff_data = []
    # Duyệt qua từng epoch
    for epoch in merged_smoothed_data['epoch'].unique():
        epoch_data = merged_smoothed_data[merged_smoothed_data['epoch'] == epoch]
        
        # Số lượng vệ tinh (I) trong epoch hiện tại
        I = len(epoch_data['PRN'].unique())
        # Trọng số ω_i = 1/I
        omega_i = 1.0 / I
        
        # Duyệt qua từng cặp vệ tinh
        for sat1 in epoch_data['PRN'].unique():
            sos = 0.0  # Sum of squares cho vệ tinh tham chiếu sat1
            
            for sat2 in epoch_data['PRN'].unique():
                if sat1 != sat2:  # Không tính chênh lệch cho chính vệ tinh

                    # Lấy giá trị difference_carrier_phase của từng vệ tinh tại epoch này
                    diff_carrier_phase_sat1 = epoch_data[epoch_data['PRN'] == sat1]['difference_carrier_phase'].values[0]
                    diff_carrier_phase_sat2 = epoch_data[epoch_data['PRN'] == sat2]['difference_carrier_phase'].values[0]

                    # Tính double_difference
                    double_diff_carrier_phase = diff_carrier_phase_sat1 - diff_carrier_phase_sat2
                    # double_diff = abs(diff_sat1) - abs(diff_sat2)

                    fract_double_diff_carrier_phase = double_diff_carrier_phase - round(double_diff_carrier_phase)
                    
                    fract_double_diff_carrier_phase = fract_double_diff_carrier_phase * Lamda_L1
                    # Tính μ_i^2 và cộng vào tổng (sos)
                    if not pd.isna(fract_double_diff_carrier_phase):
                        sos += omega_i * (fract_double_diff_carrier_phase ** 2)
                      
                    # Lấy giá trị cosAlpha của từng vệ tinh tại epoch này
                    cosAlpha_avg1 = epoch_data[epoch_data['PRN'] == sat1]['cosAlpha_file1'].values[0]
                    cosAlpha_avg2 = epoch_data[epoch_data['PRN'] == sat2]['cosAlpha_file2'].values[0]
                    # Lấy giá trị cosAlpha của từng vệ tinh tại epoch này
                    # cosAlpha_avg1 = epoch_data[epoch_data['PRN'] == sat1]['cosAlpha_avg'].values[0]
                    # cosAlpha_avg2 = epoch_data[epoch_data['PRN'] == sat2]['cosAlpha_avg'].values[0]

                    # Tính delta cosAlpha theo bản tin ephemeris
                    if pd.isna(cosAlpha_avg1) or pd.isna(cosAlpha_avg2):
                        deltaCosAlpha_ephemeris = np.nan
                    else:
                        deltaCosAlpha_ephemeris = cosAlpha_avg1 - cosAlpha_avg2
                    #tính D*deltaCosAlpha_ephemeris
                    if pd.isna(deltaCosAlpha_ephemeris):
                        D_multi_deltaCosAlpha_ephemeris = np.nan
                    else:
                        D_multi_deltaCosAlpha_ephemeris = D_rx1_rx2 * deltaCosAlpha_ephemeris
                    
                    if pd.isna(D_multi_deltaCosAlpha_ephemeris):
                        fract_D_multi_deltaCosAlpha_ephemeris = np.nan
                    else:
                        fract_D_multi_deltaCosAlpha_ephemeris = D_multi_deltaCosAlpha_ephemeris - round (D_multi_deltaCosAlpha_ephemeris)                       

                    # tìm quy luật khác biệt giữa D * delta cosalpha theo ephemeris và double_diff_carrier_phase
                    if pd.isna(fract_D_multi_deltaCosAlpha_ephemeris) or pd.isna(fract_double_diff_carrier_phase):
                        differ_rules = np.nan
                    else:
                        differ_rules = fract_D_multi_deltaCosAlpha_ephemeris - fract_double_diff_carrier_phase
                    
                    #tính fract_deltaCosAlpha_ephemeris
                    if pd.isna(deltaCosAlpha_ephemeris):
                        frac_deltaCosAlpha_ephemeris = np.nan
                    else:
                        frac_deltaCosAlpha_ephemeris = deltaCosAlpha_ephemeris - round(deltaCosAlpha_ephemeris)

                    # Tạo chuỗi satellite_difference
                    satellite_diff = f"{sat1}_{sat2}"
                    
                    double_diff_data.append({
                        'epoch': epoch,
                        'PRN_1': sat1,
                        'PRN_2': sat2,
                        # 'double_difference_pseudorange': double_diff_pseudo,
                        #'double_difference_carrier_phase': double_diff_carrier_phase,
                        #'deltaCosAlpha_pseu' : deltaCosAlpha,
                        'frac_deltaCosAlpha_ephemeris': frac_deltaCosAlpha_ephemeris,
                        'fract_D_multi_deltaCosAlpha_ephemeris' : fract_D_multi_deltaCosAlpha_ephemeris,
                        'deltaCosAlpha_ephemeris' : deltaCosAlpha_ephemeris,
                        'Lambda_SoS' : sos,
                        'differ_rules' : differ_rules,
                        'fract_DD_CP': fract_double_diff_carrier_phase,                        
                        'PRN_difference': satellite_diff
                    })

    # Chuyển danh sách thành DataFrame
    return pd.DataFrame(double_diff_data)



merged_smoothed_data_double_3x = calculate_double_difference(merged_data)

print(merged_smoothed_data_double_3x)

def plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double, folder_path, label1, label2, key):
    """
    Vẽ biểu đồ double difference cho tất cả các vệ tinh tham chiếu với tất cả các vệ tinh còn lại trên cùng một biểu đồ.
    Lọc chỉ vẽ các cặp vệ tinh mà bắt đầu từ G01, G02, ..., G32.
    """
    # Lọc dữ liệu trong phạm vi epoch
    plot_data = merged_smoothed_data_double
    


    for reference_satellite in list_satellite:  # Duyệt qua tất cả các vệ tinh trong danh sách
        # Lọc các dữ liệu cho vệ tinh tham chiếu
        reference_data = plot_data[plot_data['PRN_1'] == reference_satellite]

        # Nếu không có dữ liệu cho vệ tinh tham chiếu này, bỏ qua
        if len(reference_data) == 0:
            continue

        # Khởi tạo một biểu đồ
        plt.figure(figsize=(10, 6))
        
        # Vẽ biểu đồ cho các vệ tinh khác với vệ tinh tham chiếu, chỉ vẽ cặp vệ tinh bắt đầu từ vệ tinh tham chiếu
        for satellite_diff in plot_data['PRN_difference'].unique():
            if satellite_diff.startswith(reference_satellite):  # Lọc chỉ các vệ tinh bắt đầu từ reference_satellite
                # Lọc dữ liệu cho cặp vệ tinh
                satellite_diff_data = plot_data[plot_data['PRN_difference'] == satellite_diff]
                if len(satellite_diff_data) > 0:
                    plt.plot(satellite_diff_data['epoch'], satellite_diff_data[f'{key}'], label=satellite_diff, linewidth=3)
        
        # Thêm các yếu tố cho biểu đồ
        plt.xlabel('Epoch')
        plt.xticks(rotation=45)
        plt.ylabel(f'{label1}')
        plt.title(f'{reference_satellite} {label2} Comparison over Epochs')
        plt.legend()
        plt.grid()

        # Lưu biểu đồ vào thư mục
        plt.savefig(os.path.join(folder_path, f'{reference_satellite}_{label2}Comparison.png'))
        plt.close()

def plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double, folder_path, label1, label2, key, add, multi):
    """
    Vẽ biểu đồ double difference cho tất cả các vệ tinh tham chiếu với tất cả các vệ tinh còn lại trên cùng một biểu đồ.
    Lọc chỉ vẽ các cặp vệ tinh mà bắt đầu từ G01, G02, ..., G32.
    """
    plot_data = merged_smoothed_data_double
    
  

    for reference_satellite in list_satellite:  # Duyệt qua tất cả các vệ tinh trong danh sách
        # Lọc các dữ liệu cho vệ tinh tham chiếu
        reference_data = plot_data[plot_data['PRN_1'] == reference_satellite]

        # Nếu không có dữ liệu cho vệ tinh tham chiếu này, bỏ qua
        if len(reference_data) == 0:
            continue

        # Khởi tạo một biểu đồ
        plt.figure(figsize=(10, 6))
        
        # Vẽ biểu đồ cho các vệ tinh khác với vệ tinh tham chiếu, chỉ vẽ cặp vệ tinh bắt đầu từ vệ tinh tham chiếu
        for satellite_diff in plot_data['PRN_difference'].unique():
            if satellite_diff.startswith(reference_satellite):  # Lọc chỉ các vệ tinh bắt đầu từ reference_satellite
                # Lọc dữ liệu cho cặp vệ tinh
                satellite_diff_data = plot_data[plot_data['PRN_difference'] == satellite_diff]
                if len(satellite_diff_data) > 0:
                    plt.plot(satellite_diff_data['epoch'], (satellite_diff_data[f'{key}'] + add ) * multi , label=satellite_diff, linewidth=3)
        
        # Thêm các yếu tố cho biểu đồ
        plt.xlabel('Epoch')
        plt.xticks(rotation=45)
        plt.ylabel(f'{label1}')
        plt.title(f'{reference_satellite} {label2} Comparison over Epochs')
        plt.legend()
        plt.grid()

        # Lưu biểu đồ vào thư mục
        plt.savefig(os.path.join(folder_path, f'{reference_satellite}_{label2}Comparison.png'))
        plt.close()

# Fract_carrier_phase
plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x,r'PLOT\external\D2025_04_21_doplerConvert\carrier_phase_fract','fract_DD_CP','fract_DD_CP','fract_DD_CP')
# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x,'PLOT\external\D2025_04_21_doplerConvert\carrier_phase_fract_add1','fract_DD_CP','fract_DD_CP_add1','fract_DD_CP', 1, 1) 
# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x,'PLOT\external\D2025_04_21_doplerConvert\carrier_phase_fract_add2','fract_DD_CP','fract_DD_CP_add2','fract_DD_CP',2, 1 )
# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x,'PLOT\external\D2025_04_21_doplerConvert\carrier_phase_fract_sub1','fract_DD_CP','fract_DD_CP_sub1','fract_DD_CP',-1, 1 )
# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x,'PLOT\external\D2025_04_21_doplerConvert\carrier_phase_fract_sub2','fract_DD_CP','fract_DD_CP_sub2','fract_DD_CP',-2, 1 )



#plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x,r'PLOT\external\D2025_04_21_doplerConvert\deltaCosAlpha_ephemeris','deltaCosAlpha_ephemeris','deltaCosAlpha_ephemeris','deltaCosAlpha_ephemeris')


# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x,'PLOT\external\D2025_04_21_doplerConvert\D_multi_deltaCosAlpha_ephemeris','D_multi_deltaCosAlpha_ephemeris','D_multi_deltaCosAlpha_ephemeris','deltaCosAlpha_ephemeris',0 , D_rx1_rx2)
# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x,'PLOT\external\D2025_04_21_doplerConvert\D_multi_frac_deltaCosAlpha_ephemeris','D_multi_frac_deltaCosAlpha_ephemeris','D_multi_frac_deltaCosAlpha_ephemeris','frac_deltaCosAlpha_ephemeris',0 , D_rx1_rx2)

# plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x,'PLOT\external\D2025_04_21_doplerConvert\deltaCosAlpha_ephemeris_fract','frac_deltaCosAlpha_ephemeris','frac_deltaCosAlpha_ephemeris','frac_deltaCosAlpha_ephemeris')

plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x,r'PLOT\external\D2025_04_21_doplerConvert\dfract_D_multi_deltaCosAlpha_ephemeris','fract_D_multi_deltaCosAlpha_ephemeris','fract_D_multi_deltaCosAlpha_ephemeris','fract_D_multi_deltaCosAlpha_ephemeris')
# plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x,r'PLOT\external\D2025_04_21_doplerConvert\double_difference_carrier_phase','double_difference_carrier_phase','double_difference_carrier_phase','double_difference_carrier_phase')

def plot_fract_DD_CP_linear_regression(merged_smoothed_data_double, folder_path):
    """
    Vẽ biểu đồ fract_DD_CP theo epoch sử dụng hồi quy tuyến tính (Linear Regression).
    """
    plot_data = merged_smoothed_data_double
    
    for reference_satellite in list_satellite:  # Duyệt qua tất cả các vệ tinh trong danh sách
        # Lọc các dữ liệu cho vệ tinh tham chiếu
        reference_data = plot_data[plot_data['PRN_1'] == reference_satellite]

        # Nếu không có dữ liệu cho vệ tinh tham chiếu này, bỏ qua
        if len(reference_data) == 0:
            continue

        # Khởi tạo một biểu đồ
        plt.figure(figsize=(12, 8))
        
        # Vẽ biểu đồ cho các vệ tinh khác với vệ tinh tham chiếu, chỉ vẽ cặp vệ tinh bắt đầu từ vệ tinh tham chiếu
        for satellite_diff in plot_data['PRN_difference'].unique():
            if satellite_diff.startswith(reference_satellite):  # Lọc chỉ các vệ tinh bắt đầu từ reference_satellite
                # Lọc dữ liệu cho cặp vệ tinh
                satellite_diff_data = plot_data[plot_data['PRN_difference'] == satellite_diff]
                
                if len(satellite_diff_data) > 1:  # Cần ít nhất 2 điểm để thực hiện hồi quy
                    # Chuẩn bị dữ liệu cho hồi quy tuyến tính
                    X = satellite_diff_data['epoch'].values.reshape(-1, 1)
                    y = satellite_diff_data['fract_DD_CP'].values
                    
                    # Kiểm tra giá trị NaN
                    mask = ~np.isnan(y)
                    if np.sum(mask) > 1:  # Vẫn cần ít nhất 2 điểm không phải NaN
                        X_filtered = X[mask]
                        y_filtered = y[mask]
                        
                        # Thực hiện hồi quy tuyến tính
                        model = LinearRegression()
                        model.fit(X_filtered, y_filtered)
                        
                        # Dự đoán giá trị trên toàn bộ dải epoch
                        y_pred = model.predict(X)
                        
                        # Vẽ đường hồi quy tuyến tính
                        plt.plot(X, y_pred, label=f"{satellite_diff} (a={model.coef_[0]:.6f}, b={model.intercept_:.6f})", linewidth=2)
                        
                        # Thêm các điểm dữ liệu gốc
                        plt.scatter(X, y, alpha=0.3, s=10)
        
        # Thêm các yếu tố cho biểu đồ
        plt.xlabel('Epoch')
        plt.ylabel('fract_DD_CP')
        plt.title(f'{reference_satellite} Linear Regression of fract_DD_CP')
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Điều chỉnh layout để tránh cắt xén
        plt.tight_layout()

        # Lưu biểu đồ vào thư mục
        plt.savefig(os.path.join(folder_path, f'{reference_satellite}_fract_DD_CP_LinearRegression.png'))
        plt.close()

# Thêm lệnh gọi hàm mới
plot_fract_DD_CP_linear_regression(merged_smoothed_data_double_3x, r'PLOT\external\D2025_04_21_doplerConvert\fract_DD_CP_LinearRegression')

