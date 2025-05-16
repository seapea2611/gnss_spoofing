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

from datetime import datetime, timedelta

list_satellite = ['G01', 'G02', 'G03', 'G04','G05', 'G06', 'G07', 'G08', 'G09', 'G10', 
                  'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'G19', 'G20', 
                  'G21', 'G22', 'G23', 'G24', 'G25', 'G26', 'G27', 'G28', 'G29', 'G30', 'G31', 'G32']

# Tọa độ bộ thu trong hệ ECEF (Đã biết)
receiver_ecef = np.array([-1626584.7059, 5730519.4577, 2271864.3917])

LINE_COUNT = 20000
c = 299792458
L1 = 1575.42e6
Lamda_L1 = c/L1
D_rx1_rx2 = 3.4 # mét


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
    FOLDER_PATH = './Pseudorange_smooth_external_clock/'  # Đường dẫn tới file 2
    file1 = 'RX\RX1\RinexObservables_20180216_082105_Rx1_125dBm.rin'  # Đường dẫn tới file 1
    file2 = 'RX\RX2\RinexObservables_20180216_082103_Rx2_125dBm.rin'  # Đường dẫn tới file 2
    rx_name_01 = 'rx_01'
    # nav_file_01 = "F:\\PhDStudy\\Data\\Pseudorange_smooth_external_clock\\96813562\\96813562.24N"
    # observation_file_01 = "F:\\PhDStudy\\Data\\Pseudorange_smooth_external_clock\\96813562\\96813562.24O"


    # nav_file_01 = "Data_Clock_Smooth\Pseudorange_smooth_external_clock\96813562\96813562.24N"
    # observation_file_01 = "Data_Clock_Smooth\Pseudorange_smooth_external_clock\96813562\96813562.24O"
    nav_file_01 = "DATA\external\D2025_03_18\96810771\96810771.25O"
   
    
    # observation_file_01 = '89680592\IGS000USA_R_20250590327_01D_01S_MO.rnx'  # Đường dẫn tới file 1
    
    
    # Tọa độ bộ thu trong hệ ECEF (Đã biết)
    receiver_ecef_01 = np.array([-1626343.5660, 5730606.2916, 2271881.4271])
    
    rx_name_02 = 'rx_02'
    # nav_file_02 = "F:\\PhDStudy\\Data\\Pseudorange_smooth_external_clock\\89683562\\IGS000USA_R_20243560202_01D_01S_GN.rnx"
    # observation_file_02 = "F:\\PhDStudy\\Data\\Pseudorange_smooth_external_clock\\89683562\\IGS000USA_R_20243560202_01D_01S_MO.rnx"
    # nav_file_02 = "Data_Clock_Smooth\Pseudorange_smooth_external_clock\89683562\IGS000USA_R_20243560202_01D_01S_GN.rnx"
    # observation_file_02 = "Data_Clock_Smooth\Pseudorange_smooth_external_clock\89683562\IGS000USA_R_20243560202_01D_01S_MO.rnx"

    nav_file_02 = "DATA\external\D2025_03_13\89680723\IGS000USA_R_20250720850_01D_01S_GN.rnx"
    observation_file_02 = "DATA\Smooth_Clock_Anten\96810421\96810421.25O"
    observation_file_02_doppler = "DATA\external\D2025_03_13\89680723\convert_doppler\D20250720850_09M_01S_MO.rnx"
    # observation_file_02 = '96810592\96810592.25O'  # Đường dẫn tới file 2
    
    
    # Tọa độ bộ thu trong hệ ECEF (Đã biết)
    receiver_ecef_02 = np.array([-1626584.7059, 5730519.4577, 2271864.3917])
    
    
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
    """
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

def read_rinex_observation_v211(file_path, isClock):
    
    """
    Đọc file RINEX và trích xuất dữ liệu cần thiết.
    Trả về DataFrame chứa: thời gian (epoch), số hiệu vệ tinh, và pseudorange.
    """
    doppler = 0


    data = []
    reading_data = False
    epoch_time = ''
    clock_bias = 0
    count_line = 0
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i_tmp = 0
        for line in lines:            
            if len(line) > 79 and line[1:3].isdigit() and all(part.isdigit() for part in line[71:80].split()):
                i_tmp =0
                epoch_time = line[1:25].strip()  # Lấy thời gian epoch (thời gian quan sát)
                _, gps_seconds = cal2gpstime_v211(epoch_time) 
                parts = line.split()
                clock_bias = parts[8] 
                satellite_ids_str = parts[7]
                
                    

            if not reading_data:
                if len(line) > 79 and line[1:3].isdigit() and all(part.isdigit() for part in line[71:80].split()):
                    reading_data = True  # Bắt đầu đọc dữ liệu sau khi kiểm tra
                    continue
                    
            else:
                if len(line) > 79 and line[1:3].isdigit() and all(part.isdigit() for part in line[71:80].split()):
                    continue
                if line.strip() == '':
                    continue  # Bỏ qua các dòng trống

                # Split dòng bằng khoảng trắng hoặc tab
                parts = line.split()
               
                if len(parts) < 2:
                    continue  # Bỏ qua các dòng không đủ dữ liệu
                
                
                # Lấy số hiệu vệ tinh từ chuỗi các vệ tinh trong cột 6
                satellite_ids = [satellite_ids_str[i+1:i+4] for i in range(0, len(satellite_ids_str), 3)]  # Tách chuỗi thành các vệ tinh



                if satellite_ids[i_tmp] in list_satellite:
                    satellite_id = satellite_ids[i_tmp]
                    pseudorange = parts[1]  # Lấy pseudorange
                    carrier_phase = float(parts[0])
                    doppler = float(parts[2] )
                    

                        

                    # Điều chỉnh pseudorange nếu là clock bias
                    if pseudorange and carrier_phase != 0:
                        if isClock == 1:
                            newPseudorange = float(pseudorange)
                        else:
                            newPseudorange = float(pseudorange) - (c * float(-float(clock_bias)))

                        new_epoch = round(gps_seconds - (-float(clock_bias)))
                      

                        # data.append((epoch_time, satellite_id, newPseudorange, carrier_phase))
                        new_carrier_phase = carrier_phase - (-float(clock_bias))*doppler
                        data.append((new_epoch, epoch_time , satellite_id, newPseudorange, new_carrier_phase, doppler))
            if(i_tmp==7) : 
                i_tmp=0
            else:
                i_tmp =i_tmp+1

                        
    return pd.DataFrame(data, columns=['epoch' ,'epoch_time', 'PRN', 'pseudorange', 'carrier_phase','doppler'])
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

def read_rinex_observation_NGUYENDS(file_path, isClock):
    """
    Đọc file RINEX và trích xuất dữ liệu cần thiết.
    Trả về DataFrame chứa: thời gian (epoch), số hiệu vệ tinh, và pseudorange.
    """
    data = []
    reading_data = False
    epoch_time = ''
    clock_bias = 0
    count_line = 0
    prev_carrier_phase = {}  # Lưu giá trị carrier_phase của mỗi vệ tinh tại epoch trước
    prev_gps_seconds = {}  # Lưu giá trị gps_seconds của mỗi vệ tinh tại epoch trước

    with open(file_path, 'r') as file:
        lines = file.readlines()[20:LINE_COUNT]
        for line in lines:

            if line.startswith('>'):  # Epoch bắt đầu bằng ký tự '>'
                epoch_time = line[1:25].strip()  # Lấy thời gian epoch
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
                    #carrier_phase_str = parts[3] # lay carrier phase
                    carrier_phase_str = line[20:34]
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

                        # Tính Doppler: (carrier_phase hiện tại - carrier_phase trước đó) / (thời gian hiện tại - thời gian trước đó)
                        if satellite_id in prev_carrier_phase:
                            delta_phase = carrier_phase - prev_carrier_phase[satellite_id]
                            delta_time = gps_seconds - prev_gps_seconds[satellite_id]
                            if delta_time != 0:
                                doppler = delta_phase / delta_time  # Tính Doppler
                            else:
                                doppler = 0  # Nếu thời gian giữa hai epoch là 0 thì Doppler = 0
                        else:
                            doppler = 0  # Nếu không có dữ liệu trước đó, Doppler = 0

                        # Cập nhật giá trị carrier_phase và gps_seconds cho lần đo hiện tại
                        prev_carrier_phase[satellite_id] = carrier_phase
                        prev_gps_seconds[satellite_id] = gps_seconds
                        

                        # data.append((epoch_time, satellite_id, newPseudorange, carrier_phase))
                        new_carrier_phase = carrier_phase - (-float(clock_bias))*doppler



                        data.append((new_epoch, epoch_time, satellite_id, newPseudorange, new_carrier_phase, doppler))
    return pd.DataFrame(data, columns=['epoch', 'epoch_time', 'PRN', 'pseudorange', 'carrier_phase','doppler'])


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
                    #carrier_phase_str = parts[3] # lay carrier phase
                    carrier_phase_str = line[20:34]
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




def carrier_smoothing_NDS(df, wavelength=0.19, N=300):
    """
    Tinh toan carrier-smoothed pseudorang theo cong thuc https://gssc.esa.int/navipedia/index.php/Carrier-smoothing_of_code_pseudoranges
    wavelength: Do dai song (met) cua tin hieu GNSS; 0.19 cho tin hieu L1 GPS
    N: so epoch lam muot (alpha = 1/N)
    """

    alpha = 1/N
    smoothed_data = []
    last_smoothed = {} # luu trong thai smoothed pseudorange cua tung ve tinh
    last_carrier_phase = {} # Luu gia tri carrier phase tai epoch truoc

    for index, row in df.iterrows():
        
        satellite = row['PRN']
        pseudorange = row['pseudorange']
        
        carrier_phase = row['carrier_phase']
        new_carrier_phase = carrier_phase * Lamda_L1
        if satellite in last_smoothed and satellite in last_carrier_phase:
            if index <= N:
                alpha = 1/index
            else:
                alpha = 1/N
            # Tinh toan gia tri smoothed pseudorange
            delta_phase = new_carrier_phase - last_carrier_phase[satellite]
            smoothed = alpha * pseudorange + (1 - alpha) * (last_smoothed[satellite] + delta_phase)
        else:
            smoothed = pseudorange # Gia tri khoi tao neu khong co gia tri truoc

        # Cap nhat gia tri cho ve tinh hien tai
        last_smoothed[satellite] = smoothed
        last_carrier_phase[satellite] = new_carrier_phase

        smoothed_data.append((row['epoch'], satellite, smoothed, carrier_phase))

    return pd.DataFrame(smoothed_data, columns=['epoch', 'PRN', 'pseudorange', 'carrier_phase'])

def carrier_smoothing_NDS2(df, wavelength=0.19, N=300):
    """
    Tinh toan carrier-smoothed pseudorang theo cong thuc https://gssc.esa.int/navipedia/index.php/Carrier-smoothing_of_code_pseudoranges
    wavelength: Do dai song (met) cua tin hieu GNSS; 0.19 cho tin hieu L1 GPS
    N: so epoch lam muot (alpha = 1/N)
    """

    alpha = 1/N
    smoothed_data = []
    last_smoothed = {} # luu trong thai smoothed pseudorange cua tung ve tinh
    last_carrier_phase = {} # Luu gia tri carrier phase tai epoch truoc

    for index, row in df.iterrows():
        
        satellite = row['PRN']
        pseudorange = row['pseudorange']
        cosAlpha = row['cosAlpha']
        
        carrier_phase = row['carrier_phase']
        new_carrier_phase = carrier_phase * Lamda_L1
        if satellite in last_smoothed and satellite in last_carrier_phase:
            if index <= N:
                alpha = 1/index
            else:
                alpha = 1/N
            # Tinh toan gia tri smoothed pseudorange
            delta_phase = new_carrier_phase - last_carrier_phase[satellite]
            smoothed = alpha * pseudorange + (1 - alpha) * (last_smoothed[satellite] + delta_phase)
        else:
            smoothed = pseudorange # Gia tri khoi tao neu khong co gia tri truoc

        # Cap nhat gia tri cho ve tinh hien tai
        last_smoothed[satellite] = smoothed
        last_carrier_phase[satellite] = new_carrier_phase

        smoothed_data.append((row['epoch'], satellite, smoothed, carrier_phase, cosAlpha))

    return pd.DataFrame(smoothed_data, columns=['epoch', 'PRN', 'pseudorange', 'carrier_phase','cosAlpha'])





#Rinex2.11
# print('read_rinex_2.11')
obs_data_rx1 = read_rinex_observation_v211(file1, IS_CLOCK2)
obs_data_rx2 = read_rinex_observation_v211(file2, IS_CLOCK2)
print(obs_data_rx1[0:32])





#TÍNH GÓC

# Constants
mu = 3.986005e14  # Earth's gravitational constant (m^3/s^2)
omega_e = 7.2921151467e-5  # Earth's rotation rate (rad/s)





obs_data_01_smooth = carrier_smoothing_NDS(obs_data_rx1)
obs_data_02_smooth = carrier_smoothing_NDS(obs_data_rx2)


print('read_rinex_2.11_smooth')
print(obs_data_01_smooth[0:32])

# print('read_rinex_3.02_smooth')
# print(obs_data_02_smooth[0:32])


merged_smoothed_data_3x_smooth = pd.merge(obs_data_01_smooth, obs_data_02_smooth, on=['epoch', 'PRN'], suffixes=('_file1', '_file2'))
merged_smoothed_data_3x_smooth['difference_pseudorange'] = merged_smoothed_data_3x_smooth['pseudorange_file1'] - merged_smoothed_data_3x_smooth['pseudorange_file2'] - NORMALIZE_VALUE
merged_smoothed_data_3x_smooth['difference_carrier_phase'] = merged_smoothed_data_3x_smooth['carrier_phase_file1'] - merged_smoothed_data_3x_smooth['carrier_phase_file2'] - NORMALIZE_VALUE



def calculate_double_difference_NDS(merged_smoothed_data):
    
    double_diff_data = []

    # Duyệt qua từng epoch
    for epoch in merged_smoothed_data['epoch'].unique():
        epoch_data = merged_smoothed_data[merged_smoothed_data['epoch'] == epoch]
        
        # Duyệt qua từng cặp vệ tinh
        for sat1 in epoch_data['PRN'].unique():
            for sat2 in epoch_data['PRN'].unique():
                if sat1 != sat2:  # Không tính chênh lệch cho chính vệ tinh
                    # Lấy giá trị difference_pseudorange của từng vệ tinh tại epoch này
                    diff_pseudo_sat1 = epoch_data[epoch_data['PRN'] == sat1]['difference_pseudorange'].values[0]
                    diff_pseudo_sat2 = epoch_data[epoch_data['PRN'] == sat2]['difference_pseudorange'].values[0]

                    # Tính double_difference
                    double_diff_pseudo = diff_pseudo_sat1 - diff_pseudo_sat2
                    # double_diff = abs(diff_sat1) - abs(diff_sat2)



                    # Lấy giá trị difference_carrier_phase_pseudorange của từng vệ tinh tại epoch này
                    diff_carrier_phase_sat1 = epoch_data[epoch_data['PRN'] == sat1]['difference_carrier_phase'].values[0]
                    diff_carrier_phase_sat2 = epoch_data[epoch_data['PRN'] == sat2]['difference_carrier_phase'].values[0]

                    # Tính double_difference
                    double_diff_carrier_phase = diff_carrier_phase_sat2 - diff_carrier_phase_sat1
                    # double_diff = abs(diff_sat1) - abs(diff_sat2)
                    fract_double_diff_carrier_phase = double_diff_carrier_phase- round(double_diff_carrier_phase)


                    deltaCosAlpha = double_diff_pseudo / D_rx1_rx2

                    # Tạo chuỗi satellite_difference
                    satellite_diff = f"{sat1}_{sat2}"

                    # Thêm vào danh sách kết quả
                    double_diff_data.append({
                        'epoch': epoch,
                        'PRN_1': sat1,
                        'PRN_2': sat2,
                        'double_difference_pseudorange': double_diff_pseudo,
                        'double_difference_carrier_phase': double_diff_carrier_phase,
                        'fract_DD_CP': fract_double_diff_carrier_phase,
                        'deltaCosAlpha_pseu' : deltaCosAlpha,
                        'PRN_difference': satellite_diff
                    })

    # Chuyển danh sách thành DataFrame
    return pd.DataFrame(double_diff_data)











merged_smoothed_data_double_3x_smooth = calculate_double_difference_NDS(merged_smoothed_data_3x_smooth)
print(merged_smoothed_data_double_3x_smooth[0:32])

def plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double, folder_path, label1, label2, key):
    """
    Vẽ biểu đồ double difference cho tất cả các vệ tinh tham chiếu với tất cả các vệ tinh còn lại trên cùng một biểu đồ.
    Lọc chỉ vẽ các cặp vệ tinh mà bắt đầu từ G01, G02, ..., G32.
    """
    for reference_satellite in list_satellite:  # Duyệt qua tất cả các vệ tinh trong danh sách
        # Lọc các dữ liệu cho vệ tinh tham chiếu
        reference_data = merged_smoothed_data_double[merged_smoothed_data_double['PRN_1'] == reference_satellite]

        # Khởi tạo một biểu đồ
        plt.figure(figsize=(10, 6))
        
        # Vẽ biểu đồ cho các vệ tinh khác với vệ tinh tham chiếu, chỉ vẽ cặp vệ tinh bắt đầu từ vệ tinh tham chiếu
        for satellite_diff in merged_smoothed_data_double['PRN_difference'].unique():
            if satellite_diff.startswith(reference_satellite):  # Lọc chỉ các vệ tinh bắt đầu từ reference_satellite
                # Lọc dữ liệu cho cặp vệ tinh
                satellite_diff_data = merged_smoothed_data_double[merged_smoothed_data_double['PRN_difference'] == satellite_diff]
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
    for reference_satellite in list_satellite:  # Duyệt qua tất cả các vệ tinh trong danh sách
        # Lọc các dữ liệu cho vệ tinh tham chiếu
        reference_data = merged_smoothed_data_double[merged_smoothed_data_double['PRN_1'] == reference_satellite]

        # Khởi tạo một biểu đồ
        plt.figure(figsize=(10, 6))
        
        # Vẽ biểu đồ cho các vệ tinh khác với vệ tinh tham chiếu, chỉ vẽ cặp vệ tinh bắt đầu từ vệ tinh tham chiếu
        for satellite_diff in merged_smoothed_data_double['PRN_difference'].unique():
            if satellite_diff.startswith(reference_satellite):  # Lọc chỉ các vệ tinh bắt đầu từ reference_satellite
                # Lọc dữ liệu cho cặp vệ tinh
                satellite_diff_data = merged_smoothed_data_double[merged_smoothed_data_double['PRN_difference'] == satellite_diff]
                plt.plot(satellite_diff_data['epoch'], (satellite_diff_data[f'{key}'] + add ) * multi , label=satellite_diff,linewidth=3)
        
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
plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\carrier_phase_fract','fract_DD_CP','fract_DD_CP','fract_DD_CP')
# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\carrier_phase_fract_add1','fract_DD_CP','fract_DD_CP_add1','fract_DD_CP', 1, 1) 
# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\carrier_phase_fract_add2','fract_DD_CP','fract_DD_CP_add2','fract_DD_CP',2, 1)
# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\carrier_phase_fract_sub1','fract_DD_CP','fract_DD_CP_sub1','fract_DD_CP',-1, 1)
# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\carrier_phase_fract_sub2','fract_DD_CP','fract_DD_CP_sub2','fract_DD_CP',-2, 1)


# plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\deltaCosAlpha_pseudo','deltaCosAlpha_pseudo','deltaCosAlpha_pseudorange_smooth','deltaCosAlpha_pseu')


# plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\deltaCosAlpha_ephemeris','deltaCosAlpha_ephemeris','deltaCosAlpha_ephemeris','deltaCosAlpha_ephemeris')


# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\D_multi_deltaCosAlpha_ephemeris','D_multi_deltaCosAlpha_ephemeris','D_multi_deltaCosAlpha_ephemeris','deltaCosAlpha_ephemeris',0 , D_rx1_rx2)
# plot_all_double_difference_comparisons_nds2(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\D_multi_frac_deltaCosAlpha_ephemeris','D_multi_frac_deltaCosAlpha_ephemeris','D_multi_frac_deltaCosAlpha_ephemeris','frac_deltaCosAlpha_ephemeris',0 , D_rx1_rx2)

# plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\deltaCosAlpha_ephemeris_fract','frac_deltaCosAlpha_ephemeris','frac_deltaCosAlpha_ephemeris','frac_deltaCosAlpha_ephemeris')

# plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\dfract_D_multi_deltaCosAlpha_ephemeris','fract_D_multi_deltaCosAlpha_ephemeris','fract_D_multi_deltaCosAlpha_ephemeris','fract_D_multi_deltaCosAlpha_ephemeris')
plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\D2\double_difference_pseudorange2','double_difference_pseudorange','double_difference_pseudorange','double_difference_pseudorange')
plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\D2\double_difference_carrier_phase','double_difference_carrier_phase','double_difference_carrier_phase','double_difference_carrier_phase')
plot_all_double_difference_comparisons_nds1(merged_smoothed_data_double_3x_smooth,'PLOT\external\D2025_03_13_doplerConvert\D2\dfract_DD_CP','fract_DD_CP','fract_DD_CP','fract_DD_CP')
