import requests

# API Key từ N2YO (API Key bạn đã cung cấp)
api_key = '35DB3K-6UYCML-M5NWH3-5FSL'



import numpy as np

# Tọa độ trong hệ ECEF (Đã biết)
receiver_ecef_02 = np.array([ -1626343.5660,  5730606.2916,  2271881.4271 ])

# Các hằng số liên quan đến hệ WGS84
a = 6378137.0  # Bán kính xích đạo (m)
e = 8.1819190842622e-2  # Độ lệch tâm của elip (số không gian)

# Tính kinh độ (longitude)
longitude = np.arctan2(receiver_ecef_02[1], receiver_ecef_02[0])

# Tính vĩ độ (latitude) bằng cách sử dụng lặp
p = np.sqrt(receiver_ecef_02[0]**2 + receiver_ecef_02[1]**2)
theta = np.arctan2(receiver_ecef_02[2] * a, p * (1 - e**2))

# Lặp để cải thiện tính chính xác của vĩ độ
latitude = np.arctan2(receiver_ecef_02[2] + e**2 * a * np.sin(theta)**3,
                      p - e**2 * a * np.cos(theta)**3)

# Lặp cải tiến vĩ độ (iteration)
for i in range(10):  # Lặp để tăng độ chính xác
    sin_lat = np.sin(latitude)
    N = a / np.sqrt(1 - e**2 * sin_lat**2)  # Bán kính địa cầu tại vĩ độ hiện tại
    latitude = np.arctan2(receiver_ecef_02[2] + N * e**2 * sin_lat,
                          p)

# Chuyển đổi từ radian sang độ
latitude = np.degrees(latitude)
longitude = np.degrees(longitude)

print(f'Vĩ độ: {latitude}°')
print(f'Kinh độ: {longitude}°')

# Tọa độ của bộ thu (vị trí bạn đã cung cấp)
lat = latitude  # Vĩ độ (Decimal Degrees)
lon = longitude  # Kinh độ (Decimal Degrees)

# Chọn ID của vệ tinh (NAVSTAR 81)
sat_id_03 = 40294   # ID của vệ tinh NAVSTAR 72 (G03)
sat_id_06 = 39741   # ID của vệ tinh NAVSTAR 70 (G06)
sat_id_10 = 41019   # ID của vệ tinh NAVSTAR 75 (G10)
sat_id_11 = 48859   # ID của vệ tinh NAVSTAR 81 (G11)
sat_id_12 = 29601   # ID của vệ tinh NAVSTAR 59 (G12)
sat_id_14 = 46826   # ID của vệ tinh NAVSTAR 80 (G14)
sat_id_15 = 32260   # ID của vệ tinh NAVSTAR 60 (G15)
sat_id_17 = 28874   # ID của vệ tinh NAVSTAR 57 (G17)
sat_id_18 = 44506   # ID của vệ tinh NAVSTAR 78 (G18)
sat_id_19 = 28190   # ID của vệ tinh NAVSTAR 54 (G19)
sat_id_20 = 26360   # ID của vệ tinh NAVSTAR 47 (G20)
sat_id_22 = 26407   # ID của vệ tinh NAVSTAR 48 (G22)
sat_id_24 = 38833   # ID của vệ tinh NAVSTAR 67 (G24)
sat_id_25 = 36585   # ID của vệ tinh NAVSTAR 65 (G25)
sat_id_23 = 45854   # ID của vệ tinh NAVSTAR 79 (G23)
sat_id_28 = 55268   # ID của vệ tinh NAVSTAR 82 (G28)
sat_id_30 = 39533   # ID của vệ tinh NAVSTAR 69 (G30)
sat_id_32 = 41328   # ID của vệ tinh NAVSTAR 76 (G32)


sat_id = sat_id_24


# URL của API N2YO để lấy thông tin vệ tinh
url = f'https://api.n2yo.com/rest/v1/satellite/positions/{sat_id}/{lat}/{lon}/0/30/&apiKey={api_key}'

# Gửi yêu cầu API
response = requests.get(url)

# Kiểm tra nếu có lỗi
if response.status_code == 200:
    # In ra toàn bộ dữ liệu JSON để kiểm tra cấu trúc
    data = response.json()
    print(data)  # In toàn bộ dữ liệu trả về

    # Kiểm tra nếu dữ liệu có khóa 'positions'
    if 'positions' in data:
        positions = data['positions']
        for position in positions:
            elevation = position['elevation']
            azimuth = position['azimuth']
            print(f'Elevation: {elevation}°')
            print(f'Azimuth: {azimuth}°')
    else:
        print("Dữ liệu không chứa 'positions'. Vui lòng kiểm tra lại yêu cầu API.")
else:
    print("Lỗi khi truy vấn API.")
