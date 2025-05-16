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
