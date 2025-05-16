import numpy as np

# Tọa độ trong hệ ECEF (Đã biết)
receiver_ecef_02 = np.array([ -1626343.5660,  5730606.2916,  2271881.4271 ])

# Các hằng số liên quan đến hệ WGS84
a = 6378137.0  # Bán kính xích đạo (m)
e = 8.1819190842622e-2  # Độ lệch tâm của elip (số không gian)

# Tính kinh độ (longitude)
longitude = np.arctan2(receiver_ecef_02[1], receiver_ecef_02[0])

# Tính vĩ độ (latitude)
p = np.sqrt(receiver_ecef_02[0]**2 + receiver_ecef_02[1]**2)
theta = np.arctan2(receiver_ecef_02[2] * a, p * (1 - e**2))

# Chuyển đổi từ radian sang độ
latitude = np.arctan2(receiver_ecef_02[2] + e**2 * a * np.sin(theta)**3,
                      p - e**2 * a * np.cos(theta)**3)

latitude = np.degrees(latitude)
longitude = np.degrees(longitude)

print(f'Vĩ độ: {latitude}°')
print(f'Kinh độ: {longitude}°')
