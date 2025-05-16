from datetime import datetime

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
    gps_seconds = round(delta_seconds % 604800, 6)  # Làm tròn đến 6 chữ số thập phân

    return gps_week, gps_seconds




# ---- Ví dụ sử dụng ----

date_str = "2025  3 13  8 48 41.00"
gps_week, gps_seconds = cal2gpstime(date_str)
print(f"GPS Week: {gps_week}, GPS Seconds: {gps_seconds}")

date_str2 = "18 02 16 08 21 06.000000"
gps_week2, gps_seconds2 = cal2gpstime_v211(date_str2)
print(f"GPS Week: {gps_week2}, GPS Seconds: {gps_seconds2}")
