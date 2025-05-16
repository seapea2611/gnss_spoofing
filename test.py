print(round(0.4))
print(round(0.5001))
print(round(0.500))

print(round(-0.4))
print(round(-0.5))
print(round(-0.501))

print(round(0.5))
print(round(1.49))
print(round(-1.50))





import math

# Làm tròn theo quy tắc bình thường
def normal_round(number):
    if number % 1 >= 0.5:
        return math.ceil(number)
    else:
        return math.floor(number)

print(normal_round(0.5))  # Kết quả: 1
print(normal_round(1.5))  # Kết quả: 2

