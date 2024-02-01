from engine import Value
from vis import draw_dot

a = Value(2)
b = Value(3)
c = a + b
d = a * b
e = c + d

l = 2 + a
n = 3 * a
m = 2 / a
print(m)
print(l)
print(n)



# e.backwoard()
# draw_dot(e)