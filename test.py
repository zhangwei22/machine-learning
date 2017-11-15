x1 = [1, 2, 3]
x1 = zip(x1)
print(list(x1))

x = [1, 2, 3]

y = [4, 5, 6]

z = [7, 8, 9]

xyz = zip(x, y, z)
print(list(xyz))
u = zip(*xyz)

print(list(u))
