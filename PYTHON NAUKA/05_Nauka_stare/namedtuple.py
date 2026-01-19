from collections import namedtuple


print("Point = namedtuple('Point', ['x', 'y'])")
Point = namedtuple('Point', ["x", "y", "z"])
print("Point.__doc__")
print(Point.__doc__)
print("p = Point(11, y=22, z=3)")
p = Point(11, y=22, z=3)  
          
print(f"{p[0] + p[1]}")                     
x, y, z = p                       
print(f"x={x}, y={y}, z={z}")
print(p.x + p.y + p.z)
d = p._asdict()
print(f"d['x']: {d["x"]}")

Point(**d)
Point(x=11, y=22)
p._replace(x=100)
Point(x=100, y=22)