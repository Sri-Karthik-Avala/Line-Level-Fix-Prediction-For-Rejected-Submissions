# cross point
def cross(c1, c2):
    return c1.real * c2.imag - c1.imag * c2.real

def cross_point(p1, p2, p3, p4):
    # p1 and p2 are  points on a segment.
    # p3 and p4 are points on the other segment.
    base = p4 - p3
    hypo1 = p1 - p3
    hypo2 = p2 - p3
    d1 = cross(base, hypo1) / abs(base)
    d2 = cross(base, hypo2) / abs(base)
    cp = p1 + d1 / (d1 - d2) * (p2 - p1)
    return cp

# area of a triangle
def _area_of_triangle(c1, c2, c3):
    v1 = c2 - c1
    v2 = c3 - c1
    return abs(v1.real * v2.imag - v1.imag * v2.real) / 2

# convex cut
def convex_cut(points, c1, c2):
    points.append(points[0])
    ref_vec = c2 - c1
    cross_point1 = None
    for i, segment in enumerate(zip(points, points[1:])):
        p1, p2 = segment
        cross1 = cross(ref_vec, p1 - c1)
        cross2 = cross(ref_vec, p2 - c1)
        if cross1 <= 0 and cross2 > 0:
            cross_point1 = cross_point(c1, c2, p1, p2)
            points = points[i+1:]
            break
        elif cross1 > 0 and cross2 <= 0:
            cross_point1 = cross_point(c1, c2, p1, p2)
            points = points[i::-1] + points[:i:-1]
            break
    if cross_point1 == None:
        return 0
    cut_area = 0
    for p1, p2 in zip(points, points[1:]):
        if cross(ref_vec, p1 - c1) * cross(ref_vec, p2 - c1) <= 0:
            cross_point2 = cross_point(c1, c2, p1, p2)
            cut_area += _area_of_triangle(cross_point1, cross_point2, p1)
            break
        else:
            cut_area += _area_of_triangle(cross_point1, p1, p2)
    return cut_area

# acceptance of input
import sys

file_input = sys.stdin

n = int(file_input.readline())

def string_to_complex(s):
    x, y = map(float, s.split())
    return x + y * 1j

G = [string_to_complex(file_input.readline()) for i in range(n)]

# output
q = int(file_input.readline())

for line in file_input:
    p1x, p1y, p2x, p2y = map(int, line.split())
    p1 = p1x + p1y * 1j
    p2 = p2x + p2y * 1j
    ans = convex_cut(G.copy(), p1, p2)
    print("{:f}".format(ans))