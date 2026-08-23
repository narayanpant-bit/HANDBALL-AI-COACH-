import numpy as np

def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """तीन पॉइंट्स (a, b, c) के बीच का कोण निकालता है जहाँ b जॉइंट (Vertex) है।"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return round(float(angle), 1)
