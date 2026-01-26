#!/usr/bin/env python3
"""Test de detección temprana de obstáculos"""
import numpy as np
from perception import cluster_points_dbscan, fit_circle_least_squares, ray_cast_vectorized_full

robot_pos = np.array([100.0, 100.0])
obstacle = np.array([[350, 300, 40]])

print("=" * 50)
print("Test de detección desde posición inicial")
print("=" * 50)
print(f"Robot: {robot_pos}")
print(f"Obstáculo: {obstacle[0]}")
print(f"Distancia: {np.linalg.norm(robot_pos - obstacle[0, :2]):.0f}")
print()

# Ray cast
result = ray_cast_vectorized_full(robot_pos, 0.0, obstacle, 360, 350)
angles = np.linspace(0, 2*np.pi, 360, endpoint=False)

# Filtrar hits válidos
valid = result['distances'] < 350 * 0.95
print(f"Hits válidos: {np.sum(valid)} de 360")

dists = result['distances'][valid]
angs = angles[valid]
points = robot_pos + dists[:, np.newaxis] * np.column_stack([np.cos(angs), np.sin(angs)])
print(f"Puntos totales: {len(points)}")

# Mostrar algunos puntos
obs_center = obstacle[0, :2]
print("\nPrimeros puntos cerca del obstáculo:")
for i, p in enumerate(points):
    d = np.linalg.norm(p - obs_center)
    if d < 60:  # Solo puntos cerca del obstáculo
        print(f"  ({p[0]:.0f}, {p[1]:.0f}) dist_obs={d:.1f}")

# Clustering
print(f"\nDBSCAN clustering (eps=12, min_samples=2, min_size=2):")
clusters = cluster_points_dbscan(points, 12.0, 2, 2)
print(f"Clusters encontrados: {len(clusters)}")

for i, c in enumerate(clusters):
    center = np.mean(c, axis=0)
    dist_to_obs = np.linalg.norm(center - obs_center)
    print(f"\n  Cluster {i}: {len(c)} puntos")
    print(f"    Centro: ({center[0]:.0f}, {center[1]:.0f})")
    print(f"    Dist al obs real: {dist_to_obs:.0f}")

    if len(c) >= 2:
        fit = fit_circle_least_squares(c)
        if fit:
            cx, cy, r, var = fit
            print(f"    Circle fit: ({cx:.0f}, {cy:.0f}) r={r:.0f} var={var:.0f}")

            # Validación
            if 8.0 < r < 60.0 and var < 3000:
                print(f"    -> VÁLIDO para detección")
            else:
                print(f"    -> RECHAZADO (r fuera de rango o var alta)")
        else:
            print(f"    Circle fit: FALLÓ")

