#!/usr/bin/env python3
"""
Test de Precisión de Beliefs de Obstáculos

Monitorea cómo los beliefs detectados convergen hacia los obstáculos reales
usando el nuevo ObstacleBeliefTracker bayesiano.

Métricas:
- Error de centro (distancia euclidiana)
- Error de radio (subestimación/sobreestimación)
- Cobertura (obstáculos detectados vs totales)
- Convergencia temporal (mejora con más observaciones)
"""

import numpy as np
import sys
from typing import List, Tuple, Dict
from dataclasses import dataclass

from integrated_navigator import IntegratedNavigator, IntegratedNavConfig


@dataclass
class ObstacleMetrics:
    """Métricas de un obstáculo individual."""
    real_center: np.ndarray
    real_radius: float
    belief_center: np.ndarray = None
    belief_radius: float = 0.0
    center_error: float = float('inf')
    radius_error: float = 0.0
    matched: bool = False


def match_beliefs_to_ground_truth(
    beliefs: List[Tuple[float, float, float]],
    ground_truth: List[Tuple[float, float, float]],
    threshold: float = 60.0
) -> List[ObstacleMetrics]:
    """Empareja beliefs con obstáculos reales."""

    metrics = []
    used_beliefs = set()

    for gt in ground_truth:
        gt_center = np.array([gt[0], gt[1]])
        gt_radius = gt[2]

        m = ObstacleMetrics(
            real_center=gt_center,
            real_radius=gt_radius
        )

        # Buscar el belief más cercano
        best_dist = threshold
        best_idx = -1

        for i, belief in enumerate(beliefs):
            if i in used_beliefs:
                continue
            b_center = np.array([belief[0], belief[1]])
            dist = np.linalg.norm(gt_center - b_center)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            belief = beliefs[best_idx]
            m.belief_center = np.array([belief[0], belief[1]])
            m.belief_radius = belief[2]
            m.center_error = best_dist
            m.radius_error = belief[2] - gt_radius
            m.matched = True
            used_beliefs.add(best_idx)

        metrics.append(m)

    return metrics


def run_belief_accuracy_test():
    """Ejecuta el test de precisión de beliefs."""

    print("=" * 70)
    print("TEST DE PRECISIÓN DE BELIEFS - ObstacleBeliefTracker Bayesiano")
    print("=" * 70)

    # Configurar navegador con tracker bayesiano
    config = IntegratedNavConfig(
        room_width=800.0,
        room_height=600.0,
        use_estimated_pose=True,
        use_estimated_obstacles=True,
        use_bayesian_obstacle_tracker=True,
        execution_mode='gpu_batched',
    )

    nav = IntegratedNavigator(config)

    # Escenario con varios obstáculos
    np.random.seed(42)
    obstacles = [
        (300.0, 200.0, 35.0),
        (500.0, 350.0, 30.0),
        (200.0, 400.0, 40.0),
        (450.0, 150.0, 28.0),
        (600.0, 450.0, 32.0),
    ]

    nav.set_obstacles(obstacles)
    nav.set_goal(np.array([700.0, 300.0]))
    nav.robot_pos = np.array([100.0, 100.0])

    print(f"\nEscenario: {len(obstacles)} obstáculos")
    print(f"Posición inicial: {nav.robot_pos}")
    print(f"Meta: {nav.goal}")
    print("\nObstáculos ground truth:")
    for i, (cx, cy, r) in enumerate(obstacles):
        print(f"  [{i}] Centro: ({cx:.0f}, {cy:.0f}), Radio: {r:.0f}")

    # Historial de métricas
    history = {
        'steps': [],
        'center_errors': [],
        'radius_errors': [],
        'detection_rate': [],
        'num_beliefs': [],
    }

    print("\n" + "-" * 70)
    print("Evolución de beliefs durante navegación")
    print("-" * 70)

    max_steps = 200

    for step in range(max_steps):
        result = nav.step()

        if result.get('status') == 'goal_reached':
            print(f"\n✓ Meta alcanzada en paso {step}")
            break

        # Obtener beliefs actuales
        beliefs = [(b.mean[0], b.mean[1], b.mean[2])
                   for b in nav.obstacle_beliefs if b.confidence > 0.3]

        # Calcular métricas
        metrics = match_beliefs_to_ground_truth(beliefs, obstacles)

        matched = [m for m in metrics if m.matched]

        if matched:
            avg_center_error = np.mean([m.center_error for m in matched])
            avg_radius_error = np.mean([m.radius_error for m in matched])
        else:
            avg_center_error = float('inf')
            avg_radius_error = 0.0

        detection_rate = len(matched) / len(obstacles)

        history['steps'].append(step)
        history['center_errors'].append(avg_center_error if matched else None)
        history['radius_errors'].append(avg_radius_error if matched else None)
        history['detection_rate'].append(detection_rate)
        history['num_beliefs'].append(len(beliefs))

        # Imprimir cada 25 pasos
        if step % 25 == 0:
            print(f"\nPaso {step}:")
            print(f"  Robot: ({nav.robot_pos[0]:.0f}, {nav.robot_pos[1]:.0f})")
            print(f"  Beliefs activos: {len(beliefs)}")
            print(f"  Detección: {len(matched)}/{len(obstacles)} ({detection_rate:.0%})")
            if matched:
                print(f"  Error centro (promedio): {avg_center_error:.1f} px")
                print(f"  Error radio (promedio): {avg_radius_error:+.1f} px")

                # Detalle por obstáculo
                for m in metrics:
                    status = "✓" if m.matched else "✗"
                    if m.matched:
                        print(f"    {status} Real ({m.real_center[0]:.0f},{m.real_center[1]:.0f},r={m.real_radius:.0f})"
                              f" → Belief ({m.belief_center[0]:.0f},{m.belief_center[1]:.0f},r={m.belief_radius:.0f})"
                              f" | err_c={m.center_error:.1f}, err_r={m.radius_error:+.1f}")
                    else:
                        print(f"    {status} Real ({m.real_center[0]:.0f},{m.real_center[1]:.0f},r={m.real_radius:.0f})"
                              f" → NO DETECTADO")

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    # Filtrar valores válidos
    valid_center_errors = [e for e in history['center_errors'] if e is not None and e < 1000]
    valid_radius_errors = [e for e in history['radius_errors'] if e is not None]

    if valid_center_errors:
        print(f"\nError de centro:")
        print(f"  Media: {np.mean(valid_center_errors):.1f} px")
        print(f"  Mínimo: {np.min(valid_center_errors):.1f} px")
        print(f"  Final: {valid_center_errors[-1]:.1f} px")

    if valid_radius_errors:
        print(f"\nError de radio:")
        print(f"  Media: {np.mean(valid_radius_errors):+.1f} px")
        print(f"  Subestimación: {100*np.mean([1 if e < 0 else 0 for e in valid_radius_errors]):.0f}%")

    print(f"\nTasa de detección:")
    print(f"  Media: {np.mean(history['detection_rate']):.1%}")
    print(f"  Final: {history['detection_rate'][-1]:.1%}")

    # Análisis de convergencia
    if len(valid_center_errors) > 20:
        first_half = valid_center_errors[:len(valid_center_errors)//2]
        second_half = valid_center_errors[len(valid_center_errors)//2:]
        improvement = np.mean(first_half) - np.mean(second_half)
        print(f"\nConvergencia:")
        print(f"  Error primera mitad: {np.mean(first_half):.1f} px")
        print(f"  Error segunda mitad: {np.mean(second_half):.1f} px")
        print(f"  Mejora: {improvement:.1f} px ({100*improvement/np.mean(first_half):.0f}%)")

    print("\n" + "=" * 70)

    return history


if __name__ == "__main__":
    run_belief_accuracy_test()

