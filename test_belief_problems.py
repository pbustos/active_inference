#!/usr/bin/env python3
"""
Test específico para verificar los problemas de beliefs:
1. Beliefs creados fuera de los círculos reales
2. Beliefs buenos eliminados cuando el robot se aleja
3. Beliefs fantasma que siguen al robot

Este test simula navegación y verifica cada condición.
"""

import numpy as np
from integrated_navigator import IntegratedNavigator, IntegratedNavConfig


def test_belief_problems():
    """Test para verificar que los problemas están corregidos."""

    print("=" * 70)
    print("TEST DE PROBLEMAS DE BELIEFS")
    print("=" * 70)

    # Configurar navegador
    config = IntegratedNavConfig(
        room_width=800.0,
        room_height=600.0,
        use_estimated_pose=False,  # Ground truth para aislar problemas de beliefs
        use_estimated_obstacles=True,
        use_bayesian_obstacle_tracker=True,
        execution_mode='gpu_batched',
    )

    nav = IntegratedNavigator(config)

    # Escenario con un obstáculo bien definido
    obstacle = (400.0, 300.0, 40.0)
    nav.set_obstacles([obstacle])
    nav.set_goal(np.array([700.0, 300.0]))
    nav.robot_pos = np.array([100.0, 300.0])

    print(f"\nObstáculo real: centro=({obstacle[0]}, {obstacle[1]}), radio={obstacle[2]}")
    print(f"Robot inicial: {nav.robot_pos}")
    print(f"Meta: {nav.goal}")

    # Fase 1: Acercarse al obstáculo y observarlo múltiples veces
    print("\n" + "-" * 70)
    print("FASE 1: Acercándose al obstáculo (múltiples observaciones)")
    print("-" * 70)

    belief_when_close = None
    max_steps = 200
    observations_near_obstacle = 0

    for step in range(max_steps):
        result = nav.step()

        dist_to_obstacle = np.linalg.norm(nav.robot_pos - np.array([obstacle[0], obstacle[1]]))

        # Contar observaciones cuando estamos cerca
        if dist_to_obstacle < 150 and nav.obstacle_beliefs:
            observations_near_obstacle += 1
            belief = nav.obstacle_beliefs[0]

            # Reportar cada 10 observaciones
            if observations_near_obstacle % 10 == 1:
                center_error = np.linalg.norm(belief.mean[:2] - np.array([obstacle[0], obstacle[1]]))
                print(f"  Step {step}: dist={dist_to_obstacle:.0f}, obs_count={observations_near_obstacle}")
                print(f"    Belief: ({belief.mean[0]:.1f}, {belief.mean[1]:.1f}), r={belief.mean[2]:.1f}")
                print(f"    Error centro: {center_error:.1f} px, Confidence: {belief.confidence:.2f}")

        # Guardar belief después de varias observaciones
        if observations_near_obstacle >= 15 and nav.obstacle_beliefs:
            belief = nav.obstacle_beliefs[0]
            belief_when_close = (belief.mean[0], belief.mean[1], belief.mean[2], belief.confidence)

            center_error = np.linalg.norm(belief.mean[:2] - np.array([obstacle[0], obstacle[1]]))
            print(f"\n  Belief consolidado después de {observations_near_obstacle} observaciones:")
            print(f"    Belief: ({belief.mean[0]:.1f}, {belief.mean[1]:.1f}), r={belief.mean[2]:.1f}")
            print(f"    Error centro: {center_error:.1f} px")
            print(f"    Confidence: {belief.confidence:.2f}")

            # TEST 1: Verificar que el belief está dentro del círculo real
            if center_error > obstacle[2] * 1.5:
                print(f"    ⚠️ PROBLEMA 1: Belief fuera del círculo real!")
            else:
                print(f"    ✓ Belief dentro del círculo real")
            break

        # Si ya pasamos el obstáculo, terminar
        if nav.robot_pos[0] > obstacle[0] + 100:
            if nav.obstacle_beliefs:
                belief = nav.obstacle_beliefs[0]
                belief_when_close = (belief.mean[0], belief.mean[1], belief.mean[2], belief.confidence)
                center_error = np.linalg.norm(belief.mean[:2] - np.array([obstacle[0], obstacle[1]]))
                print(f"\n  Pasamos el obstáculo con {observations_near_obstacle} observaciones")
                print(f"    Belief: ({belief.mean[0]:.1f}, {belief.mean[1]:.1f}), r={belief.mean[2]:.1f}")
                print(f"    Error centro: {center_error:.1f} px, Confidence: {belief.confidence:.2f}")
            break

    if belief_when_close is None:
        print("  ⚠️ No se detectó ningún belief cerca del obstáculo")
        return

    # Fase 2: Alejarse del obstáculo
    print("\n" + "-" * 70)
    print("FASE 2: Alejándose del obstáculo")
    print("-" * 70)

    # Mover el robot lejos del obstáculo
    nav.robot_pos = np.array([700.0, 300.0])
    nav.set_goal(np.array([750.0, 300.0]))

    for step in range(30):
        result = nav.step()

    # Verificar que el belief persiste
    print(f"\n  Robot ahora en: {nav.robot_pos}")
    print(f"  Distancia al obstáculo: {np.linalg.norm(nav.robot_pos - np.array([obstacle[0], obstacle[1]])):.0f}")

    if nav.obstacle_beliefs:
        belief = nav.obstacle_beliefs[0]
        print(f"  Beliefs activos: {len(nav.obstacle_beliefs)}")
        print(f"  Belief persistente: ({belief.mean[0]:.1f}, {belief.mean[1]:.1f}), r={belief.mean[2]:.1f}")
        print(f"  Confidence: {belief.confidence:.2f}")

        # Verificar que el belief no se movió mucho
        belief_drift = np.linalg.norm(
            np.array([belief.mean[0], belief.mean[1]]) -
            np.array([belief_when_close[0], belief_when_close[1]])
        )

        if belief_drift < 10:
            print(f"  ✓ TEST 2 PASADO: Belief persistió sin moverse (drift={belief_drift:.1f}px)")
        else:
            print(f"  ⚠️ PROBLEMA 2: Belief se movió {belief_drift:.1f}px desde posición original")
    else:
        print(f"  ⚠️ PROBLEMA 2: ¡El belief fue eliminado cuando el robot se alejó!")

    # Fase 3: Verificar que no hay beliefs fantasma
    print("\n" + "-" * 70)
    print("FASE 3: Verificando beliefs fantasma")
    print("-" * 70)

    # Mover el robot a una zona vacía lejos del obstáculo
    nav.robot_pos = np.array([100.0, 100.0])
    nav.set_goal(np.array([150.0, 100.0]))

    phantom_count = 0
    for step in range(50):
        result = nav.step()

        # Buscar beliefs que estén cerca del robot (serían fantasmas)
        for belief in nav.obstacle_beliefs:
            dist_to_robot = np.linalg.norm(belief.mean[:2] - nav.robot_pos)
            dist_to_real = np.linalg.norm(belief.mean[:2] - np.array([obstacle[0], obstacle[1]]))

            # Si está cerca del robot pero lejos del obstáculo real, es fantasma
            if dist_to_robot < 80 and dist_to_real > obstacle[2] * 2:
                phantom_count += 1
                print(f"  ⚠️ Step {step}: Belief fantasma detectado cerca del robot!")
                print(f"     Belief en ({belief.mean[0]:.1f}, {belief.mean[1]:.1f})")
                print(f"     Dist a robot: {dist_to_robot:.1f}, Dist a obstáculo real: {dist_to_real:.1f}")

    if phantom_count == 0:
        print(f"  ✓ TEST 3 PASADO: No se detectaron beliefs fantasma")
    else:
        print(f"  ⚠️ PROBLEMA 3: Se detectaron {phantom_count} beliefs fantasma")

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)

    final_beliefs = len(nav.obstacle_beliefs)
    print(f"\nBeliefs finales: {final_beliefs}")
    for i, belief in enumerate(nav.obstacle_beliefs):
        dist_to_real = np.linalg.norm(belief.mean[:2] - np.array([obstacle[0], obstacle[1]]))
        print(f"  [{i}] Centro: ({belief.mean[0]:.1f}, {belief.mean[1]:.1f}), r={belief.mean[2]:.1f}")
        print(f"      Error al real: {dist_to_real:.1f} px, Confidence: {belief.confidence:.2f}")


if __name__ == "__main__":
    test_belief_problems()

