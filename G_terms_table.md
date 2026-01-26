# Términos G del Expected Free Energy (EFE)

El EFE total se calcula como:

```
G(π) = G_goal + G_obstacle + G_ambiguity + G_epistemic + G_structural + G_localization + G_velocity
```

## Tabla de Términos

| Término | Nombre UI | Color | Ecuación Paper | Descripción |
|---------|-----------|-------|----------------|-------------|
| **G_goal** | `G_goal` | 🟢 Verde | Eq. pragmatic_cost | **Coste pragmático hacia el objetivo.** Penaliza la distancia al goal. `G_goal = λ_goal · ‖pos - goal‖`. Cuanto más lejos del objetivo, mayor coste. |
| **G_obstacle** | `G_obs` | 🔴 Rojo | Eq. obstacle_cost | **Coste de proximidad a obstáculos.** Penaliza estar cerca de obstáculos. Tiene dos zonas: soft (repulsión suave) y hard (barrera exponencial). Evita colisiones. |
| **G_ambiguity** | `G_amb` | 🔵 Azul | Eq. 107 | **Incertidumbre en observaciones futuras.** `G_amb = λ_amb × (1/2) ln|Σ_o|` (λ=0.15). Aumenta con la velocidad (más ruido en sensores). Penaliza acciones rápidas cuando hay incertidumbre. |
| **G_epistemic** | `G_epi` | 🟠 Naranja | Eq. epistemic_value | **Ganancia de información sobre posición de obstáculos.** Negativo = incentivo para acercarse a obstáculos y reducir incertidumbre posicional. |
| **G_structural** | `G_struct` | 🟣 Púrpura | Eq. model_mutual_information | **Ganancia de información sobre identidad del modelo.** `I(m; o_{t+1}|π) = H[q(m)] - E[H[q(m|o)]]`. Incentiva movimientos que disambiguan entre círculo/rectángulo. Solo activo cuando hay incertidumbre sobre el tipo. |
| **G_localization** | `G_loc` | 🩵 Cian | Eq. localization_cost | **Coste de incertidumbre de pose.** `G_loc = λ_loc · tr(Σ_pose)` (λ=0.5). Penaliza poses con alta incertidumbre. Incentiva acciones que mejoran la localización. |
| **G_velocity** | `G_vel` | 🟤 Marrón | Section g_velocity | **Regularización de velocidad y suavidad.** `G_vel = λ_v·Σ‖v_τ‖² + λ_Δv·Σ‖v_τ - v_{τ-1}‖²`. Penaliza velocidades altas y cambios bruscos. Promueve trayectorias suaves y eficientes energéticamente. |

## Comportamiento Esperado

| Situación | Términos Dominantes | Comportamiento |
|-----------|---------------------|----------------|
| Lejos del goal, sin obstáculos | G_goal alto | Robot se mueve directamente al goal |
| Cerca de obstáculo | G_obstacle alto | Robot evita, rodea el obstáculo |
| Obstáculo con tipo incierto (50/50) | G_structural alto | Robot busca ángulo para ver mejor la forma |
| Alta velocidad | G_ambiguity alto | Robot reduce velocidad para mejor percepción |
| Pose muy incierta | G_localization alto | Robot reduce velocidad y busca landmarks para localizarse |
| Movimiento oscilatorio | G_velocity alto | Robot suaviza trayectoria, evita jerk |

## Valores Típicos

| Término | Rango Típico | Cuándo es Alto |
|---------|--------------|----------------|
| G_goal | 10 - 500 | Lejos del objetivo |
| G_obs | 0 - 100 | Muy cerca de obstáculo |
| G_amb | 1 - 50 | Alta velocidad |
| G_epi | 0 - 10 | Obstáculos en rango sensor |
| G_struct | 0 - 0.1 | Incertidumbre tipo (q(m) ≈ 0.5) |
| G_loc | 50 - 1000 | Pose muy incierta |
| G_vel | 0 - 50 | Velocidades altas o cambios bruscos |

## Notas

- **G_structural** solo es no-cero cuando hay obstáculos con incertidumbre real sobre su tipo (P(circle) ≠ 1 y P(rect) ≠ 1)
- **G_velocity** combina dos términos: magnitud de velocidad (λ_v) y cambio de velocidad (λ_Δv)
- Los términos con signo **negativo** en el cálculo interno son **incentivos** (queremos maximizar información)
- Todos se muestran como valores **absolutos** positivos en el UI para facilitar la visualización

## Uso Programático: EFEGains Dataclass

Los pesos (λ) de todos los términos G están centralizados en el dataclass `EFEGains` y pueden ser modificados dinámicamente:

```python
from unified_controller import EFEGains, UnifiedControllerConfig

# Crear config con ganancias por defecto
config = UnifiedControllerConfig()

# Acceder a las ganancias (compatibilidad hacia atrás)
print(config.lambda_goal)  # 1.0

# Modificar dinámicamente en tiempo de ejecución
config.efe_gains.lambda_goal = 2.0       # Más atracción al goal
config.efe_gains.lambda_obstacle = 200.0 # Más evitación de obstáculos
config.efe_gains.lambda_epistemic = 1.0  # Más exploración

# Crear config con ganancias personalizadas
custom_gains = EFEGains(
    lambda_goal=1.5,
    lambda_obstacle=100.0,
    lambda_epistemic=0.8,
    lambda_ambiguity=0.2,
    lambda_structural=0.5,
    lambda_localization=0.3,
    lambda_smoothness=0.01,
    lambda_velocity=0.01,
    lambda_delta_velocity=0.05
)
config = UnifiedControllerConfig(efe_gains=custom_gains)
```

### Campos de EFEGains

| Campo | Valor Default | Descripción |
|-------|---------------|-------------|
| `lambda_goal` | 1.0 | Peso de atracción al objetivo |
| `lambda_obstacle` | 200.0 | Peso de evitación de obstáculos |
| `lambda_epistemic` | 0.5 | Peso de ganancia de información posicional |
| `lambda_ambiguity` | 0.15 | Peso de incertidumbre observacional |
| `lambda_structural` | 0.5 | Peso de disambiguación de modelo |
| `lambda_localization` | 0.5 | Peso de incertidumbre de pose |
| `lambda_smoothness` | 0.02 | Peso de suavidad de acciones (legacy) |
| `lambda_velocity` | 0.01 | Peso de magnitud de velocidad (λ_v) |
| `lambda_delta_velocity` | 0.05 | Peso de cambio de velocidad / jerk (λ_Δv) |

