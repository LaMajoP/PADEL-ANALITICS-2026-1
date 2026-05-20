import pandas as pd
import numpy as np

INPUT_FILE  = "dataset.csv"
OUTPUT_FILE = "dataset_limpio.csv"

print("Cargando dataset...")
df = pd.read_csv(INPUT_FILE)
filas_originales = len(df)
print(f"Filas originales: {filas_originales:,}")

# ─────────────────────────────────────────
# PASO 1 — Eliminar frames sin jugadores
# ─────────────────────────────────────────
print("\n── Paso 1: Eliminando frames sin jugadores...")
antes = len(df)
df = df[df['j1_confianza'].notna()]
eliminados = antes - len(df)
print(f"   Eliminadas: {eliminados:,} filas")

# ─────────────────────────────────────────
# PASO 2 — Estandarizar nombres de partido
# ─────────────────────────────────────────
print("\n── Paso 2: Estandarizando nombres de partido...")
df['partido'] = df['partido'].str.capitalize()
df['punto']   = df['punto'].str.capitalize()
print("   Listo")

# ─────────────────────────────────────────
# PASO 3 — Reemplazar coordenadas fuera de rango con NaN
# ─────────────────────────────────────────
print("\n── Paso 3: Limpiando coordenadas fuera del video (1920x1080)...")
coords_corregidas = 0
partes = ['cabeza', 'hombro_d', 'codo', 'muneca', 'punta_raqueta']

for j in ['j1', 'j2', 'j3', 'j4']:
    for parte in partes:
        cx = f'{j}_{parte}_x'
        cy = f'{j}_{parte}_y'
        if cx in df.columns:
            mask_x = (df[cx] < 0) | (df[cx] > 1920)
            mask_y = (df[cy] < 0) | (df[cy] > 1080)
            coords_corregidas += mask_x.sum() + mask_y.sum()
            df.loc[mask_x, cx] = np.nan
            df.loc[mask_y, cy] = np.nan

print(f"   Coordenadas reemplazadas con NaN: {coords_corregidas:,}")

# ─────────────────────────────────────────
# PASO 4 — Borrar detecciones con confianza baja
# ─────────────────────────────────────────
print("\n── Paso 4: Eliminando columnas redundantes...")
cols_redundantes = [col for col in df.columns if col.endswith('_confianza_baja')]
df.drop(columns=cols_redundantes, inplace=True)
print(f"   Columnas eliminadas: {cols_redundantes}")

# ─────────────────────────────────────────
# PASO 5 — Marcar muñeca = raqueta (detección sospechosa)
# ─────────────────────────────────────────
print("\n── Paso 5: Marcando detecciones sospechosas (muñeca ≈ raqueta)...")
for j in ['j1', 'j2', 'j3', 'j4']:
    mx   = f'{j}_muneca_x'
    my   = f'{j}_muneca_y'
    rx   = f'{j}_punta_raqueta_x'
    ry   = f'{j}_punta_raqueta_y'
    flag = f'{j}_raqueta_sospechosa'
    if mx in df.columns:
        distancia  = np.sqrt((df[rx] - df[mx])**2 + (df[ry] - df[my])**2)
        df[flag]   = distancia < 5

total_sospechosas = sum(
    df[f'{j}_raqueta_sospechosa'].sum()
    for j in ['j1','j2','j3','j4']
    if f'{j}_raqueta_sospechosa' in df.columns
)
print(f"   Detecciones marcadas como sospechosas: {total_sospechosas:,}")

# ─────────────────────────────────────────
# PASO 6 — Interpolación lineal en j1 (NaN mínimos)
# ─────────────────────────────────────────
print("\n── Paso 6: Interpolando NaN de j1 por punto...")

# Solo las columnas de j1 que tienen NaN
cols_j1_con_nan = [
    'j1_hombro_d_x',
    'j1_codo_x',
    'j1_muneca_x',
    'j1_punta_raqueta_x',
    'j1_cabeza_x',
    'j1_cabeza_y'
]

interpolados = 0

for _, grupo in df.groupby(['cancha', 'partido', 'punto']):
    idx = grupo.index
    for col in cols_j1_con_nan:
        nans_antes = grupo[col].isna().sum()
        if nans_antes > 0:
            # Interpolación lineal dentro del grupo ordenado por frame
            df.loc[idx, col] = (
                grupo.sort_values('frame')[col]
                .interpolate(method='linear', limit_direction='both')
                .values
            )
            interpolados += nans_antes

print(f"   NaN interpolados: {interpolados:,}")

# Verificar que j1 quedó sin NaN
nans_j1_restantes = sum(df[col].isna().sum() for col in cols_j1_con_nan)
print(f"   NaN restantes en j1 después de interpolación: {nans_j1_restantes}")

# ─────────────────────────────────────────
# PASO 7 — Columnas de visibilidad j2, j3, j4
# ─────────────────────────────────────────
print("\n── Paso 7: Creando columnas de visibilidad...")
for j in ['j2', 'j3', 'j4']:
    col  = f'{j}_confianza'
    flag = f'{j}_visible'
    if col in df.columns:
        df[flag] = df[col].notna().astype(int)

print(f"   j2_visible: {df['j2_visible'].sum():,} frames con j2 ({df['j2_visible'].mean()*100:.1f}%)")
print(f"   j3_visible: {df['j3_visible'].sum():,} frames con j3 ({df['j3_visible'].mean()*100:.1f}%)")
print(f"   j4_visible: {df['j4_visible'].sum():,} frames con j4 ({df['j4_visible'].mean()*100:.1f}%)")


# ─────────────────────────────────────────
# GUARDAR
# ─────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)

# ─────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────
sep = "═" * 50
print(f"\n{sep}")
print(f"  RESUMEN DE LIMPIEZA")
print(f"{sep}")
print(f"  Filas originales         : {filas_originales:,}")
print(f"  Filas eliminadas         : {filas_originales - len(df):,}")
print(f"  Filas finales            : {len(df):,}")
print(f"  Columnas originales      : 76")
print(f"  Columnas finales         : {len(df.columns)}")
print(f"  Coordenadas corregidas   : {coords_corregidas:,}")
print(f"  Flags raqueta sospechosa : {total_sospechosas:,}")
print(f"  NaN j1 interpolados      : {interpolados:,}")
print(f"  Columnas redundantes eliminadas: {len(cols_redundantes)}")
print(f"  Columnas visibilidad     : j2_visible, j3_visible, j4_visible")
print(f"{sep}")
print(f"\n  Dataset limpio guardado en: {OUTPUT_FILE}")
print(f"{sep}\n")