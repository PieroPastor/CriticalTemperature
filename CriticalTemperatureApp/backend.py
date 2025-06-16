from mendeleev import element
from scipy.stats import gmean, entropy
from pymatgen.core.periodic_table import Element
import pandas as pd
import joblib
import numpy as np
import re
import types

# Cargar modelo
model = joblib.load("vrg.pkl")


properties = {
    'atomic_mass': 'atomic_mass',
    'atomic_radius': 'atomic_radius',
    'density': 'Density',
    'electron_affinity': 'ElectronAffinity',
    'fusion_heat': 'FusionHeat',
    'thermal_conductivity': 'ThermalConductivity',
    'nvalence': 'Valence',
    'ionenergies': 'fie',  # usaremos la primera energía de ionización
}

def parse_formula(formula):
    """
    Convierte 'Ba1.0La2.0Cu1.0O4.0' a {'Ba': 1.0, 'La': 2.0, 'Cu': 1.0, 'O': 4.0}
    """
    matches = re.findall(r"([A-Z][a-z]*)(\d+\.?\d*)", formula)
    return {elem: float(qty) for elem, qty in matches}


def compute_feature_stats(values, weights):
    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)
    total_w = weights.sum()

    # Funciones auxiliares
    w_mean = lambda x: np.average(x, weights=weights) if total_w>0 else 0.0
    w_std  = lambda x: np.sqrt(np.average((x - w_mean(x))**2, weights=weights)) if total_w>0 else 0.0
    ent    = lambda x: entropy(x/ x.sum())  if x.sum()>0 else 0.0

    stats = {
        'mean':        values.mean(),
        'wtd_mean':    w_mean(values),
        'gmean':       gmean(values[values>0]) if np.any(values>0) else 0.0,
        'wtd_gmean':   np.exp((weights * np.log(values + 1e-8)).sum()/total_w) if total_w>0 else 0.0,
        'std':         values.std(),
        'wtd_std':     w_std(values),
        'range':       values.max() - values.min(),
        'wtd_range':   w_mean(values) - values.min(),
        'entropy':     ent(values),
        'wtd_entropy': ent(weights),
    }
    return stats


def formula_to_vector(formula):
    # 1) parseamos
    comp = parse_formula(formula)
    elems, quants = zip(*comp.items())

    feat_dict = {}
    for attr, suffix in properties.items():
        vals = []
        for el in elems:
            try:
                ed = Element(el)
                ed2 = element(el)
                if attr == 'ionenergies':
                    dict_io = ed2.ionenergies
                    val = dict_io.get(1)*96.485 or 0.0 #Se debe de convertir de eV a Kj/mol
                elif attr == "fusion_heat":
                    val = ed2.fusion_heat or 0.0
                elif attr == "density":
                    val = ed2.density * 1000 or 0.0
                elif attr == "atomic_mass":
                    val = ed2.mass or 0.0
                elif attr == "thermal_conductivity":
                    val = ed2.thermal_conductivity or 0.0
                elif attr == "nvalence":
                    val = ed2.nvalence() or 0.0 #FALTA REVISAR
                elif attr == "electron_affinity":
                    val = ed2.electron_affinity*96.485 or 0.0
                else:
                    val = getattr(ed2, attr, None)
                if val is None or 0.0 or isinstance(val, (types.FunctionType, types.MethodType)):
                    print(f"{el} no tiene atributo {attr}, se usará 0.0")
                    vals.append(0.0)
                else:
                    vals.append(val)
            except Exception as e:
                print(f"Error procesando {el} - {attr}: {e}")
                vals.append(0.0)
        stats = compute_feature_stats(vals, quants)
        for stat_name, val in stats.items():
            feat_dict[f"{stat_name}_{suffix}"] = val

    # número de elementos
    feat_dict["number_of_elements"] = len(elems)
    ordered_cols = list(
        pd.read_csv("train.csv")
        .drop(columns=["critical_temp"])
        .columns
    )
    # armamos el DataFrame y reordenamos
    vec = pd.DataFrame([feat_dict])
    return vec[ordered_cols]

vector = formula_to_vector("Ba0.2La1.8Cu1.0O4.0")
zzz = pd.read_csv("train.csv")
for col in vector.columns:
    print(col, ": ", vector[col][0], "     -      ", zzz[col][0])