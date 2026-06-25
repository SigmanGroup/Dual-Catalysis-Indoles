#!/usr/bin/env python3
# coding: utf-8

"""
Parallelized ORCA-only property extraction functions.

Design choice:
- geometry-based descriptors read the sibling `.xyz`
- text/electronic descriptors read the sibling `.out` / `.OUT`
"""

from __future__ import annotations

import re
import math
import logging
import itertools
import multiprocessing
from pathlib import Path

import pandas as pd
import numpy as np

try:  # pragma: no cover - optional dependency
    from morfeus import Sterimol, BuriedVolume, Pyramidalization, SASA
    MORFEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Sterimol = BuriedVolume = Pyramidalization = SASA = None
    MORFEUS_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import dbstep.Dbstep as db
    DBSTEP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    db = None
    DBSTEP_AVAILABLE = False

from .utils import get_filecont, get_specdata, get_xyz
from .utils import FILE_COLUMN_NAME, configure_logger

logger = logging.getLogger(__name__)

# ORCA patterns
orbital_line_pattern = re.compile(r'^\s*(\d+)\s+([0-9.]+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)')
dipole_pattern = re.compile(r'Magnitude \(Debye\)\s*:\s*([-+]?\d+\.\d+)')
polar_iso_pattern = re.compile(r'Isotropic polarizability\s*:\s*([-+]?\d+\.\d+)')
polar_aniso_pattern = re.compile(r'Anisotropic polarizability\s*:\s*([-+]?\d+\.\d+)')
volume_patterns = [
    re.compile(r'Molar volume\s*=\s*([-+]?\d+(?:\.\d+)?)'),
    re.compile(r'Molecular volume\s*:\s*([-+]?\d+(?:\.\d+)?)'),
    re.compile(r'Volume\s*:\s*([-+]?\d+(?:\.\d+)?)\s*(?:Bohr\^3|a\.u\.|Ang\^3|A\^3)', re.IGNORECASE),
]
wall_time_pattern = re.compile(
    r'TOTAL RUN TIME:\s*(\d+)\s*days\s*(\d+)\s*hours\s*(\d+)\s*minutes\s*(\d+)\s*seconds(?:\s*(\d+)\s*msec)?',
    re.IGNORECASE,
)

# NPA / NBO
nbo_os_pattern = re.compile(r'beta spin orbitals', re.IGNORECASE)
npa_pattern = re.compile(r'Summary of Natural Population Analysis:')
npa_row_pattern = re.compile(r'^\s*([A-Za-z]+)\s+(\d+)\s+([-+]?\d+\.\d+)')

# NMR
nmr_header_pattern = re.compile(r'SCF GIAO Magnetic shielding tensor \(ppm\):')
nmr_row_pattern = re.compile(
    r'^\s*(\d+)\s+([A-Za-z]{1,3})\s+Isotropic\s*=\s*([-+]?\d+(?:\.\d+)?)\s+Anisotropy\s*=\s*([-+]?\d+(?:\.\d+)?)'
)

# CHELPG / Hirshfeld
chelpg1_pattern = re.compile(r'CHELPG')
chelpg2_pattern = re.compile(r'Charges from ESP fit')
hirshfeld_pattern = re.compile(r'Hirshfeld charges, spin densities, dipoles, and CM5 charges')


def _merge_results(dataframe: pd.DataFrame, results: list[pd.DataFrame], log_message: str) -> pd.DataFrame:
    results = pd.concat(results)
    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)
    logger.info(log_message)
    return dataframe


def _load_geometry(file: Path):
    elements, coordinates = get_xyz(file)
    return np.array(elements, dtype=str), np.array(coordinates, dtype=float)


def _all_atom_labels(row: pd.Series) -> list[str]:
    return [x for x in row.keys() if x != FILE_COLUMN_NAME]


def _get_frontier_orbs(file: Path) -> pd.DataFrame:
    configure_logger(debug=False)
    result = {FILE_COLUMN_NAME: file.name, 'HOMO': None, 'LUMO': None, 'μ': None, 'η': None, 'ω': None}

    try:
        filecont, error = get_filecont(file, split=True)
        if error:
            logger.error('Error reading %s\t%s', file.name, error)
            return pd.DataFrame(pd.Series(result)).transpose()

        section_starts = [i for i, line in enumerate(filecont) if 'ORBITAL ENERGIES' in line]
        if not section_starts:
            logger.error('No ORBITAL ENERGIES section found in %s', file.name)
            return pd.DataFrame(pd.Series(result)).transpose()

        start = section_starts[-1] + 2
        orbitals = []

        for line in filecont[start:]:
            match = orbital_line_pattern.match(line)
            if match:
                orbitals.append({
                    'idx': int(match.group(1)),
                    'occ': float(match.group(2)),
                    'eh': float(match.group(3)),
                    'ev': float(match.group(4)),
                })
            elif orbitals and line.strip().startswith('---'):
                break
            elif orbitals and ('MOLECULAR ORBITALS' in line or 'DENSITY' in line):
                break

        if not orbitals:
            logger.error('Could not parse ORBITAL ENERGIES block in %s', file.name)
            return pd.DataFrame(pd.Series(result)).transpose()

        lumo_idx = next((i for i, orb in enumerate(orbitals) if orb['occ'] < 1e-8), None)
        if lumo_idx is None or lumo_idx == 0:
            logger.error('Could not identify HOMO/LUMO in %s', file.name)
            return pd.DataFrame(pd.Series(result)).transpose()

        homo = orbitals[lumo_idx - 1]['ev']
        lumo = orbitals[lumo_idx]['ev']
        mu = (homo + lumo) / 2.0
        eta = lumo - homo
        omega = (mu ** 2) / (2.0 * eta) if eta != 0 else None

        result.update({'HOMO': homo, 'LUMO': lumo, 'μ': mu, 'η': eta, 'ω': omega})

    except Exception as e:
        logger.error('Exception in _get_frontier_orbs for %s\t%s', file.name, e)

    return pd.DataFrame(pd.Series(result)).transpose()


def get_frontierorbs(dataframe: pd.DataFrame,
                     data_dir: Path,
                     procs: int = 1) -> pd.DataFrame:
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]
    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_frontier_orbs, files)
    return _merge_results(dataframe, results, 'Frontier orbital function has completed.')


def _get_polarizability(file: Path) -> pd.DataFrame:
    configure_logger(debug=False)
    result = {FILE_COLUMN_NAME: file.name, 'polar_iso(au)': None, 'polar_aniso(au)': None}

    try:
        text, error = get_filecont(file, split=False)
        if error:
            logger.error('Error in _get_polarizability: %s\t%s', file.name, error)
            return pd.DataFrame(pd.Series(result)).transpose()

        iso_matches = polar_iso_pattern.findall(text)
        aniso_matches = polar_aniso_pattern.findall(text)

        if iso_matches:
            result['polar_iso(au)'] = float(iso_matches[-1])
        if aniso_matches:
            result['polar_aniso(au)'] = float(aniso_matches[-1])

        if result['polar_iso(au)'] is None and result['polar_aniso(au)'] is None:
            logger.warning('No polarizability information was found in %s', file.name)

    except Exception as e:
        logger.error('Unable to acquire polarizability for %s because %s', file.name, e)

    return pd.DataFrame(pd.Series(result)).transpose()


def get_polarizability(dataframe: pd.DataFrame,
                       data_dir: Path,
                       procs: int = 1) -> pd.DataFrame:
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]
    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_polarizability, files)
    return _merge_results(dataframe, results, 'Polarizability function has completed.')


def _get_dipole(file: Path) -> pd.DataFrame:
    configure_logger(debug=False)
    result = {FILE_COLUMN_NAME: file.name, 'dipole(Debye)': None}

    try:
        text, error = get_filecont(file, split=False)
        if error:
            logger.error('Error in _get_dipole: %s\t%s', file.name, error)
            return pd.DataFrame(pd.Series(result)).transpose()

        matches = dipole_pattern.findall(text)
        if matches:
            result['dipole(Debye)'] = float(matches[-1])
        else:
            logger.warning('No dipole magnitude was found in %s', file.name)

    except Exception as e:
        logger.error('Unable to acquire dipole for %s because %s', file.name, e)

    return pd.DataFrame(pd.Series(result)).transpose()


def get_dipole(dataframe: pd.DataFrame,
               data_dir: Path,
               procs: int = 1) -> pd.DataFrame:
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]
    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_dipole, files)
    return _merge_results(dataframe, results, 'Dipole function has completed.')


def _get_volume(file: Path) -> pd.DataFrame:
    configure_logger(debug=False)
    result = {FILE_COLUMN_NAME: file.name, 'volume(Bohr_radius^3 mol^-1)': None}

    try:
        text, error = get_filecont(file, split=False)
        if error:
            logger.error('Error in _get_volume: %s\t%s', file.name, error)
            return pd.DataFrame(pd.Series(result)).transpose()

        for pattern in volume_patterns:
            matches = pattern.findall(text)
            if matches:
                result['volume(Bohr_radius^3 mol^-1)'] = float(matches[-1])
                break

        if result['volume(Bohr_radius^3 mol^-1)'] is None:
            logger.warning('No molecular volume was found in %s', file.name)

    except Exception as e:
        logger.error('Unable to acquire volume for %s because %s', file.name, e)

    return pd.DataFrame(pd.Series(result)).transpose()


def get_volume(dataframe: pd.DataFrame,
               data_dir: Path,
               procs: int = 1) -> pd.DataFrame:
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]
    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_volume, files)
    return _merge_results(dataframe, results, 'Volume function has completed.')


def _get_SASA(file: Path) -> pd.DataFrame:
    configure_logger(debug=False)
    result = {
        FILE_COLUMN_NAME: file.name,
        'SASA_surface_area(Å^2)': None,
        'SASA_volume(Å^3)': None,
        'SASA_sphericity': None
    }

    try:
        if not MORFEUS_AVAILABLE:
            logger.warning('morfeus is not installed; SASA values for %s will be missing.', file.name)
            return pd.DataFrame(pd.Series(result)).transpose()

        elements, coordinates = _load_geometry(file)
        sasa = SASA(elements, coordinates)
        sphericity = np.cbrt((36 * math.pi * sasa.volume ** 2)) / sasa.area

        result.update({
            'SASA_surface_area(Å^2)': sasa.area,
            'SASA_volume(Å^3)': sasa.volume,
            'SASA_sphericity': sphericity
        })

    except Exception as e:
        logger.error('Unable to acquire SASA for %s because %s', file.name, e)

    return pd.DataFrame(pd.Series(result)).transpose()


def get_SASA(dataframe: pd.DataFrame,
             data_dir: Path,
             procs: int = 1) -> pd.DataFrame:
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]
    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_SASA, files)
    return _merge_results(dataframe, results, 'SASA function has completed.')


def _get_nbo(row: pd.Series) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])
    result = {f'NBO_charge_{k}': None for k in _all_atom_labels(row)}
    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True)
        if error:
            logger.error('Error in _get_nbo. %s.', error)
        else:
            nbo, nbostart, skip = [], 0, 0

            for i in range(len(filecont) - 1, 0, -1):
                if nbo_os_pattern.search(filecont[i]) and skip == 0:
                    skip = 2
                if npa_pattern.search(filecont[i]):
                    if skip != 0:
                        skip -= 1
                        continue
                    nbostart = i + 6
                    break

            if nbostart == 0:
                logger.error('Error in _get_nbo. No NPA found in %s.', file.name)
            else:
                for line in filecont[nbostart:]:
                    if '==' in line:
                        break
                    match = npa_row_pattern.match(line)
                    if match:
                        nbo.append([match.group(2), match.group(3)])

                for atom_label in _all_atom_labels(row):
                    atom_number = row[atom_label]
                    data = get_specdata([str(atom_number)], nbo)
                    if data:
                        result[f'NBO_charge_{atom_label}'] = data[0]

    except Exception as e:
        logger.error('Unable to acquire NBO for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_nbo(dataframe: pd.DataFrame,
            atom_list: list[str],
            data_dir: Path,
            procs: int = 1) -> pd.DataFrame:
    interesting_columns = [FILE_COLUMN_NAME] + atom_list
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_nbo, calculation_rows)

    return _merge_results(dataframe, results, 'NBO function has completed.')


def _get_nmr(row: pd.Series) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])
    result = {f'NMR_isotropic_shielding_{k}': None for k in _all_atom_labels(row)}
    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True)
        if error:
            logger.error('Error in _get_nmr. %s.', error)
        else:
            nmr_values = []
            in_block = False
            for line in filecont:
                if nmr_header_pattern.search(line):
                    in_block = True
                    continue
                if in_block:
                    match = nmr_row_pattern.match(line)
                    if match:
                        nmr_values.append([match.group(1), match.group(3)])
                    elif nmr_values and line.strip() == '':
                        # allow blank lines inside the block
                        continue
                    elif nmr_values and not line.strip().startswith(('XX=', 'XY=', 'XZ=', 'YX=', 'YY=', 'YZ=', 'ZX=', 'ZY=', 'ZZ=', 'Eigenvalues:')):
                        # keep scanning until no more shielding records
                        pass

            if not nmr_values:
                logger.warning('No ORCA NMR shielding block was found in %s', file.name)
            else:
                for atom_label in _all_atom_labels(row):
                    atom_number = row[atom_label]
                    data = get_specdata([str(atom_number)], nmr_values)
                    if data:
                        result[f'NMR_isotropic_shielding_{atom_label}'] = data[0]

    except Exception as e:
        logger.error('Unable to acquire NMR for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_nmr(dataframe: pd.DataFrame,
            atom_list: list[str],
            data_dir: Path,
            procs: int = 1) -> pd.DataFrame:
    interesting_columns = [FILE_COLUMN_NAME] + atom_list
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_nmr, calculation_rows)

    return _merge_results(dataframe, results, 'NMR function has completed.')


def _get_distance(row: pd.Series, dist_list: list[list[str]]) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])
    result = {f'distance_{a}_{b}(Å)': None for a, b in dist_list}
    file = Path(row[FILE_COLUMN_NAME])

    try:
        _, coordinates = get_xyz(file)
        coordinates = np.array(coordinates, dtype=float)

        for a_name, b_name in dist_list:
            a = coordinates[int(row[a_name]) - 1]
            b = coordinates[int(row[b_name]) - 1]
            result[f'distance_{a_name}_{b_name}(Å)'] = float(np.linalg.norm(a - b))

    except Exception as e:
        logger.error('Unable to acquire distance values for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_distance(dataframe: pd.DataFrame,
                 dist_list: list[list[str]],
                 data_dir: Path,
                 procs: int = 1) -> pd.DataFrame:
    if not all(isinstance(x, list) for x in dist_list) or not all(len(x) == 2 for x in dist_list):
        raise ValueError('Distances must be specified as a list of 2-membered lists of atom names.')

    interesting_columns = [FILE_COLUMN_NAME] + list(set([x for xs in dist_list for x in xs]))
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_distance, zip(calculation_rows, itertools.repeat(dist_list)))

    return _merge_results(dataframe, results, 'Distance function has completed.')


def _get_angles(row: pd.Series, angle_list: list[list[str]]) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])
    result = {f'angle_{a}_{b}_{c}(°)': None for a, b, c in angle_list}
    file = Path(row[FILE_COLUMN_NAME])

    try:
        _, coordinates = get_xyz(file)
        coordinates = np.array(coordinates, dtype=float)

        for a_name, b_name, c_name in angle_list:
            a = coordinates[int(row[a_name]) - 1]
            b = coordinates[int(row[b_name]) - 1]
            c = coordinates[int(row[c_name]) - 1]

            ba = a - b
            bc = c - b

            denom = np.linalg.norm(ba) * np.linalg.norm(bc)
            if denom == 0:
                angle = None
            else:
                cos_theta = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
                angle = float(np.degrees(np.arccos(cos_theta)))

            result[f'angle_{a_name}_{b_name}_{c_name}(°)'] = angle

    except Exception as e:
        logger.error('Unable to acquire angle values for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_angles(dataframe: pd.DataFrame,
               angle_list: list[list[str]],
               data_dir: Path,
               procs: int = 1) -> pd.DataFrame:
    interesting_columns = [FILE_COLUMN_NAME] + list(set([x for xs in angle_list for x in xs]))
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_angles, zip(calculation_rows, itertools.repeat(angle_list)))

    return _merge_results(dataframe, results, 'Angle function has completed.')


def _get_dihedral(row: pd.Series, dihedral_list: list[list[str]]) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])
    result = {f'dihedral_{a}_{b}_{c}_{d} (°)': None for a, b, c, d in dihedral_list}
    file = Path(row[FILE_COLUMN_NAME])

    try:
        _, coordinates = get_xyz(file)
        coordinates = np.array(coordinates, dtype=float)

        for a_name, b_name, c_name, d_name in dihedral_list:
            p0 = coordinates[int(row[a_name]) - 1]
            p1 = coordinates[int(row[b_name]) - 1]
            p2 = coordinates[int(row[c_name]) - 1]
            p3 = coordinates[int(row[d_name]) - 1]

            b0 = -1.0 * (p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2

            b1 /= np.linalg.norm(b1)

            v = b0 - np.dot(b0, b1) * b1
            w = b2 - np.dot(b2, b1) * b1

            x = np.dot(v, w)
            y = np.dot(np.cross(b1, v), w)
            angle = float(np.degrees(np.arctan2(y, x)))

            result[f'dihedral_{a_name}_{b_name}_{c_name}_{d_name} (°)'] = angle

    except Exception as e:
        logger.error('Unable to acquire dihedral values for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_dihedral(dataframe: pd.DataFrame,
                 dihedral_list: list[list[str]],
                 data_dir: Path,
                 procs: int = 1) -> pd.DataFrame:
    interesting_columns = [FILE_COLUMN_NAME] + list(set([x for xs in dihedral_list for x in xs]))
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_dihedral, zip(calculation_rows, itertools.repeat(dihedral_list)))

    return _merge_results(dataframe, results, 'Dihedral function has completed.')


def _get_vbur(row: pd.Series, radii: list[float], a_list: list[str]) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()
    result = {}
    for atom in a_list:
        for radius in radii:
            result[f'%Vbur_{atom}_{radius:.1f}Å'] = None

    file = Path(row[FILE_COLUMN_NAME])

    try:
        if not MORFEUS_AVAILABLE:
            logger.warning('morfeus is not installed; Vbur values for %s will be missing.', file.name)
        else:
            elements, coordinates = _load_geometry(file)
            for atom in a_list:
                for radius in radii:
                    vbur = BuriedVolume(elements, coordinates, int(row[atom]), include_hs=True, radius=radius)
                    result[f'%Vbur_{atom}_{radius:.1f}Å'] = vbur.percent_buried_volume * 100.0

    except Exception as e:
        logger.error('Unable to acquire Vbur values for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_vbur_scan(dataframe: pd.DataFrame,
                  data_dir: Path,
                  a_list: list[str],
                  start_r: float,
                  end_r: float,
                  step_size: float,
                  procs: int = 1) -> pd.DataFrame:
    assert FILE_COLUMN_NAME in dataframe.columns
    interesting_columns = [FILE_COLUMN_NAME] + a_list
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]
    radii = np.arange(start_r, end_r + step_size, step_size)

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_vbur, zip(calculation_rows, itertools.repeat(radii), itertools.repeat(a_list)))

    return _merge_results(dataframe, results, 'Vbur function has completed.')


def _get_sterimol_morfeus(row: pd.Series,
                          sterimol_list: list[list[str]],
                          radius: float | None = None) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()

    buried_radius = None if radius is None else radius - 0.5

    result = {}
    for atom_pair in sterimol_list:
        if buried_radius is None:
            result[f'Sterimol_L_{atom_pair[0]}_{atom_pair[1]}(Å)_morfeus'] = None
            result[f'Sterimol_B1_{atom_pair[0]}_{atom_pair[1]}(Å)_morfeus'] = None
            result[f'Sterimol_B5_{atom_pair[0]}_{atom_pair[1]}(Å)_morfeus'] = None
        else:
            result[f'Buried_Sterimol_L_{atom_pair[0]}_{atom_pair[1]}_{buried_radius}(Å)'] = None
            result[f'Buried_Sterimol_B1_{atom_pair[0]}_{atom_pair[1]}_{buried_radius}(Å)'] = None
            result[f'Buried_Sterimol_B5_{atom_pair[0]}_{atom_pair[1]}_{buried_radius}(Å)'] = None

    file = Path(row[FILE_COLUMN_NAME])

    try:
        if not MORFEUS_AVAILABLE:
            logger.warning('morfeus is not installed; Sterimol values for %s will be missing.', file.name)
        else:
            elements, coordinates = _load_geometry(file)
            for sterimol_pair in sterimol_list:
                if len(sterimol_pair) != 2:
                    raise ValueError(f'Number of atoms for Sterimol calculation must be 2 not {len(sterimol_pair)} ({sterimol_pair})')

                atom_a = int(row[sterimol_pair[0]])
                atom_b = int(row[sterimol_pair[1]])
                sterimol_calculation = Sterimol(elements, coordinates, atom_a, atom_b)

                if buried_radius is None:
                    result[f'Sterimol_L_{sterimol_pair[0]}_{sterimol_pair[1]}(Å)_morfeus'] = sterimol_calculation.L_value
                    result[f'Sterimol_B1_{sterimol_pair[0]}_{sterimol_pair[1]}(Å)_morfeus'] = sterimol_calculation.B_1_value
                    result[f'Sterimol_B5_{sterimol_pair[0]}_{sterimol_pair[1]}(Å)_morfeus'] = sterimol_calculation.B_5_value
                else:
                    sterimol_calculation.bury(method="delete", sphere_radius=float(buried_radius))
                    result[f'Buried_Sterimol_L_{sterimol_pair[0]}_{sterimol_pair[1]}_{buried_radius}(Å)'] = sterimol_calculation.L_value
                    result[f'Buried_Sterimol_B1_{sterimol_pair[0]}_{sterimol_pair[1]}_{buried_radius}(Å)'] = sterimol_calculation.B_1_value
                    result[f'Buried_Sterimol_B5_{sterimol_pair[0]}_{sterimol_pair[1]}_{buried_radius}(Å)'] = sterimol_calculation.B_5_value

    except Exception as e:
        logger.error('Unable to acquire Sterimol values for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_sterimol_morfeus(dataframe: pd.DataFrame,
                         data_dir: Path,
                         sterimol_list: list[list[str]],
                         radius: float | None = None,
                         procs: int = 1):
    interesting_columns = [FILE_COLUMN_NAME] + list(set([x for xs in sterimol_list for x in xs]))
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_sterimol_morfeus, zip(calculation_rows, itertools.repeat(sterimol_list), itertools.repeat(radius)))

    return _merge_results(dataframe, results, 'Sterimol MORFEUS function has completed.')


def _get_chelpg(row: pd.Series, a_list: list[str]) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])
    result = {f'ChelpG_charge_{k}': None for k in _all_atom_labels(row)}
    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True)
        if error:
            logger.error('Error in _get_chelpg. %s.', error)
        else:
            chelpgstart, chelpg = 0, False

            for i in range(len(filecont) - 1, 0, -1):
                if chelpg2_pattern.search(filecont[i]):
                    chelpgstart = i
                if chelpg1_pattern.search(filecont[i]):
                    chelpg = True
                    break

            if chelpgstart != 0 and not chelpg:
                logger.error('No ChelpG charge scheme found in %s', file.name)
            elif chelpgstart == 0:
                logger.error('No ChelpG charge scheme found in %s', file.name)
            else:
                for atom_label in a_list:
                    result[f'ChelpG_charge_{atom_label}'] = float(filecont[chelpgstart + int(row[atom_label]) + 2].split()[-1])

    except Exception as e:
        logger.error('Unable to acquire ChelpG for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_chelpg(dataframe: pd.DataFrame,
               atom_list: list[str],
               data_dir: Path,
               procs: int = 1) -> pd.DataFrame:
    interesting_columns = [FILE_COLUMN_NAME] + atom_list
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_chelpg, zip(calculation_rows, itertools.repeat(atom_list)))

    return _merge_results(dataframe, results, 'ChelpG function has completed.')


def _get_hirshfeld(row: pd.Series, a_list: list[str]) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()
    row[FILE_COLUMN_NAME] = Path(row[FILE_COLUMN_NAME])
    result = {}
    for atom_label in _all_atom_labels(row):
        result[f'Hirsh_charge_{atom_label}'] = None
        result[f'Hirsh_CM5_charge_{atom_label}'] = None
        result[f'Hirsh_atom_dipole_{atom_label}'] = None

    file = Path(row[FILE_COLUMN_NAME])

    try:
        filecont, error = get_filecont(file, split=True)
        if error:
            logger.error('Error in _get_hirshfeld. %s.', error)
        else:
            hirshstart = 0
            for i in range(len(filecont) - 1, 0, -1):
                if hirshfeld_pattern.search(filecont[i]):
                    hirshstart = i
                    break

            if hirshstart == 0:
                logger.error('No Hirshfeld population analysis found in %s', file.name)
            else:
                for atom_label in a_list:
                    cont = filecont[hirshstart + int(row[atom_label]) + 1].split()
                    qh = float(cont[2])
                    qcm5 = float(cont[7])
                    dip_components = np.array(cont[4:7], dtype=float)
                    d = float(np.linalg.norm(dip_components))

                    result[f'Hirsh_charge_{atom_label}'] = qh
                    result[f'Hirsh_CM5_charge_{atom_label}'] = qcm5
                    result[f'Hirsh_atom_dipole_{atom_label}'] = d

    except Exception as e:
        logger.error('Unable to acquire Hirshfeld for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_hirshfeld(dataframe: pd.DataFrame,
                  atom_list: list[str],
                  data_dir: Path,
                  procs: int = 1) -> pd.DataFrame:
    interesting_columns = [FILE_COLUMN_NAME] + atom_list
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_hirshfeld, zip(calculation_rows, itertools.repeat(atom_list)))

    return _merge_results(dataframe, results, 'Hirshfeld function has completed.')


def _get_pyramidalization(row: pd.Series, a_list: list[str]) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()
    result = {}
    for atom_label in _all_atom_labels(row):
        result[f'pyramidalization_Gavrish_{atom_label}(°)'] = None
        result[f'pyramidalization_Agranat-Radhakrishnan_{atom_label}'] = None

    file = Path(row[FILE_COLUMN_NAME])

    try:
        if not MORFEUS_AVAILABLE:
            logger.warning('morfeus is not installed; pyramidalization values for %s will be missing.', file.name)
        else:
            _, coordinates = get_xyz(file)
            coordinates = np.array(coordinates, dtype=float)
            for atom_label in a_list:
                pyr = Pyramidalization(coordinates, int(row[atom_label]))
                result[f'pyramidalization_Gavrish_{atom_label}(°)'] = pyr.P_angle
                result[f'pyramidalization_Agranat-Radhakrishnan_{atom_label}'] = pyr.P

    except Exception as e:
        logger.error('Unable to acquire pyramidalization for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_pyramidalization(dataframe: pd.DataFrame,
                         atom_list: list[str],
                         data_dir: Path,
                         procs: int = 1):
    interesting_columns = [FILE_COLUMN_NAME] + atom_list
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_pyramidalization, zip(calculation_rows, itertools.repeat(atom_list)))

    return _merge_results(dataframe, results, 'Pyramidalization function has completed.')


def _get_float_property_from_text(text: str, pattern: re.Pattern, return_type: str = 'last'):
    matches = pattern.findall(text)
    if len(matches) == 0:
        return None
    if return_type == 'last':
        return matches[-1]
    if return_type == 'all':
        return matches
    raise ValueError(f'Unsupported return_type {return_type}')


def _get_wall_time(file: Path) -> pd.DataFrame:
    configure_logger(debug=False)
    result = {FILE_COLUMN_NAME: file.name, 'wall_time (h)': None}

    try:
        text, error = get_filecont(file, split=False)
        if error:
            logger.error('Error in _get_wall_time: %s\t%s', file.name, error)
            return pd.DataFrame(pd.Series(result)).transpose()

        all_times = _get_float_property_from_text(text=text, pattern=wall_time_pattern, return_type='all')
        if not all_times:
            logger.warning('No TOTAL RUN TIME block was found in %s', file.name)
            return pd.DataFrame(pd.Series(result)).transpose()

        days = np.sum([int(x[0]) for x in all_times])
        hours = np.sum([int(x[1]) for x in all_times])
        minutes = np.sum([int(x[2]) for x in all_times])
        seconds = np.sum([int(x[3]) for x in all_times])
        milliseconds = np.sum([int(x[4]) if x[4] else 0 for x in all_times])

        total = hours + (days * 24) + (minutes / 60.0) + (seconds / 3600.0) + (milliseconds / 3_600_000.0)
        result['wall_time (h)'] = total

    except Exception as e:
        logger.error('Unable to acquire wall_time for %s because %s', file.name, e)

    return pd.DataFrame(pd.Series(result)).transpose()


def get_wall_time(dataframe: pd.DataFrame,
                  data_dir: Path,
                  procs: int = 1):
    files = [Path(data_dir / x) for x in dataframe[FILE_COLUMN_NAME].to_list()]
    with multiprocessing.Pool(processes=procs) as p:
        results = p.map(_get_wall_time, files)
    return _merge_results(dataframe, results, 'Wall time function has completed.')


def _get_plane_angle(row: pd.Series,
                     plane_a_names: list[str],
                     plane_b_names: list[str]) -> pd.DataFrame:
    configure_logger(debug=False)

    file = Path(row[FILE_COLUMN_NAME])
    result = {f'plane_angle_{"-".join(plane_a_names)}_{"-".join(plane_b_names)} (°)': None,
              FILE_COLUMN_NAME: file.name}

    try:
        _, coordinates = get_xyz(file)
        coordinates = np.array(coordinates, dtype=float)

        plane_a_indices = [int(row[x]) - 1 for x in plane_a_names]
        plane_b_indices = [int(row[x]) - 1 for x in plane_b_names]

        a, b, c = coordinates[plane_a_indices]
        d, e, f = coordinates[plane_b_indices]

        ab = a - b
        bc = b - c
        de = d - e
        ef = e - f

        normal_a = np.cross(ab, bc)
        normal_b = np.cross(de, ef)

        cos_theta = np.clip(
            np.dot(normal_a, normal_b) / (np.linalg.norm(normal_a) * np.linalg.norm(normal_b)),
            -1.0,
            1.0
        )
        angle = float(np.degrees(np.arccos(cos_theta)))
        result[f'plane_angle_{"-".join(plane_a_names)}_{"-".join(plane_b_names)} (°)'] = angle

    except Exception as e:
        logger.error('Unable to acquire plane angle for %s because %s', file.name, e)

    return pd.DataFrame(pd.Series(result)).transpose()


def get_plane_angle(dataframe: pd.DataFrame,
                    plane_atom_names: list[list[str]],
                    data_dir: Path,
                    procs: int = 1) -> pd.DataFrame:
    if len(plane_atom_names) != 2 or not all(len(x) == 3 for x in plane_atom_names):
        raise ValueError('Plane angle requires exactly two 3-atom planes.')

    interesting_columns = [FILE_COLUMN_NAME] + list(set([x for xs in plane_atom_names for x in xs]))
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_plane_angle, zip(calculation_rows, itertools.repeat(plane_atom_names[0]), itertools.repeat(plane_atom_names[1])))

    return _merge_results(dataframe, results, 'Plane angle function has completed.')


def _get_sterimol_dbstep(row: pd.Series,
                         sterimol_list: list[list[str]]) -> pd.DataFrame:
    configure_logger(debug=False)
    row = row.copy()

    result = {}
    for atom_pair in sterimol_list:
        result[f'Sterimol_L_{atom_pair[0]}_{atom_pair[1]}(Å)_dbstep'] = None
        result[f'Sterimol_B1_{atom_pair[0]}_{atom_pair[1]}(Å)_dbstep'] = None
        result[f'Sterimol_B5_{atom_pair[0]}_{atom_pair[1]}(Å)_dbstep'] = None

    file = Path(row[FILE_COLUMN_NAME])

    try:
        if not DBSTEP_AVAILABLE:
            logger.warning('DBSTEP is not installed; DBSTEP Sterimol values for %s will be missing.', file.name)
        else:
            xyz_file = file.with_suffix('.xyz')
            for atom_pair in sterimol_list:
                atom_a = int(row[atom_pair[0]])
                atom_b = int(row[atom_pair[1]])

                sterimol_obj = db.dbstep(
                    str(xyz_file.absolute()),
                    atom1=atom_a,
                    atom2=atom_b,
                    commandline=True,
                    verbose=False,
                    quiet=True,
                    sterimol=True,
                    measure='grid'
                )

                result[f'Sterimol_L_{atom_pair[0]}_{atom_pair[1]}(Å)_dbstep'] = sterimol_obj.L
                result[f'Sterimol_B1_{atom_pair[0]}_{atom_pair[1]}(Å)_dbstep'] = sterimol_obj.Bmin
                result[f'Sterimol_B5_{atom_pair[0]}_{atom_pair[1]}(Å)_dbstep'] = sterimol_obj.Bmax

    except Exception as e:
        logger.error('Unable to acquire DBSTEP Sterimol values for %s because %s', file.name, e)

    result[FILE_COLUMN_NAME] = file.name
    return pd.DataFrame(pd.Series(result)).transpose()


def get_sterimol_dbstep(dataframe: pd.DataFrame,
                        data_dir: Path,
                        sterimol_list: list[list[str]],
                        procs: int = 1):
    interesting_columns = [FILE_COLUMN_NAME] + list(set([x for xs in sterimol_list for x in xs]))
    calculation_df = dataframe[interesting_columns].copy(deep=True)
    calculation_df[FILE_COLUMN_NAME] = [str(Path(data_dir / x).absolute()) for x in calculation_df[FILE_COLUMN_NAME].to_list()]
    calculation_rows = [x[1] for x in calculation_df.iterrows()]

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_sterimol_dbstep, zip(calculation_rows, itertools.repeat(sterimol_list)))

    return _merge_results(dataframe, results, 'Sterimol DBSTEP function has completed.')
