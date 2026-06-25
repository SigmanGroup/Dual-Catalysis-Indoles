#!/usr/bin/env python3
# coding: utf-8

"""
ORCA-native thermochemistry parser for the ORCA-only GetProperties workflow.

Design philosophy
-----------------
- The dataframe contains the SPC filename (e.g. *_SPC.out)
- Thermochemistry is read directly from the matching ORCA OPT/FREQ file
- Final electronic energy is read directly from the matching ORCA SPC file
- No fragile GoodVibes/cclib parsing is used for ORCA outputs

Important note on the output columns
------------------------------------
ORCA already prints QRRHO thermochemistry in the THERMOCHEMISTRY block
when "Quasi RRHO ... True" is active. Therefore:

- T*S is taken directly from ORCA
- G(T)_spc is reconstructed from ORCA thermochemistry + SPC energy
- T*qh_S is set equal to T*S
- qh_G(T)_spc is set equal to G(T)_spc

This preserves compatibility with the old notebook column names while using
ORCA's own QRRHO values directly.
"""

from __future__ import annotations

import sys
import re
import logging
import itertools
import multiprocessing
from pathlib import Path

import pandas as pd

from .utils import FILE_COLUMN_NAME

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.ERROR,
    format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
    datefmt='%m/%d/%Y:%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

for noisy_logger in [
    '',
    'ccio',
    'logfileparser',
    'cclib',
    'cclib.io',
    'cclib.parser',
]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)


def _blank_thermo_row(filename: str, temp: float) -> pd.DataFrame:
    """Return an empty/None thermochemistry row."""
    thermo_data = pd.Series({
        FILE_COLUMN_NAME: filename,
        'E_spc (Hartree)': None,
        'ZPE(Hartree)': None,
        'H_spc(Hartree)': None,
        'T*S': None,
        'T*qh_S': None,
        'G(T)_spc(Hartree)': None,
        'qh_G(T)_spc(Hartree)': None,
        'T': temp
    })
    return pd.DataFrame(thermo_data).transpose()


def _read_text(file: Path) -> str:
    """Read file text safely."""
    return file.read_text(errors='ignore')


def _has_orca_freq_thermo_content(file: Path) -> bool:
    """
    Check whether an ORCA output appears to contain real frequency/thermochemistry output.
    We validate output content, not only requested input keywords.
    """
    try:
        text = _read_text(file).lower()
    except Exception:
        return False

    freq_markers = [
        'vibrational frequencies',
        'normal modes',
    ]
    thermo_markers = [
        'thermochemistry at',
        'zero point energy',
        'final gibbs free energy',
        'final entropy term',
        'total enthalpy',
    ]

    has_freq = any(marker in text for marker in freq_markers)
    has_thermo = any(marker in text for marker in thermo_markers)

    return has_freq and has_thermo


def _candidate_freq_files_from_spc(spc_file: Path) -> list[Path]:
    """
    Build deterministic candidate freq/OPT partner files from an SPC filename.

    Examples:
    - foo_OPT_SPC.out -> foo_OPT.out
    - foo_SPC.out     -> foo_OPT.out, foo.out
    """
    parent = spc_file.parent
    stem = spc_file.stem
    suffixes = [spc_file.suffix, '.out', '.OUT']

    candidates = []
    lower_stem = stem.lower()

    if lower_stem.endswith('_opt_spc'):
        base = stem[:-4]  # remove trailing "_SPC"
        candidate_stems = [base]
    elif lower_stem.endswith('_spc'):
        base = stem[:-4]  # remove trailing "_SPC"
        candidate_stems = [f'{base}_OPT', base]
    else:
        candidate_stems = [f'{stem}_OPT', stem]

    seen = set()
    for cand_stem in candidate_stems:
        for suff in suffixes:
            cand = parent / f'{cand_stem}{suff}'
            if cand not in seen:
                candidates.append(cand)
                seen.add(cand)

    return candidates


def _find_matching_freq_file(spc_file: Path) -> Path:
    """
    Find the matching ORCA freq/thermo file for a given SPC file.

    Strategy:
    1. Try deterministic candidate names first
    2. Validate candidates by checking real freq/thermo output content
    3. If needed, do a cautious same-directory fallback search
    """
    for cand in _candidate_freq_files_from_spc(spc_file):
        if cand.exists() and cand != spc_file and _has_orca_freq_thermo_content(cand):
            return cand

    stem = spc_file.stem
    lower_stem = stem.lower()

    if lower_stem.endswith('_opt_spc'):
        prefix = stem[:-8]   # remove "_OPT_SPC"
    elif lower_stem.endswith('_spc'):
        prefix = stem[:-4]   # remove "_SPC"
    else:
        prefix = stem

    matches = []
    for pattern in ['*.out', '*.OUT']:
        for cand in spc_file.parent.glob(pattern):
            if cand == spc_file:
                continue
            cand_stem_lower = cand.stem.lower()
            if 'spc' in cand_stem_lower:
                continue
            if not cand.stem.startswith(prefix):
                continue
            if _has_orca_freq_thermo_content(cand):
                matches.append(cand)

    matches = sorted(set(matches))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f'Ambiguous freq/thermo partner for {spc_file.name}: '
            f'{", ".join(x.name for x in matches)}'
        )

    raise FileNotFoundError(
        f'Could not find a matching ORCA freq/thermo file for {spc_file.name}'
    )


def _parse_last_float(pattern: str, text: str, flags: int = re.MULTILINE) -> float:
    """
    Return the last matched float for a regex pattern with a single capturing group.
    """
    matches = re.findall(pattern, text, flags=flags)
    if not matches:
        raise ValueError(f'Pattern not found: {pattern}')
    return float(matches[-1])


def _parse_orca_main_sp_energy(spc_file: Path) -> float:
    """
    Parse ONLY the main ORCA final single-point energy.

    This intentionally matches the "plain" line:
        FINAL SINGLE POINT ENERGY      -595.946204306778

    and ignores lines like:
        FINAL SINGLE POINT ENERGY (Solute-gas-phase) ...
        FINAL SINGLE POINT ENERGY (Solute-CPCM) ...
        FINAL SINGLE POINT ENERGY (Solvent-CPCM) ...
    """
    text = _read_text(spc_file)

    pattern = r'^\s*FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)\s*$'
    matches = re.findall(pattern, text, flags=re.MULTILINE)

    if len(matches) == 0:
        raise ValueError(f'Could not find main FINAL SINGLE POINT ENERGY in {spc_file.name}')

    if len(matches) > 1:
        raise ValueError(
            f'Found multiple unqualified FINAL SINGLE POINT ENERGY lines in {spc_file.name}'
        )

    return float(matches[0])


def _parse_orca_thermochemistry(freq_file: Path) -> dict:
    """
    Parse the ORCA thermochemistry block from an OPT/FREQ output.

    Values extracted:
    - Temperature
    - Electronic energy
    - Zero point energy
    - Total Enthalpy
    - Final entropy term (T*S)
    - Final Gibbs free energy

    These values are taken directly from ORCA output. In your workflow, ORCA
    already reports QRRHO thermochemistry when "Quasi RRHO ... True" is active.
    """
    text = _read_text(freq_file)

    thermo = {
        'T': _parse_last_float(
            r'^\s*Temperature\s+\.\.\.\s+(-?\d+\.\d+)\s+K\s*$',
            text
        ),
        'E_el_opt': _parse_last_float(
            r'^\s*Electronic energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh\s*$',
            text
        ),
        'ZPE': _parse_last_float(
            r'^\s*Zero point energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh\b',
            text
        ),
        'H_opt': _parse_last_float(
            r'^\s*Total Enthalpy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh\s*$',
            text
        ),
        'TS': _parse_last_float(
            r'^\s*Final entropy term\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh\b',
            text
        ),
        'G_opt': _parse_last_float(
            r'^\s*Final Gibbs free energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh\s*$',
            text
        ),
    }

    return thermo


def _get_orca_thermo_data(spc_logfile: Path | str,
                          temp: float = 298.15) -> pd.DataFrame:
    """
    Get ORCA thermochemistry using:
    - matching freq/OPT file for ZPE/H/S/G thermal data
    - matching SPC file for final electronic energy

    Reconstructed quantities:
    - H_spc = E_spc + (H_opt - E_el_opt)
    - G_spc = E_spc + (G_opt - E_el_opt)

    Since ORCA already reports QRRHO thermochemistry in this workflow:
    - T*qh_S is set equal to T*S
    - qh_G(T)_spc is set equal to G(T)_spc
    """
    try:
        spc_file = Path(spc_logfile)

        if not spc_file.exists():
            raise FileNotFoundError(f'SPC file not found: {spc_file}')

        freq_file = _find_matching_freq_file(spc_file)
        thermo = _parse_orca_thermochemistry(freq_file)
        e_spc = _parse_orca_main_sp_energy(spc_file)

        h_spc = e_spc + (thermo['H_opt'] - thermo['E_el_opt'])
        g_spc = e_spc + (thermo['G_opt'] - thermo['E_el_opt'])

        thermo_data = pd.Series({
            FILE_COLUMN_NAME: spc_file.name,
            'E_spc (Hartree)': e_spc,
            'ZPE(Hartree)': thermo['ZPE'],
            'H_spc(Hartree)': h_spc,
            'T*S': thermo['TS'],
            'T*qh_S': thermo['TS'],
            'G(T)_spc(Hartree)': g_spc,
            'qh_G(T)_spc(Hartree)': g_spc,
            'T': thermo.get('T', temp)
        })

        return pd.DataFrame(thermo_data).transpose()

    except Exception as exc:
        logger.error(
            'Could not get ORCA thermochemical data for %s because %s',
            Path(spc_logfile).name,
            exc
        )
        return _blank_thermo_row(Path(spc_logfile).name, temp)


def get_goodvibes_data(dataframe: pd.DataFrame,
                       data_dir: Path,
                       temp: float = 298.15,
                       procs: int = 1) -> pd.DataFrame:
    """
    Backward-compatible public function name.

    Even though the name is retained for notebook compatibility, this function
    no longer relies on GoodVibes parsing for ORCA outputs. Instead it uses a
    direct ORCA parser.
    """
    files = [Path(data_dir) / x for x in dataframe[FILE_COLUMN_NAME].to_list()]
    args = zip(files, itertools.repeat(temp))

    with multiprocessing.Pool(processes=procs) as pool:
        results = pool.starmap(_get_orca_thermo_data, args)

    results = pd.concat(results)

    results.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)
    dataframe = dataframe.copy()
    dataframe.set_index(FILE_COLUMN_NAME, inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    return dataframe


if __name__ == "__main__":
    pass
