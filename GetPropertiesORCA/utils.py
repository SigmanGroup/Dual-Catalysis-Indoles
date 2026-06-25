#!/usr/bin/env python3
# coding: utf-8

"""
Utility functions for the ORCA-only GetProperties workflow.

Design choice:
- read molecular geometry from sibling .xyz files
- read electronic/text properties from sibling .out / .OUT files
"""

from __future__ import annotations

import sys
import logging
import subprocess
import multiprocessing
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.Draw import rdMolDraw2D

try:
    import py3Dmol
except ImportError:
    py3Dmol = None

logger = logging.getLogger(__name__)

# This is the global name of the column that defines the Path/name of the files to process.
FILE_COLUMN_NAME = 'file'


def _candidate_paths(path: Path, suffixes: list[str]) -> list[Path]:
    """Return candidate sibling paths obtained by swapping the suffix."""
    root = path.with_suffix('')
    return [root.with_suffix(s) for s in suffixes]


def _resolve_existing_path(path: str | Path,
                           suffixes_if_missing: list[str] | None = None) -> Path:
    """
    Resolve an existing path. If `path` does not exist and has no suffix, try
    the suffixes in `suffixes_if_missing`.
    """
    p = Path(path)

    if p.exists():
        return p

    if suffixes_if_missing and p.suffix == '':
        for suffix in suffixes_if_missing:
            candidate = p.with_suffix(suffix)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f'{p.absolute()} does not exist.')


def resolve_out_file(file: str | Path) -> Path:
    """
    Resolve an ORCA output file. Bare stems are accepted and will be expanded
    to `.out` or `.OUT`.
    """
    p = Path(file)

    if p.exists():
        return p

    if p.suffix.lower() == '.xyz':
        for candidate in _candidate_paths(p, ['.out', '.OUT']):
            if candidate.exists():
                return candidate

    return _resolve_existing_path(p, suffixes_if_missing=['.out', '.OUT'])


def resolve_xyz_file(file: str | Path) -> Path:
    """
    Resolve the sibling XYZ file associated with an ORCA output.
    """
    p = Path(file)
    if p.exists() and p.suffix.lower() == '.xyz':
        return p

    for candidate in _candidate_paths(p, ['.xyz', '.XYZ']):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f'Could not locate sibling XYZ file for {p.absolute()}')


def read_xyz(file: str | Path):
    """
    Read a simple XYZ file and return (elements, coordinates).
    """
    xyz = resolve_xyz_file(file)

    with open(xyz, 'r', encoding='utf-8', errors='ignore') as infile:
        lines = [line.rstrip('\n') for line in infile]

    if len(lines) < 3:
        raise ValueError(f'{xyz.name} is not a valid XYZ file.')

    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f'First line of {xyz.name} is not the atom count.') from exc

    body = [line.split() for line in lines[2:2 + natoms] if line.strip()]
    if len(body) != natoms:
        raise ValueError(f'Expected {natoms} coordinate lines in {xyz.name}, found {len(body)}.')

    elements = [parts[0] for parts in body]
    coordinates = [[float(parts[1]), float(parts[2]), float(parts[3])] for parts in body]
    return elements, coordinates


def get_xyz(file: str | Path):
    """
    Convenience wrapper used by the property-extraction code.
    """
    return read_xyz(file)


def xyz_to_mol_file(xyz: Path) -> None:
    """
    Convert an XYZ file to a Mol file using Open Babel.
    """
    output = xyz.with_suffix('.mol')
    subprocess.run(
        args=['obabel', '-ixyz', str(xyz.absolute()), '-omol', '-O', str(output.absolute())],
        cwd=xyz.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )


def xyz_to_sdf(xyz: Path) -> None:
    """
    Convert an XYZ file to an SDF file using Open Babel.
    """
    output = xyz.with_suffix('.sdf')
    subprocess.run(
        args=['obabel', '-ixyz', str(xyz.absolute()), '-osdf', '-O', str(output.absolute())],
        cwd=xyz.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )


def _paired_output_for_structure(file: Path) -> Path | None:
    """
    Find the ORCA output corresponding to a structure file.
    """
    for candidate in _candidate_paths(file, ['.out', '.OUT']):
        if candidate.exists():
            return candidate
    return None


def _get_atom_map(file: Path,
                  substructure: Chem.Mol) -> list:
    """
    Get an atom map from either a `.mol` or `.sdf` file based on a
    substructure as an RDKit Mol object.

    The first element of the returned list is the corresponding ORCA output
    filename, and the remaining elements are the 1-index atom numbers that
    matched the provided substructure.
    """
    configure_logger(debug=False)

    if file.suffix.lower() not in ['.sdf', '.mol']:
        raise ValueError(f'Atom maps can only be extracted from .sdf/.mol files not {file.name}')

    out_file = _paired_output_for_structure(file)
    if out_file is None:
        logger.error('Could not locate sibling .out/.OUT file for %s', file.absolute())
        return [file.name]

    mapping = [out_file.name]

    file, mol = _read_in_mol_sdf_with_xyz_correction(file)
    if mol is None:
        return mapping

    matches = mol.GetSubstructMatches(substructure)

    if len(matches) == 0:
        logger.warning('Found %d matching patterns for %s\t%s', len(matches), file.name, matches)
    elif len(matches) > 1:
        logger.warning('Found %d matching patterns for %s\t%s', len(matches), file.name, matches)

    for _match in matches:
        for idx in _match:
            mapping.append(int(idx + 1))

    return mapping


def mol_to_image(mol,
                 show_atom_indices: bool = True,
                 image_size: tuple[int, int] = (400, 400)) -> str:
    """
    Draw an RDKit molecule and return SVG text.
    """
    if show_atom_indices:
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx() + 1))

    d2d = rdMolDraw2D.MolDraw2DSVG(image_size[0], image_size[1])
    dopts = d2d.drawOptions()
    dopts.bondLineWidth = 5
    dopts.baseFontSize = 3.5
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def _read_in_mol_sdf_with_xyz_correction(file: Path):
    """
    Read an RDKit Mol from `.sdf` or `.mol`. If that fails, rebuild the
    structure from the sibling `.xyz` file.
    """
    if file.suffix.lower() == '.sdf':
        mol = next(Chem.SDMolSupplier(str(file.absolute()), removeHs=False), None)
    elif file.suffix.lower() == '.mol':
        mol = Chem.MolFromMolFile(str(file.absolute()), removeHs=False)
    else:
        raise TypeError(f'File {file.name} does not have a supported extension.')

    if mol is None:
        logger.debug('Could not read %s directly with RDKit. Falling back to sibling .xyz', file.name)

        try:
            xyz_file = resolve_xyz_file(file)
        except FileNotFoundError:
            logger.error('Could not locate sibling .xyz for %s', file.name)
            return file, None

        with open(xyz_file, 'r', encoding='utf-8', errors='ignore') as infile:
            xyz_block = infile.read()

        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            logger.error('Could not reconstruct %s from %s', file.name, xyz_file.name)
            return file, None

        try:
            rdDetermineBonds.DetermineConnectivity(mol)
            rdDetermineBonds.DetermineBondOrders(mol)
        except Exception as e:
            logger.warning('Could not determine bond orders for %s because %s', file.name, e)
            return file, None

    return file, mol


def convert_files_in_directory(directory: Path) -> list[Path]:
    """
    Convert `.xyz` files in a directory to `.mol` files and identify failures.
    """
    files_to_convert = [x for x in directory.glob('*.xyz') if not x.with_suffix('.mol').exists()]

    logger.info('Converting %d xyz files to .mol files.', len(files_to_convert))

    if files_to_convert:
        with multiprocessing.Pool() as pool:
            pool.map(xyz_to_mol_file, files_to_convert)

    failed_files = []
    for file in files_to_convert:
        if not file.with_suffix('.mol').exists():
            logger.error('%s does not exist.', file.with_suffix(".mol").absolute())
            failed_files.append(file)

    if failed_files:
        logger.warning('%d xyz-to-mol conversions failed.', len(failed_files))

    return failed_files


def draw_3D_mol(mol: Chem.Mol,
                viewport_size: tuple[int, int] = (400, 300)) -> None:
    """
    Draw an interactive 3D molecule in a Jupyter notebook with 1-indexed atom labels.
    """
    if py3Dmol is None:
        raise ImportError('py3Dmol is not installed in the active environment.')

    view = py3Dmol.view(width=viewport_size[0], height=viewport_size[1])
    view.addModel(Chem.MolToMolBlock(mol), 'mol')
    view.setStyle({'stick': {}})

    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        view.addLabel(
            str(idx + 1),
            {
                'position': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                'fontColor': 'black',
                'fontSize': 14,
                'showBackground': False
            }
        )

    view.zoomTo()
    view.show()


def split_compound_name(file: str | Path,
                        delimiter: str,
                        return_key: int | list[int] = 0) -> str:
    """
    Split a file stem into parts based on `delimiter`.
    """
    stem = Path(file).stem

    if isinstance(return_key, int):
        return stem.split(delimiter)[return_key]

    return f'{delimiter}'.join([stem.split(delimiter)[i] for i in return_key])


def configure_logger(debug: bool = False) -> None:
    """
    Configure logging for the current process.
    When debug=False, keep the notebook relatively clean.
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )


def get_filecont(log: str | Path,
                 split: bool = True):
    """
    Read an ORCA output file and check whether it terminated normally.
    """
    error = ''
    out = resolve_out_file(log)

    with open(out, 'r', encoding='utf-8', errors='ignore') as infile:
        text = infile.read()

    if 'ORCA TERMINATED NORMALLY' not in text:
        error = f'****Failed or incomplete jobs for {out.name}'

    if split:
        with open(out, 'r', encoding='utf-8', errors='ignore') as infile:
            return infile.readlines(), error

    return text, error


def get_specdata(atoms, prop) -> list:
    """
    Helper used by several population-analysis extractors.
    """
    propout = []
    for atom in atoms:
        if str(atom).isdigit():
            a = int(atom) - 1
            if a < len(prop):
                propout.append(float(prop[a][1]))
    return propout
