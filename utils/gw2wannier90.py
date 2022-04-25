#!/usr/bin/env python3
#
# gw2wannier90 interface
#
# This file is distributed as part of the Wannier90 code and
# under the terms of the GNU General Public License. See the
# file `LICENSE' in the root directory of the Wannier90
# distribution, or http://www.gnu.org/copyleft/gpl.txt
#
# The webpage of the Wannier90 code is www.wannier.org
#
# The Wannier90 code is hosted on GitHub:
#
# https://github.com/wannier-developers/wannier90
#
# Designed and tested with: Quantum Espresso and Yambo
# This interface should work with any G0W0 code
# Originally written by Stepan Tsirkin
# Extended, developed and documented by Antimo Marrazzo
#
# Updated on February 19th, 2017 by Antimo Marrazzo (antimo.marrazzo@epfl.ch)
# Updated on October 7th, 2019 by Junfeng Qiao (qiaojunfeng@outlook.com)
#
import argparse
from dataclasses import dataclass
import datetime
import glob
import os
import shutil
import subprocess

import numpy as np
from scipy.io import FortranFile


def parse_args(args=None):

    parser = argparse.ArgumentParser(
        description=r"""### gw2wannier90 interface ###

Usage: gw2wannier90.py seedname options

Options can be:
  mmn, amn, spn, unk, uhu, uiu,
  spn_formatted, unk_formatted, uhu_formatted, uiu_formatted,
  write_formatted

If no options are specified, all the files are considered.

Be careful with unformatted files, they are compiler-dependent.
A safer choice is to use (bigger) formatted files, with options:
  spn_formatted, uiu_formatted, uhu_formatted, unk_formatted

In default, the output format is the same as the input format.
To generate formatted files with unformatted input, use option:
  write_formatted
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "seedname",
        metavar="seedname",
        type=str,
        help="Seedname of Wannier90 files.",
    )
    parser.add_argument(
        "-o",
        "--output_seedname",
        type=str,
        help="The seedname of output files. Default is input_seedname.gw",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        help=(
            "Comma separated list of file extensions to be converted, "
            "e.g. `-e amn,mmn` will only convert seedname.amn and seedname.mmn files. "
            "If nothing provided, all files will be converted."
        ),
    )
    parser.add_argument(
        "--no_sort",
        action="store_true",
        help="No sorting, only add GW corrections to eig.",
    )

    parsed_args = parser.parse_args(args)

    return parsed_args


def get_path_to_executable(executable: str) -> str:
    """Get path to local executable.
    :param executable: Name of executable in the $PATH variable
    :type executable: str
    :return: path to executable
    :rtype: str
    """
    path = shutil.which(executable)
    if path is None:
        raise ValueError(f"'{executable}' executable not found in PATH.")
    return path


@dataclass
class Chk:
    """Class for storing matrices in seedname.chk file."""

    header: str = None
    num_bands: int = None
    num_exclude_bands: int = None
    exclude_bands: np.ndarray = None
    real_lattice: np.ndarray = None
    recip_lattice: np.ndarray = None
    num_kpts: int = None
    mp_grid: list = None
    kpt_latt: np.ndarray = None
    nntot: int = None
    num_wann: int = None
    checkpoint: str = None
    have_disentangled: bool = None
    omega_invariant: float = None
    lwindow: np.ndarray = None
    ndimwin: np.ndarray = None
    u_matrix_opt: np.ndarray = None
    u_matrix: np.ndarray = None
    m_matrix: np.ndarray = None
    wannier_centres: np.ndarray = None
    wannier_spreads: np.ndarray = None

    def __eq__(self, other):
        if not isinstance(other, Chk):
            return NotImplemented(f"comparing {self} {other}")

        if other is self:
            return True

        eq = True

        eq = self.header == other.header
        if not eq:
            return False

        eq = self.num_bands == other.num_bands
        if not eq:
            return False

        eq = self.num_exclude_bands == other.num_exclude_bands
        if not eq:
            return False

        eq = np.allclose(self.exclude_bands, other.exclude_bands)
        if not eq:
            return False

        eq = np.allclose(self.real_lattice, other.real_lattice)
        if not eq:
            return False

        eq = np.allclose(self.recip_lattice, other.recip_lattice)
        if not eq:
            return False

        eq = self.num_kpts == other.num_kpts
        if not eq:
            return False

        eq = np.allclose(self.mp_grid, other.mp_grid)
        if not eq:
            return False

        eq = np.allclose(self.kpt_latt, other.kpt_latt)
        if not eq:
            return False

        eq = self.nntot == other.nntot
        if not eq:
            return False

        eq = self.num_wann == other.num_wann
        if not eq:
            return False

        eq = self.checkpoint == other.checkpoint
        if not eq:
            return False

        eq = self.have_disentangled == other.have_disentangled
        if not eq:
            return False

        eq = np.allclose(self.omega_invariant, other.omega_invariant)
        if not eq:
            return False

        eq = np.allclose(self.lwindow, other.lwindow)
        if not eq:
            return False

        eq = np.allclose(self.ndimwin, other.ndimwin)
        if not eq:
            return False

        eq = np.allclose(self.u_matrix_opt, other.u_matrix_opt)
        if not eq:
            return False

        eq = np.allclose(self.u_matrix, other.u_matrix)
        if not eq:
            return False

        eq = np.allclose(self.m_matrix, other.m_matrix)
        if not eq:
            return False

        eq = np.allclose(self.wannier_centres, other.wannier_centres)
        if not eq:
            return False

        eq = np.allclose(self.wannier_spreads, other.wannier_spreads)
        if not eq:
            return False

        return True


def read_chk(filename: str, formatted: bool = None, keep_temp: bool = False) -> Chk:
    """Read seedname.chk file.

    :param filename: filename
    :type filename: str
    :param formatted: defaults to None, auto detect from filename.
    :type formatted: bool, optional
    :param keep_temp: for unformatted file, creat a tempdir and run w90chk2chk.x
    in it. If True, do not remove this tempdir. Defaults to False
    :type keep_temp: bool, optional
    :return: Chk
    :rtype: Chk
    """
    import pathlib
    import tempfile

    chk = Chk()

    # From str to pathlib.Path
    filename = pathlib.Path(filename)

    if formatted is None:
        if filename.name.endswith(".chk"):
            formatted = False
        elif filename.name.endswith(".chk.fmt"):
            formatted = True
        else:
            raise ValueError(f"Cannot detect the format of {filename}")

    valid_exts = [".chk", ".chk.fmt"]
    for ext in valid_exts:
        if filename.name.endswith(ext):
            seedname = filename.name[: -len(ext)]
            break
    else:
        raise ValueError(f"{filename} not ends with {valid_exts}?")

    if not formatted:
        w90chk2chk = get_path_to_executable("w90chk2chk.x")
        tmpdir = pathlib.Path(tempfile.mkdtemp(dir="."))
        # cd tmpdir so that `w90chk2chk.log` is inside tmpdir
        os.chdir(tmpdir)
        if filename.root == "/":
            os.symlink(filename, filename.name)
        else:
            os.symlink(pathlib.Path("..") / filename, filename.name)
        call_args = [w90chk2chk, "-export", str(seedname)]
        # Some times need mpirun -n 1
        # call_args = ['mpirun', '-n', '1'] + call_args
        subprocess.check_call(call_args)
        os.chdir("..")
        filename_fmt = f"{tmpdir / filename.name}.fmt"
    else:
        filename_fmt = filename

    # Read formatted chk file
    with open(filename_fmt) as handle:
        #
        chk.header = handle.readline().strip()
        #
        chk.num_bands = int(handle.readline().strip())
        #
        chk.num_exclude_bands = int(handle.readline().strip())
        #
        chk.exclude_bands = np.zeros(chk.num_exclude_bands, dtype=int)
        #
        if chk.num_exclude_bands > 0:
            # line = handle.readline().strip().split()
            # chk.exclude_bands[:] = [int(_) for _ in line]
            for i in range(chk.num_exclude_bands):
                line = handle.readline().strip()
                chk.exclude_bands[i] = int(line)
        # Just store as a 1D array
        chk.real_lattice = np.zeros(9)
        line = handle.readline().strip().split()
        chk.real_lattice[:] = [float(_) for _ in line]
        #
        chk.recip_lattice = np.zeros(9)
        line = handle.readline().strip().split()
        chk.recip_lattice[:] = [float(_) for _ in line]
        #
        chk.num_kpts = int(handle.readline().strip())
        #
        chk.mp_grid = [int(_) for _ in handle.readline().strip().split()]
        #
        chk.kpt_latt = np.zeros((3, chk.num_kpts))
        for ik in range(chk.num_kpts):
            chk.kpt_latt[:, ik] = [float(_) for _ in handle.readline().strip().split()]
        #
        chk.nntot = int(handle.readline().strip())
        #
        chk.num_wann = int(handle.readline().strip())
        #
        chk.checkpoint = handle.readline().strip()
        # 1 -> True, 0 -> False
        chk.have_disentangled = bool(handle.readline().strip())
        if chk.have_disentangled:
            #
            chk.omega_invariant = float(handle.readline().strip())
            #
            chk.lwindow = np.zeros((chk.num_bands, chk.num_kpts), dtype=bool)
            for ik in range(chk.num_kpts):
                for ib in range(chk.num_bands):
                    # 1 -> True, 0 -> False
                    chk.lwindow[ib, ik] = bool(int(handle.readline().strip()))
            #
            chk.ndimwin = np.zeros(chk.num_kpts, dtype=int)
            for ik in range(chk.num_kpts):
                chk.ndimwin[ik] = int(handle.readline().strip())
            #
            chk.u_matrix_opt = np.zeros(
                (chk.num_bands, chk.num_wann, chk.num_kpts), dtype=complex
            )
            for ik in range(chk.num_kpts):
                for iw in range(chk.num_wann):
                    for ib in range(chk.num_bands):
                        line = [float(_) for _ in handle.readline().strip().split()]
                        chk.u_matrix_opt[ib, iw, ik] = line[0] + 1j * line[1]
        #
        chk.u_matrix = np.zeros(
            (chk.num_wann, chk.num_wann, chk.num_kpts), dtype=complex
        )
        for ik in range(chk.num_kpts):
            for iw in range(chk.num_wann):
                for ib in range(chk.num_wann):
                    line = [float(_) for _ in handle.readline().strip().split()]
                    chk.u_matrix[ib, iw, ik] = line[0] + 1j * line[1]
        #
        chk.m_matrix = np.zeros(
            (chk.num_wann, chk.num_wann, chk.nntot, chk.num_kpts), dtype=complex
        )
        for ik in range(chk.num_kpts):
            for inn in range(chk.nntot):
                for iw in range(chk.num_wann):
                    for ib in range(chk.num_wann):
                        line = [float(_) for _ in handle.readline().strip().split()]
                        chk.m_matrix[ib, iw, inn, ik] = line[0] + 1j * line[1]
        #
        chk.wannier_centres = np.zeros((3, chk.num_wann), dtype=float)
        for iw in range(chk.num_wann):
            chk.wannier_centres[:, iw] = [
                float(_) for _ in handle.readline().strip().split()
            ]
        #
        chk.wannier_spreads = np.zeros(chk.num_wann, dtype=float)
        for iw in range(chk.num_wann):
            chk.wannier_spreads[iw] = float(handle.readline().strip())

    # Read binary chk file, however its compiler dependent,
    # and it seems scipy.io.FortranFile cannot handle bool type?
    #
    #     handle = FortranFile(filename, "r")
    #     #
    #     chk.header = b"".join(handle.read_record(dtype="c"))
    #     #
    #     chk.num_bands = handle.read_record(dtype=np.int32).item()
    #     #
    #     chk.num_exclude_bands = handle.read_record(dtype=np.int32).item()
    #     #
    #     chk.exclude_bands = np.zeros(chk.num_exclude_bands, dtype=int)
    #     #
    #     if chk.num_exclude_bands > 0:
    #         line = handle.read_record(dtype=np.int32).reshape(chk.num_exclude_bands)
    #         chk.exclude_bands[:] = line[:]
    #     else:
    #         # read empty record
    #         handle.read_record(dtype=np.int32)
    #     # Just store as a 1D array
    #     chk.real_lattice = np.zeros(9)
    #     line = handle.read_record(dtype=np.float64).reshape(9)
    #     chk.real_lattice[:] = line[:]
    #     #
    #     chk.recip_lattice = np.zeros(9)
    #     line = handle.read_record(dtype=float).reshape(9)
    #     chk.recip_lattice[:] = line[:]
    #     #
    #     chk.num_kpts = handle.read_record(dtype=np.int32).item()
    #     #
    #     chk.mp_grid = handle.read_record(dtype=np.int32).reshape(3).tolist()
    #     #
    #     chk.kpt_latt = np.zeros((3, chk.num_kpts))
    #     line = handle.read_record(dtype=float).reshape((3, chk.num_kpts), order='F')
    #     chk.kpt_latt[:, :] = line[:, :]
    #     #
    #     chk.nntot = handle.read_record(dtype=np.int32).item()
    #     #
    #     chk.num_wann = handle.read_record(dtype=np.int32).item()
    #     #
    #     chk.checkpoint = b"".join(handle.read_record(dtype="c"))
    #     # 1 -> True, 0 -> False
    #     chk.have_disentangled = bool(handle.read_record(dtype=np.int32))
    #     if chk.have_disentangled:
    #         #
    #         chk.omega_invariant = handle.read_record(dtype=float).item()
    #         #
    #         chk.lwindow = np.zeros((chk.num_bands, chk.num_kpts), dtype=bool)
    #         line = handle.read_record(dtype=np.int32)
    #         line = line.reshape((chk.num_bands, chk.num_kpts), order='F')
    #         chk.lwindow[:, :] = line[:, :]
    #         #
    #         chk.ndimwin = np.array(chk.num_kpts, dtype=int)
    #         line = handle.read_record(dtype=int).reshape(chk.num_kpts)
    #         chk.ndimwin[:] = line[:]
    #         #
    #         chk.u_matrix_opt = np.array((chk.num_bands, chk.num_wann, chk.num_kpts), dtype=complex)
    #         line = handle.read_record(dtype=complex).reshape((chk.num_bands, chk.num_wann, chk.num_kpts), order='F')
    #         chk.u_matrix_opt[:, :, :] = line[:, :, :]
    #     #
    #     chk.u_matrix = np.array((chk.num_wann, chk.num_wann, chk.num_kpts), dtype=complex)
    #     line = handle.read_record(dtype=complex).reshape((chk.num_wann, chk.num_wann, chk.num_kpts), order='F')
    #     chk.u_matrix[:, :, :] = line[:, :, :]
    #     #
    #     chk.m_matrix = np.array((chk.num_wann, chk.num_wann, chk.nntot, chk.num_kpts), dtype=complex)
    #     line = handle.read_record(dtype=complex).reshape((chk.num_wann, chk.num_wann, chk.nntot, chk.num_kpts), order='F')
    #     chk.m_matrix[:, :, :, :] = line[:, :, :, :]
    #     #
    #     chk.wannier_centres = np.array((3, chk.num_wann), dtype=float)
    #     line = handle.read_record(dtype=float).reshape((3, chk.num_wann), order='F')
    #     chk.wannier_centres[:, :] = line[:, :]
    #     #
    #     chk.wannier_spreads = np.array(chk.num_wann, dtype=float)
    #     line = handle.read_record(dtype=float).reshape(chk.num_wann)
    #     chk.wannier_spreads[:] = line[:]
    #     #
    #     handle.close()

    if not formatted:
        if not keep_temp:
            shutil.rmtree(tmpdir)

    return chk


def write_chk(
    chk: Chk, filename: str, formatted: bool = None, keep_temp: bool = False
) -> None:
    """Write chk file.

    :param chk: _description_
    :type chk: Chk
    :param filename: output filename
    :type filename: str
    :param formatted: defaults to None, i.e. auto detect by filename
    :type formatted: bool, optional
    :param keep_temp: _description_, defaults to False
    :type keep_temp: bool, optional
    """
    import pathlib
    import tempfile

    # From str to pathlib.Path
    filename = pathlib.Path(filename)

    if formatted is None:
        if filename.name.endswith(".chk"):
            formatted = False
        elif filename.name.endswith(".chk.fmt"):
            formatted = True
        else:
            raise ValueError(f"Cannot detect the format of {filename}")

    valid_exts = [".chk", ".chk.fmt"]
    for ext in valid_exts:
        if filename.name.endswith(ext):
            seedname = filename.name[: -len(ext)]
            break
    else:
        raise ValueError(f"{filename} not ends with {valid_exts}?")

    if not formatted:
        tmpdir = pathlib.Path(tempfile.mkdtemp(dir="."))
        filename_fmt = f"{tmpdir / filename.name}.fmt"
    else:
        filename_fmt = filename

    # Write formatted chk file
    with open(filename_fmt, "w") as handle:
        #
        handle.write(f"{chk.header}\n")
        #
        handle.write(f"{chk.num_bands}\n")
        #
        handle.write(f"{chk.num_exclude_bands}\n")
        #
        if chk.num_exclude_bands > 0:
            # line = " ".join([str(_) for _ in chk.exclude_bands])
            # handle.write(f"{line}\n")
            for i in range(chk.num_exclude_bands):
                line = f"{chk.exclude_bands[i]}"
                handle.write(f"{line}\n")
        # Just store as a 1D array
        line = " ".join([f"{_:22.16f}" for _ in chk.real_lattice])
        handle.write(f"{line}\n")
        #
        line = " ".join([f"{_:22.16f}" for _ in chk.recip_lattice])
        handle.write(f"{line}\n")
        #
        handle.write(f"{chk.num_kpts}\n")
        #
        line = " ".join([f"{_}" for _ in chk.mp_grid])
        handle.write(f"{line}\n")
        #
        for ik in range(chk.num_kpts):
            line = " ".join([f"{_:22.16f}" for _ in chk.kpt_latt[:, ik]])
            handle.write(f"{line}\n")
        #
        handle.write(f"{chk.nntot}\n")
        #
        handle.write(f"{chk.num_wann}\n")
        #
        handle.write(f"{chk.checkpoint}\n")
        # 1 -> True, 0 -> False
        line = 1 if chk.have_disentangled else 0
        handle.write(f"{line}\n")
        if chk.have_disentangled:
            #
            handle.write(f"{chk.omega_invariant:22.16f}\n")
            #
            for ik in range(chk.num_kpts):
                for ib in range(chk.num_bands):
                    # 1 -> True, 0 -> False
                    line = 1 if chk.lwindow[ib, ik] else 0
                    handle.write(f"{line}\n")
            #
            for ik in range(chk.num_kpts):
                handle.write(f"{chk.ndimwin[ik]}\n")
            #
            for ik in range(chk.num_kpts):
                for iw in range(chk.num_wann):
                    for ib in range(chk.num_bands):
                        line = chk.u_matrix_opt[ib, iw, ik]
                        line = " ".join([f"{_:22.16f}" for _ in [line.real, line.imag]])
                        handle.write(f"{line}\n")
        #
        for ik in range(chk.num_kpts):
            for iw in range(chk.num_wann):
                for ib in range(chk.num_wann):
                    line = chk.u_matrix[ib, iw, ik]
                    line = " ".join([f"{_:22.16f}" for _ in [line.real, line.imag]])
                    handle.write(f"{line}\n")
        #
        for ik in range(chk.num_kpts):
            for inn in range(chk.nntot):
                for iw in range(chk.num_wann):
                    for ib in range(chk.num_wann):
                        line = chk.m_matrix[ib, iw, inn, ik]
                        line = " ".join([f"{_:22.16f}" for _ in [line.real, line.imag]])
                        handle.write(f"{line}\n")
        #
        for iw in range(chk.num_wann):
            line = " ".join([f"{_:22.16f}" for _ in chk.wannier_centres[:, iw]])
            handle.write(f"{line}\n")
        #
        for iw in range(chk.num_wann):
            line = f"{chk.wannier_spreads[iw]:22.16f}"
            handle.write(f"{line}\n")

    if not formatted:
        w90chk2chk = get_path_to_executable("w90chk2chk.x")
        # cd tmpdir so that `w90chk2chk.log` is inside tmpdir
        os.chdir(tmpdir)
        call_args = [w90chk2chk, "-import", str(seedname)]
        # Some times need mpirun -n 1
        # call_args = ['mpirun', '-n', '1'] + call_args
        subprocess.check_call(call_args)
        os.chdir("..")
        shutil.copy(tmpdir / f"{seedname}.chk", filename.name)

        if not keep_temp:
            shutil.rmtree(tmpdir)


def reorder_chk(seedname_in: str, seedname_out: str, bandsort: np.ndarray) -> None:
    print("----------\n CHK module  \n----------")
    filename_in = f"{seedname_in}.chk"
    filename_out = f"{seedname_out}.chk"

    if not os.path.exists(filename_in):
        print(f"WARNING: {filename_out} not written")
        return

    chk = read_chk(filename_in, formatted=False)

    # if chk.num_exclude_bands > 0:
    #     # chk.exclude_bands =
    #     # raise NotImplementedError("does not support exclude bands")

    if chk.have_disentangled:
        for ik in range(chk.num_kpts):
            chk.lwindow[:, ik] = chk.lwindow[bandsort[ik], ik]
            chk.u_matrix_opt[:, :, ik] = chk.u_matrix_opt[bandsort[ik], :, ik]
    else:
        chk.u_matrix[:, :, ik] = chk.u_matrix[bandsort[ik], :, ik]

    write_chk(chk, filename_out, formatted=False)

    print("----------\n CHK  - OK \n----------\n")


def _test_chk():
    import os
    import pathlib

    from gw2wannier90 import read_chk, write_chk

    PATH = os.environ["PATH"]
    w90_path = "/home/jqiao/git/wannier90"
    os.environ["PATH"] = f"{w90_path}:{PATH}"

    LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", "")
    mkl_path = "/opt/intel/oneapi/mpi/2021.4.0/libfabric/lib:/opt/intel/oneapi/mpi/2021.4.0/lib/release:/opt/intel/oneapi/mpi/2021.4.0/lib:/opt/intel/oneapi/mkl/2021.4.0/lib/intel64:/opt/intel/oneapi/compiler/2021.4.0/linux/lib:/opt/intel/oneapi/compiler/2021.4.0/linux/lib/x64:/opt/intel/oneapi/compiler/2021.4.0/linux/lib/emu:/opt/intel/oneapi/compiler/2021.4.0/linux/compiler/lib/intel64_lin"
    os.environ["LD_LIBRARY_PATH"] = f"{mkl_path}:{LD_LIBRARY_PATH}"

    curdir = pathlib.Path(__file__).parent

    chk = read_chk(curdir / "read_chk/silicon.chk")

    write_chk(chk, curdir / "osilicon.chk.fmt", formatted=True)

    chk2 = read_chk(curdir / "osilicon.chk.fmt")

    write_chk(chk2, curdir / "osilicon.chk")

    print(chk == chk2)


def gw2wannier90(
    seedname: str, seednameGW: str, targets: list, no_sort: bool = False
) -> None:
    print("------------------------------")
    print("##############################")
    print("### gw2wannier90 interface ###")
    print("##############################")
    print(f"Started on {datetime.datetime.now()}")

    # In case of formatted spn, uIu, uHu and UNK (mmn, amn, eig are formatted by default)
    # NB: Formatted output is strongly reccommended! Fortran binaries are compilers dependent.
    SPNformatted = "spn_formatted" in targets
    UIUformatted = "uiu_formatted" in targets
    UHUformatted = "uhu_formatted" in targets
    UNKformatted = "unk_formatted" in targets
    write_formatted = "write_formatted" in targets

    if set(targets).intersection({"spn", "uhu", "mmn", "amn", "unk", "uiu", "chk"}):
        calcAMN = "amn" in targets
        calcMMN = "mmn" in targets
        calcUHU = "uhu" in targets
        calcUIU = "uiu" in targets
        calcSPN = "spn" in targets
        calcUNK = "unk" in targets
        calcCHK = "chk" in targets
    else:
        calcAMN = True
        calcMMN = True
        calcUHU = True
        calcUIU = True
        calcSPN = True
        calcUNK = True
        calcCHK = True

    if calcUHU:
        calcMMN = True
    if calcUIU:
        calcMMN = True

    if no_sort:
        calcAMN = False
        calcMMN = False
        calcUHU = False
        calcUIU = False
        calcSPN = False
        calcUNK = False
        calcCHK = False

    # Here we open a file to dump all the intermediate steps (mainly for debugging)
    f_raw = open(seednameGW + ".gw2wannier90.raw", "w")
    # Opening seedname.nnkp file
    f = open(seedname + ".nnkp")
    # It copies the seedname.win for GW, we should make this optional
    # shutil.copy(seedname+".win",seednameGW+".win")
    while True:
        s = f.readline()
        if "begin kpoints" in s:
            break
    NKPT = int(f.readline())
    print("Kpoints number:", NKPT)
    n1 = np.array(NKPT, dtype=int)
    IKP = [
        tuple(
            np.array(
                np.round(np.array(f.readline().split(), dtype=float) * n1), dtype=int
            )
        )
        for i in range(NKPT)
    ]

    while True:
        s = f.readline()
        if "begin nnkpts" in s:
            break
    NNB = int(f.readline())

    KPNB = np.array(
        [
            [int(f.readline().split()[1]) - 1 for inb in range(NNB)]
            for ikpt in range(NKPT)
        ]
    )

    while True:
        s = f.readline()
        if "begin exclude_bands" in s:
            break
    exbands = np.array(f.readline().split(), dtype=int)
    if len(exbands) > 1 or exbands[0] != 0:
        print(
            "Exclude bands option is used: be careful to be consistent "
            "with the choice of bands for the GW QP corrections."
        )
        nexbands = exbands[0]
        exbands = np.zeros(nexbands, dtype=int)
        for i in range(nexbands):
            exbands[i] = int(f.readline().strip())
        # 0-based indexing
        exbands -= 1
    else:
        exbands = np.array([], dtype=int)

    eigenDFT = np.loadtxt(seedname + ".eig")
    nk = int(eigenDFT[:, 1].max())
    assert nk == NKPT
    nbndDFT = int(eigenDFT[:, 0].max())
    eigenDFT = eigenDFT[:, 2].reshape(NKPT, nbndDFT, order="C")
    # print(eigenDFT)
    f_raw.write("------------------------------\n")
    f_raw.write("Writing DFT eigenvalues\n")
    for line in eigenDFT:
        f_raw.write(str(line) + "\n")
    f_raw.write("------------------------------\n")

    corrections = np.loadtxt(seedname + ".gw.unsorted.eig")
    # Indexing with dict is too slow, use np.array instead.
    # corrections = {(int(l[1]) - 1, int(l[0]) - 1): l[2] for l in corrections}
    # print(corrections)
    corrections_val = np.zeros((nk, nbndDFT + len(exbands)))
    corrections_mask = np.zeros_like(corrections_val, dtype=bool)
    idx_b = corrections[:, 0].astype(int) - 1
    idx_k = corrections[:, 1].astype(int) - 1
    corrections_val[idx_k, idx_b] = corrections[:, 2]
    corrections_mask[idx_k, idx_b] = True
    # Strip excluded bands
    if len(exbands) > 0:
        corrections_val = np.delete(corrections_val, exbands, axis=1)
        corrections_mask = np.delete(corrections_mask, exbands, axis=1)
    print("G0W0 QP corrections read from ", seedname + ".gw.unsorted.eig")

    # providedGW = [
    #     ib
    #     for ib in range(nbndDFT)
    #     if all((ik, ib) in list(corrections.keys()) for ik in range(NKPT))
    # ]
    providedGW = [ib for ib in range(nbndDFT) if np.all(corrections_mask[:, ib])]
    # print(providedGW)
    f_raw.write("------------------------------\n")
    f_raw.write("List of provided GW corrections (bands indexes)\n")
    f_raw.write(str(providedGW) + "\n")
    f_raw.write("------------------------------\n")
    NBND = len(providedGW)
    print("Adding GW QP corrections to KS eigenvalues")
    # eigenDE = np.array(
    #     [[corrections[(ik, ib)] for ib in providedGW] for ik in range(NKPT)]
    # )
    # eigenDFTGW = np.array(
    #     [
    #         [eigenDFT[ik, ib] + corrections[(ik, ib)] for ib in providedGW]
    #         for ik in range(NKPT)
    #     ]
    # )
    eigenDE = corrections_val[:, providedGW]
    eigenDFTGW = eigenDFT[:, providedGW] + eigenDE

    f_raw.write("------------------------------\n")
    f_raw.write("Writing GW eigenvalues unsorted (KS + QP correction)\n")
    for line in eigenDFTGW:
        f_raw.write(str(line) + "\n")
    f_raw.write("------------------------------\n")

    if no_sort:
        print("No sorting")
    else:
        print("Sorting")
    bsort = np.array([np.argsort(eigenDFTGW[ik, :]) for ik in range(NKPT)])

    # Even if no_sort, I still output sorting list for reference
    f_raw.write("------------------------------\n")
    f_raw.write("Writing sorting list\n")
    for line in bsort:
        f_raw.write(str(line) + "\n")
    f_raw.write("------------------------------\n")

    if not no_sort:
        eigenDE = np.array([eigenDE[ik][bsort[ik]] for ik in range(NKPT)])
        eigenDFTGW = np.array([eigenDFTGW[ik][bsort[ik]] for ik in range(NKPT)])
        BANDSORT = np.array([np.array(providedGW)[bsort[ik]] for ik in range(NKPT)])

        f_raw.write("------------------------------\n")
        f_raw.write("Writing sorted GW eigenvalues\n")
        for line in eigenDFTGW:
            f_raw.write(str(line) + "\n")
        f_raw.write("------------------------------\n")

        print("GW eigenvalues sorted")

    # print eigenDFT
    print("------------------------------")
    print("writing " + seednameGW + ".eig")
    feig_out = open(seednameGW + ".eig", "w")
    for ik in range(NKPT):
        for ib in range(NBND):
            feig_out.write(f" {ib + 1:4d} {ik + 1:4d} {eigenDFTGW[ik, ib]:17.12f}\n")
    feig_out.close()
    print(seednameGW + ".eig", " written.")
    print("------------------------------\n")

    if calcAMN:
        try:
            print("----------\n AMN module  \n----------")
            f_amn_out = open(seednameGW + ".amn", "w")
            f_amn_in = open(seedname + ".amn")
            s = f_amn_in.readline().strip()
            print(s)
            f_amn_out.write(
                "{}, sorted by GW quasi-particle energies on {} \n".format(
                    s, datetime.datetime.now().isoformat()
                )
            )
            s = f_amn_in.readline()
            nb, nk, npr = np.array(s.split(), dtype=int)
            assert nk == NKPT
            assert nb == nbndDFT
            f_amn_out.write(f"  {NBND}   {nk}    {npr}   \n")

            AMN = np.loadtxt(f_amn_in, dtype=float)[:, 3:5]
            AMN = np.reshape(AMN[:, 0] + AMN[:, 1] * 1j, (nb, npr, nk), order="F")
            for ik in range(nk):
                amn = AMN[BANDSORT[ik], :, ik]
                for ipr in range(npr):
                    for ib in range(NBND):
                        f_amn_out.write(
                            " {:4d} {:4d} {:4d}  {:16.12f}  {:16.12f}\n".format(
                                ib + 1,
                                ipr + 1,
                                ik + 1,
                                amn[ib, ipr].real,
                                amn[ib, ipr].imag,
                            )
                        )
            f_amn_in.close()
            f_amn_out.close()
            print("----------\n AMN  - OK \n----------\n")
        except OSError as err:
            print(f"WARNING: {seednameGW}.amn not written : ", err)

    if calcMMN:
        try:
            print("----------\n MMN module  \n----------")

            f_mmn_out = open(os.path.join(seednameGW + ".mmn"), "w")
            f_mmn_in = open(os.path.join(seedname + ".mmn"))

            s = f_mmn_in.readline().strip()
            print(s)
            f_mmn_out.write(
                "{}, sorted by GW quasi-particle energies on {} \n".format(
                    s, datetime.datetime.now().isoformat()
                )
            )
            s = f_mmn_in.readline()
            nb, nk, nnb = np.array(s.split(), dtype=int)
            assert nb == nbndDFT
            assert nk == NKPT
            f_mmn_out.write(f"    {NBND}   {nk}    {nnb} \n")

            MMN = np.zeros((nk, nnb, NBND, NBND), dtype=complex)
            for ik in range(nk):
                for ib in range(nnb):
                    s = f_mmn_in.readline()
                    f_mmn_out.write(s)
                    ik1, ik2 = (int(i) - 1 for i in s.split()[:2])
                    assert ik == ik1
                    assert KPNB[ik][ib] == ik2
                    tmp = np.array(
                        [
                            [f_mmn_in.readline().split() for m in range(nb)]
                            for n in range(nb)
                        ],
                        dtype=str,
                    )
                    tmp = np.array(
                        tmp[BANDSORT[ik2], :, :][:, BANDSORT[ik1], :], dtype=float
                    )
                    tmp = (tmp[:, :, 0] + 1j * tmp[:, :, 1]).T
                    MMN[ik, ib, :, :] = tmp
                    for n in range(NBND):
                        for m in range(NBND):
                            f_mmn_out.write(
                                "  {:16.12f}  {:16.12f}\n".format(
                                    tmp[m, n].real, tmp[m, n].imag
                                )
                            )
            print("----------\n MMN OK  \n----------\n")
        except OSError as err:
            print(f"WARNING: {seednameGW}.mmn not written : ", err)
            if calcUHU:
                print(f"WARNING: {seednameGW}.uHu file also will not be written : ")
                calcUHU = False

    def reorder_uXu(ext, formatted=False):
        try:
            print(f"----------\n {ext} module  \n----------")

            if formatted:
                f_uXu_in = open(seedname + "." + ext)
                f_uXu_out = open(seednameGW + "." + ext, "w")
                header = f_uXu_in.readline()
                f_uXu_out.write(header)
                nbnd, NK, nnb = np.array(f_uXu_in.readline().split(), dtype=int)
                f_uXu_out.write("  ".join(str(x) for x in [NBND, NK, nnb]) + "\n")
            else:
                f_uXu_in = FortranFile(seedname + "." + ext, "r")
                header = f_uXu_in.read_record(dtype="c")
                nbnd, NK, nnb = np.array(f_uXu_in.read_record(dtype=np.int32))
                if write_formatted:
                    f_uXu_out = open(seednameGW + "." + ext, "w")
                    f_uXu_out.write("".join(header.astype(str)))
                    f_uXu_out.write("\n")
                    f_uXu_out.write("  ".join(str(x) for x in [NBND, NK, nnb]))
                    f_uXu_out.write("\n")
                else:
                    f_uXu_out = FortranFile(seednameGW + "." + ext, "w")
                    f_uXu_out.write_record(header)
                    f_uXu_out.write_record(np.array([NBND, NK, nnb], dtype=np.int32))
                header = "".join(header.astype(str))

            print(header.strip())
            print(nbnd, NK, nnb)

            assert nbnd == nbndDFT

            if formatted:
                uXu = np.loadtxt(f_uXu_in).reshape(-1)
                start = 0
                length = nbnd * nbnd

            for ik in range(NKPT):
                for ib2 in range(nnb):
                    for ib1 in range(nnb):
                        if formatted:
                            A = uXu[start : start + length]
                            start += length
                        else:
                            A = f_uXu_in.read_record(dtype=np.complex)
                        A = (
                            A.reshape(nbnd, nbnd, order="F")[
                                BANDSORT[KPNB[ik][ib2]], :
                            ][:, BANDSORT[KPNB[ik][ib1]]]
                            + np.einsum(
                                "ln,lm,l->nm",
                                MMN[ik][ib2].conj(),
                                MMN[ik][ib1],
                                eigenDE[ik],
                            )
                        ).reshape(-1, order="F")
                        if formatted or write_formatted:
                            f_uXu_out.write(
                                "".join(
                                    f"{x.real:26.16e}  {x.imag:26.16e}\n" for x in A
                                )
                            )
                        else:
                            f_uXu_out.write_record(A)
            f_uXu_out.close()
            f_uXu_in.close()
            print(f"----------\n {ext} OK  \n----------\n")
        except OSError as err:
            print(f"WARNING: {seednameGW}.{ext} not written : ", err)

    if calcUHU:
        reorder_uXu("uHu", UHUformatted)
    if calcUIU:
        reorder_uXu("uIu", UIUformatted)

    if calcSPN:
        try:
            print("----------\n SPN module  \n----------")

            if SPNformatted:
                f_spn_in = open(seedname + ".spn")
                f_spn_out = open(seednameGW + ".spn", "w")
                header = f_spn_in.readline()
                f_spn_out.write(header)
                nbnd, NK = np.array(f_spn_in.readline().split(), dtype=np.int32)
                f_spn_out.write("  ".join(str(x) for x in (NBND, NKPT)))
                f_spn_out.write("\n")
            else:
                f_spn_in = FortranFile(seedname + ".spn", "r")
                header = f_spn_in.read_record(dtype="c")
                nbnd, NK = f_spn_in.read_record(dtype=np.int32)
                if write_formatted:
                    f_spn_out = open(seednameGW + ".spn", "w")
                    f_spn_out.write("".join(header.astype(str)))
                    f_spn_out.write("\n")
                    f_spn_out.write("  ".join(str(x) for x in (NBND, NKPT)))
                    f_spn_out.write("\n")
                else:
                    f_spn_out = FortranFile(seednameGW + ".spn", "w")
                    f_spn_out.write_record(header)
                    f_spn_out.write_record(np.array([NBND, NKPT], dtype=np.int32))
                header = "".join(header.astype(str))

            print(header.strip())
            assert nbnd == nbndDFT

            indm, indn = np.tril_indices(nbnd)
            indmQP, indnQP = np.tril_indices(NBND)

            if SPNformatted:
                SPN = np.loadtxt(f_spn_in).view(complex).reshape(-1)
                start = 0
                length = (3 * nbnd * (nbnd + 1)) // 2

            for ik in range(NK):
                A = np.zeros((3, nbnd, nbnd), dtype=np.complex)
                if SPNformatted:
                    A[:, indn, indm] = SPN[start : (start + length)].reshape(
                        3, nbnd * (nbnd + 1) // 2, order="F"
                    )
                    start += length
                else:
                    A[:, indn, indm] = f_spn_in.read_record(dtype=np.complex).reshape(
                        3, nbnd * (nbnd + 1) // 2, order="F"
                    )
                A[:, indm, indn] = A[:, indn, indm].conj()
                check = np.einsum("ijj->", np.abs(A.imag))
                if check > 1e-10:
                    raise RuntimeError(f"REAL DIAG CHECK FAILED for spn: {check}")
                A = A[:, :, BANDSORT[ik]][:, BANDSORT[ik], :][
                    :, indnQP, indmQP
                ].reshape((3 * NBND * (NBND + 1) // 2), order="F")
                if SPNformatted or write_formatted:
                    f_spn_out.write(
                        "".join(f"{x.real:26.16e} {x.imag:26.16e}\n" for x in A)
                    )
                else:
                    f_spn_out.write_record(A)

            f_spn_in.close()
            f_spn_out.close()
            print("----------\n SPN OK  \n----------\n")
        except OSError as err:
            print(f"WARNING: {seednameGW}.spn not written : ", err)

    if calcUNK:
        print("----------\n UNK module  \n----------")

        unkgwdir = "UNK_GW"
        unkdftdir = "UNK_DFT"
        files_list = []
        for f_unk_name in glob.glob("UNK*.*"):
            files_list.append(f_unk_name)

        try:
            os.mkdir(unkgwdir)
            os.mkdir(unkdftdir)
        except OSError:
            pass

        for f_unk_name in files_list:
            try:
                NC = os.path.splitext(f_unk_name)[1] == ".NC"
                shutil.move("./" + f_unk_name, "./" + unkdftdir + "/")
                if UNKformatted:
                    f_unk_out = open(os.path.join(unkgwdir, f_unk_name), "w")
                    f_unk_in = open(os.path.join(unkdftdir, f_unk_name))
                    nr1, nr2, nr3, ik, nbnd = np.array(
                        f_unk_in.readline().split(), dtype=int
                    )
                    NR = nr1 * nr2 * nr3
                    if NC:
                        NR *= 2
                    f_unk_out.write(
                        " ".join(str(x) for x in (nr1, nr2, nr3, ik, NBND)) + "\n"
                    )
                    f_unk_out.write(
                        "\n".join(
                            np.array([l.rstrip() for l in f_unk_in], dtype=str)
                            .reshape((nbnd, NR), order="C")[BANDSORT[ik - 1], :]
                            .reshape(-1, order="C")
                        )
                    )
                else:
                    f_unk_in = FortranFile(os.path.join(unkdftdir, f_unk_name), "r")
                    nr1, nr2, nr3, ik, nbnd = f_unk_in.read_record(dtype=np.int32)
                    NR = nr1 * nr2 * nr3
                    unk = np.zeros((nbnd, NR), dtype=np.complex)
                    if NC:
                        unk2 = np.zeros((nbnd, NR), dtype=np.complex)
                    for ib in range(nbnd):
                        unk[ib, :] = f_unk_in.read_record(dtype=np.complex)
                        if NC:
                            unk2[ib, :] = f_unk_in.read_record(dtype=np.complex)
                    unk = unk[BANDSORT[ik - 1], :]
                    if NC:
                        unk2 = unk2[BANDSORT[ik - 1], :]
                    if write_formatted:
                        f_unk_out = open(os.path.join(unkgwdir, f_unk_name), "w")
                        f_unk_out.write(
                            " ".join(str(x) for x in (nr1, nr2, nr3, ik, NBND))
                        )
                        for i in range(NBND):
                            for j in range(NR):
                                f_unk_out.write(
                                    "\n{:21.10e} {:21.10e}".format(
                                        unk[ib, j].real, unk[ib, j].imag
                                    )
                                )
                            if NC:
                                for j in range(NR):
                                    f_unk_out.write(
                                        "\n{:21.10e} {:21.10e}".format(
                                            unk2[ib, j].real, unk2[ib, j].imag
                                        )
                                    )
                    else:
                        f_unk_out = FortranFile(os.path.join(unkgwdir, f_unk_name), "w")
                        f_unk_out.write_record(
                            np.array([nr1, nr2, nr3, ik, NBND], dtype=np.int32)
                        )
                        for i in range(NBND):
                            f_unk_out.write_record(unk[ib])
                            if NC:
                                f_unk_out.write_record(unk2[ib])
                f_unk_in.close()
                f_unk_out.close()
                shutil.move("./" + unkgwdir + "/" + f_unk_name, "./")
            except OSError as err:
                if err.errno == 21:
                    pass
                else:
                    raise err
        os.rmdir(unkgwdir)
        print(
            "UNK files have been reordered, "
            + "old files coming from DFT are available in UNK_DFT folder."
        )
        print("----------\n UNK OK  \n----------\n")

    if calcCHK:
        reorder_chk(seedname, seednameGW, BANDSORT)

    f_raw.close()


if __name__ == "__main__":
    args = parse_args()

    seedname = args.seedname  # for instance "silicon"

    if args.output_seedname is None:
        seednameGW = seedname + ".gw"  # for instance "silicon.gw"
    else:
        seednameGW = args.output_seedname

    targets = []
    if args.extensions is not None:
        targets = args.extensions.split(",")
        targets = [s.lower() for s in targets]  # options read from command line

    gw2wannier90(seedname, seednameGW, targets, args.no_sort)
