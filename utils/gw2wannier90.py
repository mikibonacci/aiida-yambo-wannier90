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
import datetime
import glob
import os
import shutil

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

    parsed_args = parser.parse_args(args)

    return parsed_args


def gw2wannier90(seedname: str, seednameGW: str, targets: list) -> None:
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

    if set(targets).intersection({"spn", "uhu", "mmn", "amn", "unk", "uiu"}):
        calcAMN = "amn" in targets
        calcMMN = "mmn" in targets
        calcUHU = "uhu" in targets
        calcUIU = "uiu" in targets
        calcSPN = "spn" in targets
        calcUNK = "unk" in targets
    else:
        calcAMN = True
        calcMMN = True
        calcUHU = True
        calcUIU = True
        calcSPN = True
        calcUNK = True

    if calcUHU:
        calcMMN = True
    if calcUIU:
        calcMMN = True

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
        # raise RuntimeError("exclude bands is not supported yet") # actually it is OK, see below
        print(
            "Exclude bands option is used: be careful to be consistent "
            + "with the choice of bands for the GW QP corrections."
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

    print("Sorting")
    bsort = np.array([np.argsort(eigenDFTGW[ik, :]) for ik in range(NKPT)])

    f_raw.write("------------------------------\n")
    f_raw.write("Writing sorting list\n")
    for line in bsort:
        f_raw.write(str(line) + "\n")
    f_raw.write("------------------------------\n")

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

    gw2wannier90(seedname, seednameGW, targets)
