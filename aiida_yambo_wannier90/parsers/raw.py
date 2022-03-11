"""
Parsers provided by aiida_yambo_wannier90.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
import typing as ty


def parse_pw_output_kpoints(filecontent: ty.List[str]) -> ty.List:
    lines = filecontent.split("\n")

    read_kpts = False
    k_fine_list = []
    for line in lines:
        if "number of k points=" in line:
            numk_line = line.strip("\n").split()
            num_kpoints = int(numk_line[4])
            # print(
            #     'Number of kpoints provided to Yambo through a NSCF calculation',
            #     num_kpoints)
        if read_kpts:
            kline = line.strip("\n").split()
            if "wk" in kline:
                a = kline[4:6]
                b = kline[6].split(")")[0]
                k_vec = [float(a[0]), float(a[1]), float(b)]
                k_fine_list.append(k_vec)
            else:
                read_kpts = False
        if "cryst. coord." in line and "site" not in line:
            read_kpts = True

    return k_fine_list
