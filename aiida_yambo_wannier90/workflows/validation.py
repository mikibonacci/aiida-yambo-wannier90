"""Validation of the inputs of the ``YamboWannier90WorkChain`` (the spec validators)."""
import typing as ty

KPATH_INPUTS = ("kpoint_path", "bands_kpoints", "bands_kpoints_distance")


def validate_kpath_inputs(inputs: dict) -> ty.Union[None, str]:
    """At most one of the sources of the band path, and labels on the k-points which need them."""
    given = [name for name in KPATH_INPUTS if name in inputs]
    if len(given) > 1:
        return f"Specify only one of {KPATH_INPUTS}, got {given}."
    for name in ("kpoint_path", "bands_kpoints"):
        if name in inputs and inputs[name].labels is None:
            return f"`{name}` must have `labels`."
    return None


def validate_inputs(  # pylint: disable=inconsistent-return-statements,too-many-return-statements,too-many-branches,too-many-locals
    inputs: dict, ctx=None  # pylint: disable=unused-argument
) -> ty.Union[None, str]:
    """Validate the inputs of the entire input namespace."""

    error = validate_kpath_inputs(inputs)
    if error:
        return error

    # Must run steps sequentially
    order = ["yambo", "yambo_qp", "ypp", "wannier90", "gw2wannier90", "wannier90_qp"]
    non_empty = [_ in inputs for _ in order]
    first_input = non_empty.index(True)
    if not all(non_empty[first_input:]):
        first_no_input = first_input + non_empty[first_input:].index(False)
        return (
            f"WorkChain must be run in order, `{order[first_input]}` is provided "
            f"but `{order[first_no_input]}` is empty."
        )

    # Check inputs if previous steps are skipped
    should_run_yambo = "yambo" in inputs
    should_run_yambo_commensurate = "GW_mesh" in inputs
    should_run_wannier90 = "wannier90" in inputs
    should_run_yambo_qp = "yambo_qp" in inputs
    should_run_ypp = "ypp" in inputs
    should_run_gw2wannier90 = "gw2wannier90" in inputs
    should_run_wannier90_qp = "wannier90_qp" in inputs

    if should_run_yambo_qp:
        yambo_qp_inputs = inputs["yambo_qp"]

        if not should_run_yambo:
            if "parent_folder" not in yambo_qp_inputs and not should_run_yambo_commensurate:
                return "`yambo_qp.parent_folder` is empty."

    if should_run_ypp:
        ypp_inputs = inputs["ypp"]

        if not should_run_yambo_qp:
            if "QP_DB" not in ypp_inputs["ypp"]:
                return "`ypp.ypp.QP_DB` is empty."
            if "parent_folder" not in ypp_inputs:
                return "`ypp.parent_folder` is empty."

        # I need `wannier90` input to run a w90 postproc before `ypp`,
        # or if there is `nnkp`, I skip the postproc.
        if not should_run_wannier90:
            if "nnkp_file" not in ypp_inputs["ypp"]:
                return "`ypp.ypp.nnkp_file` is empty."

    if should_run_gw2wannier90:
        gw2wannier90_inputs = inputs["gw2wannier90"]
        if not should_run_wannier90:
            for tag in ["nnkp", "parent_folder"]:
                if tag not in gw2wannier90_inputs:
                    return f"`gw2wannier90.{tag}` is empty."

        if not should_run_ypp:
            if "unsorted_eig" not in gw2wannier90_inputs:
                return "`gw2wannier90.unsorted_eig` is empty."

    if should_run_wannier90_qp:
        wannier90_qp_inputs = inputs["wannier90_qp"]
        if not should_run_gw2wannier90:
            if "remote_input_folder" not in wannier90_qp_inputs["wannier90"]:
                return "`wannier90_qp.wannier90.remote_input_folder` is empty."

        # The QP run reads the matrices of the first one: same bands and Wannier functions
        if should_run_wannier90:
            first = inputs["wannier90"]["wannier90"]["wannier90"]["parameters"].get_dict()
            second = wannier90_qp_inputs["wannier90"]["parameters"].get_dict()
            for key in ("num_wann", "num_bands", "exclude_bands"):
                if first.get(key) != second.get(key):
                    return (
                        f"`{key}` differs between `wannier90` ({first.get(key)}) and `wannier90_qp` ({second.get(key)}): "
                        "use `aiida_yambo_wannier90.utils.setters.set_wannier_bands`, or `sync_wannier90_qp` after changing it."
                    )
