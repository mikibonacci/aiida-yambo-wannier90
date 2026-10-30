# How-to guides

`YamboWannier90WorkChain.get_builder_from_protocol` fills every stage; the functions of `aiida_yambo_wannier90.utils.setters` change
what you usually need and keep the related inputs consistent, so you do not edit nested parameters by hand.

<div class="grid cards" markdown>

-   **Set the inputs**

    The builder, the protocol and the overrides; meshes, bands, projections and the QP splitter.

    [Inputs](inputs.md)

-   **Choose the band path**

    Seekpath, a distance, your path, labelled points or an explicit list of k-points.

    [Band path](kpath.md)

-   **Set the resources**

    Nodes, MPI, threads, walltime, queue and npool of every step.

    [Resources](resources.md)

-   **Start from the middle**

    Reuse a GW calculation, a QP database or a later stage.

    [Stages](stages.md)

</div>
