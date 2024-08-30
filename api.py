from typing import Optional
import json
import os
import uuid

from fastapi import FastAPI, BackgroundTasks, status, HTTPException
from pydantic import BaseModel, Field

from utils.sampler import SEQDIFF_sampler, cleavage_foldswitch_SAMPLER

JOB_STATUS: dict[str, str] = {}

sampler_map = {
    "default": SEQDIFF_sampler,
    "cleavage_foldswitch": cleavage_foldswitch_SAMPLER,
}

app = FastAPI()


class DesignConfig(BaseModel):
    """
    Request model to start an inference run of Protein Generator
    """

    F: int = Field(default=1, description="Noise factor")
    T: int = Field(default=25, description="Number of timesteps to use")
    aa_composition: str = Field(
        default="W0.2",
        description="Amino acid composition specified by one letter aa code and fraction to represent in sequence ex. H0.2,K0.5",
    )
    aa_spec: Optional[str] = Field(
        default=None,
        description="How to bias sequence example XXXAXL where X is mask token",
    )
    aa_weight: Optional[str] = Field(
        default=None,
        description="Weight string to use with --aa_spec for how to bias sequence",
    )
    aa_weights_json: Optional[str] = Field(
        default=None,
        description="File path the JSON file of amino acid weighting to use during inference",
    )
    add_weight_every_n: int = Field(default=1, description="Frequency to add aa weight")
    argmax_seq: bool = Field(
        default=False, description="Argmax seq after coming out of model"
    )
    cautious: bool = Field(
        default=False,
        description="If true, will not run a design if output file already exists.",
    )
    checkpoint: str = Field(
        default="./SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt",
        description="Checkpoint to pretrained RFold module",
    )
    clamp_seqout: bool = Field(
        default=False,
        description="If turned on will clamp the Xo pred sequence before sampling next t",
    )
    no_clamp_seqout_after: bool = Field(
        default=False,
        description="If turned on will clamp the Xo pred sequence before sampling next t",
    )
    contigs: Optional[str] = Field(
        default=None, description="Pieces of input protein to keep"
    )
    d_t1d: int = Field(
        default=24,
        description="The t1d dimension that is compatible with specified checkpoint",
    )
    dssp_pdb: Optional[str] = Field(default=None, description="Input protein dssp")
    dump_all: bool = Field(
        default=False, description="If true, will dump all possible outputs to outdir"
    )
    dump_npz: bool = Field(
        default=False,
        description="Whether to dump npz (disto/anglograms) files in output dir",
    )
    dump_pdb: bool = Field(default=True, description="Whether to dump pdb output")
    dump_trb: bool = Field(
        default=True, description="Whether to dump trb files in output dir"
    )
    frac_seq_to_weight: float = Field(
        default=0.0,
        description="Fraction of sequence to add AA weight bias too (will be randomly sampled)",
    )
    hal_idx: Optional[str] = Field(default=None, description="")
    helix_bias: float = Field(
        default=0.0, description="Percent of sequence to randomly bias toward helix"
    )
    hotspots: Optional[str] = Field(
        default=None, description="Specify hotspots to find i.e. B35,B44,B56"
    )
    idx_rf: Optional[str] = Field(default=None, description="")
    inpaint_seq: Optional[str] = Field(
        default=None,
        description="Predict the sequence at these residues. Similar mask (and window), but is specifically for sequence.",
    )
    inpaint_seq_tensor: Optional[str] = Field(default=None, description="")
    inpaint_str: Optional[str] = Field(
        default=None,
        description="Predict the structure at these residues. Similar mask (and window), but is specifically for structure.",
    )
    inpaint_str_tensor: Optional[str] = Field(default=None, description="")
    input_json: Optional[str] = Field(
        default=None,
        description="Path to file containing JSON-formatted list of dictionaries, each containing command-line arguments for 1 design.",
    )
    length: Optional[str] = Field(
        default=None,
        description="Specify length, or length range, you want the outputs. e.g. 100 or 95-105",
    )
    loop_bias: float = Field(
        default=0.0, description="Percent of sequence to randomly bias toward loop"
    )
    loop_design: bool = Field(
        default=False,
        description="If this arg is passed the loop design checkpoint will be used",
    )
    min_decoding_distance: int = Field(default=15, description="")
    multi_templates: Optional[str] = Field(default=None, description="")
    multi_tmpl_conf: Optional[str] = Field(default=None, description="")
    n_cycle: int = Field(
        default=4, description="Number of recycles through RFold at each step"
    )
    noise_schedule: str = Field(
        default="sqrt", description="Schedule type to add noise"
    )
    num_designs: int = Field(default=50, description="Number of designs to make")
    one_weight_per_position: bool = Field(
        default=False,
        description="Only add weight to one aa type at each residue position (will randomly sample)",
    )
    out: str = Field(default="./seqdiff", description="Output directory for file")
    potential_scale: str = Field(
        default="", description="Scale at which to guid the sequence potential"
    )
    ref_idx: Optional[int] = Field(default=None, description="")
    sampling_temp: float = Field(
        default=1.0,
        description="Temperature to sample input sequence to as a fraction of T, for partial diffusion",
    )
    save_all_steps: bool = Field(
        default=False, description="Save individual steps during diffusion"
    )
    save_best_plddt: bool = Field(
        default=True, description="Save highest plddt structure only"
    )
    save_seqs: bool = Field(default=False, description="Save in and out seqs")
    scheduled_str_cond: bool = Field(
        default=False,
        description="If turned on will self condition on x fraction of the strcutre according to schedule (jake style)",
    )
    secondary_structure: Optional[str] = Field(
        default=None,
        description="Specified secondary structure string, H-helix, E-strand, L-loop, X-mask, i.e. XXXXXXHHHHHHXXXXLLLLXXXXXEEEEXXXXX",
    )
    softmax_seqout: bool = Field(
        default=False,
        description="If turned on will softmax the Xo pred sequence before sampling next t",
    )
    start_num: int = Field(default=0, description="Number of first design to output")
    strand_bias: float = Field(
        default=0.0, description="Percent of sequence to randomly bias toward strand"
    )
    struc_cond: bool = Field(
        default=False,
        description="If turned on will struc condition on structure in sidneys style",
    )
    struc_cond_sc: bool = Field(
        default=False,
        description="If turned on will self condition on structure in sidneys style",
    )
    symmetry: int = Field(
        default=1,
        description="Integer specifying sequence repeat symmetry, e.g. 4 -> sequence composed of 4 identical repeats",
    )
    symmetry_cap: int = Field(
        default=0, description="Length for symmetry cap; assumes cap will be helix"
    )
    predict_symmetric: bool = Field(
        default=False, description="Predict explicit symmetrization after the last step"
    )
    temperature: float = Field(default=0.1, description="")
    tmpl_conf: str = Field(
        default="1", description="1D confidence value for template residues"
    )
    trb: Optional[str] = Field(
        default=None, description="Path to input trb file for partial diffusion"
    )
    sample_distribution: str = Field(
        default="normal", description="Sample distribution for q_sample()"
    )
    sample_distribution_gmm_means: list[int] = Field(
        default=[0], description="Sample distribution means for q_sample()"
    )
    sample_distribution_gmm_variances: list[int] = Field(
        default=[1], description="Sample distribution variances for q_sample()"
    )
    target_charge: int = Field(
        default=0.0, description="Set charge to guide sequence towards."
    )
    charge_loss_type: str = Field(
        default="complex", description="Type of loss to use when using charge potential"
    )
    target_pH: float = Field(default=7.4, description="Set pH to calculate charge at")
    hydrophobic_score: float = Field(
        default=0.0, description="Set GRAVY score to guide sequence towards"
    )
    hydrophobic_loss_type: str = Field(
        default="complex",
        description="Type of loss to compute when using hydrophobicity potential",
    )
    save_args: bool = Field(
        default=True, description="Will save the arguments used in a json file"
    )
    potentials: str = Field(
        default="",
        description="List of potentials to use, must be paired with potenatial_scale e.g. aa_bias,solubility,charge",
    )

    sampler: str = Field(default="default", description="Type of sampler to use")
    PSSM: str = Field(default="", description="PSSM as csv")

    pdb: str = Field(default=None, description="Path to input protein PDB file")
    sequence: str = Field(default=None, description="Input sequence to diffuse")


class HealthCheck(BaseModel):
    """
    Response model to validate and return when performing a health check.
    """

    status: str = "OK"


class JobStatus(BaseModel):
    """
    Response model to provide the user with status information about the requested or generated job.
    """

    job_id: str
    status: str


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


@app.post(
    "/inference",
    response_description="The ID and status of the created job",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobStatus,
)
def run_workflow(config: DesignConfig, background_tasks: BackgroundTasks):
    """ """
    job_id = str(uuid.uuid4())
    JOB_STATUS[job_id] = "pending"
    background_tasks.add_task(_run_workflow, job_id, config)

    return JobStatus(job_id=job_id, status=JOB_STATUS.get(job_id, "not found"))


def _run_workflow(job_id: str, config: DesignConfig):
    try:
        JOB_STATUS[job_id] = "running"
        model_args = config.model_dump()
        chosen_sampler = sampler_map[config.sampler]
        S = chosen_sampler(model_args)

        # get JSON args
        if config.input_json is not None:
            with open(config.input_json) as f_json:
                argdicts = json.load(f_json)
            print(f"JSON args loaded {config.input_json}")
            # wrap argdicts in a list if not inputed as one
            if isinstance(argdicts, dict):
                argdicts = [argdicts]
            S.set_args(argdicts[0])
        else:
            # no json input, spoof list of argument dicts
            argdicts = [{}]

        # build model
        S.model_init()

        # diffuser init
        S.diffuser_init()

        for i_argdict, argdict in enumerate(argdicts):

            if config.input_json is not None:
                print(
                    f"\nAdding argument dict {i_argdict} from input JSON ({len(argdicts)} total):"
                )

                ### HERE IS WHERE ARGUMENTS SHOULD GET SET
                S.set_args(argdict)
                S.diffuser_init()

            for i_des in range(
                S.args["start_num"], S.args["start_num"] + S.args["num_designs"]
            ):

                out_prefix = f"{config.out}_{i_des:06}"

                if config.cautious and os.path.exists(out_prefix + ".pdb"):
                    print(
                        f"CAUTIOUS MODE: Skipping design because output file "
                        f'{out_prefix + ".pdb"} already exists.'
                    )
                    continue

                S.generate_sample()

        JOB_STATUS[job_id] = "completed"
    except Exception as e:
        JOB_STATUS[job_id] = "failed"
        raise e


@app.get(
    "/job-status/{job_id}",
    response_description="The ID and status of the requested job",
    response_model=JobStatus,
)
def get_job_status(job_id: str):
    """
    Returns the current status of the job represented by `job_id`.

    If `job_id` does not exist, return a status code 404
    """
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Job not found")
    status = JOB_STATUS[job_id]
    return JobStatus(job_id=job_id, status=status)
