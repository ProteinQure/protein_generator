from typing import Optional
import json
import os
import uuid

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from utils.sampler import SEQDIFF_sampler, cleavage_foldswitch_SAMPLER

JOB_STATUS: dict[str, str] = {}

sampler_map = {
    "default": SEQDIFF_sampler,
    "cleavage_foldswitch": cleavage_foldswitch_SAMPLER,
}


class DesignConfig(BaseModel):
    F: int = 1
    T: int = 25
    aa_composition: str = "W0.2"
    aa_spec: Optional[str] = None
    aa_weight: Optional[str] = None
    aa_weights_json: Optional[str] = None
    add_weight_every_n: int = 1
    argmax_seq: bool = False
    cautious: bool = False
    checkpoint: str = "./SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt"
    clamp_seqout: bool = False
    contigs: Optional[str] = None
    d_t1d: int = 24
    dssp_pdb: Optional[str] = None
    dump_all: bool = False
    dump_npz: bool = False
    dump_pdb: bool = True
    dump_trb: bool = True
    frac_seq_to_weight: float = 0.0
    hal_idx: Optional[str] = None
    helix_bias: float = 0.0
    hotspots: Optional[str] = None
    idx_rf: Optional[str] = None
    inpaint_seq: Optional[str] = None
    inpaint_seq_tensor: Optional[str] = None
    inpaint_str: Optional[str] = None
    inpaint_str_tensor: Optional[str] = None
    input_json: Optional[str] = None
    length: Optional[str] = None
    loop_bias: float = 0.0
    loop_design: bool = False
    min_decoding_distance: int = 15
    multi_templates: Optional[str] = None
    multi_tmpl_conf: Optional[str] = None
    n_cycle: int = 4
    noise_schedule: str = "sqrt"
    num_designs: int = 500
    one_weight_per_position: bool = False
    out: str = "./examples/out/design"
    pdb: Optional[str] = None
    potential_scale: str = ""
    ref_idx: Optional[int] = None
    sampling_temp: float = 1.0
    save_all_steps: bool = False
    save_best_plddt: bool = True
    save_seqs: bool = False
    scheduled_str_cond: bool = False
    secondary_structure: Optional[str] = None
    softmax_seqout: bool = False
    start_num: int = 0
    strand_bias: float = 0.0
    struc_cond_sc: bool = False
    symmetry: int = 1
    symmetry_cap: int = 0
    temperature: float = 0.1
    tmpl_conf: str = "1"
    trb: Optional[str] = None
    sample_distribution: str = "normal"
    sample_distribution_gmm_means: list[int] = [0]
    sample_distribution_gmm_variances: list[int] = [1]
    target_charge: int = -10
    charge_loss_type: str = "complex"
    target_pH: float = 7.4
    hydrophobic_score: int = -10
    hydrophobic_loss_type: str = "complex"
    save_args: bool = True
    potentials: str = ""
    sequence: str = "XXXXXXXXXXXXXXXXPEPSEQXXXXXXXXXXXXXXXX"
    sampler: str = "default"


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/workflow")
async def run_workflow(config: DesignConfig, background_tasks: BackgroundTasks):
    # config = DesignConfig()
    job_id = str(uuid.uuid4())
    JOB_STATUS[job_id] = "pending"
    background_tasks.add_task(_run_workflow, job_id, config)
    _run_workflow(job_id, config)

    return {"job_id", job_id}


def _run_workflow(job_id: str, config: DesignConfig):
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


@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    status = JOB_STATUS.get(job_id, "not found")
    return {"job_id": job_id, "status": status}
