
"""Full AlphaFold protein structure prediction script."""
import enum
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, Union

from absl import app
from absl import flags
from absl import logging
from alphafold.common import confidence
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
import jax.numpy as jnp
import numpy as np

if __name__ == '__main__':
    print("ok")
    if max_sequences is None:
        with open(sto_path) as f:
            sto = f.read()
    else:
        sto = parsers.truncate_stockholm_msa(sto_path, max_sequences)
    raw_output = dict(
        sto=sto,
        tbl=tbl,
        stderr=stderr,
        n_iter=self.n_iter,
        e_value=self.e_value,
    )

    output_dir = os.path.join(output_dir_base, fasta_name)
    msa_output_dir = os.path.join(output_dir, 'msas')

    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path, msa_output_dir=msa_output_dir
    )


    msa_for_templates = jackhmmer_uniref90_result['sto']

    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
    bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniref_hits.a3m')
    # Q13557