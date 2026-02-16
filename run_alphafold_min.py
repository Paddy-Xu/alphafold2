"""
Lightweight subset of run_alphafold.py for single_process_fasta.py.
Defines only the flags and constants needed for precomputed-MSA feature
generation and template search; omits model loading / inference code.
"""

import shutil
from absl import flags

FLAGS = flags.FLAGS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TEMPLATE_HITS = 20

# ---------------------------------------------------------------------------
# Flag definitions (minimal set used by single_process_fasta.py)
# ---------------------------------------------------------------------------

flags.DEFINE_list(
    'fasta_paths',
    None,
    'Paths to FASTA files. single_process_fasta.py expects exactly one.',
)

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string(
    'output_dir', None, 'Path to a directory that will store the results.'
)

flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the JackHMMER executable.',
)
flags.DEFINE_string(
    'hhblits_binary_path',
    shutil.which('hhblits'),
    'Path to the HHblits executable.',
)
flags.DEFINE_string(
    'hhsearch_binary_path',
    shutil.which('hhsearch'),
    'Path to the HHsearch executable.',
)
flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the hmmsearch executable.',
)
flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the hmmbuild executable.',
)
flags.DEFINE_string(
    'kalign_binary_path',
    shutil.which('kalign'),
    'Path to the Kalign executable.',
)

flags.DEFINE_string('uniref90_database_path', None, 'Uniref90 database path.')
flags.DEFINE_string('mgnify_database_path', None, 'MGnify database path.')
flags.DEFINE_string('bfd_database_path', None, 'BFD database path.')
flags.DEFINE_string('small_bfd_database_path', None, 'Small BFD path.')
flags.DEFINE_string('uniref30_database_path', None, 'UniRef30 database path.')
flags.DEFINE_string('uniprot_database_path', None, 'Uniprot database path.')
flags.DEFINE_string('pdb70_database_path', None, 'PDB70 database path.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'PDB seqres file path.')
flags.DEFINE_string('template_mmcif_dir', None, 'Directory of template mmCIFs.')
flags.DEFINE_string('max_template_date', None, 'Max template release date.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Obsolete PDB mapping file.')

flags.DEFINE_enum(
    'db_preset',
    'full_dbs',
    ['full_dbs', 'reduced_dbs'],
    'Genetic database configuration.',
)
flags.DEFINE_enum(
    'model_preset',
    'monomer',
    ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
    'Model configuration preset.',
)

flags.DEFINE_boolean(
    'use_precomputed_msas',
    True,
    'Use precomputed MSAs (expected by single_process_fasta).',
)

flags.DEFINE_integer(
    'jackhmmer_n_cpu', 8, 'CPUs for jackhmmer / jackhmmer runners.'
)
flags.DEFINE_integer('hhsearch_n_cpu', 8, 'CPUs for hhsearch.')
flags.DEFINE_integer('hmmsearch_n_cpu', 8, 'CPUs for hmmsearch.')

# Multimer uses this; keep for completeness.
flags.DEFINE_boolean(
    'benchmark',
    False,
    'Unused here; kept for flag compatibility.',
)

