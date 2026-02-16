"""
Minimal configure_run_alphafold_flags that depends only on run_alphafold_min.
Avoids importing the full run_alphafold (and duplicate flag definitions).
"""

import os
import pathlib
from absl import flags, app

import run_alphafold_min  # ensure flags are defined

FLAGS = flags.FLAGS


def configure_run_alphafold_flags():
    """Populate FLAGS paths similarly to run_no_docker.configure_run_alphafold_flags."""
    # Paths relative to FLAGS.data_dir
    uniref90_database_path = os.path.join(FLAGS.data_dir, 'uniref90', 'uniref90.fasta')
    uniprot_database_path = os.path.join(FLAGS.data_dir, 'uniprot', 'uniprot.fasta')
    mgnify_database_path = os.path.join(FLAGS.data_dir, 'mgnify', 'mgy_clusters_2022_05.fa')
    bfd_database_path = os.path.join(FLAGS.data_dir, 'bfd', 'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')
    small_bfd_database_path = os.path.join(FLAGS.data_dir, 'small_bfd', 'bfd-first_non_consensus_sequences.fasta')
    uniref30_database_path = os.path.join(FLAGS.data_dir, 'uniref30', 'UniRef30_2021_03')
    pdb70_database_path = os.path.join(FLAGS.data_dir, 'pdb70', 'pdb70')
    pdb_seqres_database_path = os.path.join(FLAGS.data_dir, 'pdb_seqres', 'pdb_seqres.txt')
    template_mmcif_dir = os.path.join(FLAGS.data_dir, 'pdb_mmcif', 'mmcif_files')
    obsolete_pdbs_path = os.path.join(FLAGS.data_dir, 'pdb_mmcif', 'obsolete.dat')

    alphafold_path = pathlib.Path(__file__).parent
    data_dir_path = pathlib.Path(FLAGS.data_dir)

    # Set defaults if not already provided.
    if FLAGS.uniref90_database_path is None:
        FLAGS['uniref90_database_path'].value = uniref90_database_path
    if FLAGS.mgnify_database_path is None:
        FLAGS['mgnify_database_path'].value = mgnify_database_path
    if FLAGS.bfd_database_path is None:
        FLAGS['bfd_database_path'].value = bfd_database_path
    if FLAGS.small_bfd_database_path is None:
        FLAGS['small_bfd_database_path'].value = small_bfd_database_path
    if FLAGS.uniref30_database_path is None:
        FLAGS['uniref30_database_path'].value = uniref30_database_path
    if FLAGS.uniprot_database_path is None:
        FLAGS['uniprot_database_path'].value = uniprot_database_path
    if FLAGS.pdb70_database_path is None:
        FLAGS['pdb70_database_path'].value = pdb70_database_path
    if FLAGS.pdb_seqres_database_path is None:
        FLAGS['pdb_seqres_database_path'].value = pdb_seqres_database_path
    if FLAGS.template_mmcif_dir is None:
        FLAGS['template_mmcif_dir'].value = template_mmcif_dir
    if FLAGS.obsolete_pdbs_path is None:
        FLAGS['obsolete_pdbs_path'].value = obsolete_pdbs_path

    # Convenience accessors.
    # FLAGS['template_searcher_mmseqs_cache_dir'].value = None if 'template_searcher_mmseqs_cache_dir' in FLAGS else None

