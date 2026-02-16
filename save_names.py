import os
import pathlib

output_dir_base  = ''

fasta_path = ''

fasta_name = pathlib.Path(fasta_path).stem

output_dir = os.path.join(output_dir_base, fasta_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
msa_output_dir = os.path.join(output_dir, 'msas')

bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')

mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')

mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')

# bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniref_hits.a3m')
#