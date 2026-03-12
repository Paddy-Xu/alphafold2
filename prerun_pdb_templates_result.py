import os
from alphafold.data.pipeline import *
import shutil
from pathlib import Path

uniref_max_hits: int = 10000

uniprot_id = 'P0DTC2'

msa_out_path = 'uniref90.sto'

msa_output_dir = f'output_dir/{uniprot_id}/msas/'

data_dir = Path('/scratch/project_465002572/uniprot_test/deep_mind_dataset').absolute()

os.makedirs(msa_output_dir, exist_ok=True)


pdb_seqres_database_path = os.path.join(
    data_dir, 'pdb_seqres', 'pdb_seqres.txt'
)
hmmsearch_binary_path =  shutil.which('hmmsearch')
hmmbuild_binary_path = shutil.which('hmmbuild')
hhsearch_binary_path = shutil.which('hhsearch')
pdb70_database_path = os.path.join(data_dir, 'pdb70', 'pdb70')

hmmsearch_n_cpu = 8
hhsearch_n_cpu = 8



run_multimer_system = True

if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        database_path=pdb_seqres_database_path,
        cpu=hmmsearch_n_cpu,

    )


else:
    template_searcher = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path,
        databases=[pdb70_database_path],
        cpu=hhsearch_n_cpu,
    )

precomputed_msa = parsers.truncate_stockholm_msa(
    msa_out_path, uniref_max_hits
)
msa_for_templates = precomputed_msa
msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
    msa_for_templates
)

if template_searcher.input_format == 'sto':
    pdb_templates_result = template_searcher.query(msa_for_templates)
elif template_searcher.input_format == 'a3m':
    uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
    pdb_templates_result = template_searcher.query(uniref90_msa_as_a3m)
else:
    raise ValueError(
        'Unrecognized template input format: '
        f'{template_searcher.input_format}'
    )

pdb_hits_out_path = os.path.join(
    msa_output_dir, f'pdb_hits.{template_searcher.output_format}'
)
with open(pdb_hits_out_path, 'w') as f:
    f.write(pdb_templates_result)
