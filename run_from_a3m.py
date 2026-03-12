from run_no_docker import *

# from run_alphafold import *

from pipeline_pre_run import *

from alphafold.data import pipeline
from alphafold.data import pipeline_multimer

DataPipelineNew.process = DataPipelineNew.process_a3m
DataPipelineMultimerNew.process = DataPipelineMultimerNew.process_a3m

pipeline.DataPipeline = DataPipelineNew
pipeline_multimer.DataPipeline = DataPipelineMultimerNew

if __name__ == '__main__':

    flags.mark_flags_as_required([
        'data_dir',
        'fasta_paths',
        'max_template_date',
    ])

    fasta_paths = 'O15552.fasta'

    output_dir = "/scratch/project_465002572/af3_self/runs/precompute_msa/"

    db_dir = '/scratch/project_465002572/uniprot_test/deep_mind_dataset'

    model_dir = '../alphafold_models'

    new_argv = sys.argv[:]

    if not any(a.startswith("--fasta_paths=") for a in new_argv):
        new_argv.append(f"--fasta_paths={fasta_paths}")

    if not any(a.startswith("--max_template_date=") for a in new_argv):
        new_argv.append(f"--max_template_date=2020-05-14")

    if not any(a.startswith("--model_preset=") for a in new_argv):
        # new_argv.append(f"--model_preset=monomer")
        new_argv.append(f"--model_preset=multimer")


    if not any(a.startswith("--db_preset=") for a in new_argv):
        new_argv.append(f"--db_preset=reduced_dbs")
    if not any(a.startswith("--data_dir=") for a in new_argv):
        new_argv.append(f"--data_dir={db_dir}")
    if not any(a.startswith("--use_precomputed_msas=") for a in new_argv):
        new_argv.append(f"--use_precomputed_msas=True")

    if not any(a.startswith("--output_dir=") for a in new_argv):
        new_argv.append(f"--output_dir={output_dir}")

    sys.argv = new_argv

    flags.mark_flags_as_required([
        'fasta_paths',
        'output_dir',
        'data_dir',
        # 'use_gpu_relax',
    ])

    app.run(main)


