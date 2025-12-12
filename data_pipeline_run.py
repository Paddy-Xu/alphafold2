from run_alphafold import *
from alphafold.data.pipeline import *
from alphafold.data.pipeline_pre_run import DataPipelineNew
from pathlib import Path

from run_no_docker import configure_run_alphafold_flags




def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    configure_run_alphafold_flags()

    run_multimer_system = 'multimer' in FLAGS.model_preset
    model_type = 'Multimer' if run_multimer_system else 'Monomer'

    use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
    if FLAGS.model_preset == 'monomer_casp14':
        num_ensemble = 8
    else:
        num_ensemble = 1
    if not isinstance(FLAGS.fasta_paths, list):
        FLAGS.fasta_paths = [FLAGS.fasta_paths]
    fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]

    if len(fasta_names) != len(set(fasta_names)):
        raise ValueError('All FASTA paths must have a unique basename.')
    if run_multimer_system:
        template_searcher = hmmsearch.Hmmsearch(
            binary_path=FLAGS.hmmsearch_binary_path,
            hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
            database_path=FLAGS.pdb_seqres_database_path,
            cpu=FLAGS.hmmsearch_n_cpu,
        )
        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir,
            max_template_date=FLAGS.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=FLAGS.kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
        )
    else:
        template_searcher = hhsearch.HHSearch(
            binary_path=FLAGS.hhsearch_binary_path,
            databases=[FLAGS.pdb70_database_path],
            cpu=FLAGS.hhsearch_n_cpu,
        )
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir,
            max_template_date=FLAGS.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=FLAGS.kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
        )

    monomer_data_pipeline = DataPipelineNew(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniref30_database_path=FLAGS.uniref30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        msa_tools_n_cpu=FLAGS.jackhmmer_n_cpu,
    )

    if run_multimer_system:
        num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
        data_pipeline = pipeline_multimer.DataPipeline(
            monomer_data_pipeline=monomer_data_pipeline,
            jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
            uniprot_database_path=FLAGS.uniprot_database_path,
            use_precomputed_msas=FLAGS.use_precomputed_msas,
            jackhmmer_n_cpu=FLAGS.jackhmmer_n_cpu,
        )
    else:
        num_predictions_per_model = 1
        data_pipeline = monomer_data_pipeline

    model_runners = {}
    model_names = config.MODEL_PRESETS[FLAGS.model_preset]
    for model_name in model_names:
        model_config = config.model_config(model_name)
        if run_multimer_system:
            model_config.model.num_ensemble_eval = num_ensemble
        else:
            model_config.data.eval.num_ensemble = num_ensemble
        model_params = data.get_model_haiku_params(
            model_name=model_name, data_dir=FLAGS.data_dir
        )
        model_runner = model.RunModel(model_config, model_params)
        for i in range(num_predictions_per_model):
            model_runners[f'{model_name}_pred_{i}'] = model_runner

    logging.info(
        'Have %d models: %s', len(model_runners), list(model_runners.keys())
    )

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=FLAGS.use_gpu_relax,
    )

    random_seed = FLAGS.random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_runners))
    logging.info('Using random seed %d for the data pipeline', random_seed)

    # Predict structure for each of the sequences.
    for i, fasta_path in enumerate(FLAGS.fasta_paths):
        fasta_name = fasta_names[i]

        fasta_path=fasta_path
        fasta_name=fasta_name
        output_dir_base=FLAGS.output_dir

        model_runners=model_runners
        amber_relaxer=amber_relaxer

        timings = {}
        output_dir = os.path.join(output_dir_base, fasta_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        msa_output_dir = os.path.join(output_dir, 'msas')
        if not os.path.exists(msa_output_dir):
            os.makedirs(msa_output_dir)

        t_0 = time.time()

        input_file = Path(fasta_path).stem
        if input_file.strip() in all_accession:
            sto_filename_prefix = "output_" + input_file
            sto_filename = os.path.join(output_sto_root, sto_filename_prefix)

        else:
            continue
            res = find_sequence(all_huamn_protein_database_path, protein_sequence=query_sequence)
            target_id = res["id"]

            # print(find_sto_results(output_sto_root, res))

            sto_filename_prefix = "output_" + target_id
            sto_filename = os.path.join(output_sto_root, sto_filename_prefix)

        if USE_a3m:
            a3m_filename = os.path.join(output_a3m_root, sto_filename_prefix)
            all_dbs = [f"{a3m_filename}_on_{db}_a3m.a3m"
                        for db in all_database_exact_names]
        else:
            all_dbs = [f"{sto_filename}_on_{db}.sto"
                       for db in all_database_exact_names]


        all_dbs = [Path(i).absolute() for i in all_dbs]

        if USE_a3m:
            assert all([os.path.exists(i) for i in all_dbs]), f"some a3m files not found on {all_dbs}"
        else:
            assert all([os.path.exists(i) for i in all_dbs]), f"some sto files not found on {all_dbs}"



        feature_dict = data_pipeline.process(
            input_fasta_path=fasta_path, msa_output_dir=msa_output_dir, all_dbs=all_dbs
        )

        timings['features'] = time.time() - t_0

        # Write out features as a pickled dictionary.
        features_output_path = os.path.join(output_dir, 'features.pkl')
        with open(features_output_path, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)

        unrelaxed_pdbs = {}
        unrelaxed_proteins = {}
        relaxed_pdbs = {}
        relax_metrics = {}
        ranking_confidences = {}

        print(f'done for {fasta_name}, no model running needed')

if __name__ == '__main__':


    check_original = False

    force_rerun_jackhmmer = False

    use_msa_from_original = True

    force_executor = True

    DEBUG = False

    Truncate = False

    from Bio import SeqIO

    all_huamn_protein_database_path = "all_proteins_uniprotkb_organism_id_9606_AND_reviewed_2025_10_21.fasta"
    records = list(SeqIO.parse(all_huamn_protein_database_path, "fasta"))
    all_ids = [record.id for record in records]
    all_accession = [id.split("|")[1].replace(" ", "") for id in all_ids]

    output_sto_root = "../af3_self/1000_2000_results_jackhmmer_on_public_dbs/sto/"

    output_a3m_root = "../af3_self/1000_2000_results_jackhmmer_on_public_dbs/a3m/"
    USE_a3m = True

    all_database_exact_names = ["uniref90_2022_05", "bfd-first_non_consensus_sequences",
                                "mgy_clusters_2022_05", "uniprot_all_2021_04"]




    # fasta_paths = 'O15552.fasta'
    fasta_paths = 'O14818.fasta'
    # model_dir = Path(model_dir).absolute()
    model_dir = '../alphafold_models'
    new_argv = sys.argv[:]
    db_dir = Path('/scratch/project_465001728/uniprot_test/deep_mind_dataset').absolute()
    output_dir = "../output_sb"

    if not any(a.startswith("--fasta_paths=") for a in new_argv):
        new_argv.append(f"--fasta_paths={fasta_paths}")

    if not any(a.startswith("--max_template_date=") for a in new_argv):
        new_argv.append(f"--max_template_date=2020-05-14")

    if not any(a.startswith("--model_preset=") for a in new_argv):
        new_argv.append(f"--model_preset=monomer")

    if not any(a.startswith("--db_preset=") for a in new_argv):
        new_argv.append(f"--db_preset=reduced_dbs")

    if not any(a.startswith("--data_dir=") for a in new_argv):
        new_argv.append(f"--data_dir={db_dir}")
    #
    # if not any(a.startswith("--db_dir=") for a in new_argv):
    #     new_argv.append(f"--db_dir={db_dir}")

    if not any(a.startswith("--output_dir=") for a in new_argv):
        new_argv.append(f"--output_dir={output_dir}")

    sys.argv = new_argv

    app.run(main)
