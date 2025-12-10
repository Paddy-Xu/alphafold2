from run_alphafold import *
from alphafold.data.pipeline import *

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
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

    # monomer_data_pipeline = pipeline.DataPipeline(
    #     jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
    #     hhblits_binary_path=FLAGS.hhblits_binary_path,
    #     uniref90_database_path=FLAGS.uniref90_database_path,
    #     mgnify_database_path=FLAGS.mgnify_database_path,
    #     bfd_database_path=FLAGS.bfd_database_path,
    #     uniref30_database_path=FLAGS.uniref30_database_path,
    #     small_bfd_database_path=FLAGS.small_bfd_database_path,
    #     template_searcher=template_searcher,
    #     template_featurizer=template_featurizer,
    #     use_small_bfd=use_small_bfd,
    #     use_precomputed_msas=FLAGS.use_precomputed_msas,
    #     msa_tools_n_cpu=FLAGS.jackhmmer_n_cpu,
    # )

    if run_multimer_system:
        num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
        # data_pipeline = pipeline_multimer.DataPipeline(
        #     monomer_data_pipeline=monomer_data_pipeline,
        #     jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        #     uniprot_database_path=FLAGS.uniprot_database_path,
        #     use_precomputed_msas=FLAGS.use_precomputed_msas,
        #     jackhmmer_n_cpu=FLAGS.jackhmmer_n_cpu,
        # )
    else:
        num_predictions_per_model = 1
        # data_pipeline = monomer_data_pipeline

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
        # data_pipeline=data_pipeline
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

        if not run_multimer_system:

            jackhmmer_binary_path = FLAGS.jackhmmer_binary_path
            hhblits_binary_path = FLAGS.hhblits_binary_path
            uniref90_database_path = FLAGS.uniref90_database_path
            mgnify_database_path = FLAGS.mgnify_database_path
            bfd_database_path = FLAGS.bfd_database_path
            uniref30_database_path = FLAGS.uniref30_database_path
            small_bfd_database_path = FLAGS.small_bfd_database_path
            template_searcher = template_searcher
            template_featurizer = template_featurizer
            use_small_bfd = use_small_bfd
            use_precomputed_msas = FLAGS.use_precomputed_msas
            msa_tools_n_cpu = FLAGS.jackhmmer_n_cpu


            input_fasta_path=fasta_path
            msa_output_dir=msa_output_dir
            with open(input_fasta_path) as f:
                input_fasta_str = f.read()
            input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
            if len(input_seqs) != 1:
                raise ValueError(
                    f'More than one input sequence found in {input_fasta_path}.'
                )
            input_sequence = input_seqs[0]
            input_description = input_descs[0]
            num_res = len(input_sequence)

            uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
            jackhmmer_uniref90_result = run_msa_tool(
                msa_runner=self.jackhmmer_uniref90_runner,
                input_fasta_path=input_fasta_path,
                msa_out_path=uniref90_out_path,
                msa_format='sto',
                use_precomputed_msas=self.use_precomputed_msas,
                max_sto_sequences=self.uniref_max_hits,
            )
            mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
            jackhmmer_mgnify_result = run_msa_tool(
                msa_runner=self.jackhmmer_mgnify_runner,
                input_fasta_path=input_fasta_path,
                msa_out_path=mgnify_out_path,
                msa_format='sto',
                use_precomputed_msas=self.use_precomputed_msas,
                max_sto_sequences=self.mgnify_max_hits,
            )

            msa_for_templates = jackhmmer_uniref90_result['sto']
            msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
            msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
                msa_for_templates
            )

            if self.template_searcher.input_format == 'sto':
                pdb_templates_result = self.template_searcher.query(msa_for_templates)
            elif self.template_searcher.input_format == 'a3m':
                uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
                pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
            else:
                raise ValueError(
                    'Unrecognized template input format: '
                    f'{self.template_searcher.input_format}'
                )

            pdb_hits_out_path = os.path.join(
                msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}'
            )
            with open(pdb_hits_out_path, 'w') as f:
                f.write(pdb_templates_result)

            uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
            mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])

            pdb_template_hits = self.template_searcher.get_template_hits(
                output_string=pdb_templates_result, input_sequence=input_sequence
            )

            if self._use_small_bfd:
                bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
                jackhmmer_small_bfd_result = run_msa_tool(
                    msa_runner=self.jackhmmer_small_bfd_runner,
                    input_fasta_path=input_fasta_path,
                    msa_out_path=bfd_out_path,
                    msa_format='sto',
                    use_precomputed_msas=self.use_precomputed_msas,
                )
                bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
            else:
                bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniref_hits.a3m')
                hhblits_bfd_uniref_result = run_msa_tool(
                    msa_runner=self.hhblits_bfd_uniref_runner,
                    input_fasta_path=input_fasta_path,
                    msa_out_path=bfd_out_path,
                    msa_format='a3m',
                    use_precomputed_msas=self.use_precomputed_msas,
                )
                bfd_msa = parsers.parse_a3m(hhblits_bfd_uniref_result['a3m'])

            templates_result = self.template_featurizer.get_templates(
                query_sequence=input_sequence, hits=pdb_template_hits
            )

            sequence_features = make_sequence_features(
                sequence=input_sequence, description=input_description, num_res=num_res
            )

            msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))

            logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
            logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
            logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
            logging.info(
                'Final (deduplicated) MSA size: %d sequences.',
                msa_features['num_alignments'][0],
            )
            logging.info(
                'Total number of templates (NB: this can include bad '
                'templates and is later filtered to top 4): %d.',
                templates_result.features['template_domain_names'].shape[0],
            )

            feature_dict = {**sequence_features, **msa_features, **templates_result.features}

        else:
            feature_dict = data_pipeline.process(
                input_fasta_path=fasta_path, msa_output_dir=msa_output_dir
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
