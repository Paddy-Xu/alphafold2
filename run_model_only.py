from run_alphafold import *
# force import underlined _ functions from run_alphafold
from run_alphafold import _jnp_to_np, _save_confidence_json_file, _save_mmcif_file, _save_pae_json_file

from alphafold.data.pipeline import *
from pathlib import Path

from run_no_docker import configure_run_alphafold_flags



def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    features_output_path: str,
    # data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    models_to_relax: ModelsToRelax,
    model_type: str,
):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  # output_dir = os.path.join(output_dir_base, fasta_name)
  # if not os.path.exists(output_dir):
  #   os.makedirs(output_dir)
  # msa_output_dir = os.path.join(output_dir, 'msas')
  # if not os.path.exists(msa_output_dir):
  #   os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  # feature_dict = data_pipeline.process(
  #     input_fasta_path=fasta_path, msa_output_dir=msa_output_dir
  # )

  timings['features'] = time.time() - t_0

  # Write out features as a pickled dictionary.

  # with open(features_output_path, 'wb') as f:
  #   pickle.dump(feature_dict, f, protocol=4)
  with open(features_output_path, 'rb') as f:
      feature_dict = pickle.load(f)


  unrelaxed_pdbs = {}
  unrelaxed_proteins = {}
  relaxed_pdbs = {}
  relax_metrics = {}
  ranking_confidences = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()
  ):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed
    )
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(
        processed_feature_dict, random_seed=model_random_seed
    )
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see'
        ' --benchmark): %.1fs',
        model_name,
        fasta_name,
        t_diff,
    )

    if benchmark:
      t_0 = time.time()
      model_runner.predict(
          processed_feature_dict, random_seed=model_random_seed
      )
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (excludes compilation time):'
          ' %.1fs',
          model_name,
          fasta_name,
          t_diff,
      )

    plddt = prediction_result['plddt']
    _save_confidence_json_file(plddt, output_dir, model_name)
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    if (
        'predicted_aligned_error' in prediction_result
        and 'max_predicted_aligned_error' in prediction_result
    ):
      pae = prediction_result['predicted_aligned_error']
      max_pae = prediction_result['max_predicted_aligned_error']
      _save_pae_json_file(pae, float(max_pae), output_dir, model_name)

    # Remove jax dependency from results.
    np_prediction_result = _jnp_to_np(dict(prediction_result))

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(np_prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1
    )
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode,
    )

    unrelaxed_proteins[model_name] = unrelaxed_protein
    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

    _save_mmcif_file(
        prot=unrelaxed_protein,
        output_dir=output_dir,
        model_name=f'unrelaxed_{model_name}',
        file_id=str(model_index),
        model_type=model_type,
    )

  # Rank by model confidence.
  ranked_order = [
      model_name
      for model_name, confidence in sorted(
          ranking_confidences.items(), key=lambda x: x[1], reverse=True
      )
  ]

  # Relax predictions.
  if models_to_relax == ModelsToRelax.BEST:
    to_relax = [ranked_order[0]]
  elif models_to_relax == ModelsToRelax.ALL:
    to_relax = ranked_order
  elif models_to_relax == ModelsToRelax.NONE:
    to_relax = []

  for model_name in to_relax:
    t_0 = time.time()
    relaxed_pdb_str, _, violations = amber_relaxer.process(
        prot=unrelaxed_proteins[model_name]
    )
    relax_metrics[model_name] = {
        'remaining_violations': violations,
        'remaining_violations_count': sum(violations),
    }
    timings[f'relax_{model_name}'] = time.time() - t_0

    relaxed_pdbs[model_name] = relaxed_pdb_str

    # Save the relaxed PDB.
    relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
    with open(relaxed_output_path, 'w') as f:
      f.write(relaxed_pdb_str)

    relaxed_protein = protein.from_pdb_string(relaxed_pdb_str)
    _save_mmcif_file(
        prot=relaxed_protein,
        output_dir=output_dir,
        model_name=f'relaxed_{model_name}',
        file_id='0',
        model_type=model_type,
    )

  # Write out relaxed PDBs in rank order.
  for idx, model_name in enumerate(ranked_order):
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if model_name in relaxed_pdbs:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

    if model_name in relaxed_pdbs:
      protein_instance = protein.from_pdb_string(relaxed_pdbs[model_name])
    else:
      protein_instance = protein.from_pdb_string(unrelaxed_pdbs[model_name])

    _save_mmcif_file(
        prot=protein_instance,
        output_dir=output_dir,
        model_name=f'ranked_{idx}',
        file_id=str(idx),
        model_type=model_type,
    )

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(
        json.dumps(
            {label: ranking_confidences, 'order': ranked_order}, indent=4
        )
    )

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))
  if models_to_relax != ModelsToRelax.NONE:
    relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
    with open(relax_metrics_path, 'w') as f:
      f.write(json.dumps(relax_metrics, indent=4))


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
        num_predictions_per_model = FLAGS.num_multimer_predictions_per_model

    else:
        num_predictions_per_model = 1

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

        output_dir_base = FLAGS.output_dir
        logging.info('Predicting %s', fasta_name)
        output_dir = os.path.join(output_dir_base, fasta_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        msa_output_dir = os.path.join(output_dir, 'msas')
        if not os.path.exists(msa_output_dir):
            os.makedirs(msa_output_dir)
        features_output_path = os.path.join(output_dir, 'features.pkl')

        assert os.path.exists(fasta_path), f"{fasta_path} not found"

        predict_structure(
            fasta_path=fasta_path,
            fasta_name=fasta_name,
            output_dir_base=FLAGS.output_dir,
            features_output_path=features_output_path,
            # data_pipeline=data_pipeline,
            model_runners=model_runners,
            amber_relaxer=amber_relaxer,
            benchmark=FLAGS.benchmark,
            random_seed=random_seed,
            models_to_relax=FLAGS.models_to_relax,
            model_type=model_type,
        )


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
