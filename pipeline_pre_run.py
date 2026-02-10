from alphafold.data.pipeline import *
import alphafold.data.pipeline as pipeline

import alphafold.data.pipeline_multimer as pipeline_multimer
from alphafold.data.pipeline_multimer import *

class DataPipelineNew(pipeline.DataPipeline):
    """Runs the alignment tools and assembles the input features."""

    def __init__(
            self,
            *,
            jackhmmer_binary_path: str,
            hhblits_binary_path: str,
            uniref90_database_path: str,
            mgnify_database_path: str,
            bfd_database_path: Optional[str],
            uniref30_database_path: Optional[str],
            small_bfd_database_path: Optional[str],
            template_searcher: TemplateSearcher,
            template_featurizer: templates.TemplateHitFeaturizer,
            use_small_bfd: bool,
            mgnify_max_hits: int = 501,
            uniref_max_hits: int = 10000,
            use_precomputed_msas: bool = False,
            msa_tools_n_cpu: int = 8,
    ):
        """Initializes the data pipeline."""
        self._use_small_bfd = use_small_bfd
        self.jackhmmer_uniref90_runner = None
        if use_small_bfd:
            self.jackhmmer_small_bfd_runner = None
        else:
            self.hhblits_bfd_uniref_runner = hhblits.HHBlits(
                binary_path=hhblits_binary_path,
                databases=[bfd_database_path, uniref30_database_path],
                n_cpu=msa_tools_n_cpu,
            )
        self.jackhmmer_mgnify_runner = None
        self.template_searcher = template_searcher
        self.template_featurizer = template_featurizer
        self.mgnify_max_hits = mgnify_max_hits
        self.uniref_max_hits = uniref_max_hits
        self.use_precomputed_msas = use_precomputed_msas

    def process(self, input_fasta_path: str, msa_output_dir: str, all_dbs) -> FeatureDict:
        """Runs alignment tools on the input sequence and creates features."""
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

        # uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
        for path in all_dbs:
            if "uniref90" in path.stem:
                uniref90_out_path = path
            elif "mgy" in path.stem or "mgnify" in path.stem:
                mgnify_out_path = path
            elif "bfd" in path.stem:
                if self._use_small_bfd:
                    bfd_out_path = path
                else:
                    raise ValueError("ful bfd database not supported yet")
            else:
                assert "uniprot" in path.stem
                uniprot_out_path = path

        if 'sto' in uniref90_out_path.suffix:
            msa_format = 'sto'
        else:
            assert 'a3m' in uniref90_out_path.suffix
            msa_format = 'a3m'

        print('msa format = ', msa_format, '')
        print('msa_output_dir = ', msa_output_dir, '')

        pdb_hits_out_path = os.path.join(
            msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}'
        )

        # breakpoint()
        jackhmmer_uniref90_result = run_msa_tool(
            msa_runner=self.jackhmmer_uniref90_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=uniref90_out_path,
            msa_format=msa_format,
            use_precomputed_msas=True,
            max_sto_sequences=self.uniref_max_hits,
        )
        # mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')

        jackhmmer_mgnify_result = run_msa_tool(
            msa_runner=self.jackhmmer_mgnify_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=mgnify_out_path,
            msa_format=msa_format,
            use_precomputed_msas=True,
            max_sto_sequences=self.mgnify_max_hits,
        )

        if os.path.exists(pdb_hits_out_path):
            print(f'pdb_hits_out_path exists at {pdb_hits_out_path}, reading from it')
            with open(pdb_hits_out_path, 'r') as f:
                pdb_templates_result = f.read()

        else:
            msa_for_templates = jackhmmer_uniref90_result[msa_format]
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

            with open(pdb_hits_out_path, 'w') as f:
                f.write(pdb_templates_result)


        if self._use_small_bfd:
            # bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
            jackhmmer_small_bfd_result = run_msa_tool(
                msa_runner=self.jackhmmer_small_bfd_runner,
                input_fasta_path=input_fasta_path,
                msa_out_path=bfd_out_path,
                msa_format=msa_format,
                use_precomputed_msas=True,
            )
            bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result[msa_format])
        else:
            raise ValueError("ful bfd database not supported yet")
            # bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniref_hits.a3m')
            hhblits_bfd_uniref_result = run_msa_tool(
                msa_runner=self.hhblits_bfd_uniref_runner,
                input_fasta_path=input_fasta_path,
                msa_out_path=bfd_out_path,
                msa_format='a3m',
                use_precomputed_msas=True,
            )
            bfd_msa = parsers.parse_a3m(hhblits_bfd_uniref_result['a3m'])
        uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result[msa_format])
        mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result[msa_format])


        # templates_result -> TemplateSearchResult => a dataclass object

        templates_result_out_path = os.path.join(
            msa_output_dir, f'templates_result.pkl'
        )

        import pickle
        if os.path.exists(templates_result_out_path):
            print(f'templates_result_out_path exists at {templates_result_out_path}, reading from it')
            with open(templates_result_out_path, 'rb') as f:
                templates_result = pickle.load(f)
        else:
            pdb_template_hits = self.template_searcher.get_template_hits(
                output_string=pdb_templates_result, input_sequence=input_sequence
            )

            templates_result = self.template_featurizer.get_templates(
                query_sequence=input_sequence, hits=pdb_template_hits
            )
        with open(templates_result_out_path, 'wb') as f:
            pickle.dump(templates_result, f, protocol=4)



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

        return {**sequence_features, **msa_features, **templates_result.features}


### actually only adding all_dbs in process and _process_single_chain


@dataclasses.dataclass(frozen=True)
class _FastaChain:
  sequence: str
  description: str
def _make_chain_id_map(
    *,
    sequences: Sequence[str],
    descriptions: Sequence[str],
) -> Mapping[str, _FastaChain]:
  """Makes a mapping from PDB-format chain ID to sequence and description."""
  if len(sequences) != len(descriptions):
    raise ValueError(
        'sequences and descriptions must have equal length. '
        f'Got {len(sequences)} != {len(descriptions)}.'
    )
  if len(sequences) > protein.PDB_MAX_CHAINS:
    raise ValueError(
        'Cannot process more chains than the PDB format supports. '
        f'Got {len(sequences)} chains.'
    )
  chain_id_map = {}
  for chain_id, sequence, description in zip(
      protein.PDB_CHAIN_IDS, sequences, descriptions
  ):
    chain_id_map[chain_id] = _FastaChain(
        sequence=sequence, description=description
    )
  return chain_id_map


class DataPipelineMultimerNew(pipeline_multimer.DataPipeline):
    """Runs the alignment tools and assembles the input features."""

    def __init__(
            self,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

    def _process_single_chain(
            self,
            chain_id: str,
            sequence: str,
            description: str,
            msa_output_dir: str,
            is_homomer_or_monomer: bool,
            all_dbs: Sequence[str],
    ) -> pipeline.FeatureDict:
        """Runs the monomer pipeline on a single chain."""
        chain_fasta_str = f'>chain_{chain_id}\n{sequence}\n'
        chain_msa_output_dir = os.path.join(msa_output_dir, chain_id)
        if not os.path.exists(chain_msa_output_dir):
            os.makedirs(chain_msa_output_dir)
        with temp_fasta_file(chain_fasta_str) as chain_fasta_path:
            logging.info(
                'Running monomer pipeline on chain %s: %s', chain_id, description
            )

            chain_features = self._monomer_data_pipeline.process(
                input_fasta_path=chain_fasta_path, msa_output_dir=chain_msa_output_dir, all_dbs=all_dbs
            )

            # We only construct the pairing features if there are 2 or more unique
            # sequences.
            if not is_homomer_or_monomer:
                all_seq_msa_features = self._all_seq_msa_features(
                    chain_fasta_path, chain_msa_output_dir
                )
                chain_features.update(all_seq_msa_features)
        return chain_features

    def process(
            self, input_fasta_path: str, msa_output_dir: str, all_dbs
    ) -> pipeline.FeatureDict:
        """Runs alignment tools on the input sequences and creates features."""
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)

        chain_id_map = _make_chain_id_map(
            sequences=input_seqs, descriptions=input_descs
        )
        chain_id_map_path = os.path.join(msa_output_dir, 'chain_id_map.json')
        with open(chain_id_map_path, 'w') as f:
            chain_id_map_dict = {
                chain_id: dataclasses.asdict(fasta_chain)
                for chain_id, fasta_chain in chain_id_map.items()
            }
            json.dump(chain_id_map_dict, f, indent=4, sort_keys=True)

        all_chain_features = {}
        sequence_features = {}
        is_homomer_or_monomer = len(set(input_seqs)) == 1
        for chain_id, fasta_chain in chain_id_map.items():
            if fasta_chain.sequence in sequence_features:
                all_chain_features[chain_id] = copy.deepcopy(
                    sequence_features[fasta_chain.sequence]
                )
                continue

            chain_features = self._process_single_chain(
                chain_id=chain_id,
                sequence=fasta_chain.sequence,
                description=fasta_chain.description,
                msa_output_dir=msa_output_dir,
                is_homomer_or_monomer=is_homomer_or_monomer,

                all_dbs=all_dbs

            )


            chain_features = convert_monomer_features(
                chain_features, chain_id=chain_id
            )
            all_chain_features[chain_id] = chain_features
            sequence_features[fasta_chain.sequence] = chain_features

        all_chain_features = add_assembly_features(all_chain_features)

        np_example = feature_processing.pair_and_merge(
            all_chain_features=all_chain_features
        )

        # Pad MSA to avoid zero-sized extra_msa.
        np_example = pad_msa(np_example, 512)

        return np_example
