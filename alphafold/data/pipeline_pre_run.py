from alphafold.data.pipeline import *

class DataPipelineNew(DataPipeline):
    """Runs the alignment tools and assembles the input features."""

    def __init__(
            self,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

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
        # breakpoint()
        jackhmmer_uniref90_result = run_msa_tool(
            msa_runner=self.jackhmmer_uniref90_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=uniref90_out_path,
            msa_format='sto',
            use_precomputed_msas=True,
            max_sto_sequences=self.uniref_max_hits,
        )
        # mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')

        jackhmmer_mgnify_result = run_msa_tool(
            msa_runner=self.jackhmmer_mgnify_runner,
            input_fasta_path=input_fasta_path,
            msa_out_path=mgnify_out_path,
            msa_format='sto',
            use_precomputed_msas=True,
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
            # bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
            jackhmmer_small_bfd_result = run_msa_tool(
                msa_runner=self.jackhmmer_small_bfd_runner,
                input_fasta_path=input_fasta_path,
                msa_out_path=bfd_out_path,
                msa_format='sto',
                use_precomputed_msas=True,
            )
            bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
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

        return {**sequence_features, **msa_features, **templates_result.features}
