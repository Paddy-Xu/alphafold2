from run_alphafold import *
from alphafold.data.pipeline import *
from pipeline_pre_run import DataPipelineNew, DataPipelineMultimerNew
from pathlib import Path
import pathlib
import shutil
import tempfile
import zstandard as zstd
import sys
import typing
from typing import List
from absl import flags

from run_no_docker import configure_run_alphafold_flags


def _normalize_db_from_path(path: pathlib.Path) -> str:
    name = path.name.lower()
    if 'mgy' in name or 'mgnify' in name:
        return 'mgnify'
    if 'uniref' in name:
        return 'uniref90'
    if 'uniprot' in name:
        return 'uniprot'
    if 'bfd' in name:
        return 'small_bfd' if 'small' in name else 'bfd'
    return 'unknown'


def _infer_tbl_dom_paths(msa_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    name = msa_path.name
    if name.endswith('.zst'):
        name = name[:-4]
    if name.endswith('.sto') or name.endswith('.a3m'):
        name = name[:-4]
    parent = msa_path.parent.parent
    tbl = parent / 'tbl_zstd' / f'{name}.tbl.zst'
    dom = parent / 'dom_zstd' / f'{name}.dom.zst'
    return tbl, dom


def _maybe_decompress_zst(src: pathlib.Path, *, suffix: str, dst_dir: pathlib.Path) -> typing.Optional[pathlib.Path]:
    if not src.exists():
        return None
    if src.suffix != '.zst':
        dst = dst_dir / src.name
        shutil.copy(src, dst)
        return dst
    dst = dst_dir / f'{src.stem}{suffix}'
    with open(src, 'rb') as fin, zstd.open(fin, 'rt') as zin, open(dst, 'w') as fout:
        fout.write(zin.read())
    return dst


def _copy_or_decompress_msa(msa_path: pathlib.Path, dst_dir: pathlib.Path) -> pathlib.Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    suffixes = ''.join(msa_path.suffixes)
    if suffixes.endswith('.zst'):
        # Drop the trailing .zst for the decompressed target.
        target = dst_dir / msa_path.name[:-4]
        with open(msa_path, 'rb') as fin, zstd.open(fin, 'rt') as zin, open(target, 'w') as fout:
            fout.write(zin.read())
    else:
        target = dst_dir / msa_path.name
        shutil.copy(msa_path, target)
    return target


def _truncate_stockholm_if_possible(msa_path: pathlib.Path, tbl_path: typing.Optional[pathlib.Path],
                                    dom_path: typing.Optional[pathlib.Path], max_depth: int,
                                    work_dir: pathlib.Path) -> pathlib.Path:
    """Truncate .sto using tbl/dom if present; otherwise return original path."""
    if msa_path.suffix != '.sto':
        return msa_path

    # Lazy import to avoid hard dependency if alphafold3 is absent.
    try:
        from alphafold3.filter_seq_limit import filter_seq_limit
    except ImportError:
        af3_src = pathlib.Path(__file__).resolve().parent / 'alphafold3' / 'src'
        if af3_src.exists():
            sys.path.insert(0, str(af3_src))
        try:
            from alphafold3.filter_seq_limit import filter_seq_limit  # type: ignore
        except Exception:
            logging.warning('alphafold3.filter_seq_limit unavailable; skipping truncation')
            return msa_path

    if tbl_path is None or dom_path is None or not tbl_path.exists() or not dom_path.exists():
        logging.warning(f'Missing tbl/dom for {msa_path.name}; skipping truncation')
        return msa_path

    truncated = work_dir / f'{msa_path.stem}.truncated.sto'
    truncated.parent.mkdir(parents=True, exist_ok=True)

    try:
        filter_seq_limit(
            TBL_IN=str(tbl_path),
            DOM_IN=str(dom_path),
            STO_IN=str(msa_path),
            STO_OUT=str(truncated),
            N_LIMIT=max_depth,
        )
        return truncated if truncated.exists() else msa_path
    except Exception as exc:
        logging.warning(f'Failed to truncate {msa_path}: {exc}')
        return msa_path


def prepare_precomputed_msas(all_dbs: list[pathlib.Path], data_pipeline, cache_dir: pathlib.Path,
                             tbl_paths: typing.Optional[List[pathlib.Path]] = None,
                             dom_paths: typing.Optional[List[pathlib.Path]] = None) -> list[pathlib.Path]:
    """Fetch MSAs locally, truncate where possible, and return new paths."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[pathlib.Path] = []
    tmp_dir = cache_dir / 'tmp'
    tmp_dir.mkdir(exist_ok=True)

    for idx, msa_path in enumerate(all_dbs):
        if not msa_path.exists():
            raise FileNotFoundError(f"MSA file not found: {msa_path}")

        db_key = _normalize_db_from_path(msa_path)
        max_depth = data_pipeline.uniref_max_hits if db_key in ['uniref90', 'uniprot', 'bfd',
                                                                'small_bfd'] else data_pipeline.mgnify_max_hits

        local_msa = _copy_or_decompress_msa(msa_path, cache_dir)

        # Use provided tbl/dom when available; otherwise infer next to original source.
        if tbl_paths and idx < len(tbl_paths):
            tbl_candidate = tbl_paths[idx]
        else:
            tbl_candidate, _ = _infer_tbl_dom_paths(msa_path)
        if dom_paths and idx < len(dom_paths):
            dom_candidate = dom_paths[idx]
        else:
            _, dom_candidate = _infer_tbl_dom_paths(msa_path)
        tbl_local = _maybe_decompress_zst(tbl_candidate, suffix='.tbl', dst_dir=tmp_dir)
        dom_local = _maybe_decompress_zst(dom_candidate, suffix='.dom', dst_dir=tmp_dir)

        truncated = _truncate_stockholm_if_possible(local_msa, tbl_local, dom_local, max_depth, cache_dir)
        prepared.append(truncated)

    return prepared


def main(argv, msa_list: list[pathlib.Path], tbl_list: list[pathlib.Path], dom_list: list[pathlib.Path]):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    run_multimer_system = 'multimer' in FLAGS.model_preset

    use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
    # Simplify: this script now supports exactly one input FASTA.
    if isinstance(FLAGS.fasta_paths, list):
        if len(FLAGS.fasta_paths) != 1:
            raise ValueError('Provide exactly one FASTA path.')
        fasta_path = FLAGS.fasta_paths[0]
    else:
        fasta_path = FLAGS.fasta_paths
    fasta_name = pathlib.Path(fasta_path).stem

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

        data_pipeline = DataPipelineMultimerNew(
            monomer_data_pipeline=monomer_data_pipeline,
            jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
            uniprot_database_path=FLAGS.uniprot_database_path,
            use_precomputed_msas=FLAGS.use_precomputed_msas,
            jackhmmer_n_cpu=FLAGS.jackhmmer_n_cpu,
        )
    else:
        data_pipeline = monomer_data_pipeline

    logging.info(f'result will be saved to {FLAGS.output_dir}')

    output_dir_base = FLAGS.output_dir

    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    msa_output_dir = os.path.join(output_dir, 'msas')
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)

    input_file = Path(fasta_path).stem

    if input_file.strip() not in all_accession:
        raise Exception(
            " input sequence not found in human protein database, please check the fasta file and make sure the sequence is correct and the fasta header is in the format of >sp|accession|entry_name")

    all_dbs = msa_list

    # Fetch/decompress MSAs locally and truncate depth similar to AF3 get_pickle_single
    prepared_dir = Path(msa_output_dir) / 'precomputed_msas'
    prepared_all_dbs = prepare_precomputed_msas(all_dbs, data_pipeline, prepared_dir,
                                                tbl_paths=tbl_list, dom_paths=dom_list)

    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path, msa_output_dir=msa_output_dir, all_dbs=prepared_all_dbs
    )
    features_output_path = os.path.join(output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)

    print(f'done for {fasta_name}, no model running needed')


if __name__ == '__main__':



    force_executor = True

    from Bio import SeqIO

    all_huamn_protein_database_path = "all_proteins_uniprotkb_organism_id_9606_AND_reviewed_2025_10_21.fasta"
    records = list(SeqIO.parse(all_huamn_protein_database_path, "fasta"))
    all_ids = [record.id for record in records]
    all_accession = [id.split("|")[1].replace(" ", "") for id in all_ids]

    root = "../../../../data"

    # root = "../../../all_msa/backup/"
    output_a3m_root = None
    output_sto_root = f'{root}/random_1000_inputs_results_jackhmmer_on_public_dbs/zstd_sto'


    USE_a3m = False

    # model_preset = 'monomer'
    model_preset = 'multimer'

    fasta_paths = 'O15552.fasta'
    # fasta_paths = 'O14818.fasta'
    # model_dir = Path(model_dir).absolute()

    new_argv = sys.argv[:]
    db_dir = Path(f'{root}/public_dataset').absolute()


    output_dir = "output_sb"

    if not any(a.startswith("--fasta_paths=") for a in new_argv):
        new_argv.append(f"--fasta_paths={fasta_paths}")

    if not any(a.startswith("--max_template_date=") for a in new_argv):
        new_argv.append(f"--max_template_date=2020-05-14")

    if not any(a.startswith("--model_preset=") for a in new_argv):
        new_argv.append(f"--model_preset={model_preset}")

    if not any(a.startswith("--db_preset=") for a in new_argv):
        new_argv.append(f"--db_preset=reduced_dbs")

    if not any(a.startswith("--data_dir=") for a in new_argv):
        new_argv.append(f"--data_dir={db_dir}")


    if not any(a.startswith("--output_dir=") for a in new_argv):
        new_argv.append(f"--output_dir={output_dir}")


    # Build explicit msa paths (and inferred tbl/dom paths) for this single fasta.
    def build_msa_paths(fasta_path_str: str) -> tuple[list[pathlib.Path], list[pathlib.Path], list[pathlib.Path]]:
        fasta_stem = pathlib.Path(fasta_path_str).stem
        sto_prefix = f"output_{fasta_stem}"
        if USE_a3m:
            root = pathlib.Path(output_a3m_root)
            suffix_map = {
                "uniref90": "_on_uniref90_2022_05_a3m.a3m",
                "small_bfd": "_on_bfd-first_non_consensus_sequences_a3m.a3m",
                "mgnify": "_on_mgy_clusters_2022_05_a3m.a3m",
                "uniprot": "_on_uniprot_all_2021_04_a3m.a3m",
            }
        else:
            root = pathlib.Path(output_sto_root)
            suffix_map = {
                "uniref90": "_on_uniref90_2022_05.sto",
                "small_bfd": "_on_bfd-first_non_consensus_sequences.sto",
                "mgnify": "_on_mgy_clusters_2022_05.sto",
                "uniprot": "_on_uniprot_all_2021_04.sto",
            }
        paths: List[pathlib.Path] = []
        for suffix in suffix_map.values():
            paths.append((root / f"{sto_prefix}{suffix}").absolute())
        tbls: List[pathlib.Path] = []
        doms: List[pathlib.Path] = []
        for p in paths:
            tbl, dom = _infer_tbl_dom_paths(p)
            tbls.append(tbl)
            doms.append(dom)
        return paths, tbls, doms


    # Single fasta_paths is enforced; accept string or single-item list for compatibility.
    if isinstance(fasta_paths, list):
        fasta_paths = fasta_paths[0]

    paths, tbls, doms = build_msa_paths(fasta_paths)

    # Parse flags, then configure paths that depend on parsed values, and call main directly.
    parsed_argv = flags.FLAGS(new_argv)  # returns remaining argv after parsing
    configure_run_alphafold_flags()
    FLAGS['pdb_seqres_database_path'].value = os.path.join(FLAGS.data_dir, 'mmcif_files')
    FLAGS['obsolete_pdbs_path'].valye = os.path.join(FLAGS.data_dir, 'obsolete.dat')

    main(parsed_argv, paths, tbls, doms)
