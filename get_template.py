from pipeline_pre_run import *
import os
import tempfile
import zstandard as zstd
import shutil
import boto3
import sqlite3
import resource
import sys
import gc
from pathlib import Path
import pathlib
from Bio import SeqIO


print("import pickle_single done", flush=True)

DB_PATH = "../../af3_self/src/api/index_s3_all_0303_local.sqlite"
DB_PATH  = (Path(__file__).resolve().parent / DB_PATH).resolve()
s3 = boto3.client("s3")


def is_s3(path: str) -> bool:
    return path.startswith("s3://")


def parse_s3_url(url: str):
    _, _, rest = url.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def cleanup_files(paths) -> None:
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def ensure_local(path_str: str, cleanup_paths: list[Path]) -> Path:
    if is_s3(path_str):
        bucket, key = parse_s3_url(path_str)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(path_str).suffix)
        s3.download_file(bucket, key, tmp.name)
        cleanup_paths.append(Path(tmp.name))
        return Path(tmp.name)
    return Path(path_str)


def decompress_zst(path: Path, cleanup_paths: list[Path], suffix: str | None = None) -> Path:
    if path.suffix != ".zst":
        return path
    target_suffix = suffix or "".join(path.suffixes[:-1]) or ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=target_suffix, mode="wb")
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fin, dctx.stream_reader(fin) as reader:
        shutil.copyfileobj(reader, tmp, length=1024 * 1024)
    tmp.close()
    out_path = Path(tmp.name)
    cleanup_paths.append(out_path)
    return out_path


def decompress_zst_to_sto(path: Path, cleanup_paths: list[Path]) -> Path:
    return decompress_zst(path, cleanup_paths, suffix=".sto")



if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print(os.path.abspath('.'), flush=True)

    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Memory usage: {mem_kb / 1024 / 1024:.2f} GB", flush=True)

    all_huamn_protein_database_path = "../../all_proteins_uniprotkb_organism_id_9606_AND_reviewed_2025_10_21.fasta"

    all_huamn_protein_database_path = (Path(__file__).resolve().parent / all_huamn_protein_database_path).resolve()


    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Memory usage: {mem_kb / 1024 / 1024:.2f} GB", flush=True)

    save_root = (Path(__file__).resolve().parent / "../../../all_a3ms").resolve()

    print("Processing UniProt IDs in streaming mode...", flush=True)


    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Memory usage after DataPipeline init: {mem_kb / 1024 / 1024:.2f} GB", flush=True)

    # ✅ CHANGE 2: open sqlite once
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    i = 0
    for record in SeqIO.parse(all_huamn_protein_database_path, "fasta"):
        fields = record.id.split("|")
        if len(fields) < 2:
            continue
        uniprot_id = fields[1].replace(" ", "")
        sequence = str(record.seq)

        mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Memory usage: {mem_kb / 1024 / 1024:.2f} GB at start of {i} loop for {uniprot_id}", flush=True)

        file_paths = {}
        dom_paths = {}
        tbl_paths = {}

        output_dir = save_root
        output_path = pathlib.Path(output_dir)
        output_path = output_path / uniprot_id / 'msas'

        output_path.mkdir(parents=True, exist_ok=True)

        unpaired_path = output_path / f"unpaired.a3m"
        paired_path = output_path / f"paired.a3m"
        force_recompute = False
        if (not force_recompute) and unpaired_path.exists() and paired_path.exists() and unpaired_path.stat().st_size > 0 and paired_path.stat().st_size > 0:
            print(f"MSA files already exist for {uniprot_id}, skipping generation.")
            i += 1
            continue

        missing_db = False

        too_large = False

        for db_name in ["uniref90", "uniprot_all", "mgnify", "small_bfd"]:

            cur.execute("""
                        SELECT file_path, file_size, dom_path, tbl_path, n_seqs
                        FROM artifacts
                        WHERE uniprot = ?
                          AND db_name = ?
                          AND format = 'sto' LIMIT 1
                        """, (uniprot_id, db_name))

            row = cur.fetchone()

            # ✅ CHANGE 4: avoid None crash
            if row is None:
                print(f"No DB entry for {uniprot_id} {db_name}, skipping this UniProt.")
                missing_db = True
                break

            file_paths[db_name] = row["file_path"]
            dom_paths[db_name] = row["dom_path"]
            tbl_paths[db_name] = row["tbl_path"]

            n_seqs = row["n_seqs"]
            if n_seqs is None:
                print(f"Skipping {uniprot_id} {db_name}: no n_seqs",
                      flush=True)
                too_large = True
                break

            elif n_seqs > 10000:
                print(f"Skipping {uniprot_id} {db_name}: n_seqs is {n_seqs}, which is too large",
                      flush=True)
                too_large = True
                break

        if missing_db or too_large:
            i += 1
            continue

        cleanup_paths: list[Path] = []

        final_dict = {}
        final_dict_dom = {}
        final_dict_tbl = {}

        try:
            for db_name, path in file_paths.items():
                base_path = ensure_local(path, cleanup_paths)
                base_path = decompress_zst_to_sto(base_path, cleanup_paths)
                final_dict[db_name] = str(base_path)

            # for db_name, path in dom_paths.items():
            #     base_path = ensure_local(path, cleanup_paths)
            #     base_path = decompress_zst_to_sto(base_path, cleanup_paths)
            #     final_dict_dom[db_name] = str(base_path)
            #
            # for db_name, path in tbl_paths.items():
            #     base_path = ensure_local(path, cleanup_paths)
            #     base_path = decompress_zst_to_sto(base_path, cleanup_paths)
            #     final_dict_tbl[db_name] = str(base_path)

            print(f'completed decompression and local preparation for {uniprot_id}, now saving MSAs...', flush=True)

            uniref90_out_path = final_dict["uniref90"]
            pdb_hits_out_path = output_path / f"{uniprot_id}_pdb_hits.txt"

            uniref_max_hits: int = 10000
            template_searcher =  TemplateSearcher
            template_featurizer = templates.TemplateHitFeaturizer

            jackhmmer_uniref90_result = run_msa_tool(
                msa_runner=None,
                input_fasta_path=None,
                msa_out_path=uniref90_out_path,
                msa_format='sto',
                use_precomputed_msas=True,
                max_sto_sequences=uniref_max_hits,
            )

            msa_for_templates = jackhmmer_uniref90_result['sto']
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

            with open(pdb_hits_out_path, 'w') as f:
                f.write(pdb_templates_result)


        except Exception as e:
            print(f"Error processing {uniprot_id}, skipping. {e}", flush=True)

        finally:
            cleanup_files(cleanup_paths)
            del final_dict, final_dict_dom, final_dict_tbl, file_paths, dom_paths, tbl_paths
            gc.collect()
            i += 1

    # close db once
    conn.close()


