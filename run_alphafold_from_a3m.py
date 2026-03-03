import sys

# Import original module
import run_alphafold as run_alphafold

from pipeline_pre_run import *

# Replace the class reference
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer

pipeline.DataPipeline = DataPipelineNew
pipeline_multimer.DataPipeline = DataPipelineMultimerNew

DataPipelineNew.process = DataPipelineNew.process_a3m


# Now call original main
if __name__ == "__main__":
    # app.run(main)

    run_alphafold.main(sys.argv)