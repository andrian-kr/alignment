import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from adaptation.spivavtor_pipeline import SpivavtorPipeline

spivavtor_pipeline = SpivavtorPipeline()

## ETHICS
spivavtor_pipeline.improve_gec(
    file_path="../data/ethics/ethics_commonsense_claude.csv",
    output_file_path="../data/ethics/ethics_commonsense_gec.csv",
    input_column_name="input_ukr",
    output_column_name="input_ukr_spivavtor",
)

## Social Chemistry 101
spivavtor_pipeline.improve_gec(
    file_path="../data/social-chem-101/social-chem-101_claude.csv",
    output_file_path="../data/social-chem-101/social-chem-101_gec.csv",
    input_column_name="action_ukr",
    output_column_name="action_ukr_spivavtor",
)
