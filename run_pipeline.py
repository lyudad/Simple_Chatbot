from fine_tune_llm import FineTuneLLM
import requests


def run_fine_tuning_pipeline():
    fine_tune_llm = FineTuneLLM(train_file_path='chat_train.jsonl', validation_file_path='chat_validation.jsonl')
    fine_tune_llm.run_ft_job()


if __name__ == '__main__':
    run_fine_tuning_pipeline()
