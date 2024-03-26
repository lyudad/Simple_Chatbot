from openai import OpenAI
import os
import signal
import datetime
import time

OPENAI_API_KEY = ''
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


class FineTuneLLM:

    def __init__(self, train_file_path, validation_file_path):
        self.client = OpenAI()
        self.train_file_path = train_file_path
        self.validation_file_path = validation_file_path
        self.model_id = "gpt-3.5-turbo"

    def _define_files_id(self) -> (str, str):
        training_file_id = self.client.files.create(
            file=open(self.train_file_path, "rb"),
            purpose="fine-tune"
        )

        validation_file_id = self.client.files.create(
            file=open(self.validation_file_path, "rb"),
            purpose="fine-tune"
        )
        return training_file_id, validation_file_id

    def _define_ft_job(self, train_file_id, val_file_id) -> str:
        response = self.client.fine_tuning.jobs.create(
            training_file=train_file_id,
            validation_file=val_file_id,
            model=self.model_id,
            hyperparameters={
                "n_epochs": 15,
                "batch_size": 3,
                "learning_rate_multiplier": 0.3
            }
        )
        print(f'Fine-tunning model with jobID: {response.id}.')
        print(f"Training Status: {response.status}")
        return response.id

    def _signal_handler(self, job_id: str) -> str:
        status = self.client.fine_tuning.jobs.retrieve(job_id).status
        print(f"Stream interrupted. Job is still {status}.")
        return status

    def run_ft_job(self):
        train_file_id, validation_file_id = self._define_files_id()
        job_id = self._define_ft_job(
            train_file_id=train_file_id.id,
            val_file_id=validation_file_id.id
        )

        signal.signal(signal.SIGINT, self.signal_handler(job_id))

        events = self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
        try:
            for event in events:
                print(f'{datetime.datetime.fromtimestamp(event.created_at)} {event.message}')
        except Exception:
            print("Stream interrupted (client disconnected).")

    # def _monitor_job_status(self, job_id: str) -> str:
    #     status = self.client.fine_tuning.jobs.retrieve(job_id).status
    #     if status not in ["succeeded", "failed"]:
    #         print(f"Job not in terminal status: {status}. Waiting.")
    #         while status not in ["succeeded", "failed"]:
    #             time.sleep(2)
    #             status = self.client.fine_tuning.jobs.retrieve(job_id).status
    #     else:
    #         print(f"Finetune job {job_id} finished with status: {status}")
    #     result = self.client.fine_tuning.jobs.list()
    #     print(f"Found {len(result.data)} finetune jobs.")
    #     return result

