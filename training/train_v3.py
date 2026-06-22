import subprocess
from pathlib import Path

from dragon_baseline.main import AutoTokenizer, DragonBaseline, ProblemType


class ReportAnonymizerTrainer(DragonBaseline):
    def __init__(self, **kwargs):
        # Example of how to adapt the DRAGON baseline to use a different model
        """
        Adapt the DRAGON baseline to use the joeranbosma/dragon-roberta-large-mixed-domain model.
        """
        super().__init__(**kwargs)
        self.model_name = "joeranbosma/dragon-roberta-large-mixed-domain"
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 8
        self.gradient_checkpointing = True
        self.max_seq_length = 512
        self.learning_rate = 1e-05

    def train(self):
        """Train the model."""
        # save the preprocessed data for training through command line interface of the HuggingFace library
        for path in [
            self.nlp_dataset_train_preprocessed_path,
            self.nlp_dataset_val_preprocessed_path,
            self.nlp_dataset_test_preprocessed_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.df_train.to_json(self.nlp_dataset_train_preprocessed_path, orient="records")
        self.df_val.to_json(self.nlp_dataset_val_preprocessed_path, orient="records")
        self.df_test.to_json(self.nlp_dataset_test_preprocessed_path, orient="records")

        # load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, truncation_side=self.task.recommended_truncation_side)
        tokenizer.model_max_length = self.max_seq_length  # set the maximum sequence length, if not already set

        # train the model
        cmd = [
            "python", "run_ner_strided.py",
            "--do_train",
            "--do_eval",
            "--do_predict",
            "--learning_rate", self.learning_rate,
            "--model_name_or_path", self.model_name,
            "--ignore_mismatched_sizes",
            "--num_train_epochs", self.num_train_epochs,
            "--warmup_ratio", self.warmup_ratio,
            "--max_seq_length", self.max_seq_length,
            "--truncation_side", self.task.recommended_truncation_side,
            "--load_best_model_at_end", self.load_best_model_at_end,
            "--save_strategy", "epoch",
            "--evaluation_strategy", "epoch",
            "--per_device_train_batch_size", self.per_device_train_batch_size,
            "--gradient_accumulation_steps", self.gradient_accumulation_steps,
            "--gradient_checkpointing", self.gradient_checkpointing,
            "--train_file", self.nlp_dataset_train_preprocessed_path,
            "--validation_file", self.nlp_dataset_val_preprocessed_path,
            "--test_file", self.nlp_dataset_test_preprocessed_path,
            "--output_dir", self.model_save_dir,
            "--overwrite_output_dir",
            "--save_total_limit", "2",
            "--seed", self.task.jobid,
            "--report_to", "none",
            "--text_column_name", self.task.input_name,
            "--remove_columns", "uid",
        ]
        if self.task.target.problem_type in [
            ProblemType.MULTI_LABEL_REGRESSION,
            ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION,
        ]:
            label_names = [col for col in self.df_train.columns if col.startswith(f"{self.task.target.label_name}_")]
            cmd.extend([
                "--label_column_names", ",".join(label_names),
            ])
        else:
            cmd.extend([
                "--label_column_name", self.task.target.label_name,
            ])
        if not self.task.target.problem_type in [ProblemType.SINGLE_LABEL_NER, ProblemType.MULTI_LABEL_NER]:
            cmd.extend([
                "--text_column_delimiter", tokenizer.sep_token,
            ])
        if self.metric_for_best_model is not None:
            cmd.extend([
                "--metric_for_best_model", str(self.metric_for_best_model),
            ])
        if self.fp16:
            cmd.append("--fp16")

        cmd = [str(arg) for arg in cmd]
        print("Training command:")
        print(" ".join(cmd))
        subprocess.check_call(cmd)


if __name__ == "__main__":
    for fold in range(1):
        ReportAnonymizerTrainer(
            input_path=Path(f"/input/algorithm-input/Task301_anonymisation_ner-fold{fold}"),
            output_path=Path(f"/output/Task301_anonymisation_ner-fold{fold}"),
            workdir=Path(f"/workdir/Task301_anonymisation_ner-fold{fold}"),
        ).process()
