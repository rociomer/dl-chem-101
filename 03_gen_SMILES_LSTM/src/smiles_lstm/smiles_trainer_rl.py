"""
SMILES LSTM reinforcement learning trainer class.
"""
import glob
import csv
from typing import Union, Tuple
import os
from os import remove
from copy import deepcopy
from statistics import mean
import rdkit
import torch
import numpy as np
from malmo.models.smiles_lstm import SmilesLSTM
from malmo.benchmarks.scoring_function import ScoringFunction
from malmo.utils import suppress_warnings, draw_smiles, plot_scores, save_smiles

# suppress minor warnings
suppress_warnings()

rdkit.rdBase.DisableLog("rdApp.error")


class SmilesTrainerRL():
    """
    Trains a SMILES-based generative model using reinforcement learning.
    """
    def __init__(self, model: SmilesLSTM, scoring_function: ScoringFunction,
                 score_type : str="binary",
                 learning_rate : float=0.0001, batch_size : int=64,
                 n_steps : int=10, sigma : float=0.5, eval_num_samples : int=64,
                 eval_batch_size : int=64, output_model_path : str="./output/",
                 label : str="", learning_rate_scheduler : str="StepLR", gamma : float=0.8, 
                 inner_loop_model : bool=False, inner_loop_label : str="outerStepX_subtaskY") -> None:
        """
        Args:
        ----
            model (SmilesLSTM)                 : The SMILES LSTM generative model.
            scoring_function (ScoringFunction) : Scoring function to use.
            score_type (str)                   : TODO
            learning_rate (float, optional)    : Initial learning rate. Defaults
                                                 to 0.0001.
            n_steps (int, optional)            : Number of steps to train for.
                                                 Defaults to 10.
            batch_size (int, optional)         : Batch size to use for training.
                                                 Defaults to 250.
            sigma (int, optional)              : Weight for the score in the augmented
                                                 log likelihood. Defaults to 20.
            eval_num_samples (int, optional)   : Number of samples to use for evaluating
                                                 the model. Defaults to 16.
            eval_batch_size (int, optional)    : Batch size to use for evaluating
                                                 the model. Defaults to 16.
            output_model_path (str, optional)  : Directory in which to save the
                                                 results. Defaults to "./output/".
            label (str, optional)              : A label indicating something
                                                 about the model (e.g. the
                                                 scoring function). Defaults
                                                 to an empty string.
            learning_rate_scheduler (str)      : TODO
            gamma (float)                      : TODO
            inner_loop_model (bool)            : Indicates if this model occurs in the inner loop of a meta learning job.
        """
        # define the model
        self._prior = model
        self._agent = deepcopy(model)
        self._scoring_function = scoring_function
        self._score_type       = score_type
        self._label            = label

        # define parameters
        self._batch_size        = batch_size
        self._learning_rate     = learning_rate
        self._output_model_path = output_model_path
        self._n_steps           = n_steps
        self._sigma             = sigma
        self._eval_num_samples  = eval_num_samples
        self._eval_batch_size   = eval_batch_size
        self._save_models       = bool(not inner_loop_model)

        # define the optimizer and scheduler
        self._optimizer = torch.optim.Adam(params=self._agent.network.parameters(),
                                           lr=self._learning_rate)

        if learning_rate_scheduler == "CosineAnnealingLR":
            # cosine annealing schedule
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self._optimizer,
                T_max=self._n_steps,
            )
        elif learning_rate_scheduler == "StepLR":
            # step learning rate annealing schedule
            self._scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self._optimizer,
                step_size=1,
                gamma=gamma,  # source: https://github.com/MolecularAI/Reinvent/blob/master/running_modes/configurations/transfer_learning/adaptive_learning_rate_configuration.py
            )
        else:
            raise ValueError("Please enter a valid value for ´learning_rate_scheduler´.")

        # create a directory for the model if one does not exist already
        if not os.path.exists(self._output_model_path):
            os.makedirs(self._output_model_path)
        
        # define paths for the output files
        if not inner_loop_model:
            params_filename                  = f"{self._output_model_path}SmilesTrainerRL_params.csv"
            self._training_output_filename   = f"{self._output_model_path}SmilesTrainerRL_training.csv"
            self._evaluation_output_filename = f"{self._output_model_path}SmilesTrainerRL_evaluation.csv"
        else:
            params_filename                  = f"{self._output_model_path}SmilesTrailerRL_{inner_loop_label}_params.csv"
            self._training_output_filename   = f"{self._output_model_path}SmilesTrailerRL_{inner_loop_label}_training.csv"
            self._evaluation_output_filename = f"{self._output_model_path}SmilesTrailerRL_{inner_loop_label}_evaluation.csv"

        # write the basic hyperparameters to a CSV file
        with open(params_filename, "w") as params_file:
            params_writer = csv.writer(params_file)
            param_names  = ["model type", "batch size", "learning rate", "num steps", "sigma", "eval num samples", "eval batch size", "score type", "label", "learning rate scheduler"]
            param_values = ["SmilesTrainerRL", self._batch_size, self._learning_rate, self._n_steps, self._sigma, self._eval_num_samples, self._eval_batch_size, self._score_type, self._label, learning_rate_scheduler]
            params_writer.writerow(param_names)  # the header is the parameter names
            params_writer.writerow(param_values)

        # placeholders for the loss and likelihoods
        self._loss                  = None
        self._batch_loss            = None
        self._agent_likelihood      = None
        self._augmented_likelihood  = None
        self._prior_likelihood      = None

        # create the output file for training metrics
        with open(self._training_output_filename, "w") as training_file:
            training_writer = csv.writer(training_file)
            header = ["step", "training task", "learning rate", "training loss", "agent likelihood", "prior likelihood", "augmented likelihood"]
            training_writer.writerow(header)

        # create the output file for evaluation metrics
        with open(self._evaluation_output_filename, "w") as evaluation_file:
            evaluation_writer = csv.writer(evaluation_file)
            header = ["step", "evaluation task", "mean docking score", "fraction valid"]
            evaluation_writer.writerow(header)

    def run(self, start_step : int=0, update : bool=True) -> None:
        """
        Fine-tune the model for the specified number of policy gradient steps.

        Args:
        ----
            start_step (int)    : Index at which to start counting steps. Defaults to 0.
            update (bool)       : If True, updates the network after computing the loss.
                                  Defaults to True.
        Returns:
        -------
            loss (float)        : Returns the loss. # TODO previously returned False e.g. if `update` is False.
        """
        self._disable_prior_gradients()
        
        # begin training
        for step in range(start_step, start_step + self._n_steps):
            self._train_step(update_grads=update)

            # write progress to the output CSV
            learning_rate = self._optimizer.param_groups[0]["lr"]
            with open(self._training_output_filename, "a") as training_file:
                training_writer = csv.writer(training_file)
                progress = [step, self._label, self._get_float(learning_rate), self._get_float(self._loss), self._get_float(self._agent_likelihood), self._get_float(self._prior_likelihood), self._get_float(self._augmented_likelihood)]
                training_writer.writerow(progress)

            # evaluate the model for the task
            self.task_eval(steps=step)

            # save a checkpoint if desired
            if self._save_models:
                self._save_checkpoint(step=step)

        return self._loss

    def _get_float(self, value : Union[torch.Tensor, float]) -> float:
        try:
            float_value = value.item()
        except:
            float_value = value
        return float_value

    def _save_checkpoint(self, step : int) -> None:
        """
        Save a checkpoint of the current agent, and also keep track of the best
        agent likelihood seen so far.

        Args:
        ----
            step (int) : Current reinforcement learning step.
        """
        self._save_agent(step)

    def _disable_prior_gradients(self) -> None:
        """
        Disable prior gradients (there may be a more elegant way).
        """
        for param in self._prior.network.parameters():
            param.requires_grad = False

    def _train_step(self, update_grads) -> None:
        """
        Perform a single training step.
        """
        def to_tensor(tensor):
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            if torch.cuda.is_available():
                return torch.autograd.Variable(tensor).cuda()
            return torch.autograd.Variable(tensor)

        self._agent.network.train()
        seqs, smiles, agent_likelihood = self._sample_unique_sequences(agent=self._agent,
                                                                       batch_size=self._batch_size)
        # switch signs
        agent_likelihood = -agent_likelihood
        prior_likelihood = -self._prior.likelihood(seqs)
        score            = self._scoring_function.compute_score(smiles_list=smiles,
                                                                seqs=seqs)

        augmented_likelihood = prior_likelihood + self._sigma * to_tensor(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        self._batch_loss           = loss
        self._loss                 = self._batch_loss.mean()
        self._agent_likelihood     = agent_likelihood.mean()
        self._augmented_likelihood = augmented_likelihood.mean()
        self._prior_likelihood     = prior_likelihood.mean()

        if update_grads:
            self._optimizer.zero_grad()
            self._loss.backward()
            self._optimizer.step()
            self._scheduler.step(self._loss)

    def _save_agent(self, step : int, path : Union[str, None]=None) -> None:
        """
        Save the current agent.

        Args:
        ----
            step (int) : Current reinforcement learning step.
        """
        if path is None:
            self._agent.save_state(path=self._agent_path(step))
        else:
            self._agent.save_state(path=path)

    def _agent_path(self, step : Union[int, str]="*") -> str:
        """
        Return the path in which to save the current agent.

        Args:
        ----
            step (int or str) : Current reinforcement learning step. Use * for
                                regex. Defaults to "*".

        Returns:
        -------
            path (str) : Path to use for saving the agent.
        """
        path = f"{self._output_model_path}model.agent.{step}.{self._label}.pth"
        return path

    def _sample_unique_sequences(self, agent : SmilesLSTM, batch_size : int) -> \
                                 Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """
        Sample unique SMILES sequences from the current agent.

        Args:
        ----
            agent (SmilesLSTM) : The current agent.
            batch_size (int)   : Batch size to use for sampling.

        Returns:
        -------
            seqs_unique (torch.Tensor)             : Contains unique sequences.
            smiles_unique (np.ndarray)             : Contains unique SMILES.
            agent_likelihood_unique (torch.Tensor) : Contains negative log-likelihoods
                                                     for sampling the output
                                                     sequences via the current agent.
        """
        def get_indices_of_unique_smiles(smiles : list) -> np.ndarray:
            """
            Returns an array of indices corresponding to the first entries in
            a list of SMILES strings.
            """
            _, idxs = np.unique(smiles, return_index=True)
            sorted_indices = np.sort(idxs)
            return sorted_indices

        seqs, smiles, agent_likelihood = agent.sample_sequences_and_smiles(batch_size)
        unique_idxs = get_indices_of_unique_smiles(smiles)
        seqs_unique = seqs[unique_idxs]
        smiles_np = np.array(smiles)
        smiles_unique = smiles_np[unique_idxs]
        agent_likelihood_unique = agent_likelihood[unique_idxs]
        return seqs_unique, smiles_unique, agent_likelihood_unique

    def task_eval(self, tasks : Union[list, None]=None, steps : int=0):
        """
        Sample molecules from the current model for evaluation. Uses multiprocessing
        for computing the docking scores.

        Args:
        ----
            tasks (list) :
            steps (int)  : Step number used as a label in the output file.
        """
        if tasks is None:
            tasks = self._scoring_function.score_components

        tasks.sort()

        # sample some molecules using the fine-tuned model and compute the fraction valid
        sampled_smiles, nlls = self._agent.sample_smiles(num=self._eval_num_samples,
                                                         batch_size=self._eval_batch_size)

        fraction_valid = draw_smiles(
            path=f"{self._output_model_path}sampled_step{steps}_{self._label}.png",
            smiles_list=sampled_smiles
        )

        # predict the docking scores for each of the targets
        for subtask in tasks:

            # define the scoring function
            scoring_function = ScoringFunction(score_components=[subtask],
                                               score_thresholds=None,
                                               score_type=self._score_type)
            docking_scores = [float(i) for i in
                list(scoring_function.get_contributions_to_score(smiles_list=sampled_smiles,
                                                                 validity_list=[None])[0])
            ]
            mean_docking_score = -1.0 * mean(docking_scores)  # flip the sign

            if subtask in self._label:
                scores = [-1.0 * i for i in docking_scores]
                plot_scores(scores=scores,
                            output_filename=f"{self._output_model_path}scores_step{steps}_{subtask}.png")
                save_smiles(smiles=sampled_smiles,
                            output_filename=f"{self._output_model_path}sampled_smiles_step{steps}_{subtask}.smi")

            # write the evaluation metrics to output
            with open(self._evaluation_output_filename, "a") as evaluation_file:
                evaluation_writer = csv.writer(evaluation_file)
                progress = [steps, subtask, mean_docking_score, fraction_valid]
                evaluation_writer.writerow(progress)
