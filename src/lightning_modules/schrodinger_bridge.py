import torch
from typing import Tuple, Callable, List
from torch import Tensor
from src.lightning_modules.baselightningmodule import BaseLightningModule
from src.lightning_modules.utils import ckpt_path_from_id
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, ConstantLR
from torch.nn import Module
from torch_ema import ExponentialMovingAverage
from pytorch_lightning.utilities import grad_norm
import random
import copy

class Cache:
    def __init__(self, max_size : int):
        self.cache = []
        self.max_size = max_size
        
    def add(self, sample: Tensor) -> None:
        """
        Add a sample to the cache.
        """
        if len(self) >= self.max_size:
            del self.cache[0]
            
        self.cache.append(sample)
        
    def sample(self) -> Tensor:
        """
        Randomly sample a sample from the cache. The sample is removed from the cache.
        """
        randint = random.randint(0, len(self.cache) - 1)
        return self.cache[randint]
    
    def clear(self) -> None:
        """
        Clears the cache
        """
        self.cache = []
    
    def is_full(self) -> bool:
        """
        Returns whether the cache is full
        """
        return len(self) == self.max_size
        
    def __len__(self) -> int:
        return len(self.cache)

class StandardDSB(BaseLightningModule):
    def __init__(
        self,
        forward_model : torch.nn.Module,
        optimizer : Optimizer,
        scheduler : LRScheduler,
        max_gamma : float,
        min_gamma : float,
        num_steps : int,
        cache_max_size : int,
        cache_num_iters : int,
        backward_model : torch.nn.Module | None = None,
        T : int | None = None,
        lr_multiplier : float = 1.0,
        min_init_lr : float = 0.0,
        patience : int = float("inf"),
        max_iterations : int | list = float("inf"),
        min_iterations : int | list = 0,
        max_dsb_iterations : int | None = 20,
        max_norm : float = float("inf"),
        initial_forward_sampling: str | None = None,
        forward_model_id : str | None = None,
        backward_model_id : str | None = None,
        **kwargs
    ):
        """
        Initializes the StandardDSB model
        """
        
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore = ["forward_model", "backward_model", "optimizer", "scheduler"])
        self.hparams.update({
            "training_backward": True,
            "curr_num_iters": 0,
            "DSB_iteration": 1,
            "val_losses": [],
        })

        assert max_gamma >= min_gamma, f"{max_gamma = } must be greater than {min_gamma = }"
        
        gammas = torch.zeros(num_steps)
        half_steps = num_steps // 2

        if self.hparams.num_steps % 2 == 0:
            gammas[:half_steps] = torch.linspace(min_gamma, max_gamma, half_steps)
            gammas[half_steps:] = torch.linspace(max_gamma, min_gamma, half_steps)
        else:
            gammas[:half_steps + 1] = torch.linspace(min_gamma, max_gamma, half_steps + 1)
            gammas[half_steps:] = torch.flip(gammas[:half_steps + 1], [0])

        gammas = torch.cat((torch.tensor([0.0]), gammas)) # add 0 to make the correct indexing

        # make sure gammas add up to T
        if T is not None:
            gammas *= T / gammas.sum()
        else:
            self.hparams.T = gammas.sum()
            
        if isinstance(max_iterations, int):
            self.hparams.max_iterations = [max_iterations]
        if isinstance(min_iterations, int):
            self.hparams.min_iterations = [min_iterations]

        self.gammas = gammas
        self.gammas_bar = torch.cumsum(self.gammas, 0)
        
        # if the backward model is not provided, make it a copy of the forward model 
        self.forward_model : torch.nn.Module = forward_model
        self.backward_model : torch.nn.Module = copy.deepcopy(forward_model) if backward_model is None else backward_model
        
        def state_dict_from_ckpt_path(ckpt_path : str) -> dict:
            # all the keys in the state dict have the prefix "model."
            # therefore, we need to remove the prefix
            state_dict = torch.load(ckpt_path, weights_only=True)['state_dict']
            new_state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
            return new_state_dict
        
        if forward_model_id is not None:
            forward_ckpt_path = ckpt_path_from_id(forward_model_id)
            forward_state_dict = state_dict_from_ckpt_path(forward_ckpt_path)
            self.forward_model.load_state_dict(forward_state_dict)
            print(f"Successfully loaded forward model from {forward_ckpt_path}")
        else:
            self.init_weights(self.forward_model)
        
        if backward_model_id is not None:
            backward_ckpt_path = ckpt_path_from_id(backward_model_id)
            backward_state_dict = state_dict_from_ckpt_path(backward_ckpt_path)
            self.backward_model.load_state_dict(backward_state_dict)
            print(f"Successfully loaded backward model from {backward_ckpt_path}")
        else:
            self.init_weights(self.backward_model)
        
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
        
        self.mse : Callable[[Tensor, Tensor], Tensor] = torch.nn.MSELoss()  
        self.cache = Cache(max_size = self.hparams.cache_max_size)
    

    def on_fit_start(self) -> None:
        # make the ema for the forward and backward models
        # we do it here and not in init because we need the device
        forward_params = [p for p in self.forward_model.parameters() if p.requires_grad]
        backward_params = [p for p in self.backward_model.parameters() if p.requires_grad]
        self.forward_ema = ExponentialMovingAverage(forward_params, decay = 0.9999)
        self.backward_ema = ExponentialMovingAverage(backward_params, decay = 0.9999)

        self.trainer.datamodule.hparams.cache_num_iters = self.hparams.cache_num_iters
        self.trainer.datamodule.hparams.training_backward = self.hparams.training_backward
        self.trainer.datamodule.hparams.cache_max_size = self.hparams.cache_max_size

    def _has_converged(self) -> bool:
        def get_iteration_element(dsb_iteration : int, iterations_list : list[int]) -> int:
            idx = min(dsb_iteration - 1, len(iterations_list) - 1)
            return iterations_list[idx]
        
        params = self.hparams
        max_iters, min_iters, patience, curr_iters, dsb_iteration, val_losses = (
            params.max_iterations, 
            params.min_iterations, 
            params.patience, 
            params.curr_num_iters,
            params.DSB_iteration,
            params.val_losses
        )
        max_iters = get_iteration_element(dsb_iteration, max_iters)
        min_iters = get_iteration_element(dsb_iteration, min_iters)
        
        has_converged = False

        if dsb_iteration > self.hparams.max_dsb_iterations:
            print("Stopping training as max_dsb_iterations has been reached")
            raise KeyboardInterrupt

        if len(val_losses) > patience: 
            prior_losses = val_losses[:-patience]
            min_prior_loss = min(prior_losses)
            recent_losses = val_losses[-patience:]
            if all(l > min_prior_loss for l in recent_losses):
                has_converged = True
        
        if curr_iters >= max_iters:
            has_converged = True

        if curr_iters < min_iters:
            has_converged = False

        return has_converged

    def on_train_batch_start(self, batch : Tensor, batch_idx : int) -> None:
        """
        We check if the model has converged before each batch
        if the model has converged we do the following:
        - switch the training direction
        - reset the learning rate of the optimizer
        - reset the learning rate scheduler
        
        we also return -1 to skip the rest of the epoch if converged (pytorch lightning behavior)
        
        """

        if self._has_converged():
            self.hparams.curr_num_iters = 0
            self.hparams.val_losses = []
            self.hparams.training_backward = not self.hparams.training_backward
            self.trainer.datamodule.hparams.training_backward = self.hparams.training_backward
            self.cache.clear()

            if self.hparams.training_backward: 
                self.hparams.DSB_iteration += 1
                # resetting the optimizers and schedulers
                self._reset_optim_and_scheduler()

            return -1
        
    def _reset_optim_and_scheduler(self) -> None:
        """
        Resets the optimizer and the scheduler
        """
        optimizers, schedulers = self.configure_optimizers()
        self.trainer.optimizers[0] = optimizers[0]
        self.trainer.optimizers[1] = optimizers[1]
        
        self.trainer.lr_scheduler_configs[0].scheduler = schedulers[0]['scheduler']
        self.trainer.lr_scheduler_configs[1].scheduler = schedulers[1]['scheduler']
    
    def k_to_tensor(self, k : int, size : Tuple[int]) -> Tensor:
        """
        Given k, return a tensor of size 'size' filled with k

        :param int k: the value to fill the tensor with
        :param Tuple[int] size: the size of the tensor

        :return Tensor: the tensor filled with k
        """
        return torch.full((size, ), k, dtype = torch.int64, device = self.device)
    
    def forward_call(self, x : Tensor, k: Tensor) -> Tensor:
        """
        Calls the forward model
        """
        return self.forward_model(x, k)
    
    def backward_call(self, x : Tensor, k : Tensor) -> Tensor:
        """
        Calls the backward model
        """
        return self.backward_model(x, k)
    
    def ornstein_uhlenbeck(self, xk : Tensor, k : int, alpha : float = 1.0) -> Tensor:        
        mu = (1 - alpha * self.gammas[k + 1]) * xk
        sigma = torch.sqrt(2 * self.gammas[k + 1])
        return mu + sigma * torch.randn_like(xk)

    def _initial_go_forward(self, xk : Tensor, k : int) -> Tensor:
        if self.hparams.initial_forward_sampling is None:
            return self.go_forward(xk, k)
        
        if "ornstein" in self.hparams.initial_forward_sampling:
            # the string comes in the format "ornstein_0.5"
            alpha = float(self.hparams.initial_forward_sampling.split("_")[1])
            return self.ornstein_uhlenbeck(xk, k, alpha)
        
        raise ValueError(f"Invalid initial_forward_sampling: {self.hparams.initial_forward_sampling}")
        
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        """        
        Get :math:`x_{k + 1} | x_{k}`
        
        :param Tensor xk: the current point
        :param int k: the current step
        
        :return xk_plus_one: the next point
        """
        raise NotImplementedError
    
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        """
        Get :math:`x_{k} | x_{k + 1}`
        
        :param Tensor xk_plus_one: the current point
        :param int k_plus_one: the current step
        
        :return xk: the previous point
        """
        raise NotImplementedError
    
    def _forward_loss(self, xk : Tensor, ks : int, xN : Tensor) -> Tensor:
        """
        Compute the loss for the forward model.
        Uses the backward model to 'walk' backwards and optimize the forward model to 'walk' back to the original point
        
        :param Tensor xk: the current point
        :param Tensor ks: the current step
        :param Tensor xN: the end point
        
        :return loss: the loss and the previous point
        """
        raise NotImplementedError
    
    def _backward_loss(self, xk_plus_one : Tensor, ks_plus_one : int, x0 : Tensor) -> Tensor:
        """
        Compute the loss for the backward model.
        Uses the forward model to 'walk' forward and optimize the backward model to 'walk' back to the original point
        
        :param Tensor xk: the current point
        :param Tensor ks: the current step
        :param Tensor x0: the start point
        
        :return loss: the loss and the next point
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def sample(self, x_start : Tensor, forward : bool = True, return_trajectory : bool = False, clamp : bool = False, ema_scope : bool = False) -> Tensor:
        """
        Given the start point x_start, sample the final point xN / x0 by going forward / backward.
        Also, return the trajectory if return_trajectory is True.
        
        :param Tensor x_start: the start point
        :param bool forward: whether to go forward or backward
        :param bool return_trajectory: whether to return the trajectory
        
        :return Tensor: the final point xN / x0 or the trajectory
        """
        if ema_scope:
            ema = self.forward_ema if forward else self.backward_ema
            with ema.average_parameters():
                return self.sample(x_start, forward = forward, return_trajectory = return_trajectory, clamp = clamp, ema_scope = False)
        
        trajectory = torch.zeros(self.hparams.num_steps + 1, *x_start.size()).to(self.device)
        
        if forward:
            self.forward_model.eval()
            xk = x_start
            trajectory[0] = xk
            for k in range(self.hparams.num_steps): # 0, 1, 2, ..., num_steps - 1
                if self.hparams.DSB_iteration == 1 and self.hparams.training_backward:
                    xk_plus_one = self._initial_go_forward(xk, k)
                else:
                    xk_plus_one = self.go_forward(xk, k)
                trajectory[k + 1] = xk_plus_one
                xk = xk_plus_one

        else:
            self.backward_model.eval()
            xk_plus_one = x_start
            trajectory[-1] = xk_plus_one
            for k in reversed(range(self.hparams.num_steps)): # num_steps - 1, num_steps - 2, ..., 1, 0
                xk = self.go_backward(xk_plus_one, k + 1)
                trajectory[k] = xk
                xk_plus_one = xk
        
        if clamp:
            trajectory = torch.clamp(trajectory, -1, 1)
            xk = torch.clamp(xk, -1, 1)

        return trajectory if return_trajectory else xk
    
    def _get_training_components(self, is_backward : bool) -> Tuple[Module, Optimizer, ExponentialMovingAverage, LRScheduler]:
        """
        Given the model, return the optimizer and the ema

        :param bool is_backward: whether to get the backward model or the forward model

        :return Tuple[Module, Optimizer, ExponentialMovingAverage]: the model, optimizer and ema
        """
        backward_optim, forward_optim = self.optimizers()
        backward_scheduler, forward_scheduler = self.lr_schedulers()

        if is_backward:
            return self.backward_model, backward_optim, self.backward_ema, backward_scheduler
        else:
            return self.forward_model, forward_optim, self.forward_ema, forward_scheduler

    def _get_loss_name(self, is_backward : bool, is_training : bool):
        iteration = self.hparams.DSB_iteration
        direction = "backward" if is_backward else "forward"
        training = "train" if is_training else "val"
        return f"iteration_{iteration}/{direction}_loss/{training}"
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        training_backward = self.hparams.training_backward
        
        trajectory = self.sample(batch, forward = training_backward, return_trajectory = True, ema_scope=True)
        self.cache.add(trajectory)
        
        cache_is_full = self.cache.is_full()
        
        num_cache_iters = self.hparams.cache_num_iters if cache_is_full else 1
        for _ in range(num_cache_iters):
            self.hparams.curr_num_iters += 1
            
            if cache_is_full:
                trajectory = self.cache.sample()

            x0, xN = trajectory[0], trajectory[-1]
            batch_size = trajectory.size(1)
            
            ks = torch.randint(0, self.hparams.num_steps, (batch_size,)).to(self.device)
            if training_backward:
                ks += 1
            all_samples = torch.arange(batch_size).to(self.device)
            sampled_batch = trajectory[ks, all_samples].to(self.device)

            # -- OPTIMIZATION --
            # get model, optimizer and ema
            model, optimizer, ema, lr_scheduler = self._get_training_components(training_backward)
            model.train()

            # calculate loss and do backward pass
            optimizer.zero_grad()
            # if training backward, then sampled_batch = xk + 1 else sampled_batch = xk
            loss = self._backward_loss(sampled_batch, ks, x0) if training_backward else self._forward_loss(sampled_batch, ks, xN)
            self.manual_backward(loss)
            
            # raise exception if loss is nan
            if torch.isnan(loss).any():
                raise ValueError(f"Loss is nan: {loss.item() = }")
            
            if loss.item() > 1.e6:
                raise ValueError(f"Loss is too high: {loss.item() = }")

            # clip the gradients. first, save the norm for later logging
            norm = grad_norm(model, norm_type=2).get('grad_2.0_norm_total', 0)
            self.clip_gradients(optimizer, self.hparams.max_norm, "norm")

            # step the optimizer and update the ema
            optimizer.step()
            ema.update()
            
            model_name = "backward_model" if training_backward else "forward_model"
            self.logger.log_metrics({
                f"{model_name}_grad_norm_before_clip": norm,
                self._get_loss_name(is_backward = training_backward, is_training = True): loss.item(),
                "Training iteration": self.hparams.curr_num_iters
            }, step = self.hparams.curr_num_iters)
            
            # update scheduler
            if isinstance(lr_scheduler, (CosineAnnealingWarmRestarts, ConstantLR)):
                lr_scheduler.step(self.hparams.curr_num_iters)

    def validation_step(self, batch : Tensor, batch_idx : int, dataloader_idx : int) -> Tensor:
        self.eval()
        
        if self.hparams.training_backward and dataloader_idx == 0:
            trajectory = self.sample(batch, forward = True, return_trajectory = True, ema_scope=True)
            batch_size = trajectory.size(1)
            x0 = trajectory[0]

            for k in range(1, self.hparams.num_steps + 1):
                ks = self.k_to_tensor(k, batch_size)
                loss = self._backward_loss(trajectory[k], ks, x0)
                self.log(self._get_loss_name(is_backward = True, is_training = False), loss.item(), prog_bar=True, add_dataloader_idx=False)

        elif not self.hparams.training_backward and dataloader_idx == 1:
            trajectory = self.sample(batch, forward = False, return_trajectory = True, ema_scope=True)
            batch_size = trajectory.size(1)
            xN = trajectory[-1]

            for k in range(0, self.hparams.num_steps):
                ks = self.k_to_tensor(k, batch_size)
                loss = self._forward_loss(trajectory[k], ks, xN)
                self.log(self._get_loss_name(is_backward = False, is_training = False), loss.item(), prog_bar=True, add_dataloader_idx=False)
 
    def on_validation_epoch_end(self) -> None:
        # after validation we want to update the learning rate
        is_backward = self.hparams.training_backward
        backward_scheduler, forward_scheduler = self.lr_schedulers()
        scheduler = backward_scheduler if is_backward else forward_scheduler

        metrics = self.trainer.callback_metrics

        # if no losses were computed, return
        if len(metrics) == 0:
            return
        
        val_loss = metrics[self._get_loss_name(is_backward = is_backward, is_training = False)].item()
        self.hparams.val_losses.append(val_loss)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
    
    def configure_optimizers(self):
        # make the optimizers
        backward_opt = self.partial_optimizer(self.backward_model.parameters())
        forward_opt = self.partial_optimizer(self.forward_model.parameters())

        # the learning rate should drop every DSB iteration
        dsb_iteration = self.hparams.DSB_iteration
        lr_multiplier = self.hparams.lr_multiplier
        min_init_lr = self.hparams.min_init_lr
        
        lr = backward_opt.param_groups[0]['lr']
        new_lr = max(lr * lr_multiplier ** (dsb_iteration - 1), min_init_lr)
        backward_opt.param_groups[0]['lr'] = new_lr
        forward_opt.param_groups[0]['lr'] = new_lr

        backward_scheduler = {'scheduler': self.partial_scheduler(backward_opt), 'name': 'lr_scheduler_backward'}
        forward_scheduler = {'scheduler': self.partial_scheduler(forward_opt), 'name': 'lr_scheduler_forward'}
        
        sch = backward_scheduler['scheduler']
        assert isinstance(sch, (CosineAnnealingWarmRestarts, ReduceLROnPlateau, ConstantLR)), f"The scheduler must be either CosineAnnealingWarmRestarts or ReduceLROnPlateau, but got {type(sch)}"
        
        return [backward_opt, forward_opt], [backward_scheduler, forward_scheduler]