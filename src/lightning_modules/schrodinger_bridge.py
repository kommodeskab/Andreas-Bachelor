import torch
from typing import Tuple, Any, Callable, Literal
from torch import Tensor
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import Module
from torch_ema import ExponentialMovingAverage
from pytorch_lightning.utilities import grad_norm

class StandardDSB(BaseLightningModule):
    def __init__(
        self,
        forward_model : torch.nn.Module,
        backward_model : torch.nn.Module,
        optimizer : Optimizer,
        scheduler : LRScheduler,
        max_gamma : float,
        min_gamma : float,
        num_steps : int,
        patience : int | None = None,
        max_iterations : int | None = None,
        max_dsb_iterations : int | None = 20,
        min_iterations : int = 1,
        max_norm : float = 1,
        initial_forward_sampling: Literal["diffuse"] | None = None,
    ):
        """
        Initializes the StandardDSB model
        """
        
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore = ["forward_model", "backward_model", "optimizer"])
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
        gammas_sum = gammas.sum()
        gammas = gammas / gammas_sum
        self.gammas = gammas
        
        gammas_sum = gammas.sum()
        assert abs(1 - gammas_sum.item()) <= 1e-2, f"The sum of gammas must be equal to 1, but got {gammas_sum = }"
        
        self.forward_model : torch.nn.Module = forward_model
        self.backward_model : torch.nn.Module = backward_model

        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler
        
        self.mse : Callable[[Tensor, Tensor], Tensor] = torch.nn.MSELoss()    

    def on_fit_start(self) -> None:
        # make the ema for the forward and backward models
        # we do it here and not in init because we need the device
        forward_params = [p for p in self.forward_model.parameters() if p.requires_grad]
        backward_params = [p for p in self.backward_model.parameters() if p.requires_grad]
        self.forward_ema = ExponentialMovingAverage(forward_params, decay = 0.9999)
        self.backward_ema = ExponentialMovingAverage(backward_params, decay = 0.9999)
    
    def on_train_epoch_start(self) -> None:
        # check if the training direction is the same for the model and the datamodule
        assert self.trainer.datamodule.hparams.training_backward == self.hparams.training_backward, "The training direction must be the same for datamodule and model"
    
    def _has_converged(self) -> bool:
        params = self.hparams
        max_iters, min_iters, patience, curr_iters = params.max_iterations, params.min_iterations, params.patience, params.curr_num_iters
        has_converged = False

        if self.hparams.max_dsb_iterations is not None:
            if self.hparams.DSB_iteration > self.hparams.max_dsb_iterations:
                return True

        if patience is not None:
            val_losses = params.val_losses
            if len(val_losses) > patience: 
                prior_losses = val_losses[:-patience]
                min_prior_loss = min(prior_losses)
                recent_losses = val_losses[-patience:]
                if all(l > min_prior_loss for l in recent_losses):
                    has_converged = True
        
        if max_iters is not None:
            if curr_iters >= max_iters:
                has_converged = True

        if min_iters is not None:
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

            if self.hparams.training_backward: 
                self.hparams.DSB_iteration += 1
                
                # resetting the optimizer and scheduler
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
    
    def ornstein_uhlenbeck(self, xk : Tensor, k : int, alpha : float = 1) -> Tensor:
        mu = xk - alpha * self.gammas[k + 1] * xk
        sigma = torch.sqrt(2 * self.gammas[k + 1])
        return mu + sigma * torch.randn_like(xk)
    
    
    def _initial_go_forward(self, xk : Tensor, k : int) -> Tensor:
        if self.hparams.initial_forward_sampling is None:
            return self.go_forward(xk, k)
        
        if self.hparams.initial_forward_sampling == "diffuse":
            return self.ornstein_uhlenbeck(xk, k, alpha=3)
        
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
    
    def _backward_loss(self, xk : Tensor, ks : int, x0 : Tensor) -> Tensor:
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
        Given the start point x_start, sample the final point xN / x0 by going forward / backward
        Also, return the trajectory if return_trajectory is True
        
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
    
    def _get_model_optim_and_ema(self, is_backward : bool) -> Tuple[Module, Optimizer, ExponentialMovingAverage]:
        """
        Given the model, return the optimizer and the ema

        :param Literal["forward", "backward"] which_model: the model to get

        :return Tuple[Module, Optimizer, ExponentialMovingAverage]: the model, optimizer and ema
        """

        if is_backward:
            return self.backward_model, self.optimizers()[0], self.backward_ema
        else:
            return self.forward_model, self.optimizers()[1], self.forward_ema

    def _get_loss_name(self, is_backward : bool, is_training : bool):
        iteration = self.hparams.DSB_iteration
        direction = "backward" if is_backward else "forward"
        training = "train" if is_training else "val"
        return f"iteration_{iteration}/{direction}_loss/{training}"
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        self.hparams.curr_num_iters += 1
        training_backward = self.hparams.training_backward
        
        # using custom "cachedataloader" to deliver batches
        # therefore, most batches will be 0 (meaning we need to use cache)
        # if the batch is not 0, and therefore a tensor, then we create a new cache
        if not isinstance(batch, int):
            self.cache = self.sample(batch, forward = training_backward, return_trajectory = True)
        
        # trajectory.shape = (num_steps + 1, batch_size, *x_start.size())
        trajectory = self.cache
        
        x0, xN = trajectory[0], trajectory[-1]
        batch_size = trajectory.size(1)
        
        ks = torch.randint(0, self.hparams.num_steps, (batch_size,)).to(self.device)
        if training_backward:
            ks += 1
        all_samples = torch.arange(batch_size).to(self.device)
        sampled_batch = trajectory[ks, all_samples].to(self.device)

        # -- OPTIMIZATION --
        # get model, optimizer and ema
        model, optimizer, ema = self._get_model_optim_and_ema(training_backward)
        model.train()

        # calculate loss and do backward pass
        optimizer.zero_grad()
        loss = self._backward_loss(sampled_batch, ks, x0) if training_backward else self._forward_loss(sampled_batch, ks, xN)
        self.manual_backward(loss)

        # clip gradients and log gradients
        self.clip_gradients(optimizer, self.hparams.max_norm, "norm")
        norm = grad_norm(model, norm_type=2).get('grad_2.0_norm_total', 0)
        model_name = "backward_model" if training_backward else "forward_model"
        self.log(f"{model_name}_grad_norm", norm, prog_bar=True)

        # step the optimizer and update the ema
        optimizer.step()
        ema.update()
        self.log(self._get_loss_name(is_backward = training_backward, is_training = True), loss.item(), prog_bar=True)

    def validation_step(self, batch : Tensor, batch_idx : int, dataloader_idx : int) -> Tensor:
        if self.hparams.training_backward and dataloader_idx == 0:
            trajectory = self.sample(batch, forward = True, return_trajectory = True)
            batch_size = trajectory.size(1)
            x0 = trajectory[0]

            for k in range(1, self.hparams.num_steps + 1):
                ks = self.k_to_tensor(k, batch_size)
                loss = self._backward_loss(trajectory[k], ks, x0)
                self.log(self._get_loss_name(is_backward = True, is_training = False), loss.item(), prog_bar=True, add_dataloader_idx=False)

        elif not self.hparams.training_backward and dataloader_idx == 1:
            trajectory = self.sample(batch, forward = False, return_trajectory = True)
            batch_size = trajectory.size(1)
            xN = trajectory[-1]

            for k in range(0, self.hparams.num_steps):
                ks = self.k_to_tensor(k, batch_size)
                loss = self._forward_loss(trajectory[k], ks, xN)
                self.log(self._get_loss_name(is_backward = False, is_training = False), loss.item(), prog_bar=True, add_dataloader_idx=False)
 
    def on_validation_epoch_end(self) -> None:
        # after validation we want to update the learning rate
        backward_scheduler, forward_scheduler = self.lr_schedulers()
        metrics = self.trainer.callback_metrics

        # if no losses were computed, return
        if len(metrics) == 0:
            return
        
        # take a step based on the validation loss
        if self.hparams.training_backward:
            val_loss = metrics[self._get_loss_name(is_backward = True, is_training = False)].item()
            backward_scheduler.step(val_loss)
        else:
            val_loss = metrics[self._get_loss_name(is_backward = False, is_training = False)].item()
            forward_scheduler.step(val_loss)
        
        self.hparams.val_losses.append(val_loss)
    
    def configure_optimizers(self):
        backward_opt = self.partial_optimizer(self.backward_model.parameters())
        forward_opt = self.partial_optimizer(self.forward_model.parameters())
        backward_scheduler = {'scheduler': self.partial_scheduler(backward_opt), 'name': 'lr_scheduler_backward'}
        forward_scheduler = {'scheduler': self.partial_scheduler(forward_opt), 'name': 'lr_scheduler_forward'}
        
        return [backward_opt, forward_opt], [backward_scheduler, forward_scheduler]