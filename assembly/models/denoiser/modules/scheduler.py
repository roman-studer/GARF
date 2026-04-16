import math
from typing import Union, Tuple, Literal
from dataclasses import dataclass

import torch
from diffusers import SchedulerMixin, ConfigMixin, DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils import BaseOutput
from diffusers.configuration_utils import register_to_config
from pytorch3d import transforms as p3dt


@dataclass
class SE3FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, 7)` for translations and scalar first quaternions):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class SE3FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        stochastic_paths: bool = False,
        stochastic_level: float = 0.1,
        min_stochastic_epsilon: float = 0.01,
        sigma_schedule: Literal[
            "linear",
            "piecewise-linear",
            "piecewise-quadratic",
            "exponential",
        ] = "linear",
    ):
        super().__init__()
        timesteps = torch.flip(
            torch.linspace(1, num_train_timesteps, num_train_timesteps), dims=[0]
        )
        sigmas = torch.tensor(
            [self._sigma_schedule(t, num_train_timesteps) for t in timesteps],
            dtype=torch.float32,
        )
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self._step_index = None
        self._begin_index = None

    def _calc_stochastic_epsilon(self, sigma: torch.FloatTensor):
        return torch.sqrt(
            self.config.get("stochastic_level") ** 2 * sigma * (1 - sigma)
            + self.config.get("min_stochastic_epsilon")
        )

    def _sigma_schedule(
        self,
        t: torch.FloatTensor,
        num_timesteps: int = 1000,
    ):
        t = t * 1000 / num_timesteps  # rescale t to [0, 1000]
        if self.config.get("sigma_schedule") == "linear":
            return t / 1000
        elif self.config.get("sigma_schedule") == "piecewise-linear":
            if t <= 700:
                return t / 700 * 0.1
            else:
                return 0.1 + (t - 700) / 300 * 0.9
        elif self.config.get("sigma_schedule") == "piecewise-quadratic":
            if t <= 700:
                return 0.1 * (t / 700) ** 2
            else:
                return 0.1 + 0.9 * ((t - 700) / 300) ** 2
        elif self.config.get("sigma_schedule") == "exponential":
            return math.exp(-5 * (1 - t / 1000))
        else:
            raise ValueError(
                f"Invalid sigma schedule: {self.config.get('sigma_schedule')}"
            )

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma: torch.FloatTensor) -> torch.FloatTensor:
        return sigma * self.config.get("num_train_timesteps")

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        sigmas: torch.FloatTensor = None,
    ):
        """
        Set the timesteps for the scheduler.

        By default, we use linspace from sigma_max to sigma_min with num_inference_steps.
        If sigmas is provided, we will use it directly.
        """
        if sigmas is None:
            timesteps = torch.flip(
                torch.linspace(1, num_inference_steps, num_inference_steps), dims=[0]
            )
            sigmas = torch.tensor(
                [self._sigma_schedule(t, num_inference_steps) for t in timesteps],
                dtype=torch.float32,
            )
        else:
            num_inference_steps = len(sigmas)

        self.timesteps = sigmas * self.config.get("num_train_timesteps")

        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def _scale_noise_for_translation(
        self,
        x_0_trans: torch.FloatTensor,  # (B, 3)
        sigma: torch.FloatTensor,  # (B)
        x_1_trans: torch.FloatTensor,  # (B, 3)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward process for translations from x_0 to x_1.
        Args:
            x_0_trans: (B, 3) tensor, ground truth translations.
            sigma: (B) tensor, sigmas for each sample
            x_1_trans: (B, 3) tensor, pure noise translations.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]:
                - x_t_trans: (B, 3) tensor, noisy translations.
                - trans_vec_field: (B, 3) tensor, translation vector field.
        """
        sigma = sigma.unsqueeze(-1)
        x_t_trans = (1 - sigma) * x_0_trans + sigma * x_1_trans
        if self.config.get("stochastic_paths"):
            x_t_trans += torch.randn_like(x_t_trans) * self._calc_stochastic_epsilon(
                sigma
            )

        trans_vec_field = x_1_trans - x_0_trans
        return (x_t_trans, trans_vec_field)

    def _scale_noise_for_rotation(
        self,
        x_0_rot: torch.FloatTensor,  # (B, 4)
        sigma: torch.FloatTensor,  # (B)
        x_1_rot: torch.FloatTensor,  # (B, 4)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward process for rotations from x_0 to x_1.
        Args:
            x_0_rot: (B, 4) tensor, ground truth rotations, scalar first quaternions.
            sigma: (B) tensor, sigmas for each sample
            x_1_rot: (B, 4) tensor, pure noise rotations, scalar first quaternions.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]:
                - x_t_rot: (B, 4) tensor, noisy rotations, scalar first quaternions.
                - rot_vec_field: (B, 3) tensor, rotation vector field.
        """
        sigma = sigma.unsqueeze(-1)
        x_0_rot_mat = p3dt.quaternion_to_matrix(x_0_rot)
        x_1_rot_mat = p3dt.quaternion_to_matrix(x_1_rot)

        # Calculate the rotation vector field
        rot_vec_field = p3dt.matrix_to_axis_angle(x_1_rot_mat)
        x_t_rot_mat = p3dt.axis_angle_to_matrix(sigma * rot_vec_field) @ x_0_rot_mat
        if self.config.get("stochastic_paths"):
            epsilon_t = self._calc_stochastic_epsilon(sigma)
            x_t_rot_mat = x_t_rot_mat @ p3dt.axis_angle_to_matrix(
                epsilon_t * torch.randn_like(rot_vec_field)
            )

        x_t_rot = p3dt.matrix_to_quaternion(x_t_rot_mat)
        return (x_t_rot, rot_vec_field)

    def scale_noise(
        self,
        sample: torch.FloatTensor,  # (B, 7)
        timestep: torch.FloatTensor,  # (B)
        noise: torch.FloatTensor,  # (B, 7)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            sample (`torch.FloatTensor`): (B, 7) tensor, translations and scalar first quaternions.
            timesteps (`torch.FloatTensor`): (B) tensor, timesteps for each sample.
            noise (`torch.FloatTensor`): (B, 7) tensor, noise for each sample.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]:
                - x_t: (B, 7) tensor, noisy translations and scalar first quaternions.
                - vec_field: (B, 6) tensor, translation and rotation vector fields.
        """
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)
        schedule_timesteps = self.timesteps.to(sample.device)
        timestep = timestep.to(sample.device)

        step_indices = [
            self.index_for_timestep(t, schedule_timesteps) for t in timestep
        ]
        sigma = sigmas[step_indices].flatten()

        x_t_trans, trans_vec_field = self._scale_noise_for_translation(
            sample[..., :3], sigma, noise[..., :3]
        )
        x_t_rots, rot_vec_field = self._scale_noise_for_rotation(
            sample[..., 3:], sigma, noise[..., 3:]
        )

        return torch.cat([x_t_trans, x_t_rots], dim=-1), torch.cat(
            [trans_vec_field, rot_vec_field], dim=-1
        )

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def _step_for_translation(
        self,
        vec_field: torch.FloatTensor,
        delta_sigma: torch.FloatTensor,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            vec_field (`torch.FloatTensor`): (B, 3) tensor, translation vector field.
            delta_sigma (`torch.FloatTensor`): (B) tensor.
            sample (`torch.FloatTensor`): (B, 3) tensor, sample translations.

        Returns:
            prev_sample (`torch.FloatTensor`): (B, 3) tensor, denoised translations.
        """
        prev_sample = sample + delta_sigma * vec_field
        if self.config.get("stochastic_paths"):
            prev_sample += (
                self.config.get("stochastic_level")
                * torch.sqrt(-delta_sigma)
                * torch.randn_like(vec_field)
            )
        return prev_sample

    def _step_for_rotation(
        self,
        vec_field: torch.FloatTensor,
        delta_sigma: torch.FloatTensor,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            vec_field (`torch.FloatTensor`): (B, 3) tensor, rotation vector field.
            delta_sigma (`torch.FloatTensor`): (B) tensor.
            sample (`torch.FloatTensor`): (B, 4) tensor, sample rotations, scalar first quaternions.

        Returns:
            prev_sample (`torch.FloatTensor`): (B, 4) tensor, denoised rotations, scalar first quaternions.
        """
        prev_sample = p3dt.axis_angle_to_matrix(
            delta_sigma * vec_field
        ) @ p3dt.quaternion_to_matrix(sample)
        if self.config.get("stochastic_paths"):
            z = (
                self.config.get("stochastic_level")
                * torch.sqrt(-delta_sigma)
                * torch.randn_like(vec_field)
            )
            prev_sample = prev_sample @ p3dt.axis_angle_to_matrix(z)

        prev_sample = p3dt.matrix_to_quaternion(prev_sample)
        return prev_sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
    ) -> SE3FlowMatchEulerDiscreteSchedulerOutput:
        """
        Args:
            model_output (`torch.FloatTensor`):
                The model output. Should be a tuple of (trans_vec_field, rot_vec_field), each of shape (B, 3).
            timestep (`Union[float, torch.FloatTensor]`):
                The current timestep.
            sample (`torch.FloatTensor`):
                The sample for the current timestep of shape (B, 7).
                First 3 elements are translation
                Last 4 elements are quaternion, scalar first.

        Returns:
            `torch.FloatTensor`:
                The denoised sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        # Avoid precision issues
        sample = sample.to(torch.float32)
        # sigma_next - sigma < 0
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        delta_sigma = sigma_next - sigma

        prev_sample_trans = self._step_for_translation(
            model_output[..., :3],
            delta_sigma,
            sample[..., :3],
        )

        prev_sample_rot = self._step_for_rotation(
            model_output[..., 3:],
            delta_sigma,
            sample[..., 3:],
        )

        prev_sample = torch.cat([prev_sample_trans, prev_sample_rot], dim=-1)
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        return SE3FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps


"""
Following are adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""


def betas_for_alpha_bar(
    num_diffusion_timesteps=1000,
    max_beta=0.999,
    alpha_transform_type="piece_wise",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    elif alpha_transform_type == "piece_wise":

        def alpha_bar_fn(t):
            t = t * 1000
            if t <= 700:
                # Quadratic decrease from 1 to 0.9 between x = 0 to 700
                return 1 - 0.1 * (t / 700) ** 2
            else:
                # Quadratic decrease from 0.9 to 0 between x = 700 to 1000
                return 0.9 * (1 - ((t - 700) / 300) ** 2)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class PiecewiseScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.betas = betas_for_alpha_bar(alpha_transform_type="piece_wise")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)


class SE3PiecewiseScheduler(PiecewiseScheduler):
    def add_noise(
        self,
        original_samples: torch.Tensor,  # (B, 7) trans+quat
        noise: torch.Tensor,  # (B, 6)
        timesteps: torch.Tensor,
    ):
        translations = original_samples[:, :3]
        quaternions = original_samples[:, 3:]
        rot_matrics = p3dt.quaternion_to_matrix(quaternions)
        log_rot = p3dt.so3_log_map(rot_matrics)  # (B, 3)

        trans_noise = noise[:, :3]
        rot_noise = noise[:, 3:]

        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_translations = (
            sqrt_alpha_prod * translations + sqrt_one_minus_alpha_prod * trans_noise
        )
        noisy_log_rot = (
            sqrt_alpha_prod * log_rot + sqrt_one_minus_alpha_prod * rot_noise
        )
        noisy_rot_matrics = p3dt.so3_exp_map(noisy_log_rot)
        noisy_quaternions = p3dt.matrix_to_quaternion(noisy_rot_matrics)

        noisy_samples = torch.cat([noisy_translations, noisy_quaternions], dim=1)
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator=None,
        return_dict=True,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        translations = sample[:, :3]
        quaternions = sample[:, 3:]
        rot_matrics = p3dt.quaternion_to_matrix(quaternions)
        log_rot = p3dt.so3_log_map(rot_matrics)  # (B, 3)

        if self.config.prediction_type == "epsilon":
            pred_trans = (
                translations - beta_prod_t**0.5 * model_output[:, :3]
            ) / alpha_prod_t**0.5
            pred_log_rot = (
                log_rot - beta_prod_t**0.5 * model_output[:, 3:6]
            ) / alpha_prod_t**0.5
        elif self.config.prediction_type == "sample":
            pred_trans = model_output[:, :3]
            pred_log_rot = model_output[:, 3:6]
        elif self.config.prediction_type == "v_prediction":
            pred_trans = (alpha_prod_t**0.5) * translations - (
                beta_prod_t**0.5
            ) * model_output[:, :3]
            pred_log_rot = (alpha_prod_t**0.5) * log_rot - (
                beta_prod_t**0.5
            ) * model_output[:, 3:6]
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_trans_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_trans_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        pred_log_rot_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_log_rot_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_trans = (
            pred_trans_coeff * pred_trans + current_trans_coeff * translations
        )
        pred_prev_log_rot = (
            pred_log_rot_coeff * pred_log_rot + current_log_rot_coeff * log_rot
        )

        # 6. Add noise
        variance = torch.zeros_like(model_output)
        if t > 0:
            device = model_output.device
            variance_noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            if self.variance_type == "fixed_small_log":
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance)
                    * variance_noise
                )
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance) ** 0.5
                ) * variance_noise

        pred_prev_trans = pred_prev_trans + variance[:, :3]
        pred_prev_log_rot = pred_prev_log_rot + variance[:, 3:6]

        pred_prev_rot_matrices = p3dt.so3_exp_map(pred_prev_log_rot)
        pred_prev_quaternions = p3dt.matrix_to_quaternion(pred_prev_rot_matrices)

        pred_prev_sample = torch.cat(
            [
                pred_prev_trans,
                pred_prev_quaternions,
            ],
            dim=1,
        )

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample,
        )


class SE3DDPMScheduler(DDPMScheduler):
    def add_noise(
        self,
        original_samples: torch.Tensor,  # (B, 7) trans+quat
        noise: torch.Tensor,  # (B, 6)
        timesteps: torch.Tensor,
    ):
        translations = original_samples[:, :3]
        quaternions = original_samples[:, 3:]
        rot_matrics = p3dt.quaternion_to_matrix(quaternions)
        log_rot = p3dt.so3_log_map(rot_matrics)  # (B, 3)

        trans_noise = noise[:, :3]
        rot_noise = noise[:, 3:]

        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_translations = (
            sqrt_alpha_prod * translations + sqrt_one_minus_alpha_prod * trans_noise
        )
        noisy_log_rot = (
            sqrt_alpha_prod * log_rot + sqrt_one_minus_alpha_prod * rot_noise
        )
        noisy_rot_matrics = p3dt.so3_exp_map(noisy_log_rot)
        noisy_quaternions = p3dt.matrix_to_quaternion(noisy_rot_matrics)

        noisy_samples = torch.cat([noisy_translations, noisy_quaternions], dim=1)
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator=None,
        return_dict=True,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        translations = sample[:, :3]
        quaternions = sample[:, 3:]
        rot_matrics = p3dt.quaternion_to_matrix(quaternions)
        log_rot = p3dt.so3_log_map(rot_matrics)  # (B, 3)

        if self.config.prediction_type == "epsilon":
            pred_trans = (
                translations - beta_prod_t**0.5 * model_output[:, :3]
            ) / alpha_prod_t**0.5
            pred_log_rot = (
                log_rot - beta_prod_t**0.5 * model_output[:, 3:6]
            ) / alpha_prod_t**0.5
        elif self.config.prediction_type == "sample":
            pred_trans = model_output[:, :3]
            pred_log_rot = model_output[:, 3:6]
        elif self.config.prediction_type == "v_prediction":
            pred_trans = (alpha_prod_t**0.5) * translations - (
                beta_prod_t**0.5
            ) * model_output[:, :3]
            pred_log_rot = (alpha_prod_t**0.5) * log_rot - (
                beta_prod_t**0.5
            ) * model_output[:, 3:6]
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_trans_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_trans_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        pred_log_rot_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_log_rot_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_trans = (
            pred_trans_coeff * pred_trans + current_trans_coeff * translations
        )
        pred_prev_log_rot = (
            pred_log_rot_coeff * pred_log_rot + current_log_rot_coeff * log_rot
        )

        # 6. Add noise
        variance = torch.zeros_like(model_output)
        if t > 0:
            device = model_output.device
            variance_noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            if self.variance_type == "fixed_small_log":
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance)
                    * variance_noise
                )
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance) ** 0.5
                ) * variance_noise

        pred_prev_trans = pred_prev_trans + variance[:, :3]
        pred_prev_log_rot = pred_prev_log_rot + variance[:, 3:6]

        pred_prev_rot_matrices = p3dt.so3_exp_map(pred_prev_log_rot)
        pred_prev_quaternions = p3dt.matrix_to_quaternion(pred_prev_rot_matrices)

        pred_prev_sample = torch.cat(
            [
                pred_prev_trans,
                pred_prev_quaternions,
            ],
            dim=1,
        )

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample,
        )


if __name__ == "__main__":
    scheduler = SE3FlowMatchEulerDiscreteScheduler()
    print(scheduler.sigmas)
