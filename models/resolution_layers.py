import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np

OptionalDim = Union[int, None]
Number = Union[float, int]
Stride = Union[Number, Tuple[Number, Number]]
Rate = Union[Number, Tuple[Number, Number]]
Shape = Union[Tuple[int, int], Tuple[None, None]]
Shape = Union[Tuple[int, int], Tuple[None, None]]


try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))

    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)

    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = (
        -torch.arange(N, dtype=x.dtype, device=x.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


# %%
def compute_adaptive_span_mask(
    threshold, ramp_softness, pos, dtype=torch.float32
):
    """Adaptive mask as proposed in https://arxiv.org/pdf/1905.07799.pdf.

    Parameters
    ----------
    threshold : torch.float32
         Threshold that starts the ramp.
    ramp_softness : torch.float32
        Smoothness of the ramp.
    pos : torch.Tensor
        Position indices.
    dtype : torch.dtype, optional
        Datatype to return, by default torch.complex64

    Returns
    -------
    torch.Tensor
        A mask for the given threshold, ramp softness, and position
    """
    output = (1.0 / ramp_softness) * (ramp_softness + threshold - pos)
    return torch.clamp(output, 0.0, 1.0).to(dtype)


class DCTResolution2D(nn.Module):
    def __init__(
        self,
        rates: Rate = (1.0, 1.0),
        smoothness_factor: float = 4.0,
        trainable: bool = True,
        shared_rate: bool = False,
        maximum_shape: Shape = (None, None),
        minimum_shape: Shape = (1, 1),
        maximum_rate: Rate = 2.0,
        minimum_rate: Rate = 0.0,
        constrain_max_shape: bool = False,
        **kwargs,
    ):
        super().__init__()

        self._smoothness_factor = torch.tensor(smoothness_factor, device="cuda")
        self._shared_rate = shared_rate
        self.trainable = trainable

        self.constrain_max_shape = constrain_max_shape
        if maximum_shape[0] is not None and maximum_shape[1] is not None:
            self.constrain_max_shape = True

        # If one element of the maximum shape is None, then all elements should be None
        if any([x is None for x in maximum_shape]) and not all(
            [x is None for x in maximum_shape]
        ):
            raise ValueError(
                "If one element of maximum_shape is None, then all elements should be "
                "None."
            )
        self._maximum_shape = maximum_shape

        # Ensure minimum shape is greater than equal to 1
        if minimum_shape[0] < 1 or minimum_shape[1] < 1:
            raise ValueError(
                f"Minimum shape should be >=1 but got {minimum_shape}"
            )
        self._minimum_shape = minimum_shape

        # Ensures a tuple of floats for the maximum and minimum rates
        maximum_rate = (
            (maximum_rate, maximum_rate)
            if isinstance(maximum_rate, (int, float))
            else maximum_rate
        )
        minimum_rate = (
            (minimum_rate, minimum_rate)
            if isinstance(minimum_rate, (int, float))
            else minimum_rate
        )

        # Validate max and min rates
        if maximum_rate[0] < 0 or maximum_rate[1] < 0:
            raise ValueError(
                f"Maximum rates should be >=0 but got {maximum_rate}"
            )
        if minimum_rate[0] < 0 or minimum_rate[1] < 0:
            raise ValueError(
                f"Minimum rates should be >=0 but got {minimum_rate}"
            )
        self._maximum_rate_height, self._maximum_rate_width = maximum_rate
        self._minimum_rate_height, self._minimum_rate_width = minimum_rate
        self._maximum_rate = torch.tensor(
            [maximum_rate[0], maximum_rate[1]],
            dtype=torch.float32,
            device="cuda",
        )
        self._minimum_rate = torch.tensor(
            [minimum_rate[0], minimum_rate[1]],
            dtype=torch.float32,
            device="cuda",
        )

        # Ensures a tuple of floats for the initial rates
        rates = (rates, rates) if isinstance(rates, (int, float)) else rates
        rates = tuple(map(float, rates))
        if rates[0] != rates[1] and shared_rate:
            raise ValueError(
                "shared_rate requires the same initialization for "
                f"vertical and horizontal rates but got {rates}"
            )
        if rates[0] < 0 or rates[1] < 0:
            raise ValueError(f"Both rates should be >=0 but got {rates}")
        if smoothness_factor < 0.0:
            raise ValueError(
                "Smoothness factor should be >= 0 but got "
                f"{smoothness_factor}."
            )
        self._rates = rates

        maximum_height, maximum_width = self._maximum_shape
        minimum_height, minimum_width = self._minimum_shape
        self._minimum_shape = torch.tensor(
            [minimum_height, minimum_width], dtype=torch.float32, device="cuda"
        )

        # Initialize trainable parameters if needed
        if self.trainable:
            self.rate_weights = nn.Parameter(
                torch.tensor(rates)
            )
        else:
            self.register_buffer(
                "rate_weights", torch.tensor(rates, device="cuda")
            )

    def _get_frequency_representation(self, x: torch.Tensor):
        return dct_2d(x, norm="ortho")

    def _get_inv_frequency_representation(self, x: torch.Tensor):
        return idct_2d(x, norm="ortho")

    def _get_positions(self, height, width):
        vertical_positions = torch.arange(height, dtype=torch.float32, device="cuda")
        horizontal_positions = torch.arange(width, dtype=torch.float32, device="cuda")

        return vertical_positions, horizontal_positions

    def _get_masks(
        self, crop_height, crop_width, vertical_positions, horizontal_positions
    ):
        vertical_mask = compute_adaptive_span_mask(
            crop_height,
            self._smoothness_factor,
            vertical_positions,
            dtype=torch.float32,
        )
        horizontal_mask = compute_adaptive_span_mask(
            crop_width,
            self._smoothness_factor,
            horizontal_positions,
            dtype=torch.float32,
        )

        return vertical_mask, horizontal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Get the shape of x
        *_, current_height, current_width = x.shape
        current_shape = torch.tensor(
            [current_height, current_width],
            dtype=torch.float32,
            device=x.device,
        )

        if self._maximum_shape[0] is None or self._maximum_shape[1] is None:
            self._maximum_shape = (
                torch.tensor(
                    [current_height, current_width],
                    dtype=torch.float32,
                    device=x.device,
                )
                * self._maximum_rate
            )
        else:
            self._maximum_shape = torch.tensor(
                [self._maximum_shape[0], self._maximum_shape[1]],
                dtype=torch.float32,
                device=x.device,
            )

        # Constrain the rates
        min_shape_rate = (
            self._minimum_shape - self._smoothness_factor
        ) / current_shape
        min_allowed_rate = torch.max(min_shape_rate, self._minimum_rate)

        max_allowed_rate = self._maximum_rate
        if self.constrain_max_shape:
            max_shape_rate = (
                self._maximum_shape - self._smoothness_factor
            ) / current_shape
            max_allowed_rate = torch.min(max_shape_rate, self._maximum_rate)

        # Constrain the rates
        if self._shared_rate:
            self.rate_weights.data = torch.clamp(
                self.rate_weights, min_allowed_rate[0], max_allowed_rate[0]
            )
        else:
            self.rate_weights.data[0] = torch.clamp(
                self.rate_weights[0], min_allowed_rate[0], max_allowed_rate[0]
            )
            self.rate_weights.data[1] = torch.clamp(
                self.rate_weights[1], min_allowed_rate[1], max_allowed_rate[1]
            )

        # Determine the new height and width
        new_shape_smooth = (
            current_shape * self.rate_weights + self._smoothness_factor
        )

        # Get the integer shape of the output
        new_shape = new_shape_smooth.to(torch.int32) + 1

        # Get the frequency representation
        frequency_inputs = self._get_frequency_representation(x)

        # If upsampling, zero pad the frequency representation
        frequency_inputs = F.pad(
            frequency_inputs,
            (
                0,
                max(new_shape[1] - current_width, 0),
                0,
                max(new_shape[0] - current_height, 0),
            ),
            "constant",
            0,
        )
        *_, new_height, new_width = frequency_inputs.shape

        # Configure positions for the masks
        vertical_positions, horizontal_positions = self._get_positions(
            new_height,
            new_width,
        )

        # Get the masks
        crop_shape = new_shape_smooth - self._smoothness_factor
        vertical_mask, horizontal_mask = self._get_masks(
            crop_shape[0],
            crop_shape[1],
            vertical_positions,
            horizontal_positions,
        )

        # Find the indices to keep
        with torch.no_grad():
            # Find the indices to keep
            horizontal_to_keep = torch.nonzero(horizontal_mask > 0.0).squeeze()
            vertical_to_keep = torch.nonzero(vertical_mask > 0.0).squeeze()
            new_height = torch.max(vertical_to_keep) + 1
            new_width = torch.max(horizontal_to_keep) + 1

        # Apply the masks
        output = (
            frequency_inputs
            * horizontal_mask[None, None, None, :]
            * vertical_mask[None, None, :, None]
        )

        # Crop
        output = output[..., :new_height, :new_width]

        # Return to spatial representation
        result = self._get_inv_frequency_representation(output)

        if not self.training:
            print(f"HxW for layer: {new_height}x{new_width}")

        return result


class ResolutionWithConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        dilation=1,
        bias=True,
        stride=1,
    ):

        super().__init__()

        rates: Rate = (1.0, 1.0)
        smoothness_factor: float = 4.0
        trainable: bool = True
        shared_rate: bool = False
        maximum_shape: Shape = (128, 128)
        minimum_shape: Shape = (1, 1)
        maximum_rate: Rate = 2.0
        minimum_rate: Rate = 0.0
        constrain_max_shape: bool = True

        self.dct = DCTResolution2D(
            rates,
            smoothness_factor,
            trainable,
            shared_rate,
            maximum_shape,
            minimum_shape,
            maximum_rate,
            minimum_rate,
            constrain_max_shape,
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = self.dct(x)
        x = self.conv(x)
        return x


class FixedDCTResizing(nn.Module):
    def __init__(
        self,
        output_shape: Shape,
        **kwargs,
    ):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, current_height, current_width = x.shape

        # Get the frequency representation
        frequency_inputs = dct_2d(x, norm="ortho")

        # If upsampling, zero pad the frequency representation
        frequency_inputs = F.pad(
            frequency_inputs,
            (
                0,
                max(self.output_shape[1] - current_width, 0),
                0,
                max(self.output_shape[0] - current_height, 0),
            ),
            "constant",
            0,
        )

        # Inverse DCT
        out_height, out_width = self.output_shape
        result = idct_2d(
            frequency_inputs[..., :out_height, :out_width], norm="ortho"
        )

        return result


class VariableDCTResizing(nn.Module):

    def forward(
        self, x: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        *_, current_height, current_width = x.shape

        # Get the frequency representation
        frequency_inputs = dct_2d(x, norm="ortho")

        # If upsampling, zero pad the frequency representation
        frequency_inputs = F.pad(
            frequency_inputs,
            (
                0,
                max(height - current_width, 0),
                0,
                max(width - current_height, 0),
            ),
            "constant",
            0,
        )

        # Inverse DCT
        result = idct_2d(frequency_inputs[..., :height, :width], norm="ortho")

        return result


class DCTConcat2D(nn.Module):
    def __init__(self, mode="max"):
        super().__init__()
        self.mode = mode
        self.variable_dct_resizing = VariableDCTResizing()

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if self.mode == "max":
            max_height = max(x.shape[-2] for x in args)
            max_width = max(x.shape[-1] for x in args)
        elif self.mode == "min":
            max_height = min(x.shape[-2] for x in args)
            max_width = min(x.shape[-1] for x in args)

        resized_tensors = [
            self.variable_dct_resizing(x, max_height, max_width) for x in args
        ]

        return torch.cat(resized_tensors, dim=1)
