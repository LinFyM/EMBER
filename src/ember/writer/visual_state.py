"""Anchored, non-recursive visual state for PI05 teacher forecasts."""

from __future__ import annotations

import torch

from ember.pi05_processing import PI05_DIGIT_TOKEN_IDS
from ember.writer.temporal import RMSNorm


class VisualStateError(RuntimeError):
    """Raised when the sealed visual-state interface changes."""


def frozen_digit_embedding_basis(
    paligemma_model: torch.nn.Module,
    *,
    image_width: int,
) -> torch.Tensor:
    """Read only the ten native digit embeddings from frozen PaliGemma."""

    if not hasattr(paligemma_model, "get_input_embeddings"):
        raise VisualStateError("PaliGemma input embedding interface changed")
    embedding = paligemma_model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    device = weight.device if isinstance(weight, torch.Tensor) else torch.device("cpu")
    token_ids = torch.tensor(
        PI05_DIGIT_TOKEN_IDS,
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        digits = embedding(token_ids).detach().to(dtype=torch.float32)
    if digits.shape != (10, image_width):
        raise VisualStateError("PaliGemma digit embeddings changed shape")
    # Centering changes neither the spanned subspace nor its tokenizer owner.
    return digits - digits.mean(dim=0, keepdim=True)


class RoutedCoordinateReader(torch.nn.Module):
    """Read eight scalars while routing identities never enter value content."""

    COORDINATES = 8

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        key_width: int,
        value_width: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            min(width, heads, key_width, value_width) <= 0
            or width % heads
        ):
            raise VisualStateError("invalid coordinate-reader dimensions")
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        routing = torch.empty(self.COORDINATES, width, dtype=torch.float32)
        routing.normal_(mean=0.0, std=0.02, generator=generator)
        self.routing = torch.nn.Parameter(routing)
        self.routing_norm = RMSNorm(width)
        self.key_norm = RMSNorm(key_width)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
            kdim=key_width,
            vdim=value_width,
        )
        self.ffn_norm = RMSNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(4 * width, width, bias=False),
        )
        self.coordinate_head = torch.nn.Linear(width, 1, bias=False)
        # Exact native-state initialization.  The nonzero content path and
        # renderer basis give this gate a gradient as soon as downstream
        # identity heads open; deeper reader weights then receive gradients.
        self.output_gate = torch.nn.Parameter(
            torch.zeros(self.COORDINATES, dtype=torch.float32)
        )

    def forward(
        self,
        key_context: torch.Tensor,
        value_content: torch.Tensor,
    ) -> torch.Tensor:
        if (
            key_context.ndim != 3
            or value_content.ndim != 3
            or key_context.shape[:2] != value_content.shape[:2]
        ):
            raise VisualStateError("coordinate-reader memory changed shape")
        routing = self.routing_norm(self.routing)[None].expand(
            key_context.shape[0],
            -1,
            -1,
        )
        content, _ = self.attention(
            routing,
            self.key_norm(key_context),
            value_content,
            need_weights=False,
        )
        content = content + self.ffn(self.ffn_norm(content))
        raw = self.coordinate_head(content).squeeze(-1)
        return raw * self.output_gate.to(dtype=raw.dtype)


class NativeStateRenderer(torch.nn.Module):
    """Render eight scalars into offsets for the 32 native state tokens."""

    COORDINATES = 8
    TOKENS_PER_COORDINATE = 4
    DIGIT_POSITIONS = 3
    DIGIT_COUNT = 10

    def __init__(
        self,
        *,
        digit_basis: torch.Tensor,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            digit_basis.ndim != 2
            or digit_basis.shape[0] != self.DIGIT_COUNT
            or digit_basis.shape[1] <= 0
        ):
            raise VisualStateError("invalid native digit embedding basis")
        self.image_width = int(digit_basis.shape[1])
        self.register_buffer(
            "digit_basis",
            digit_basis.detach().to(dtype=torch.float32).contiguous(),
            persistent=True,
        )
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        coefficients = torch.empty(
            self.COORDINATES,
            self.DIGIT_POSITIONS,
            self.DIGIT_COUNT,
            dtype=torch.float32,
        )
        coefficients.normal_(mean=0.0, std=0.02, generator=generator)
        self.coefficients = torch.nn.Parameter(coefficients)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != self.COORDINATES
        ):
            raise VisualStateError("visual-state coordinates changed shape")
        coefficients = self.coefficients.to(dtype=coordinates.dtype)
        basis = self.digit_basis.to(dtype=coordinates.dtype)
        digit_offsets = torch.einsum(
            "bd,djn,nw->bdjw",
            coordinates,
            coefficients,
            basis,
        )
        offsets = digit_offsets.new_zeros(
            coordinates.shape[0],
            self.COORDINATES,
            self.TOKENS_PER_COORDINATE,
            self.image_width,
        )
        offsets[:, :, 1:] = digit_offsets
        return offsets.reshape(coordinates.shape[0], 32, self.image_width)


class AnchoredVisualState(torch.nn.Module):
    """Initial anchor plus per-frame anchor/local change, without recurrence."""

    COORDINATES = 8
    STATE_SLOTS = 32

    def __init__(
        self,
        *,
        image_width: int,
        state_width: int,
        heads: int,
        digit_basis: torch.Tensor,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            min(image_width, state_width, heads) <= 0
            or state_width % heads
        ):
            raise VisualStateError("invalid anchored visual-state dimensions")
        self.image_width = int(image_width)
        self.state_width = int(state_width)
        self.image_projection = torch.nn.Linear(
            image_width,
            state_width,
            bias=False,
        )
        self.initial_reader = RoutedCoordinateReader(
            width=state_width,
            heads=heads,
            key_width=state_width,
            value_width=state_width,
            initialization_seed=initialization_seed,
        )
        self.change_key_projection = torch.nn.Linear(
            4 * state_width,
            state_width,
            bias=False,
        )
        self.change_value_projection = torch.nn.Linear(
            2 * state_width,
            state_width,
            bias=False,
        )
        self.change_reader = RoutedCoordinateReader(
            width=state_width,
            heads=heads,
            key_width=state_width,
            value_width=state_width,
            initialization_seed=initialization_seed + 1,
        )
        self.renderer = NativeStateRenderer(
            digit_basis=digit_basis,
            initialization_seed=initialization_seed + 2,
        )

    def _validate_images(
        self,
        current: torch.Tensor,
        anchor: torch.Tensor,
        previous: torch.Tensor,
    ) -> None:
        if (
            current.ndim != 3
            or current.shape != anchor.shape
            or current.shape != previous.shape
            or current.shape[1] <= 1
            or current.shape[2] != self.image_width
        ):
            raise VisualStateError("visual-state image tokens changed shape")

    def encode_coordinates(
        self,
        current: torch.Tensor,
        anchor: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        """Return bounded state-and-motion coordinates ``[B,8]``."""

        self._validate_images(current, anchor, previous)
        projected_anchor = self.image_projection(anchor)
        projected_previous = self.image_projection(previous)
        projected_current = self.image_projection(current)

        initial = self.initial_reader(projected_anchor, projected_anchor)
        anchor_delta = projected_current - projected_anchor
        local_delta = projected_current - projected_previous
        key_context = self.change_key_projection(
            torch.cat(
                (
                    0.5 * (projected_anchor + projected_current),
                    anchor_delta.abs(),
                    0.5 * (projected_previous + projected_current),
                    local_delta.abs(),
                ),
                dim=-1,
            )
        )
        signed_values = self.change_value_projection(
            torch.cat((anchor_delta, local_delta), dim=-1)
        )
        change = self.change_reader(key_context, signed_values)
        return torch.tanh(initial + change)

    def forward(
        self,
        current: torch.Tensor,
        anchor: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        coordinates = self.encode_coordinates(current, anchor, previous)
        return self.renderer(coordinates)
