from typing import Union, Optional, Dict, Any, Tuple, List

import torch
from diffusers.utils.torch_utils import apply_freeu
from diffusers.models.controlnets import ControlNetXSOutput
from diffusers import UNetControlNetXSModel, UNet2DConditionModel
from diffusers.models.controlnets.controlnet_xs import ControlNetXSAdapter, ControlNetXSCrossAttnUpBlock2D


class TensorControlNetXSCrossAttnUpBlock2D(ControlNetXSCrossAttnUpBlock2D):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple_base: Tuple[torch.Tensor, ...],
            res_hidden_states_tuple_ctrl: Tuple[torch.Tensor, ...],
            temb: torch.Tensor,
            conditioning_scale: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            upsample_size: Optional[int] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            apply_control: bool = True,
    ) -> torch.Tensor:

        if cross_attention_kwargs is not None and "scale" in cross_attention_kwargs:
            cross_attention_kwargs = dict(cross_attention_kwargs)
            cross_attention_kwargs.pop("scale", None)

        is_freeu_enabled = (
                getattr(self, "s1", None) is not None
                and getattr(self, "s2", None) is not None
                and getattr(self, "b1", None) is not None
                and getattr(self, "b2", None) is not None
        )

        def maybe_apply_freeu_to_subblock(hidden_states, res_h_base):
            if is_freeu_enabled:
                return apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_h_base,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )
            else:
                return hidden_states, res_h_base

        num_layers = len(self.resnets)
        for i in range(num_layers):
            resnet = self.resnets[i]
            attn = self.attentions[i]
            c2b = self.ctrl_to_base[i]

            reverse_idx = num_layers - 1 - i
            res_h_base = res_hidden_states_tuple_base[reverse_idx]
            res_h_ctrl = res_hidden_states_tuple_ctrl[reverse_idx]

            if apply_control:
                hidden_states += c2b(res_h_ctrl) * conditioning_scale

            hidden_states, res_h_base = maybe_apply_freeu_to_subblock(hidden_states, res_h_base)
            hidden_states = torch.cat([hidden_states, res_h_base], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            if attn is not None:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            hidden_states = self.upsamplers(hidden_states, upsample_size)

        return hidden_states


class TensorUNetControlNetXSModel(UNetControlNetXSModel):

    @classmethod
    def from_unet(
            cls,
            unet: "UNet2DConditionModel",
            controlnet: Optional["ControlNetXSAdapter"] = None,
            size_ratio: Optional[float] = None,
            ctrl_block_out_channels: Optional[List[float]] = None,
            time_embedding_mix: Optional[float] = None,
            ctrl_optional_kwargs: Optional[Dict] = None,
    ):
        model = super().from_unet(
            unet=unet,
            controlnet=controlnet,
            size_ratio=size_ratio,
            ctrl_block_out_channels=ctrl_block_out_channels,
            time_embedding_mix=time_embedding_mix,
            ctrl_optional_kwargs=ctrl_optional_kwargs,
        )

        for up_block in model.up_blocks:
            up_block.forward = TensorControlNetXSCrossAttnUpBlock2D.forward.__get__(up_block, up_block.__class__)

        return model

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            conditioning_scale: torch.Tensor,
            controlnet_cond: Optional[torch.Tensor] = None,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            return_dict: bool = True,
            apply_control: bool = True,
    ) -> Union[ControlNetXSOutput, Tuple]:

        if cross_attention_kwargs is not None and "scale" in cross_attention_kwargs:
            cross_attention_kwargs = dict(cross_attention_kwargs)
            cross_attention_kwargs.pop("scale", None)

        if self.config.ctrl_conditioning_channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.base_time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.config.ctrl_learn_time_embedding and apply_control:
            ctrl_temb = self.ctrl_time_embedding(t_emb, timestep_cond)
            base_temb = self.base_time_embedding(t_emb, timestep_cond)
            interpolation_param = self.config.time_embedding_mix ** 0.3
            temb = ctrl_temb * interpolation_param + base_temb * (1 - interpolation_param)
        else:
            temb = self.base_time_embedding(t_emb)

        aug_emb = None
        if self.config.addition_embed_type is None:
            pass
        elif self.config.addition_embed_type == "text_time":
            if added_cond_kwargs is None:
                raise ValueError(f"{self.__class__} has addition_embed_type='text_time' but added_cond_kwargs is None")
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(f"{self.__class__} requires 'text_embeds' in added_cond_kwargs")
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(f"{self.__class__} requires 'time_ids' in added_cond_kwargs")
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.base_add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(temb.dtype)
            aug_emb = self.base_add_embedding(add_embeds)
        else:
            raise ValueError(f"addition_embed_type={self.config.addition_embed_type} not supported")

        temb = temb + aug_emb if aug_emb is not None else temb
        cemb = encoder_hidden_states

        h_ctrl = h_base = sample
        guided_hint = self.controlnet_cond_embedding(controlnet_cond)

        h_base = self.base_conv_in(h_base)
        h_ctrl = self.ctrl_conv_in(h_ctrl)
        if guided_hint is not None:
            h_ctrl += guided_hint
        if apply_control:
            h_base = h_base + self.control_to_base_for_conv_in(h_ctrl) * conditioning_scale

        skip_connections_base = [h_base]
        skip_connections_ctrl = [h_ctrl]

        for down in self.down_blocks:
            h_base, h_ctrl, residual_hb, residual_hc = down(
                hidden_states_base=h_base,
                hidden_states_ctrl=h_ctrl,
                temb=temb,
                encoder_hidden_states=cemb,
                conditioning_scale=conditioning_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                apply_control=apply_control,
            )
            for i in range(len(residual_hb)):
                skip_connections_base.append(residual_hb[i])
            for i in range(len(residual_hc)):
                skip_connections_ctrl.append(residual_hc[i])

        h_base, h_ctrl = self.mid_block(
            hidden_states_base=h_base,
            hidden_states_ctrl=h_ctrl,
            temb=temb,
            encoder_hidden_states=cemb,
            conditioning_scale=conditioning_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            attention_mask=attention_mask,
            apply_control=apply_control,
        )

        total_skips = len(skip_connections_base)
        skip_idx = total_skips

        for up in self.up_blocks:
            n_resnets = len(up.resnets)

            skips_hb = []
            skips_hc = []
            for i in range(n_resnets):
                idx = skip_idx - n_resnets + i
                skips_hb.append(skip_connections_base[idx])
                skips_hc.append(skip_connections_ctrl[idx])

            skip_idx = skip_idx - n_resnets

            h_base = up(
                hidden_states=h_base,
                res_hidden_states_tuple_base=tuple(skips_hb),
                res_hidden_states_tuple_ctrl=tuple(skips_hc),
                temb=temb,
                encoder_hidden_states=cemb,
                conditioning_scale=conditioning_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                apply_control=apply_control,
            )

        h_base = self.base_conv_norm_out(h_base)
        h_base = self.base_conv_act(h_base)
        h_base = self.base_conv_out(h_base)

        if not return_dict:
            return (h_base,)

        return ControlNetXSOutput(sample=h_base)
