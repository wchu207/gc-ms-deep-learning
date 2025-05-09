from torch import nn
from mae.models_mae import MaskedAutoencoderViT

class MaskedAutoencoder(MaskedAutoencoderViT):
    def __init__(self, img_size=(4096, 512), patch_size=(128, 16), in_chans=1,
                 embed_dim=1024, depth=16, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.ReLU, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans,
                         embed_dim, depth, num_heads, decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, act_layer, norm_pix_loss)