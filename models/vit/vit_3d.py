import torch
import torch.nn.functional as F
from torch import nn


class PatchEmbedding3D(nn.Module):
    """Convierte un volumen 3D en una secuencia de embeddings de parches lineales."""

    def __init__(self, vol_size=96, patch_size=16, in_channels=1, embed_dim=768) -> None:
        super().__init__()
        self.vol_size = vol_size
        self.patch_size = patch_size
        self.n_patches = (vol_size // patch_size) ** 3

        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        assert D == H == W == self.vol_size, (
            f"Input volume size ({D}*{H}*{W}) doesn't match expected size ({self.vol_size}*{self.vol_size}*{self.vol_size})"
        )

        # [B, C, D, H, W] -> [B, embed_dim, D//patch_size, H//patch_size, W//patch_size]
        x = self.proj(x)
        # [B, embed_dim, D', H', W'] -> [B, embed_dim, D'*H'*W']
        x = x.flatten(2)
        # [B, embed_dim, D'*H'*W'] -> [B, D'*H'*W', embed_dim]
        return x.transpose(1, 2)


class MultiHeadAttention3D(nn.Module):
    """Multi-head Attention para Vision Transformer 3D."""

    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop=0.0, proj_drop=0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape

        # [B, N, 3*C] -> [B, N, 3, num_heads, C//num_heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # [B, 3, num_heads, N, C//num_heads]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # [B, num_heads, N, C//num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [B, num_heads, N, C//num_heads] -> [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class MLP3D(nn.Module):
    """MLP para Vision Transformer 3D."""

    def __init__(self, in_features, hidden_features, out_features, drop=0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TransformerBlock3D(nn.Module):
    """Bloque Transformer para ViT 3D."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.0,
        attn_p=0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention3D(
            embed_dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_p,
            proj_drop=p,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP3D(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            drop=p,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT_3D(nn.Module):
    """Vision Transformer para volumetrías médicas 3D."""

    def __init__(
        self,
        vol_size=96,
        patch_size=16,
        in_channels=1,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.0,
        attn_p=0.0,
    ) -> None:
        super().__init__()

        # Embedding de parches
        self.patch_embed = PatchEmbedding3D(
            vol_size=vol_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Token de clase [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Embedding de posición
        num_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=p)

        # Bloques Transformer
        self.blocks = nn.ModuleList(
            [
                TransformerBlock3D(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ],
        )

        # Normalización final
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Clasificador
        self.head = nn.Linear(embed_dim, num_classes)

        # Inicialización
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # [B, C, D, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        B, N, _ = x.shape

        # Añadir token CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, embed_dim]

        # Añadir embedding posicional
        x = (
            x + self.pos_embed[:, : x.size(1), :]
        )  # Puede requerir interpolación si el volumen cambia de tamaño
        x = self.pos_drop(x)

        # Aplicar bloques Transformer
        for block in self.blocks:
            x = block(x)

        # Normalización final
        x = self.norm(x)

        # Solo usamos el token CLS para clasificación
        return x[:, 0]

    def forward(self, x):
        # Asegurarse que el volumen tiene el tamaño correcto
        _, _, D, H, W = x.shape
        if (
            self.patch_embed.vol_size != D
            or self.patch_embed.vol_size != H
            or self.patch_embed.vol_size != W
        ):
            x = F.interpolate(
                x,
                size=(
                    self.patch_embed.vol_size,
                    self.patch_embed.vol_size,
                    self.patch_embed.vol_size,
                ),
                mode="trilinear",
                align_corners=False,
            )

        x = self.forward_features(x)
        return self.head(x)


def get_vit_3d(config):
    """Función para instanciar un modelo ViT 3D con configuración específica.

    Args:
        config (dict): Diccionario con parámetros de configuración

    Returns:
        ViT_3D: Modelo instanciado

    """
    num_classes = config.get("num_classes", 2)  # Por defecto binario CN/AD
    vol_size = config.get("vol_size", 96)  # Un tamaño típico para escaneos cerebrales 3D
    patch_size = config.get("patch_size", 16)
    embed_dim = config.get("embed_dim", 768)
    depth = config.get("depth", 12)
    num_heads = config.get("num_heads", 12)

    return ViT_3D(
        vol_size=vol_size,
        patch_size=patch_size,
        in_channels=1,  # PET scans
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )
