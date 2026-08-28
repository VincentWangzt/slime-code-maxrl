# Megatron architecture matching ``python -m maze.model``.
MODEL_ARGS=(
   --swiglu
   --num-layers 4
   --hidden-size 256
   --ffn-hidden-size 1024
   --num-attention-heads 4
   --group-query-attention
   --num-query-groups 2
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization RMSNorm
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 32
   --make-vocab-size-divisible-by 32
   --seq-length 512
   --max-position-embeddings 512
)
